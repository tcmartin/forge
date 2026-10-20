//! Transport-side bounds owned by Forge rather than inherited from rmcp.
//!
//! rmcp's stdio reader uses `read_until` with an unlimited line buffer and its
//! SSE helpers accumulate an unlimited event.  These adapters enforce limits
//! before those buffers can grow past the MCP catalog budget.

use crate::raw_mcp_observer::{ObservedReader, ObservedWriter, RawMcpObserver};
use futures::StreamExt;
use rmcp::service::{RoleClient, RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::{async_rw::AsyncRwTransport, Transport};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::VecDeque;
use std::pin::Pin;
use std::process::Stdio;
use std::task::{Context, Poll};
use std::{future::Future, io};
use tokio::io::{AsyncRead, ReadBuf};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

pub(crate) const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_SSE_LINE_BYTES: usize = 256 * 1024;
pub(crate) const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_HTTP_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_HTTP_SSE_EVENTS: usize = 4096;

// Child teardown is finite. Raw receipts are flushed by the owning backend
// even when one of these bounds fails, preserving observed partial frames.
const CHILD_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);
const CHILD_GROUP_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1000);
const CHILD_CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const CHILD_GROUP_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

fn quarantine_admission(permit: Option<std::sync::Arc<tokio::sync::OwnedSemaphorePermit>>) {
    // A failed or unproven cleanup must not return capacity to the admission
    // pool. Leaking this one permit is deliberate fail-closed quarantine; the
    // process may still be in an unknown state and must not admit another child.
    if let Some(permit) = permit {
        std::mem::forget(permit);
    }
}

fn limit_error(kind: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, kind)
}

/// Reject the one resource shape that rmcp's untagged ResourceContents enum
/// cannot distinguish safely: an object carrying both `text` and `blob`.
/// Check the raw JSON before rmcp deserializes it, otherwise serde accepts the
/// text variant and silently drops the blob field.
pub(crate) fn reject_ambiguous_resource_payload(value: &Value) -> Result<(), &'static str> {
    let Some(result) = value.get("result") else {
        return Ok(());
    };
    if let Some(contents) = result.get("contents").and_then(Value::as_array) {
        reject_resource_array(contents)?;
    }
    if let Some(content) = result.get("content").and_then(Value::as_array) {
        for item in content {
            if item.get("type").and_then(Value::as_str) == Some("resource") {
                if let Some(resource) = item.get("resource") {
                    reject_resource_object(resource)?;
                }
            }
        }
    }
    Ok(())
}

fn reject_resource_array(contents: &[Value]) -> Result<(), &'static str> {
    for resource in contents {
        reject_resource_object(resource)?;
    }
    Ok(())
}

fn reject_resource_object(resource: &Value) -> Result<(), &'static str> {
    let Some(object) = resource.as_object() else {
        return Ok(());
    };
    if object.contains_key("text") && object.contains_key("blob") {
        return Err("MCP resource contains both text and blob payloads");
    }
    Ok(())
}

fn parse_bounded_message<T>(data: &str) -> anyhow::Result<T>
where
    T: DeserializeOwned,
{
    let raw: Value = serde_json::from_str(data)?;
    reject_ambiguous_resource_payload(&raw).map_err(|error| anyhow::anyhow!(error))?;
    Ok(serde_json::from_value(raw)?)
}

/// An AsyncRead that rejects a JSON-RPC line before rmcp's `read_until`
/// buffer can exceed `MAX_FRAME_BYTES`.
pub(crate) struct BoundedLineReader<R> {
    inner: R,
    line_bytes: usize,
    validation_frame: Vec<u8>,
    pending: VecDeque<u8>,
}

impl<R> BoundedLineReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self {
            inner,
            line_bytes: 0,
            validation_frame: Vec::new(),
            pending: VecDeque::new(),
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for BoundedLineReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        destination: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if destination.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        if !self.pending.is_empty() {
            let count = destination.remaining().min(self.pending.len());
            for byte in self.pending.drain(..count) {
                destination.put_slice(&[byte]);
            }
            return Poll::Ready(Ok(()));
        }
        let mut chunk = [0u8; 8192];
        let mut read = ReadBuf::new(&mut chunk);
        match Pin::new(&mut self.inner).poll_read(cx, &mut read) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => {
                if read.filled().is_empty() {
                    let frame = std::mem::take(&mut self.validation_frame);
                    if !frame.is_empty() {
                        if let Ok(raw) = serde_json::from_slice::<Value>(&frame) {
                            if let Err(error) = reject_ambiguous_resource_payload(&raw) {
                                return Poll::Ready(Err(limit_error(error)));
                            }
                        }
                    }
                    return Poll::Ready(Ok(()));
                }
                for &value in read.filled() {
                    if value != b'\n' && self.line_bytes >= MAX_FRAME_BYTES {
                        return Poll::Ready(Err(limit_error(
                            "MCP JSON-RPC frame exceeds byte limit",
                        )));
                    }
                    self.pending.push_back(value);
                    if value == b'\n' {
                        let frame = std::mem::take(&mut self.validation_frame);
                        if !frame.is_empty() {
                            if let Ok(raw) = serde_json::from_slice::<Value>(&frame) {
                                if let Err(error) = reject_ambiguous_resource_payload(&raw) {
                                    return Poll::Ready(Err(limit_error(error)));
                                }
                            }
                        }
                        self.line_bytes = 0;
                    } else {
                        self.validation_frame.push(value);
                        self.line_bytes += 1;
                    }
                }
                let count = destination.remaining().min(self.pending.len());
                for byte in self.pending.drain(..count) {
                    destination.put_slice(&[byte]);
                }
                Poll::Ready(Ok(()))
            }
        }
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

#[cfg(unix)]
fn signal_owned_group(group: i32, signal: i32) -> io::Result<()> {
    if group <= 1 {
        return Err(io::Error::other("Invalid owned MCP process group"));
    }
    // SAFETY: POSIX kill takes integer arguments, not pointers. This private
    // call uses only the positive PID captured from a child we spawned with
    // process_group(0); negating it addresses that owned group, never group0.
    if unsafe { kill(-group, signal) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Child process transport with bounded stdout framing and owned shutdown.
pub(crate) struct BoundedChildProcess {
    child: Option<Child>,
    admission: Option<std::sync::Arc<tokio::sync::OwnedSemaphorePermit>>,
    #[cfg(unix)]
    group: i32,
    transport: AsyncRwTransport<
        RoleClient,
        ObservedReader<BoundedLineReader<ChildStdout>>,
        ObservedWriter<ChildStdin>,
    >,
    observer: RawMcpObserver,
}

impl BoundedChildProcess {
    #[cfg(test)]
    pub(crate) fn spawn(command: Command) -> io::Result<Self> {
        Self::spawn_admitted(command, None)
    }

    #[cfg(test)]
    pub(crate) fn spawn_admitted(
        command: Command,
        admission: Option<std::sync::Arc<tokio::sync::OwnedSemaphorePermit>>,
    ) -> io::Result<Self> {
        Self::spawn_admitted_with_raw_observation(command, admission, false)
    }

    pub(crate) fn spawn_admitted_with_raw_observation(
        mut command: Command,
        admission: Option<std::sync::Arc<tokio::sync::OwnedSemaphorePermit>>,
        capture_raw: bool,
    ) -> io::Result<Self> {
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::inherit());
        command.kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn()?;
        #[cfg(unix)]
        let group = match child
            .id()
            .and_then(|pid| i32::try_from(pid).ok())
            .filter(|pid| *pid > 1)
        {
            Some(group) => group,
            None => {
                let _ = child.start_kill();
                quarantine_admission(admission);
                return Err(io::Error::other("MCP child has no owned process group"));
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                #[cfg(unix)]
                let _ = signal_owned_group(group, 9);
                let _ = child.start_kill();
                quarantine_admission(admission);
                return Err(io::Error::other("MCP child stdout was not piped"));
            }
        };
        let stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                #[cfg(unix)]
                let _ = signal_owned_group(group, 9);
                let _ = child.start_kill();
                quarantine_admission(admission);
                return Err(io::Error::other("MCP child stdin was not piped"));
            }
        };
        let observer = if capture_raw {
            RawMcpObserver::new()
        } else {
            RawMcpObserver::disabled()
        };
        Ok(Self {
            child: Some(child),
            admission,
            #[cfg(unix)]
            group,
            transport: AsyncRwTransport::new(
                ObservedReader::new(BoundedLineReader::new(stdout), observer.clone()),
                ObservedWriter::new(stdin, observer.clone()),
            ),
            observer,
        })
    }

    pub(crate) fn raw_observer(&self) -> RawMcpObserver {
        self.observer.clone()
    }

    /// Start before the close future's first poll. The cleanup task retains the
    /// admission permit until child reaping and complete group disappearance are
    /// both proved. Failure paths quarantine it instead of reopening capacity.
    fn begin_shutdown(&mut self) -> Option<tokio::task::JoinHandle<io::Result<()>>> {
        let mut child = self.child.take()?;
        let admission = self.admission.take();
        #[cfg(unix)]
        let group = self.group;
        // Signal before spawning cleanup: runtime shutdown may drop an
        // unpolled cleanup task, but owned descendants must still be stopped.
        #[cfg(unix)]
        let signal_error = signal_owned_group(group, 9).err(); // POSIX SIGKILL.
        let start_kill_error = child.start_kill().err();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            Some(handle.spawn(async move {
                #[cfg(unix)]
                if let Some(error) = signal_error {
                    // ESRCH is already the desired terminal state; any other
                    // signal failure is retained as an ownership/cleanup error.
                    if error.raw_os_error() != Some(3) {
                        quarantine_admission(admission);
                        return Err(io::Error::other(format!(
                            "MCP owned group kill failed: {error}"
                        )));
                    }
                }
                if let Some(error) = start_kill_error {
                    if error.kind() != io::ErrorKind::InvalidInput {
                        quarantine_admission(admission);
                        return Err(io::Error::other(format!("MCP child kill failed: {error}")));
                    }
                }
                let wait_result = tokio::time::timeout(CHILD_REAP_TIMEOUT, child.wait()).await;
                match wait_result {
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => {
                        quarantine_admission(admission);
                        return Err(io::Error::other(format!("MCP child reap failed: {error}")));
                    }
                    Err(_) => {
                        if let Err(error) = child.start_kill() {
                            if error.kind() != io::ErrorKind::InvalidInput {
                                quarantine_admission(admission);
                                return Err(io::Error::other(format!(
                                    "MCP child kill retry failed: {error}"
                                )));
                            }
                        }
                        match tokio::time::timeout(CHILD_REAP_TIMEOUT, child.wait()).await {
                            Ok(Ok(_)) => {}
                            Ok(Err(error)) => {
                                quarantine_admission(admission);
                                return Err(io::Error::other(format!(
                                    "MCP child reap failed after retry: {error}"
                                )));
                            }
                            Err(_) => {
                                quarantine_admission(admission);
                                return Err(io::Error::new(
                                    io::ErrorKind::TimedOut,
                                    "MCP child reap timed out",
                                ));
                            }
                        }
                    }
                }
                #[cfg(unix)]
                {
                    let deadline = std::time::Instant::now() + CHILD_GROUP_REAP_TIMEOUT;
                    loop {
                        match signal_owned_group(group, 0) {
                            // ESRCH means every member of the owned group is gone.
                            Err(error) if error.raw_os_error() == Some(3) => break,
                            Ok(()) if std::time::Instant::now() < deadline => {
                                tokio::time::sleep(CHILD_GROUP_PROBE_INTERVAL).await;
                            }
                            Ok(()) => {
                                quarantine_admission(admission);
                                return Err(io::Error::new(
                                    io::ErrorKind::TimedOut,
                                    "MCP owned process group did not drain",
                                ));
                            }
                            Err(error) => {
                                quarantine_admission(admission);
                                return Err(io::Error::other(format!(
                                    "MCP owned process group probe failed: {error}"
                                )));
                            }
                        }
                    }
                }
                // Only this path returns capacity to the pool: child reap and
                // complete owned process-group disappearance were both proved.
                drop(admission);
                Ok(())
            }))
        } else {
            // There is no executor to finish ownership proof. Keep capacity
            // quarantined rather than allowing another untracked child.
            quarantine_admission(admission);
            None
        }
    }
}

impl Transport<RoleClient> for BoundedChildProcess {
    type Error = io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.transport.send(item)
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleClient>>> + Send {
        self.transport.receive()
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        let cleanup = self.begin_shutdown();
        let transport = &mut self.transport;
        async move {
            // Bound the transport close itself. The R2 order awaited this
            // future without a deadline, so a blocked stdin/stdout close could
            // bypass the finite child cleanup contract.
            let transport_result =
                match tokio::time::timeout(CHILD_CLOSE_TIMEOUT, transport.close()).await {
                    Ok(result) => result,
                    Err(_) => Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "MCP transport close deadline exceeded",
                    )),
                };
            let cleanup_result = if let Some(mut cleanup) = cleanup {
                match tokio::time::timeout(CHILD_CLOSE_TIMEOUT, &mut cleanup).await {
                    Ok(Ok(result)) => result,
                    Ok(Err(_)) => Err(io::Error::other("MCP child cleanup task failed")),
                    Err(_) => {
                        // Do not abort: the detached task still owns the
                        // permit and may finish its bounded reaping proof.
                        // Aborting here would drop the permit on an unknown
                        // process state and reopen admission unsafely.
                        Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "MCP child cleanup deadline exceeded",
                        ))
                    }
                }
            } else {
                Err(io::Error::other("MCP child cleanup could not start"))
            };
            match (transport_result, cleanup_result) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(transport), Ok(())) => Err(transport),
                (Ok(()), Err(cleanup)) => Err(cleanup),
                (Err(transport), Err(cleanup)) => Err(io::Error::other(format!(
                    "MCP transport close failed: {transport}; child cleanup failed: {cleanup}"
                ))),
            }
        }
    }
}

impl Drop for BoundedChildProcess {
    fn drop(&mut self) {
        drop(self.begin_shutdown());
    }
}

/// Bounded SSE parser. Event and line storage are capped before UTF-8 or JSON
/// conversion, while an otherwise valid long-lived notification stream remains
/// usable across many finite events.
pub(crate) struct BoundedSseParser {
    line: Vec<u8>,
    event: Option<String>,
    event_id: Option<String>,
    data: Vec<String>,
    event_bytes: usize,
    field_count: usize,
    saw_cr: bool,
}

impl BoundedSseParser {
    pub(crate) fn new() -> Self {
        Self {
            line: Vec::new(),
            event: None,
            event_id: None,
            data: Vec::new(),
            event_bytes: 0,
            field_count: 0,
            saw_cr: false,
        }
    }

    pub(crate) fn push<F>(&mut self, chunk: &[u8], on_event: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(Option<&str>, Option<&str>, &str) -> anyhow::Result<()>,
    {
        for byte in chunk {
            if self.saw_cr {
                self.saw_cr = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            if *byte == b'\n' {
                self.finish_line(on_event)?;
            } else if *byte == b'\r' {
                self.finish_line(on_event)?;
                self.saw_cr = true;
            } else {
                if self.line.len() >= MAX_SSE_LINE_BYTES {
                    anyhow::bail!("MCP SSE line exceeds byte limit");
                }
                if self.event_bytes.saturating_add(self.line.len() + 2) > MAX_SSE_EVENT_BYTES {
                    anyhow::bail!("MCP SSE event exceeds byte limit");
                }
                self.line.push(*byte);
            }
        }
        Ok(())
    }

    pub(crate) fn finish<F>(&mut self, on_event: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(Option<&str>, Option<&str>, &str) -> anyhow::Result<()>,
    {
        if !self.line.is_empty() {
            self.finish_line(on_event)?;
        }
        self.dispatch(on_event)
    }

    fn finish_line<F>(&mut self, on_event: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(Option<&str>, Option<&str>, &str) -> anyhow::Result<()>,
    {
        if !self.line.is_empty() {
            self.event_bytes = self
                .event_bytes
                .checked_add(self.line.len() + 1)
                .ok_or_else(|| anyhow::anyhow!("MCP SSE event exceeds byte limit"))?;
            if self.event_bytes > MAX_SSE_EVENT_BYTES {
                anyhow::bail!("MCP SSE event exceeds byte limit");
            }
        }
        let line = String::from_utf8(std::mem::take(&mut self.line))
            .map_err(|_| anyhow::anyhow!("MCP SSE line is not UTF-8"))?;
        if line.is_empty() {
            self.dispatch(on_event)
        } else {
            self.process_line(&line)
        }
    }

    fn process_line(&mut self, line: &str) -> anyhow::Result<()> {
        if line.starts_with(':') {
            return Ok(());
        }
        self.field_count = self
            .field_count
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("MCP SSE field count exceeds limit"))?;
        if self.field_count > 4096 {
            anyhow::bail!("MCP SSE field count exceeds limit");
        }
        let (field, value) = line
            .split_once(':')
            .map(|(field, value)| (field, value.strip_prefix(' ').unwrap_or(value)))
            .unwrap_or((line, ""));
        match field {
            "event" => self.event = Some(value.to_owned()),
            "id" if !value.contains('\0') => self.event_id = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        Ok(())
    }

    fn dispatch<F>(&mut self, on_event: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(Option<&str>, Option<&str>, &str) -> anyhow::Result<()>,
    {
        if self.event.is_none() && self.data.is_empty() {
            return Ok(());
        }
        let data = self.data.join("\n");
        on_event(self.event.as_deref(), self.event_id.as_deref(), &data)?;
        self.event = None;
        self.event_id = None;
        self.data.clear();
        self.event_bytes = 0;
        self.field_count = 0;
        Ok(())
    }
}

pub(crate) async fn read_response_body(
    response: reqwest::Response,
    limit: usize,
) -> anyhow::Result<String> {
    read_response_body_with_timeout(response, limit, std::time::Duration::from_secs(10)).await
}

async fn read_response_body_with_timeout(
    response: reqwest::Response,
    limit: usize,
    timeout: std::time::Duration,
) -> anyhow::Result<String> {
    tokio::time::timeout(timeout, async move {
        let mut stream = response.bytes_stream();
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|error| anyhow::anyhow!("failed reading HTTP body: {error}"))?;
            if body
                .len()
                .checked_add(chunk.len())
                .is_none_or(|size| size > limit)
            {
                anyhow::bail!("MCP HTTP response body exceeds byte limit");
            }
            body.extend_from_slice(&chunk);
        }
        String::from_utf8(body).map_err(|_| anyhow::anyhow!("MCP HTTP response body is not UTF-8"))
    })
    .await
    .map_err(|_| anyhow::anyhow!("MCP HTTP response body timed out"))?
}

#[cfg(test)]
struct StartGetGate {
    checked: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[cfg(not(test))]
struct StartGetGate;

struct GetReaderSlot {
    closing: bool,
    handle: Option<tokio::task::JoinHandle<()>>,
    start_gate: Option<std::sync::Arc<StartGetGate>>,
}

const GET_READER_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const GET_READER_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);

struct GetReaderJoinGuard {
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl GetReaderJoinGuard {
    fn new(handle: Option<tokio::task::JoinHandle<()>>) -> Self {
        Self { handle }
    }

    async fn join(mut self) -> io::Result<()> {
        if self.handle.is_none() {
            return Ok(());
        }
        let join_result = tokio::time::timeout(
            GET_READER_JOIN_TIMEOUT,
            self.handle.as_mut().expect("reader handle checked above"),
        )
        .await;
        match join_result {
            Ok(Ok(())) => {
                self.handle.take();
                Ok(())
            }
            Ok(Err(_)) => {
                self.handle.take();
                Err(io::Error::other("MCP HTTP GET reader teardown failed"))
            }
            Err(_) => {
                self.handle
                    .as_mut()
                    .expect("reader handle remains after join timeout")
                    .abort();
                let reap_result = tokio::time::timeout(
                    GET_READER_REAP_TIMEOUT,
                    self.handle
                        .as_mut()
                        .expect("reader handle remains after abort"),
                )
                .await;
                if reap_result.is_err() {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "MCP HTTP GET reader teardown unresolved after abort",
                    ));
                }
                self.handle.take();
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "MCP HTTP GET reader join timed out",
                ))
            }
        }
    }
}

impl Drop for GetReaderJoinGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn join_get_reader(reader: Option<tokio::task::JoinHandle<()>>) -> io::Result<()> {
    GetReaderJoinGuard::new(reader).join().await
}

fn abort_get_reader_on_drop(reader_slot: &std::sync::Arc<std::sync::Mutex<GetReaderSlot>>) {
    if let Ok(mut slot) = reader_slot.lock() {
        slot.closing = true;
        if let Some(reader) = slot.handle.take() {
            reader.abort();
        }
    }
}

struct StreamableHttpState {
    client: reqwest::Client,
    endpoint: String,
    auth_token: Option<String>,
    session_id: Option<String>,
    protocol_version: Option<String>,
    cancellation: CancellationToken,
    fatal: CancellationToken,
    get_started: bool,
}

/// Forge-owned streamable HTTP transport.  rmcp's reqwest adapter delegates
/// SSE framing to an unbounded third-party stream parser; this transport keeps
/// the JSON-RPC wire shape and session headers while routing every body/event
/// through the bounded parser above.
pub(crate) struct BoundedStreamableHttpTransport {
    state: std::sync::Arc<Mutex<StreamableHttpState>>,
    incoming: std::sync::Arc<Mutex<mpsc::Receiver<RxJsonRpcMessage<RoleClient>>>>,
    incoming_tx: mpsc::Sender<RxJsonRpcMessage<RoleClient>>,
    reader_slot: std::sync::Arc<std::sync::Mutex<GetReaderSlot>>,
    close_error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    close_result: std::sync::Arc<tokio::sync::OnceCell<Result<(), String>>>,
    #[cfg(test)]
    test_force_reader_failure: std::sync::Arc<std::sync::atomic::AtomicBool>,
    cancellation: CancellationToken,
    fatal: CancellationToken,
}

impl BoundedStreamableHttpTransport {
    pub(crate) fn new(endpoint: String, auth_token: Option<String>) -> anyhow::Result<Self> {
        let (incoming_tx, incoming_rx) = mpsc::channel(256);
        let cancellation = CancellationToken::new();
        let fatal = CancellationToken::new();
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|error| anyhow::anyhow!("failed to build MCP HTTP client: {error}"))?;
        Ok(Self {
            state: std::sync::Arc::new(Mutex::new(StreamableHttpState {
                client,
                endpoint,
                auth_token,
                session_id: None,
                protocol_version: None,
                cancellation: cancellation.clone(),
                fatal: fatal.clone(),
                get_started: false,
            })),
            incoming: std::sync::Arc::new(Mutex::new(incoming_rx)),
            incoming_tx,
            reader_slot: std::sync::Arc::new(std::sync::Mutex::new(GetReaderSlot {
                closing: false,
                handle: None,
                start_gate: None,
            })),
            close_error: std::sync::Arc::new(std::sync::Mutex::new(None)),
            close_result: std::sync::Arc::new(tokio::sync::OnceCell::new()),
            #[cfg(test)]
            test_force_reader_failure: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                false,
            )),
            cancellation,
            fatal,
        })
    }

    pub(crate) fn close_error_handle(&self) -> std::sync::Arc<std::sync::Mutex<Option<String>>> {
        std::sync::Arc::clone(&self.close_error)
    }

    #[cfg(test)]
    pub(crate) fn test_force_reader_failure_handle(
        &self,
    ) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        std::sync::Arc::clone(&self.test_force_reader_failure)
    }

    async fn start_get_if_ready(
        state: &std::sync::Arc<Mutex<StreamableHttpState>>,
        incoming_tx: &mpsc::Sender<RxJsonRpcMessage<RoleClient>>,
        reader_slot: &std::sync::Arc<std::sync::Mutex<GetReaderSlot>>,
    ) {
        let (state, incoming_tx, _start_gate) = {
            let guard = state.lock().await;
            let slot = reader_slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if guard.get_started || guard.session_id.is_none() || slot.closing {
                return;
            }
            (
                std::sync::Arc::clone(state),
                incoming_tx.clone(),
                slot.start_gate.clone(),
            )
        };
        #[cfg(test)]
        if let Some(start_gate) = _start_gate {
            start_gate.checked.notify_one();
            start_gate.release.notified().await;
        }
        let mut guard = state.lock().await;
        let mut slot = reader_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.get_started || guard.session_id.is_none() || slot.closing {
            return;
        }
        guard.get_started = true;
        let reader_state = std::sync::Arc::clone(&state);
        let handle =
            tokio::spawn(async move { Self::run_get_stream(reader_state, incoming_tx).await });
        slot.handle = Some(handle);
    }

    fn protocol_version(message: &RxJsonRpcMessage<RoleClient>) -> Option<String> {
        serde_json::to_value(message)
            .ok()?
            .get("result")?
            .get("protocolVersion")?
            .as_str()
            .map(str::to_owned)
    }

    async fn run_get_stream(
        state: std::sync::Arc<Mutex<StreamableHttpState>>,
        incoming_tx: mpsc::Sender<RxJsonRpcMessage<RoleClient>>,
    ) {
        let mut last_event_id = None;
        loop {
            let (client, endpoint, auth_token, session_id, protocol_version, cancellation, fatal) = {
                let guard = state.lock().await;
                (
                    guard.client.clone(),
                    guard.endpoint.clone(),
                    guard.auth_token.clone(),
                    guard.session_id.clone(),
                    guard.protocol_version.clone(),
                    guard.cancellation.clone(),
                    guard.fatal.clone(),
                )
            };
            let Some(session_id) = session_id else { return };
            let mut request = client
                .get(endpoint)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header("Mcp-Session-Id", session_id);
            if let Some(last_event_id) = &last_event_id {
                request = request.header("Last-Event-ID", last_event_id);
            }
            if let Some(version) = protocol_version {
                request = request.header("MCP-Protocol-Version", version);
            }
            if let Some(token) = auth_token {
                request = request.bearer_auth(token);
            }
            let response = tokio::select! {
                _ = cancellation.cancelled() => return,
                response = request.send() => match response {
                    Ok(response) => response,
                    Err(_) => {
                        fatal.cancel();
                        return;
                    }
                },
            };
            if !response.status().is_success() {
                if matches!(
                    response.status(),
                    reqwest::StatusCode::BAD_REQUEST
                        | reqwest::StatusCode::METHOD_NOT_ALLOWED
                        | reqwest::StatusCode::NOT_FOUND
                ) {
                    if response.status() == reqwest::StatusCode::NOT_FOUND {
                        fatal.cancel();
                    }
                    return;
                }
                fatal.cancel();
                return;
            }
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !content_type.starts_with("text/event-stream") {
                fatal.cancel();
                return;
            }
            let mut stream = response.bytes_stream();
            let mut parser = BoundedSseParser::new();
            loop {
                let Some(chunk) = (tokio::select! {
                    _ = cancellation.cancelled() => return,
                    chunk = stream.next() => chunk,
                }) else {
                    break;
                };
                let Ok(chunk) = chunk else {
                    fatal.cancel();
                    return;
                };
                let mut messages = Vec::new();
                let mut batch_bytes = 0usize;
                let mut message_count = 0usize;
                let parsed = parser.push(&chunk, &mut |_, event_id, data| {
                    if let Some(event_id) = event_id {
                        last_event_id = Some(event_id.to_owned());
                    }
                    if data.trim().is_empty() {
                        return Ok(());
                    }
                    batch_bytes = batch_bytes
                        .checked_add(data.len())
                        .ok_or_else(|| anyhow::anyhow!("MCP GET event batch exceeds byte limit"))?;
                    if batch_bytes > MAX_HTTP_BODY_BYTES {
                        anyhow::bail!("MCP GET event batch exceeds byte limit");
                    }
                    message_count = message_count
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("MCP GET event count exceeds limit"))?;
                    if message_count > MAX_HTTP_SSE_EVENTS {
                        anyhow::bail!("MCP GET event count exceeds limit");
                    }
                    messages.push(parse_bounded_message(data)?);
                    Ok(())
                });
                if parsed.is_err() {
                    fatal.cancel();
                    return;
                }
                for message in messages {
                    if tokio::select! {
                        _ = cancellation.cancelled() => true,
                        _ = fatal.cancelled() => true,
                        result = incoming_tx.send(message) => result.is_err(),
                    } {
                        return;
                    }
                }
            }
            let mut messages = Vec::new();
            let mut final_bytes = 0usize;
            let mut final_count = 0usize;
            if parser
                .finish(&mut |_, event_id, data| {
                    if let Some(event_id) = event_id {
                        last_event_id = Some(event_id.to_owned());
                    }
                    if !data.trim().is_empty() {
                        final_bytes = final_bytes
                            .checked_add(data.len())
                            .ok_or_else(|| anyhow::anyhow!("MCP GET event exceeds byte limit"))?;
                        if final_bytes > MAX_HTTP_BODY_BYTES {
                            anyhow::bail!("MCP GET event exceeds byte limit");
                        }
                        final_count = final_count
                            .checked_add(1)
                            .ok_or_else(|| anyhow::anyhow!("MCP GET event count exceeds limit"))?;
                        if final_count > MAX_HTTP_SSE_EVENTS {
                            anyhow::bail!("MCP GET event count exceeds limit");
                        }
                        messages.push(parse_bounded_message(data)?);
                    }
                    Ok(())
                })
                .is_err()
            {
                fatal.cancel();
                return;
            }
            for message in messages {
                if tokio::select! {
                    _ = cancellation.cancelled() => true,
                    _ = fatal.cancelled() => true,
                    result = incoming_tx.send(message) => result.is_err(),
                } {
                    return;
                }
            }
            if last_event_id.is_none() {
                // A clean EOF without an event id cannot be resumed safely.
                // Keep the catalog fail-closed rather than continuing to serve
                // authority from a stream whose notification history is lost.
                fatal.cancel();
                return;
            }
            tokio::select! {
                _ = cancellation.cancelled() => return,
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
            }
        }
    }

    async fn post(
        state: std::sync::Arc<Mutex<StreamableHttpState>>,
        incoming_tx: mpsc::Sender<RxJsonRpcMessage<RoleClient>>,
        reader_slot: std::sync::Arc<std::sync::Mutex<GetReaderSlot>>,
        message: TxJsonRpcMessage<RoleClient>,
    ) -> io::Result<()> {
        let (cancellation, fatal) = {
            let guard = state.lock().await;
            (guard.cancellation.clone(), guard.fatal.clone())
        };
        let (client, endpoint, auth_token, session_id, protocol_version) = {
            let guard = state.lock().await;
            (
                guard.client.clone(),
                guard.endpoint.clone(),
                guard.auth_token.clone(),
                guard.session_id.clone(),
                guard.protocol_version.clone(),
            )
        };
        let body = serde_json::to_vec(&message).map_err(io::Error::other)?;
        let mut request = client
            .post(endpoint)
            .timeout(std::time::Duration::from_secs(30))
            .header(
                reqwest::header::ACCEPT,
                "text/event-stream, application/json",
            )
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body);
        if let Some(session_id) = session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }
        if let Some(version) = protocol_version {
            request = request.header("MCP-Protocol-Version", version);
        }
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
            }
            _ = fatal.cancelled() => {
                return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
            }
            response = request.send() => match response {
                Ok(response) => response,
                Err(error) => {
                    fatal.cancel();
                    return Err(io::Error::other(error));
                }
            },
        };
        if let Some(session_id) = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|value| value.to_str().ok())
        {
            state.lock().await.session_id = Some(session_id.to_string());
        }
        if response.status() == reqwest::StatusCode::ACCEPTED
            || response.status() == reqwest::StatusCode::NO_CONTENT
        {
            Self::start_get_if_ready(&state, &incoming_tx, &reader_slot).await;
            return Ok(());
        }
        if !response.status().is_success() {
            let status = response.status();
            let body = tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                }
                _ = fatal.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                }
                body = read_response_body(response, MAX_HTTP_BODY_BYTES) => body
                    .unwrap_or_else(|error| error.to_string()),
            };
            fatal.cancel();
            return Err(io::Error::other(format!("MCP HTTP {status}: {body}")));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if content_type.starts_with("application/json") {
            let body = tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                }
                _ = fatal.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                }
                body = read_response_body(response, MAX_HTTP_BODY_BYTES) => match body {
                    Ok(body) => body,
                    Err(error) => {
                        fatal.cancel();
                        return Err(io::Error::other(error));
                    }
                },
            };
            match parse_bounded_message(&body) {
                Ok(message) => {
                    if let Some(version) = Self::protocol_version(&message) {
                        state.lock().await.protocol_version = Some(version);
                    }
                    tokio::select! {
                        _ = cancellation.cancelled() => {
                            return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                        }
                        _ = fatal.cancelled() => {
                            return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                        }
                        result = incoming_tx.send(message) => result.map_err(io::Error::other)?,
                    }
                }
                Err(error) => {
                    fatal.cancel();
                    return Err(io::Error::other(error));
                }
            }
        } else if content_type.starts_with("text/event-stream") {
            let mut stream = response.bytes_stream();
            let mut parser = BoundedSseParser::new();
            let mut events = 0usize;
            let mut sse_bytes = 0usize;
            while let Some(chunk) = tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                }
                _ = fatal.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                }
                chunk = stream.next() => chunk,
            } {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        fatal.cancel();
                        return Err(io::Error::other(error));
                    }
                };
                sse_bytes = sse_bytes
                    .checked_add(chunk.len())
                    .ok_or_else(|| io::Error::other("MCP HTTP SSE body exceeds byte limit"))?;
                if sse_bytes > MAX_HTTP_BODY_BYTES {
                    fatal.cancel();
                    return Err(io::Error::other("MCP HTTP SSE body exceeds byte limit"));
                }
                let mut messages = Vec::new();
                if let Err(error) = parser.push(&chunk, &mut |_, _, data| {
                    if data.trim().is_empty() {
                        return Ok(());
                    }
                    events = events.saturating_add(1);
                    if events > MAX_HTTP_SSE_EVENTS {
                        anyhow::bail!("MCP HTTP SSE event count exceeds limit");
                    }
                    messages.push(parse_bounded_message(data)?);
                    Ok(())
                }) {
                    fatal.cancel();
                    return Err(io::Error::other(error));
                }
                for message in messages {
                    if let Some(version) = Self::protocol_version(&message) {
                        state.lock().await.protocol_version = Some(version);
                    }
                    tokio::select! {
                        _ = cancellation.cancelled() => {
                            return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                        }
                        _ = fatal.cancelled() => {
                            return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                        }
                        result = incoming_tx.send(message) => result.map_err(io::Error::other)?,
                    }
                }
            }
            let mut messages = Vec::new();
            if let Err(error) = parser.finish(&mut |_, _, data| {
                if !data.trim().is_empty() {
                    events = events.saturating_add(1);
                    if events > MAX_HTTP_SSE_EVENTS {
                        anyhow::bail!("MCP HTTP SSE event count exceeds limit");
                    }
                    messages.push(parse_bounded_message(data)?);
                }
                Ok(())
            }) {
                fatal.cancel();
                return Err(io::Error::other(error));
            }
            for message in messages {
                tokio::select! {
                    _ = cancellation.cancelled() => {
                        return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                    }
                    _ = fatal.cancelled() => {
                        return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                    }
                    result = incoming_tx.send(message) => result.map_err(io::Error::other)?,
                }
            }
        } else {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "MCP HTTP transport cancelled"));
                }
                _ = fatal.cancelled() => {
                    return Err(io::Error::new(io::ErrorKind::NotConnected, "MCP HTTP transport is unavailable"));
                }
                body = read_response_body(response, MAX_HTTP_BODY_BYTES) => {
                    if body.is_err() {
                        fatal.cancel();
                    }
                }
            }
            fatal.cancel();
            return Err(io::Error::other(
                "MCP HTTP response has unsupported content type",
            ));
        }
        Self::start_get_if_ready(&state, &incoming_tx, &reader_slot).await;
        Ok(())
    }
}

impl Transport<RoleClient> for BoundedStreamableHttpTransport {
    type Error = io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let state = std::sync::Arc::clone(&self.state);
        let incoming_tx = self.incoming_tx.clone();
        let reader_slot = std::sync::Arc::clone(&self.reader_slot);
        async move { Self::post(state, incoming_tx, reader_slot, item).await }
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleClient>>> + Send {
        let incoming = std::sync::Arc::clone(&self.incoming);
        let fatal = self.fatal.clone();
        async move {
            tokio::select! {
                _ = fatal.cancelled() => None,
                message = async { incoming.lock().await.recv().await } => message,
            }
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        let state = std::sync::Arc::clone(&self.state);
        let reader_slot = std::sync::Arc::clone(&self.reader_slot);
        let close_error = std::sync::Arc::clone(&self.close_error);
        let close_result = std::sync::Arc::clone(&self.close_result);
        #[cfg(test)]
        let test_force_reader_failure = std::sync::Arc::clone(&self.test_force_reader_failure);
        let cancellation = self.cancellation.clone();
        let fatal = self.fatal.clone();
        async move {
            let result = close_result
                .get_or_init(|| async move {
                    cancellation.cancel();
                    fatal.cancel();
                    let reader = {
                        let mut slot = reader_slot
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        slot.closing = true;
                        slot.handle.take()
                    };
                    let reader_error = join_get_reader(reader).await.err();
                    #[cfg(test)]
                    let reader_error =
                        if test_force_reader_failure.load(std::sync::atomic::Ordering::SeqCst) {
                            Some(io::Error::other("MCP HTTP GET reader teardown failed"))
                        } else {
                            reader_error
                        };
                    let (client, endpoint, auth_token, session_id) = {
                        let guard = state.lock().await;
                        (
                            guard.client.clone(),
                            guard.endpoint.clone(),
                            guard.auth_token.clone(),
                            guard.session_id.clone(),
                        )
                    };
                    let delete_error = if let Some(session_id) = session_id {
                        let mut request = client
                            .delete(endpoint)
                            .timeout(std::time::Duration::from_secs(5))
                            .header("Mcp-Session-Id", session_id);
                        if let Some(token) = auth_token {
                            request = request.bearer_auth(token);
                        }
                        request.send().await.is_err()
                    } else {
                        false
                    };
                    let reader_message = reader_error.as_ref().map(|error| {
                        if error.kind() == io::ErrorKind::TimedOut {
                            "MCP HTTP reader teardown timed out or remained unresolved"
                        } else {
                            "MCP HTTP reader teardown failed"
                        }
                    });
                    let error_message = match (reader_message, delete_error) {
                        (Some(_), true) => Some("MCP HTTP reader and session teardown failed"),
                        (Some(message), false) => Some(message),
                        _ => None,
                    };
                    if let Some(message) = error_message {
                        if let Ok(mut recorded) = close_error.lock() {
                            *recorded = Some(message.to_owned());
                        }
                        return Err(message.to_owned());
                    }
                    Ok(())
                })
                .await;
            match result {
                Ok(()) => Ok(()),
                Err(error) => Err(io::Error::other(error.clone())),
            }
        }
    }
}

impl Drop for BoundedStreamableHttpTransport {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.fatal.cancel();
        abort_get_reader_on_drop(&self.reader_slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct CountingReader {
        bytes: Vec<u8>,
        offset: usize,
        reads: Arc<AtomicUsize>,
    }

    impl AsyncRead for CountingReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            destination: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            if self.offset == self.bytes.len() {
                return Poll::Ready(Ok(()));
            }
            let count = destination.remaining().min(self.bytes.len() - self.offset);
            destination.put_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn line_reader_reads_large_frame_in_bounded_chunks() {
        let reads = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader {
            bytes: vec![b'x'; 1024 * 1024],
            offset: 0,
            reads: Arc::clone(&reads),
        };
        let mut bounded = BoundedLineReader::new(reader);
        let mut output = Vec::new();
        bounded.read_to_end(&mut output).await.unwrap();
        assert_eq!(output.len(), 1024 * 1024);
        assert!(reads.load(Ordering::Relaxed) < 300);
    }

    #[cfg(unix)]
    #[test]
    fn child_drop_without_runtime_uses_kill_on_drop_fallback() {
        let mut command = Command::new("sh");
        command.args(["-c", "exec sleep 10"]);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let child = runtime.block_on(async { BoundedChildProcess::spawn(command).unwrap() });
        // Process construction needs an I/O driver; Drop deliberately runs
        // outside the runtime context while that driver remains alive.
        drop(child);
        drop(runtime);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unpolled_close_keeps_admission_until_owned_child_is_reaped() {
        let capacity = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let admission = std::sync::Arc::new(capacity.clone().acquire_owned().await.unwrap());
        let mut command = Command::new("sh");
        command.args(["-c", "exec sleep 30"]);
        let mut transport = BoundedChildProcess::spawn_admitted(command, Some(admission)).unwrap();
        assert_eq!(capacity.available_permits(), 0);
        let close = transport.close();
        drop(close);
        drop(transport);
        let _reclaimed =
            tokio::time::timeout(std::time::Duration::from_secs(2), capacity.acquire_owned())
                .await
                .expect("unpolled close leaked its owned child/admission")
                .unwrap();
    }

    #[test]
    fn sse_parser_rejects_overlong_line_and_event() {
        let mut parser = BoundedSseParser::new();
        let mut callback = |_: Option<&str>, _: Option<&str>, _: &str| Ok(());
        assert!(parser
            .push(&vec![b'x'; MAX_SSE_LINE_BYTES + 1], &mut callback)
            .is_err());

        let mut parser = BoundedSseParser::new();
        let mut callback = |_: Option<&str>, _: Option<&str>, _: &str| Ok(());
        let line = format!("data: {}\n", "x".repeat(2040));
        let mut data = Vec::with_capacity(MAX_SSE_EVENT_BYTES + line.len());
        for _ in 0..=4096 {
            data.extend_from_slice(line.as_bytes());
        }
        assert!(parser.push(&data, &mut callback).is_err());

        let mut parser = BoundedSseParser::new();
        let mut callback = |_: Option<&str>, _: Option<&str>, _: &str| Ok(());
        assert!(parser.push(b"\n", &mut callback).is_ok());
        let mut empty_data = Vec::new();
        for _ in 0..4097 {
            empty_data.extend_from_slice(b"data:\n");
        }
        assert!(parser.push(&empty_data, &mut callback).is_err());
    }

    #[test]
    fn raw_resource_validation_rejects_ambiguous_text_and_blob() {
        let value = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "contents": [{
                    "uri": "file:///ambiguous",
                    "mimeType": "text/plain",
                    "text": "text",
                    "blob": "aGVsbG8="
                }]
            }
        });
        assert_eq!(
            reject_ambiguous_resource_payload(&value),
            Err("MCP resource contains both text and blob payloads")
        );

        let value = serde_json::json!({
            "result": {
                "content": [{
                    "type": "resource",
                    "resource": {
                        "uri": "file:///ambiguous",
                        "text": "text",
                        "blob": "aGVsbG8="
                    }
                }]
            }
        });
        assert!(reject_ambiguous_resource_payload(&value).is_err());
    }

    #[test]
    fn sse_parser_accepts_lf_crlf_and_cr_records() {
        let mut parser = BoundedSseParser::new();
        let mut records = Vec::new();
        parser
            .push(b"data: one\n\n", &mut |_, _, data| {
                records.push(data.to_owned());
                Ok(())
            })
            .unwrap();
        parser
            .push(b"data: two\r\n\r\n", &mut |_, _, data| {
                records.push(data.to_owned());
                Ok(())
            })
            .unwrap();
        parser
            .push(b"data: three\r\r", &mut |_, _, data| {
                records.push(data.to_owned());
                Ok(())
            })
            .unwrap();
        assert_eq!(records, ["one", "two", "three"]);
    }

    async fn loopback_response(
        content_type: &str,
        body: &[u8],
        hold_body: bool,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_vec();
        let content_type = content_type.to_owned();
        let task = tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut request = vec![0_u8; 4096];
            let _ = stream.read(&mut request).await;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(headers.as_bytes()).await;
            if hold_body {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            } else {
                let _ = stream.write_all(&body).await;
            }
        });
        (format!("http://{address}/mcp"), task)
    }

    #[tokio::test]
    async fn loopback_response_body_rejects_overflow_before_accumulation() {
        let (url, server) = loopback_response("text/plain", b"0123456789", false).await;
        let response = reqwest::get(url).await.unwrap();
        let error = read_response_body(response, 9).await.unwrap_err();
        assert!(error.to_string().contains("exceeds byte limit"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn loopback_response_body_timeout_is_bounded() {
        let (url, server) = loopback_response("text/plain", b"late", true).await;
        let response = reqwest::get(url).await.unwrap();
        let error =
            read_response_body_with_timeout(response, 64, std::time::Duration::from_millis(25))
                .await
                .unwrap_err();
        assert!(error.to_string().contains("timed out"));
        server.abort();
    }

    #[tokio::test]
    async fn loopback_sse_body_uses_bounded_parser() {
        let body = b"id: 7\r\nevent: message\r\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\r\n\r\n";
        let (url, server) = loopback_response("text/event-stream", body, false).await;
        let response = reqwest::get(url).await.unwrap();
        let mut parser = BoundedSseParser::new();
        let mut messages = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            parser
                .push(&chunk, &mut |event, event_id, data| {
                    messages.push((
                        event.map(str::to_owned),
                        event_id.map(str::to_owned),
                        data.to_owned(),
                    ));
                    Ok(())
                })
                .unwrap();
        }
        parser
            .finish(&mut |event, event_id, data| {
                messages.push((
                    event.map(str::to_owned),
                    event_id.map(str::to_owned),
                    data.to_owned(),
                ));
                Ok(())
            })
            .unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].0.as_deref(), Some("message"));
        assert_eq!(messages[0].1.as_deref(), Some("7"));
        assert!(messages[0].2.contains("list_changed"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn loopback_streamable_http_preserves_auth_session_and_json_response() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return false;
            };
            let mut request = vec![0_u8; 8192];
            let length = stream.read(&mut request).await.unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..length]);
            let authorized = request.contains("authorization: Bearer test-token")
                || request.contains("Authorization: Bearer test-token");
            let body = br#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: session-1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let wrote = stream.write_all(response.as_bytes()).await.is_ok()
                && stream.write_all(body).await.is_ok();
            let get_was_rejected = if let Ok((mut get_stream, _)) = listener.accept().await {
                let mut get_request = [0_u8; 2048];
                let _ = get_stream.read(&mut get_request).await;
                get_stream
                    .write_all(
                        b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .is_ok()
            } else {
                false
            };
            authorized && wrote && get_was_rejected
        });
        let endpoint = format!("http://{address}/mcp");
        let mut transport =
            BoundedStreamableHttpTransport::new(endpoint, Some("test-token".to_owned())).unwrap();
        let initialize: TxJsonRpcMessage<RoleClient> = serde_json::from_value(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "fixture", "version": "1"}
            }
        }))
        .unwrap();
        transport.send(initialize).await.unwrap();
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), transport.receive())
            .await
            .unwrap()
            .expect("JSON response should be delivered");
        assert_eq!(serde_json::to_value(response).unwrap()["id"], 1);
        assert!(server.await.unwrap());
        transport.close().await.unwrap();
    }

    async fn start_get_fixture(status: &str, body: &[u8]) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_vec();
        let status = status.to_owned();
        let task = tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).await;
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.write_all(&body).await;
        });
        (format!("http://{address}/mcp"), task)
    }

    async fn start_held_chunked_get_fixture(
        body: &[u8],
    ) -> (
        String,
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
        tokio::sync::oneshot::Receiver<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = body.to_vec();
        let (body_sent_tx, body_sent_rx) = tokio::sync::oneshot::channel();
        let (release_eof_tx, release_eof_rx) = tokio::sync::oneshot::channel();
        let (eof_sent_tx, eof_sent_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).await;
            if stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .await
                .is_err()
            {
                return;
            }
            let chunk_header = format!("{:X}\r\n", body.len());
            if stream.write_all(chunk_header.as_bytes()).await.is_err()
                || stream.write_all(&body).await.is_err()
                || stream.write_all(b"\r\n").await.is_err()
                || stream.flush().await.is_err()
            {
                return;
            }
            if body_sent_tx.send(()).is_err() {
                return;
            }
            if release_eof_rx.await.is_err() {
                return;
            }
            if stream.write_all(b"0\r\n\r\n").await.is_err() || stream.flush().await.is_err() {
                return;
            }
            let _ = stream.shutdown().await;
            let _ = eof_sent_tx.send(());
        });
        (
            format!("http://{address}/mcp"),
            body_sent_rx,
            release_eof_tx,
            eof_sent_rx,
            task,
        )
    }

    #[tokio::test]
    async fn streamable_get_404_closes_receive_and_marks_transport_fatal() {
        let (endpoint, server) = start_get_fixture("404 Not Found", b"expired").await;
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        let message = tokio::time::timeout(std::time::Duration::from_secs(1), transport.receive())
            .await
            .unwrap();
        assert!(message.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn streamable_get_405_keeps_post_only_receive_open() {
        let (endpoint, server) = start_get_fixture("405 Method Not Allowed", b"").await;
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), transport.receive(),)
                .await
                .is_err()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn streamable_get_overlong_frame_closes_receive() {
        let body = format!("data: {}\n\n", "x".repeat(MAX_SSE_LINE_BYTES + 1));
        let (endpoint, server) = start_get_fixture("200 OK", body.as_bytes()).await;
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        let message = tokio::time::timeout(std::time::Duration::from_secs(1), transport.receive())
            .await
            .unwrap();
        assert!(message.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn streamable_get_eof_without_event_id_fails_closed() {
        let body =
            b"data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n";
        let (endpoint, body_sent_rx, release_eof_tx, eof_sent_rx, server) =
            start_held_chunked_get_fixture(body).await;
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        tokio::time::timeout(std::time::Duration::from_secs(1), body_sent_rx)
            .await
            .unwrap()
            .expect("fixture should hold the stream after sending the notification");
        let first = tokio::time::timeout(std::time::Duration::from_secs(1), transport.receive())
            .await
            .unwrap()
            .expect("message should be delivered before EOF");
        assert_eq!(
            serde_json::to_value(first).unwrap()["method"],
            "notifications/tools/list_changed"
        );
        release_eof_tx
            .send(())
            .expect("fixture should still be waiting to release EOF");
        tokio::time::timeout(std::time::Duration::from_secs(1), eof_sent_rx)
            .await
            .unwrap()
            .expect("fixture should send EOF after the notification is consumed");
        let after_eof =
            tokio::time::timeout(std::time::Duration::from_secs(1), transport.receive())
                .await
                .unwrap();
        assert!(after_eof.is_none());
        tokio::time::timeout(std::time::Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn streamable_get_reconnects_with_last_event_id() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 4096];
            let length = first.read(&mut request).await.unwrap();
            let first_request = String::from_utf8_lossy(&request[..length]).to_ascii_lowercase();
            assert!(!first_request.contains("last-event-id:"));
            let body = b"id: 7\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n";
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            first.write_all(headers.as_bytes()).await.unwrap();
            first.write_all(body).await.unwrap();
            drop(first);

            let (mut second, _) = listener.accept().await.unwrap();
            let length = second.read(&mut request).await.unwrap();
            let second_request = String::from_utf8_lossy(&request[..length]).to_ascii_lowercase();
            assert!(second_request.contains("last-event-id: 7"));
            second
                .write_all(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
        });
        let endpoint = format!("http://{address}/mcp");
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        let message = tokio::time::timeout(std::time::Duration::from_secs(1), transport.receive())
            .await
            .unwrap()
            .expect("first SSE message should be delivered");
        assert_eq!(
            serde_json::to_value(message).unwrap()["method"],
            "notifications/tools/list_changed"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn streamable_get_close_joins_reader_before_return_and_blocks_events() {
        let body =
            b"data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n";
        let (endpoint, body_sent_rx, release_eof_tx, _eof_sent_rx, server) =
            start_held_chunked_get_fixture(body).await;
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        tokio::time::timeout(std::time::Duration::from_secs(1), body_sent_rx)
            .await
            .unwrap()
            .expect("fixture should hold the active GET reader");
        transport.state.lock().await.session_id = None;

        transport.close().await.unwrap();
        assert!(transport.reader_slot.lock().unwrap().handle.is_none());
        assert!(transport.receive().await.is_none());

        release_eof_tx
            .send(())
            .expect("fixture should still be waiting for release");
        tokio::time::timeout(std::time::Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn dropped_close_future_aborts_taken_reader_instead_of_detaching() {
        struct DropSignal(Option<tokio::sync::oneshot::Sender<()>>);

        impl Drop for DropSignal {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.send(());
                }
            }
        }

        let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
        let reader = tokio::spawn(async move {
            let _signal = DropSignal(Some(dropped_tx));
            std::future::pending::<()>().await;
        });
        let mut transport =
            BoundedStreamableHttpTransport::new("http://127.0.0.1:9/mcp".to_owned(), None).unwrap();
        transport.reader_slot.lock().unwrap().handle = Some(reader);
        let mut close = Box::pin(transport.close());
        tokio::select! {
            result = &mut close => panic!("close unexpectedly completed: {result:?}"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
        }
        drop(close);
        tokio::time::timeout(std::time::Duration::from_secs(1), dropped_rx)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn repeated_close_returns_sticky_reader_failure() {
        let reader = tokio::spawn(async {
            panic!("synthetic GET reader failure");
        });
        let mut transport =
            BoundedStreamableHttpTransport::new("http://127.0.0.1:9/mcp".to_owned(), None).unwrap();
        transport.reader_slot.lock().unwrap().handle = Some(reader);

        let first = transport.close().await.unwrap_err();
        let second = transport.close().await.unwrap_err();
        assert_eq!(first.to_string(), "MCP HTTP reader teardown failed");
        assert_eq!(second.to_string(), first.to_string());
        assert_eq!(
            *transport.close_error_handle().lock().unwrap(),
            Some("MCP HTTP reader teardown failed".to_owned())
        );
    }

    #[tokio::test]
    async fn start_get_and_close_race_does_not_publish_reader() {
        let mut transport =
            BoundedStreamableHttpTransport::new("http://127.0.0.1:9/mcp".to_owned(), None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        let gate = Arc::new(StartGetGate {
            checked: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        });
        transport.reader_slot.lock().unwrap().start_gate = Some(Arc::clone(&gate));

        let state = Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        let reader_slot = Arc::clone(&transport.reader_slot);
        let start = tokio::spawn(async move {
            BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &reader_slot).await;
        });
        gate.checked.notified().await;
        transport.state.lock().await.session_id = None;
        transport.close().await.unwrap();
        gate.release.notify_one();
        start.await.unwrap();

        assert!(!transport.state.lock().await.get_started);
        assert!(transport.reader_slot.lock().unwrap().handle.is_none());
    }

    #[tokio::test]
    async fn start_get_does_not_spawn_after_close_claims_slot() {
        let transport =
            BoundedStreamableHttpTransport::new("http://127.0.0.1:9/mcp".to_owned(), None).unwrap();
        transport.state.lock().await.session_id = Some("session".to_owned());
        transport.reader_slot.lock().unwrap().closing = true;
        let state = std::sync::Arc::clone(&transport.state);
        let sender = transport.incoming_tx.clone();
        BoundedStreamableHttpTransport::start_get_if_ready(&state, &sender, &transport.reader_slot)
            .await;
        assert!(!transport.state.lock().await.get_started);
        assert!(transport.reader_slot.lock().unwrap().handle.is_none());
    }

    #[tokio::test]
    async fn get_reader_join_reports_timeout_and_fatal_without_endpoint_text() {
        let reader = tokio::spawn(async {
            std::future::pending::<()>().await;
        });
        let error = join_get_reader(Some(reader)).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(!error.to_string().contains("127.0.0.1"));

        let reader = tokio::spawn(async {
            panic!("synthetic GET reader failure");
        });
        let error = join_get_reader(Some(reader)).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(error.to_string(), "MCP HTTP GET reader teardown failed");
    }

    #[tokio::test]
    async fn pending_post_headers_do_not_block_a_concurrent_refresh() {
        async fn read_request(stream: &mut tokio::net::TcpStream) {
            let mut bytes = Vec::new();
            loop {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).await.unwrap();
                assert!(read > 0, "request ended before its body");
                bytes.extend_from_slice(&chunk[..read]);
                assert!(bytes.len() <= 16 * 1024);
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let header = std::str::from_utf8(&bytes[..end]).unwrap();
                    let length = header
                        .lines()
                        .find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if bytes.len() >= end + 4 + length {
                        return;
                    }
                }
            }
        }

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let endpoint = format!("http://{}/mcp", listener.local_addr().unwrap());
        let (first_seen, first_ready) = tokio::sync::oneshot::channel();
        let mut server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            read_request(&mut first).await;
            first_seen.send(()).unwrap();
            // The first response has no headers yet. A second request must
            // reach the server before the first call can complete.
            let (mut second, _) = listener.accept().await.unwrap();
            read_request(&mut second).await;
            for (stream, id) in [(&mut second, 2), (&mut first, 1)] {
                let body = format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{{}}}}");
                let header = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                stream.write_all(header.as_bytes()).await.unwrap();
                stream.write_all(body.as_bytes()).await.unwrap();
            }
        });
        let mut transport = BoundedStreamableHttpTransport::new(endpoint, None).unwrap();
        let first = transport.send(
            serde_json::from_value(serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "ping"
            }))
            .unwrap(),
        );
        let second = transport.send(
            serde_json::from_value(serde_json::json!({
                "jsonrpc": "2.0", "id": 2, "method": "ping"
            }))
            .unwrap(),
        );
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            let gated_second = async {
                first_ready.await.unwrap();
                second.await
            };
            tokio::try_join!(first, gated_second).unwrap();
            (&mut server).await.unwrap();
        })
        .await;
        server.abort();
        result.expect("concurrent POST was blocked behind pending response headers");
        let mut ids = Vec::new();
        for _ in 0..2 {
            let message = transport.receive().await.unwrap();
            ids.push(
                serde_json::to_value(message).unwrap()["id"]
                    .as_u64()
                    .unwrap(),
            );
        }
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }
}
