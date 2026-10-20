//! Bidirectional newline-delimited JSON-RPC 2.0 connection over arbitrary
//! `AsyncRead` / `AsyncWrite` half-duplex pair (typically stdin/stdout).
//!
//! Both sides can send requests, responses, and notifications. Outbound
//! requests are tracked in a pending map so their responses can be routed
//! back to the awaiting caller.
//!
//! The wire format matches the Agent Client Protocol: each message is a
//! single UTF-8 line terminated by `\n` carrying a JSON-RPC 2.0 envelope.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema as acp;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, Mutex, OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, trace, warn};

const JSONRPC_VERSION: &str = "2.0";
const MAX_PENDING_REQUESTS: usize = 64;
pub(crate) const MAX_INBOUND_QUEUE: usize = 8;
const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;
const MAX_JSON_UNITS: usize = 100_000;
const MAX_JSON_DEPTH: usize = 128;
const INBOUND_BUDGET: usize = 128 * 1024 * 1024;
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

struct FrameBuffer(Vec<u8>);

impl std::io::Write for FrameBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_FRAME_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("ACP output frame limit exceeded"));
        }
        if bytes.len() > self.0.capacity().saturating_sub(self.0.len()) {
            let capacity = (self.0.len() + bytes.len())
                .max(self.0.capacity().saturating_mul(2))
                .clamp(4096, MAX_FRAME_BYTES);
            self.0.reserve_exact(capacity - self.0.len());
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn input_charge(bytes: &[u8]) -> anyhow::Result<usize> {
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    let mut units = 0usize;
    for &byte in bytes {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'{' | b'[' => {
                depth += 1;
                units += 1;
                if depth > MAX_JSON_DEPTH {
                    anyhow::bail!("ACP JSON nesting limit exceeded");
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            b',' | b':' => units += 1,
            _ => {}
        }
        if units > MAX_JSON_UNITS {
            anyhow::bail!("ACP JSON complexity limit exceeded");
        }
    }
    Ok(bytes.len() * 2 + units * 128 + 1024)
}

#[derive(Default)]
struct PendingRequests {
    closed: bool,
    requests: HashMap<String, oneshot::Sender<PendingResponse>>,
}

struct PendingResponse {
    payload: Result<Value, acp::Error>,
    _budget: OwnedSemaphorePermit,
}

struct PendingRequest<'a> {
    connection: &'a Connection,
    key: String,
}

impl Drop for PendingRequest<'_> {
    fn drop(&mut self) {
        self.connection.pending.lock().requests.remove(&self.key);
    }
}

struct ReaderOwner(Arc<Connection>);

impl Drop for ReaderOwner {
    fn drop(&mut self) {
        self.0.disconnect();
    }
}

struct FrameWrite<'a> {
    connection: &'a Connection,
    complete: bool,
}

impl Drop for FrameWrite<'_> {
    fn drop(&mut self) {
        if !self.complete {
            // An interrupted write may have emitted only part of a JSON line.
            // Further writes cannot safely reuse that framing boundary.
            self.connection.output_failed.store(true, Ordering::Release);
            self.connection.disconnect();
        }
    }
}

/// One parsed inbound JSON-RPC message — request, notification, or response.
///
/// Responses are NOT included here because the connection routes them
/// internally to the awaiting `send_request` future.
#[derive(Debug)]
pub enum Inbound {
    Request {
        id: acp::RequestId,
        method: String,
        params: Option<Value>,
    },
    Notification {
        method: String,
        params: Option<Value>,
    },
}

/// Keep the connection's conservative memory charge through dispatch.
#[derive(Debug)]
pub(crate) struct QueuedInbound {
    pub message: Inbound,
    pub _budget: OwnedSemaphorePermit,
}

/// Bidirectional connection over newline-delimited JSON-RPC.
///
/// The caller drives `run_reader` and receives parsed inbound messages through
/// its channel. Responses resolve at most 64 pending outbound requests.
pub struct Connection {
    writer: Mutex<Box<dyn AsyncWrite + Send + Unpin>>,
    pending: parking_lot::Mutex<PendingRequests>,
    disconnected: CancellationToken,
    output_failed: AtomicBool,
    inbound_budget: Arc<Semaphore>,
    next_outbound_id: AtomicI64,
}

impl Connection {
    /// Create a new connection over the given writer. The caller drives the
    /// reader by calling `run_reader` separately.
    pub fn new<W>(writer: W) -> Arc<Self>
    where
        W: AsyncWrite + Send + Unpin + 'static,
    {
        Arc::new(Self {
            writer: Mutex::new(Box::new(writer)),
            pending: parking_lot::Mutex::new(PendingRequests::default()),
            disconnected: CancellationToken::new(),
            output_failed: AtomicBool::new(false),
            inbound_budget: Arc::new(Semaphore::new(INBOUND_BUDGET)),
            next_outbound_id: AtomicI64::new(1),
        })
    }

    /// Send a successful JSON-RPC response to a previously received request.
    pub async fn send_response<R: Serialize>(
        self: &Arc<Self>,
        id: acp::RequestId,
        result: R,
    ) -> anyhow::Result<()> {
        let value = serde_json::to_value(result)?;
        let msg = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": id,
            "result": value,
        });
        self.write_line(&msg).await
    }

    /// Send a JSON-RPC error response.
    pub async fn send_error_response(
        self: &Arc<Self>,
        id: acp::RequestId,
        error: acp::Error,
    ) -> anyhow::Result<()> {
        let msg = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": id,
            "error": error,
        });
        self.write_line(&msg).await
    }

    /// Send a notification (no id, no response expected).
    pub async fn send_notification<P: Serialize>(
        self: &Arc<Self>,
        method: &str,
        params: P,
    ) -> anyhow::Result<()> {
        let value = serde_json::to_value(params)?;
        let msg = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": value,
        });
        self.write_line(&msg).await
    }

    /// Send an outbound request and await the response.
    ///
    /// The caller is blocked until the matching response arrives via the
    /// reader loop, or the connection is closed (in which case `Err(...)`).
    pub async fn send_request<P: Serialize, R: DeserializeOwned>(
        self: &Arc<Self>,
        method: &str,
        params: P,
    ) -> anyhow::Result<Result<R, acp::Error>> {
        let params_value = serde_json::to_value(params)?;
        let raw_id = self.next_outbound_id.fetch_add(1, Ordering::Relaxed);
        let id_key = format!("n:{}", raw_id);
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock();
            if pending.closed {
                anyhow::bail!("ACP connection is closed");
            }
            if pending.requests.len() >= MAX_PENDING_REQUESTS {
                anyhow::bail!("ACP pending request limit reached");
            }
            pending.requests.insert(id_key.clone(), tx);
        }
        let _request = PendingRequest {
            connection: self,
            key: id_key,
        };

        let msg = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": raw_id,
            "method": method,
            "params": params_value,
        });
        tokio::select! {
            biased;
            result = self.write_line(&msg) => result?,
            _ = self.disconnected.cancelled() => anyhow::bail!("ACP connection is closed"),
        }

        match rx.await {
            Ok(response) => {
                let PendingResponse { payload, _budget } = response;
                match payload {
                    Ok(value) => Ok(Ok(serde_json::from_value(value)?)),
                    Err(error) => Ok(Err(error)),
                }
            }
            Err(_) => Err(anyhow::anyhow!(
                "connection closed while awaiting response to '{}'",
                method
            )),
        }
    }

    async fn write_line(self: &Arc<Self>, value: &Value) -> anyhow::Result<()> {
        let mut buffer = FrameBuffer(Vec::new());
        serde_json::to_writer(&mut buffer, value)?;
        std::io::Write::write_all(&mut buffer, b"\n")?;
        let write = async {
            let mut w = self.writer.lock().await;
            if self.output_failed.load(Ordering::Acquire) {
                anyhow::bail!("ACP output is unavailable");
            }
            let mut frame = FrameWrite {
                connection: self,
                complete: false,
            };
            w.write_all(&buffer.0).await?;
            w.flush().await?;
            frame.complete = true;
            trace!(bytes = buffer.0.len(), "ACP wire send");
            Ok(())
        };
        match tokio::time::timeout(WRITE_TIMEOUT, write).await {
            Ok(result) => result,
            Err(_) => {
                self.output_failed.store(true, Ordering::Release);
                self.disconnect();
                anyhow::bail!("ACP output deadline exceeded");
            }
        }
    }

    pub(crate) fn admit_inbound(
        &self,
        message: Inbound,
        charge: usize,
    ) -> anyhow::Result<QueuedInbound> {
        let charge = u32::try_from(charge)?;
        let permit = self
            .inbound_budget
            .clone()
            .try_acquire_many_owned(charge)
            .map_err(|_| anyhow::anyhow!("ACP retained input budget exceeded"))?;
        Ok(QueuedInbound {
            message,
            _budget: permit,
        })
    }

    pub(crate) fn disconnected(&self) -> CancellationToken {
        self.disconnected.clone()
    }

    /// Look up and resolve a pending outbound request when a response arrives.
    fn complete_pending(
        &self,
        id: &acp::RequestId,
        payload: Result<Value, acp::Error>,
        charge: usize,
    ) -> anyhow::Result<()> {
        let key = id_to_key(id);
        let sender = self.pending.lock().requests.remove(&key);
        if let Some(tx) = sender {
            let charge = u32::try_from(charge)?;
            let permit = self
                .inbound_budget
                .clone()
                .try_acquire_many_owned(charge)
                .map_err(|_| anyhow::anyhow!("ACP retained input budget exceeded"))?;
            let _ = tx.send(PendingResponse {
                payload,
                _budget: permit,
            });
        } else {
            warn!("ACP: received response for unknown request id");
        }
        Ok(())
    }

    fn disconnect(&self) {
        let mut pending = self.pending.lock();
        pending.closed = true;
        pending.requests.clear();
        self.disconnected.cancel();
    }
}

fn id_to_key(id: &acp::RequestId) -> String {
    match id {
        acp::RequestId::Null => "null".to_string(),
        acp::RequestId::Number(n) => format!("n:{}", n),
        acp::RequestId::Str(s) => format!("s:{}", s),
    }
}

/// Spawn the reader loop. Inbound requests and notifications are forwarded
/// to `tx`. Responses are matched against `connection.pending` internally.
/// The future completes when stdin reaches EOF.
pub(crate) fn run_reader<R>(
    connection: Arc<Connection>,
    reader: R,
    tx: mpsc::Sender<QueuedInbound>,
) -> impl std::future::Future<Output = anyhow::Result<()>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    // Construct ownership before the future is first polled. Aborting a task
    // immediately after spawn must still close its pending requests.
    let reader_owner = ReaderOwner(connection);
    async move {
        let connection = &reader_owner.0;
        let mut buffered = BufReader::new(reader);
        let mut line = Vec::new();

        loop {
            line.clear();
            let n = (&mut buffered)
                .take((MAX_FRAME_BYTES + 1) as u64)
                .read_until(b'\n', &mut line)
                .await?;
            if n == 0 {
                debug!("ACP: stdin EOF, shutting down reader");
                break;
            }
            if line.len() > MAX_FRAME_BYTES {
                anyhow::bail!("ACP input frame limit exceeded");
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let charge = input_charge(&line)?;
            // One bounded parser frame is independent of the retained queue
            // budget, allowing responses to release waiting handlers at capacity.
            let mut v: Value = match serde_json::from_slice(&line) {
                Ok(v) => v,
                Err(_) => {
                    // Spec says: if the message can't be parsed, send a parse-error
                    // response with id=null. Best-effort here since we may not even
                    // know if it was a request or notification.
                    error!(bytes = line.len(), "ACP: invalid JSON message");
                    let _ = connection
                        .send_error_response(acp::RequestId::Null, acp::Error::parse_error())
                        .await;
                    continue;
                }
            };

            // Distinguish response (has id + (result|error)) from request (has id + method)
            // from notification (no id, has method).
            let has_id = v.get("id").is_some();
            let has_method = v.get("method").is_some();
            let has_result = v.get("result").is_some();
            let has_error = v.get("error").is_some();

            if has_id && (has_result || has_error) && !has_method {
                // Response — route to pending.
                let id: acp::RequestId =
                    serde_json::from_value(v["id"].clone()).unwrap_or(acp::RequestId::Null);
                if has_result {
                    let value = v["result"].take();
                    connection.complete_pending(&id, Ok(value), charge)?;
                } else {
                    let raw_error = v["error"].take();
                    let err: acp::Error = match serde_json::from_value(raw_error.clone()) {
                        Ok(e) => e,
                        Err(_) => acp::Error::internal_error().data(Some(raw_error)),
                    };
                    connection.complete_pending(&id, Err(err), charge)?;
                }
            } else if has_id && has_method {
                // Request.
                let id: acp::RequestId =
                    serde_json::from_value(v["id"].clone()).unwrap_or(acp::RequestId::Null);
                let method = v
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let params = v.get_mut("params").map(Value::take);
                drop(v);
                let message =
                    connection.admit_inbound(Inbound::Request { id, method, params }, charge)?;
                match tx.try_send(message) {
                    Ok(()) => tokio::task::yield_now().await,
                    Err(mpsc::error::TrySendError::Closed(_)) => break,
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        anyhow::bail!("ACP inbound queue limit exceeded")
                    }
                }
            } else if has_method {
                // Notification.
                let method = v
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let params = v.get_mut("params").map(Value::take);
                drop(v);
                let message =
                    connection.admit_inbound(Inbound::Notification { method, params }, charge)?;
                match tx.try_send(message) {
                    Ok(()) => tokio::task::yield_now().await,
                    Err(mpsc::error::TrySendError::Closed(_)) => break,
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        anyhow::bail!("ACP inbound queue limit exceeded")
                    }
                }
            } else {
                warn!(
                    bytes = line.len(),
                    "ACP: unrecognized JSON-RPC message shape"
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{duplex, AsyncReadExt};

    async fn waiting_request(
        connection: &Arc<Connection>,
    ) -> tokio::task::JoinHandle<anyhow::Result<Result<Value, acp::Error>>> {
        let request_connection = connection.clone();
        let request = tokio::spawn(async move {
            request_connection
                .send_request("session/request_permission", serde_json::json!({}))
                .await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while connection.pending.lock().requests.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("request should be waiting for the client");
        request
    }

    #[tokio::test]
    async fn cancelled_request_releases_its_pending_entry() {
        let connection = Connection::new(tokio::io::sink());
        let request = waiting_request(&connection).await;
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert!(
            connection.pending.lock().requests.is_empty(),
            "cancelled request leaked its sender"
        );
    }

    #[tokio::test]
    async fn serialization_failure_does_not_register_a_request() {
        struct InvalidParams;
        impl Serialize for InvalidParams {
            fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("fixture serialization failure"))
            }
        }
        let connection = Connection::new(tokio::io::sink());
        let result: anyhow::Result<Result<Value, acp::Error>> = connection
            .send_request("session/request_permission", InvalidParams)
            .await;
        assert!(result.is_err());
        assert!(
            connection.pending.lock().requests.is_empty(),
            "serialization failure leaked a sender"
        );
    }

    #[tokio::test]
    async fn reader_eof_releases_waiters_and_rejects_later_requests() {
        let connection = Connection::new(tokio::io::sink());
        let request = waiting_request(&connection).await;
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        run_reader(connection.clone(), &b""[..], tx).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .expect("EOF must release an outstanding request")
            .unwrap();
        assert!(result.is_err());
        assert!(connection.pending.lock().requests.is_empty());
        let result: anyhow::Result<Result<Value, acp::Error>> = tokio::time::timeout(
            Duration::from_secs(1),
            connection.send_request("session/request_permission", serde_json::json!({})),
        )
        .await
        .expect("closed connection must reject registration immediately");
        assert!(result.is_err());
        assert!(connection.pending.lock().requests.is_empty());
    }

    #[tokio::test]
    async fn reader_io_error_releases_waiters() {
        struct BrokenReader;
        impl tokio::io::AsyncRead for BrokenReader {
            fn poll_read(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
                _buf: &mut tokio::io::ReadBuf<'_>,
            ) -> std::task::Poll<std::io::Result<()>> {
                std::task::Poll::Ready(Err(std::io::Error::other("fixture read failure")))
            }
        }
        let connection = Connection::new(tokio::io::sink());
        let request = waiting_request(&connection).await;
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        assert!(run_reader(connection.clone(), BrokenReader, tx)
            .await
            .is_err());
        assert!(tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .expect("reader error must release the waiter")
            .unwrap()
            .is_err());
        assert!(connection.pending.lock().requests.is_empty());
    }

    #[tokio::test]
    async fn cancelled_reader_releases_waiters() {
        let connection = Connection::new(tokio::io::sink());
        let request = waiting_request(&connection).await;
        let (_client, server_reader) = duplex(8192);
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let (started_tx, started_rx) = oneshot::channel();
        let reader_connection = connection.clone();
        let reader = tokio::spawn(async move {
            let reader = run_reader(reader_connection, server_reader, tx);
            tokio::pin!(reader);
            tokio::select! {
                biased;
                result = &mut reader => result,
                _ = async { let _ = started_tx.send(()); std::future::pending::<()>().await } => unreachable!(),
            }
        });
        started_rx.await.unwrap();
        reader.abort();
        assert!(reader.await.unwrap_err().is_cancelled());
        assert!(tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .expect("dropping reader must release the waiter")
            .unwrap()
            .is_err());
        assert!(connection.pending.lock().requests.is_empty());
    }

    #[tokio::test]
    async fn dropping_an_unpolled_reader_releases_waiters() {
        let connection = Connection::new(tokio::io::sink());
        let request = waiting_request(&connection).await;
        let (_client, server_reader) = duplex(8192);
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        drop(run_reader(connection.clone(), server_reader, tx));
        assert!(tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .expect("unpolled reader must own connection shutdown")
            .unwrap()
            .is_err());
        assert!(connection.pending.lock().requests.is_empty());
    }

    #[tokio::test]
    async fn outbound_request_saturation_is_explicit_and_does_not_replace_waiters() {
        let connection = Connection::new(tokio::io::sink());
        let mut requests = Vec::new();
        for _ in 0..64 {
            let count = connection.pending.lock().requests.len();
            let request = waiting_request(&connection).await;
            tokio::time::timeout(Duration::from_secs(1), async {
                while connection.pending.lock().requests.len() == count {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            requests.push(request);
        }
        let result: anyhow::Result<Result<Value, acp::Error>> = tokio::time::timeout(
            Duration::from_secs(1),
            connection.send_request("session/request_permission", serde_json::json!({})),
        )
        .await
        .expect("saturated connection must fail immediately");
        assert!(result.is_err());
        assert_eq!(connection.pending.lock().requests.len(), 64);
        for request in requests {
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
        }
        assert!(connection.pending.lock().requests.is_empty());
    }

    #[tokio::test]
    async fn post_eof_request_emits_no_bytes() {
        let (mut client, writer) = duplex(8192);
        let connection = Connection::new(writer);
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        run_reader(connection.clone(), &b""[..], tx).await.unwrap();
        let result: anyhow::Result<Result<Value, acp::Error>> = connection
            .send_request("session/request_permission", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        drop(connection);
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.unwrap();
        assert!(bytes.is_empty(), "closed connection emitted a new request");
    }

    #[tokio::test]
    async fn eof_interrupts_a_request_blocked_on_output() {
        let (mut client, writer) = duplex(1);
        let connection = Connection::new(writer);
        let request_connection = connection.clone();
        let request = tokio::spawn(async move {
            request_connection
                .send_request::<_, Value>("session/request_permission", serde_json::json!({}))
                .await
        });
        let mut byte = [0];
        client.read_exact(&mut byte).await.unwrap();
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        run_reader(connection.clone(), &b""[..], tx).await.unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .expect("EOF must also interrupt a blocked request write")
            .unwrap()
            .is_err());
        assert!(connection.pending.lock().requests.is_empty());
    }

    #[tokio::test]
    async fn cancelling_a_partial_frame_prevents_framing_reuse() {
        let (mut client, writer) = duplex(1);
        let connection = Connection::new(writer);
        let request_connection = connection.clone();
        let request = tokio::spawn(async move {
            request_connection
                .send_request::<_, Value>("session/request_permission", serde_json::json!({}))
                .await
        });
        let mut byte = [0];
        client.read_exact(&mut byte).await.unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert!(connection.pending.lock().requests.is_empty());
        assert!(connection
            .send_response(acp::RequestId::Number(1), serde_json::json!({}))
            .await
            .is_err());
        drop(connection);
        let mut remainder = Vec::new();
        client.read_to_end(&mut remainder).await.unwrap();
        assert!(
            remainder.len() <= 1,
            "a second frame followed an interrupted frame"
        );
    }

    #[tokio::test]
    async fn write_failure_releases_pending_entry_and_closes_connection() {
        let (client, writer) = duplex(1);
        drop(client);
        let connection = Connection::new(writer);
        let result = connection
            .send_request::<_, Value>("session/request_permission", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        assert!(connection.pending.lock().requests.is_empty());
        assert!(connection.pending.lock().closed);
    }

    #[tokio::test]
    async fn oversized_inbound_frame_is_rejected_before_dispatch() {
        let connection = Connection::new(tokio::io::sink());
        let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let mut input = b"{\"id\":1,\"method\":\"initialize\",\"params\":\"".to_vec();
        input.resize(32 * 1024 * 1024 + 1, b'x');
        input.extend_from_slice(b"\"}\n");
        let result = run_reader(connection, input.as_slice(), tx).await;
        assert!(result.is_err(), "oversized frame was accepted");
        assert_eq!(rx.len(), 0, "oversized frame reached dispatch");
    }

    #[tokio::test]
    async fn dense_json_is_rejected_before_dispatch() {
        let connection = Connection::new(tokio::io::sink());
        let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let mut input = b"{\"id\":1,\"method\":\"initialize\",\"params\":[".to_vec();
        input.extend_from_slice("0,".repeat(100_001).as_bytes());
        input.extend_from_slice(b"0]}\n");
        assert!(run_reader(connection, input.as_slice(), tx).await.is_err());
        assert_eq!(rx.len(), 0, "dense frame reached dispatch");
    }

    #[tokio::test]
    async fn inbound_queue_saturation_closes_instead_of_growing() {
        let connection = Connection::new(tokio::io::sink());
        let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let input = b"{\"id\":1,\"method\":\"initialize\",\"params\":{}}\n".repeat(9);
        assert!(run_reader(connection, input.as_slice(), tx).await.is_err());
        assert_eq!(rx.len(), 8);
    }

    #[tokio::test]
    async fn oversized_outbound_frame_fails_without_closing_healthy_output() {
        let connection = Connection::new(tokio::io::sink());
        let text = "x".repeat(32 * 1024 * 1024);
        assert!(connection.send_notification("fixture", text).await.is_err());
        connection
            .send_response(acp::RequestId::Number(1), serde_json::json!({}))
            .await
            .unwrap();
    }

    #[test]
    fn json_charge_ignores_quoted_and_escaped_punctuation() {
        let value = serde_json::json!({"text":"[,]:\\\"".repeat(100_001)});
        let bytes = serde_json::to_vec(&value).unwrap();
        let charge = input_charge(&bytes).unwrap();
        assert!(charge <= bytes.len() * 2 + 4096);
        let too_deep = format!("{}0{}", "[".repeat(129), "]".repeat(129));
        assert!(input_charge(too_deep.as_bytes()).is_err());
    }

    #[test]
    fn output_buffer_bounds_capacity_growth_for_small_serializer_writes() {
        let mut buffer = FrameBuffer(Vec::new());
        let mut growths = 0;
        for _ in 0..100_000 {
            let previous = buffer.0.capacity();
            std::io::Write::write_all(&mut buffer, b"x").unwrap();
            growths += usize::from(buffer.0.capacity() != previous);
            assert!(buffer.0.capacity() <= MAX_FRAME_BYTES);
        }
        assert!(
            growths <= 16,
            "small serializer writes repeatedly reallocated: {growths}"
        );
        assert_eq!(buffer.0.len(), 100_000);
    }

    #[tokio::test]
    async fn exact_inbound_frame_boundary_is_accepted() {
        let connection = Connection::new(tokio::io::sink());
        let (tx, mut rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let mut input = b"{\"id\":1,\"method\":\"fixture\",\"params\":\"".to_vec();
        input.resize(MAX_FRAME_BYTES - 3, b'x');
        input.extend_from_slice(b"\"}\n");
        assert_eq!(input.len(), MAX_FRAME_BYTES);
        run_reader(connection, input.as_slice(), tx).await.unwrap();
        assert!(matches!(
            rx.recv().await.unwrap().message,
            Inbound::Request { .. }
        ));
    }

    #[tokio::test]
    async fn exact_outbound_frame_boundary_is_accepted() {
        let connection = Connection::new(tokio::io::sink());
        connection
            .write_line(&Value::String("x".repeat(MAX_FRAME_BYTES - 3)))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn input_budget_rejection_releases_waiters_without_dispatch() {
        let mut connection = Connection::new(tokio::io::sink());
        Arc::get_mut(&mut connection).unwrap().inbound_budget = Arc::new(Semaphore::new(1));
        let request = waiting_request(&connection).await;
        let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        assert!(run_reader(
            connection.clone(),
            &b"{\"id\":1,\"method\":\"fixture\"}\n"[..],
            tx
        )
        .await
        .is_err());
        assert_eq!(rx.len(), 0);
        assert!(request.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn blocked_output_has_a_finite_deadline() {
        let (_client, writer) = duplex(1);
        let connection = Connection::new(writer);
        let result = tokio::time::timeout(
            WRITE_TIMEOUT + Duration::from_secs(1),
            connection.send_notification("fixture", serde_json::json!({})),
        )
        .await
        .expect("blocked output must observe the connection deadline");
        assert!(result.is_err());
        assert!(connection.pending.lock().closed);
    }

    #[tokio::test]
    async fn responses_in_waiting_channels_keep_their_input_charge() {
        let mut connection = Connection::new(tokio::io::sink());
        Arc::get_mut(&mut connection).unwrap().inbound_budget = Arc::new(Semaphore::new(4096));
        let first = waiting_request(&connection).await;
        let second = waiting_request(&connection).await;
        tokio::time::timeout(Duration::from_secs(1), async {
            while connection.pending.lock().requests.len() != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        connection
            .complete_pending(&acp::RequestId::Number(1), Ok(serde_json::json!({})), 3000)
            .unwrap();
        assert!(connection
            .complete_pending(&acp::RequestId::Number(2), Ok(serde_json::json!({})), 3000)
            .is_err());
        assert_eq!(connection.inbound_budget.available_permits(), 1096);
        assert!(first.await.unwrap().unwrap().is_ok());
        assert!(second.await.unwrap().is_err());
        assert_eq!(connection.inbound_budget.available_permits(), 4096);
    }

    #[tokio::test]
    async fn responses_route_when_the_inbound_queue_is_full() {
        let connection = Connection::new(tokio::io::sink());
        let request = waiting_request(&connection).await;
        let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let mut input = b"{\"id\":99,\"method\":\"fixture\"}\n".repeat(MAX_INBOUND_QUEUE);
        input.extend_from_slice(b"{\"id\":1,\"result\":{\"ok\":true}}\n");
        run_reader(connection, input.as_slice(), tx).await.unwrap();
        assert_eq!(rx.len(), MAX_INBOUND_QUEUE);
        assert_eq!(request.await.unwrap().unwrap().unwrap()["ok"], true);
    }

    #[test]
    fn id_to_key_distinguishes_number_and_string_ids_with_the_same_text() {
        // A numeric id and a string id that render the same digits must not
        // collide in the pending-request map.
        let number_key = id_to_key(&acp::RequestId::Number(7));
        let string_key = id_to_key(&acp::RequestId::Str("7".into()));
        assert_ne!(number_key, string_key);
        assert_eq!(number_key, id_to_key(&acp::RequestId::Number(7)));
        assert_eq!(string_key, id_to_key(&acp::RequestId::Str("7".into())));
    }

    #[test]
    fn id_to_key_null_is_stable() {
        assert_eq!(
            id_to_key(&acp::RequestId::Null),
            id_to_key(&acp::RequestId::Null)
        );
    }

    /// `send_response` must emit a newline-delimited JSON object containing
    /// `jsonrpc: "2.0"`, the request id, and the result payload.
    #[tokio::test]
    async fn send_response_writes_newline_delimited_jsonrpc() {
        let (client_read, server_write) = duplex(8192);
        let connection = Connection::new(server_write);

        connection
            .send_response(acp::RequestId::Number(7), serde_json::json!({"ok": true}))
            .await
            .unwrap();
        drop(connection);

        let mut buf = Vec::new();
        let mut client_read = client_read;
        client_read.read_to_end(&mut buf).await.unwrap();
        let line = std::str::from_utf8(&buf).unwrap();
        assert!(line.ends_with('\n'), "response missing newline: {line:?}");
        let parsed: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 7);
        assert_eq!(parsed["result"]["ok"], true);
    }

    /// `send_notification` must omit `id` so the receiver knows not to reply.
    #[tokio::test]
    async fn notifications_have_no_id() {
        let (client_read, server_write) = duplex(8192);
        let connection = Connection::new(server_write);

        connection
            .send_notification("session/update", serde_json::json!({"sessionId": "s1"}))
            .await
            .unwrap();
        drop(connection);

        let mut buf = Vec::new();
        let mut client_read = client_read;
        client_read.read_to_end(&mut buf).await.unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(std::str::from_utf8(&buf).unwrap().trim()).unwrap();
        assert!(parsed.get("id").is_none(), "notification must not carry id");
        assert_eq!(parsed["method"], "session/update");
    }

    /// The reader must distinguish requests, notifications, and responses by
    /// shape and route each to the right callback path.
    #[tokio::test]
    async fn reader_routes_inbound_messages() {
        // Drive the reader from an in-memory pipe.
        let (writer_handle, server_reader) = duplex(8192);
        let (_to_client, client_writer) = duplex(8192);
        let connection = Connection::new(client_writer);
        let (tx, mut rx) = mpsc::channel(MAX_INBOUND_QUEUE);

        let reader_handle = tokio::spawn(run_reader(connection.clone(), server_reader, tx));

        // Inject a request, a notification, and an unsolicited response.
        let mut writer_handle = writer_handle;
        writer_handle
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":1}}\n",
            )
            .await
            .unwrap();
        writer_handle
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"method\":\"session/cancel\",\"params\":{\"sessionId\":\"s1\"}}\n",
            )
            .await
            .unwrap();
        writer_handle
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{\"orphan\":true}}\n")
            .await
            .unwrap();
        drop(writer_handle); // EOF the reader

        let first = rx.recv().await.expect("expected request");
        match first.message {
            Inbound::Request { id, method, .. } => {
                assert_eq!(method, "initialize");
                assert!(matches!(id, acp::RequestId::Number(1)));
            }
            other => panic!("expected Request, got {other:?}"),
        }
        let second = rx.recv().await.expect("expected notification");
        match second.message {
            Inbound::Notification { method, .. } => {
                assert_eq!(method, "session/cancel");
            }
            other => panic!("expected Notification, got {other:?}"),
        }
        // Orphan response is routed internally; nothing surfaces here.
        assert!(rx.recv().await.is_none(), "no further messages expected");
        let _ = reader_handle.await;
    }

    /// `send_request` resolves when a response with the matching id arrives.
    #[tokio::test]
    async fn send_request_resolves_on_matching_response() {
        let (client_to_server, server_reader) = duplex(8192);
        let (server_to_client_reader, server_to_client) = duplex(8192);
        let connection = Connection::new(server_to_client);
        let (tx, _rx) = mpsc::channel(MAX_INBOUND_QUEUE);
        let reader_handle = tokio::spawn(run_reader(connection.clone(), server_reader, tx));

        // Background: as a fake client, read the outbound request and write a
        // matching response.
        let client_loop = tokio::spawn(async move {
            let mut buf = Vec::new();
            let mut server_to_client_reader = server_to_client_reader;
            let mut byte = [0u8; 1];
            // Read up to one full line.
            loop {
                let n = server_to_client_reader.read(&mut byte).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.push(byte[0]);
                if byte[0] == b'\n' {
                    break;
                }
            }
            let outbound: serde_json::Value = serde_json::from_slice(buf.trim_ascii_end()).unwrap();
            let id = outbound["id"].clone();
            // Send the response back through the client_to_server pipe.
            let response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {"echo": outbound["method"].as_str().unwrap()}
            });
            let mut bytes = serde_json::to_vec(&response).unwrap();
            bytes.push(b'\n');
            let mut client_to_server = client_to_server;
            client_to_server.write_all(&bytes).await.unwrap();
            drop(client_to_server);
        });

        let response: Result<serde_json::Value, acp::Error> = connection
            .send_request("fs/read_text_file", serde_json::json!({"path": "/x"}))
            .await
            .unwrap();
        let response = response.unwrap();
        assert_eq!(response["echo"], "fs/read_text_file");
        let _ = client_loop.await;
        let _ = reader_handle.await;
    }
}
