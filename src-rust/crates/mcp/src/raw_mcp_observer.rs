//! Bounded raw MCP JSON-RPC observation for stdio transports.
//!
//! This is deliberately below rmcp's typed decoder: it observes the exact
//! bytes written to and read from the child pipes, then separately parses a
//! bounded summary for request/response identity and argument/result digests.
//! The raw frame is retained for the controller ledger; it is never used as a
//! replacement for typed rmcp validation.

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

pub(crate) const MAX_RAW_FRAME_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_RAW_TOTAL_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const MAX_RAW_FRAMES: usize = 4096;
pub(crate) const MAX_RECEIPT_FILE_BYTES: usize = 40 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawMcpDirection {
    Request,
    Response,
}

#[derive(Debug, Clone)]
pub(crate) struct RawMcpFrame {
    pub(crate) direction: RawMcpDirection,
    pub(crate) bytes: Vec<u8>,
    pub(crate) sha256: String,
    pub(crate) id: Option<Value>,
    pub(crate) method: Option<String>,
    pub(crate) cursor: Option<String>,
    pub(crate) arguments_sha256: Option<String>,
    pub(crate) result_sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct RawMcpReceipt {
    pub(crate) frames: Vec<RawMcpFrame>,
    pub(crate) request_partial: Vec<u8>,
    pub(crate) response_partial: Vec<u8>,
    pub(crate) total_bytes: usize,
    pub(crate) overflow: bool,
}

#[derive(Debug, Default)]
struct RawMcpTrace {
    request_partial: Vec<u8>,
    response_partial: Vec<u8>,
    frames: Vec<RawMcpFrame>,
    total_bytes: usize,
    overflow: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct RawMcpObserver {
    // None is the zero-retention default: no trace mutex or frame buffers are
    // allocated until the controller supplies an explicit receipt sink.
    trace: Option<Arc<Mutex<RawMcpTrace>>>,
}

/// A controller-selected, create-once receipt destination.
///
/// The path is never guessed or inherited from the agent workspace.  The
/// controller validates and supplies it explicitly; this type repeats the
/// no-symlink/regular-file checks at the native write boundary.
#[derive(Debug, Clone)]
pub(crate) struct RawMcpReceiptSink {
    path: PathBuf,
}

impl RawMcpReceiptSink {
    pub(crate) fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "receipt sink must be absolute",
            ));
        }
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "receipt sink has no parent")
        })?;
        let parent_meta = fs::symlink_metadata(parent)?;
        if !parent_meta.file_type().is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "receipt sink parent is not a directory",
            ));
        }
        let mut current = PathBuf::new();
        for component in path.components() {
            current.push(component);
            if let Ok(meta) = fs::symlink_metadata(&current) {
                if meta.file_type().is_symlink() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "receipt sink path contains a symlink",
                    ));
                }
            }
        }
        Ok(Self {
            path: path.to_owned(),
        })
    }

    pub(crate) fn write(&self, receipt: &RawMcpReceipt) -> io::Result<()> {
        self.write_with_operation_contexts(receipt, &[])
    }

    pub(crate) fn write_with_operation_contexts(
        &self,
        receipt: &RawMcpReceipt,
        operation_contexts: &[Value],
    ) -> io::Result<()> {
        if operation_contexts.len() > 256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "operation context count exceeds bound",
            ));
        }
        let bytes = receipt_json_bytes(receipt, operation_contexts)?;
        if bytes.len() > MAX_RECEIPT_FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "receipt serialization exceeds bound",
            ));
        }
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(&self.path)?;
        file.write_all(&bytes)?;
        file.flush()?;
        file.sync_all()?;
        let metadata = file.metadata()?;
        if !metadata.file_type().is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "receipt sink is not regular",
            ));
        }
        #[cfg(unix)]
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "receipt sink mode is not private",
            ));
        }
        Ok(())
    }
}

fn receipt_json_bytes(
    receipt: &RawMcpReceipt,
    operation_contexts: &[Value],
) -> io::Result<Vec<u8>> {
    if receipt.frames.len() > MAX_RAW_FRAMES
        || receipt.total_bytes > MAX_RAW_TOTAL_BYTES
        || receipt.request_partial.len() > MAX_RAW_FRAME_BYTES
        || receipt.response_partial.len() > MAX_RAW_FRAME_BYTES
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "receipt exceeds capture bound",
        ));
    }
    let frames: Vec<Value> = receipt
        .frames
        .iter()
        .map(|frame| {
            json!({
                "direction": match frame.direction {
                    RawMcpDirection::Request => "request",
                    RawMcpDirection::Response => "response",
                },
                "bytes_b64": base64::engine::general_purpose::STANDARD.encode(&frame.bytes),
                "sha256": frame.sha256.clone(),
                "id": frame.id.clone(),
                "method": frame.method.clone(),
                "cursor": frame.cursor.clone(),
                "arguments_sha256": frame.arguments_sha256.clone(),
                "result_sha256": frame.result_sha256.clone(),
            })
        })
        .collect();
    let value = json!({
        "schema": "forge.p07.raw-mcp-receipt.v1",
        "frames": frames,
        "request_partial_b64": base64::engine::general_purpose::STANDARD.encode(&receipt.request_partial),
        "response_partial_b64": base64::engine::general_purpose::STANDARD.encode(&receipt.response_partial),
        "total_bytes": receipt.total_bytes,
        "overflow": receipt.overflow,
        "operation_contexts": operation_contexts,
    });
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if bytes.len() > MAX_RECEIPT_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "receipt serialization exceeds bound",
        ));
    }
    Ok(bytes)
}

impl RawMcpObserver {
    pub(crate) fn new() -> Self {
        Self {
            trace: Some(Arc::new(Mutex::new(RawMcpTrace::default()))),
        }
    }

    pub(crate) fn disabled() -> Self {
        Self { trace: None }
    }

    pub(crate) fn record(&self, direction: RawMcpDirection, bytes: &[u8]) {
        let Some(trace) = self.trace.as_ref() else {
            return;
        };
        let Ok(mut trace) = trace.lock() else {
            return;
        };
        if trace.overflow {
            return;
        }
        trace.total_bytes = match trace.total_bytes.checked_add(bytes.len()) {
            Some(total) if total <= MAX_RAW_TOTAL_BYTES => total,
            _ => {
                trace.overflow = true;
                return;
            }
        };
        for byte in bytes {
            let too_long = {
                let partial = match direction {
                    RawMcpDirection::Request => &trace.request_partial,
                    RawMcpDirection::Response => &trace.response_partial,
                };
                *byte != b'\n' && partial.len() >= MAX_RAW_FRAME_BYTES
            };
            if too_long {
                trace.overflow = true;
                return;
            }
            let mut frame = None;
            {
                let partial = match direction {
                    RawMcpDirection::Request => &mut trace.request_partial,
                    RawMcpDirection::Response => &mut trace.response_partial,
                };
                partial.push(*byte);
                if *byte == b'\n' {
                    frame = Some(std::mem::take(partial));
                }
            }
            let Some(mut frame) = frame else {
                continue;
            };
            frame.pop();
            if frame.last() == Some(&b'\r') {
                frame.pop();
            }
            if frame.is_empty() {
                continue;
            }
            if trace.frames.len() >= MAX_RAW_FRAMES {
                trace.overflow = true;
                return;
            }
            trace.frames.push(summarize(direction, frame));
        }
    }

    pub(crate) fn receipt(&self) -> RawMcpReceipt {
        let Some(trace) = self.trace.as_ref() else {
            return RawMcpReceipt {
                frames: Vec::new(),
                request_partial: Vec::new(),
                response_partial: Vec::new(),
                total_bytes: 0,
                overflow: false,
            };
        };
        let Ok(trace) = trace.lock() else {
            return RawMcpReceipt {
                frames: Vec::new(),
                request_partial: Vec::new(),
                response_partial: Vec::new(),
                total_bytes: 0,
                overflow: true,
            };
        };
        RawMcpReceipt {
            frames: trace.frames.clone(),
            request_partial: trace.request_partial.clone(),
            response_partial: trace.response_partial.clone(),
            total_bytes: trace.total_bytes,
            overflow: trace.overflow,
        }
    }
}

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn value_digest(value: Option<&Value>) -> Option<String> {
    value
        .and_then(|value| serde_json::to_value(value).ok())
        .and_then(|value| serde_json::to_vec(&value).ok())
        .map(|bytes| digest(&bytes))
}

fn summarize(direction: RawMcpDirection, bytes: Vec<u8>) -> RawMcpFrame {
    let parsed = serde_json::from_slice::<Value>(&bytes).ok();
    let id = parsed.as_ref().and_then(|value| value.get("id").cloned());
    let method = parsed
        .as_ref()
        .and_then(|value| value.get("method"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let params = parsed.as_ref().and_then(|value| value.get("params"));
    let result = parsed.as_ref().and_then(|value| value.get("result"));
    let cursor = params
        .and_then(|value| value.get("cursor"))
        .or_else(|| result.and_then(|value| value.get("nextCursor")))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let arguments = params.and_then(|value| value.get("arguments"));
    RawMcpFrame {
        direction,
        sha256: digest(&bytes),
        bytes,
        id,
        method,
        cursor,
        arguments_sha256: value_digest(arguments),
        result_sha256: value_digest(result),
    }
}

pub(crate) struct ObservedReader<R> {
    inner: R,
    observer: RawMcpObserver,
}

impl<R> ObservedReader<R> {
    pub(crate) fn new(inner: R, observer: RawMcpObserver) -> Self {
        Self { inner, observer }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for ObservedReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        destination: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.as_mut().get_mut();
        let before = destination.filled().len();
        let result = Pin::new(&mut this.inner).poll_read(cx, destination);
        if let Poll::Ready(Ok(())) = &result {
            let filled = destination.filled();
            if filled.len() > before {
                this.observer
                    .record(RawMcpDirection::Response, &filled[before..]);
            }
        }
        result
    }
}

pub(crate) struct ObservedWriter<W> {
    inner: W,
    observer: RawMcpObserver,
}

impl<W> ObservedWriter<W> {
    pub(crate) fn new(inner: W, observer: RawMcpObserver) -> Self {
        Self { inner, observer }
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for ObservedWriter<W> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.as_mut().get_mut();
        match Pin::new(&mut this.inner).poll_write(cx, bytes) {
            Poll::Ready(Ok(written)) => {
                if written != 0 {
                    this.observer
                        .record(RawMcpDirection::Request, &bytes[..written]);
                }
                Poll::Ready(Ok(written))
            }
            other => other,
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.as_mut().get_mut().inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.as_mut().get_mut().inner).poll_shutdown(cx)
    }
}
