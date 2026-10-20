//! Source-bound MCP operation receipts for the opt-in rmcp capture path.
//!
//! The transport observer remains the authority for serialized JSON-RPC
//! frames.  This module only binds the ID returned by rmcp's RequestHandle to
//! one request/response pair already present in that observer snapshot.  It
//! never pairs frames by ordering or by matching arguments.

use crate::raw_mcp_observer::{RawMcpDirection, RawMcpReceipt};
use rmcp::model as rmcp_model;
use rmcp::service::{PeerRequestOptions, RoleClient};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

const CALL_TOOL_METHOD: &str = "tools/call";
const MAX_OPERATION_CONTEXTS: usize = 256;
const MAX_OPERATION_CONTEXT_BYTES: usize = 1024 * 1024;

/// Exact source descriptor retained by the backend before the wire call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpOperationBackend {
    pub server_name: String,
    pub raw_name: String,
    pub callable_name: String,
    pub source_identity: String,
    pub connection_generation: u64,
    pub catalog_generation: u64,
    pub definition_digest: String,
}

impl McpOperationBackend {
    pub fn from_descriptor(descriptor: &crate::catalog_state::McpToolDescriptor) -> Self {
        Self {
            server_name: descriptor.server_name().to_owned(),
            raw_name: descriptor.raw_name().to_owned(),
            callable_name: descriptor.callable_name().to_owned(),
            source_identity: descriptor.source_identity().to_string(),
            connection_generation: descriptor.connection_generation(),
            catalog_generation: descriptor.catalog_generation(),
            definition_digest: descriptor.definition_digest().to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum McpSessionCompleteness {
    /// Every observed frame was individually framed and no partial bytes or
    /// hard overflow were reported.  This is a snapshot property only; the
    /// observer has no EOF/transport-close bit and this variant makes no
    /// claim that the session is complete.
    SnapshotFramed,
    Unavailable,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpQueryOperationIdentity {
    /// Provider content-block tool ID, when a later query adapter supplies it.
    pub provider_tool_id: Option<String>,
    /// Provider/native assistant response ID, kept separate from MCP JSON-RPC.
    pub observed_assistant_response_id: Option<String>,
    /// Query-local generated UUID, when a later query adapter supplies it.
    pub query_generated_uuid: Option<String>,
    /// A query-owned selection must match the backend that actually dispatches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_backend: Option<McpOperationBackend>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpOperationContext {
    /// The exact JSON-RPC ID returned by rmcp::RequestHandle::id.
    pub json_rpc_request_id: Value,
    /// The exact response-frame ID; it must equal json_rpc_request_id.
    pub json_rpc_response_id: Value,
    pub method: String,
    pub tool_name: String,
    pub arguments_sha256: Option<String>,
    pub result_sha256: String,
    pub request_frame_sha256: String,
    pub response_frame_sha256: String,
    pub request_frame_ordinal: usize,
    pub response_frame_ordinal: usize,
    pub session_completeness: McpSessionCompleteness,
    pub query_identity: McpQueryOperationIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<McpOperationBackend>,
}

pub type McpOperationObserver = Arc<dyn Fn(&McpOperationContext) + Send + Sync>;

tokio::task_local! {
    static CURRENT_QUERY_OPERATION_IDENTITY: McpQueryOperationIdentity;
    static CURRENT_QUERY_OPERATION_OBSERVER: McpOperationObserver;
}

/// Observe only a completed, raw-ID-bound operation on the selected backend.
pub async fn with_mcp_query_operation_observer<F, T>(
    identity: McpQueryOperationIdentity,
    observer: McpOperationObserver,
    future: F,
) -> T
where
    F: std::future::Future<Output = T>,
{
    CURRENT_QUERY_OPERATION_OBSERVER
        .scope(
            observer,
            with_mcp_query_operation_identity(identity, future),
        )
        .await
}

pub(crate) fn current_query_operation_observer() -> Option<McpOperationObserver> {
    CURRENT_QUERY_OPERATION_OBSERVER.try_with(Arc::clone).ok()
}

/// Run one query operation with an exact caller-supplied identity.  The
/// identity is read by the backend before dispatch and copied into the
/// corresponding ID-bound context; no post-hoc inference is performed.
pub async fn with_mcp_query_operation_identity<F, T>(
    identity: McpQueryOperationIdentity,
    future: F,
) -> T
where
    F: std::future::Future<Output = T>,
{
    CURRENT_QUERY_OPERATION_IDENTITY
        .scope(identity, future)
        .await
}

pub(crate) fn current_query_operation_identity() -> Option<McpQueryOperationIdentity> {
    CURRENT_QUERY_OPERATION_IDENTITY.try_with(Clone::clone).ok()
}

/// A bounded query integration surface.  Callers may provide an exact
/// task-local identity before dispatch. Recorded identities are immutable;
/// no post-hoc identity attachment or completion-order inference is allowed.
#[derive(Debug, Clone)]
pub(crate) struct McpOperationContextStore {
    contexts: Arc<Mutex<Vec<McpOperationContext>>>,
    context_bytes: Arc<Mutex<usize>>,
}

impl Default for McpOperationContextStore {
    fn default() -> Self {
        Self {
            contexts: Arc::new(Mutex::new(Vec::new())),
            context_bytes: Arc::new(Mutex::new(0)),
        }
    }
}

impl McpOperationContextStore {
    pub(crate) fn record(&self, context: McpOperationContext) -> anyhow::Result<()> {
        let mut contexts = self
            .contexts
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP operation context store poisoned"))?;
        if contexts
            .iter()
            .any(|existing| existing.json_rpc_request_id == context.json_rpc_request_id)
        {
            return Err(anyhow::anyhow!(
                "duplicate MCP operation context for JSON-RPC request ID"
            ));
        }
        if contexts.len() >= MAX_OPERATION_CONTEXTS {
            return Err(anyhow::anyhow!("MCP operation context count exceeds bound"));
        }
        let encoded_len = bounded_context_size(&context)?;
        let mut context_bytes = self
            .context_bytes
            .lock()
            .map_err(|_| anyhow::anyhow!("MCP operation context size store poisoned"))?;
        let next = context_bytes
            .checked_add(encoded_len)
            .ok_or_else(|| anyhow::anyhow!("MCP operation context byte count overflow"))?;
        if next > MAX_OPERATION_CONTEXT_BYTES {
            return Err(anyhow::anyhow!("MCP operation context bytes exceed bound"));
        }
        contexts.push(context);
        *context_bytes = next;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> anyhow::Result<Vec<McpOperationContext>> {
        self.contexts
            .lock()
            .map(|contexts| contexts.clone())
            .map_err(|_| anyhow::anyhow!("MCP operation context store poisoned"))
    }
}

/// Send a typed tools/call request while retaining rmcp's actual request ID.
/// This helper is used only when the controller opted into raw capture.
pub(crate) async fn send_call_tool_with_request_id(
    peer: &rmcp::Peer<RoleClient>,
    params: rmcp_model::CallToolRequestParams,
) -> anyhow::Result<(rmcp_model::CallToolResult, rmcp_model::RequestId)> {
    let handle = peer
        .send_cancellable_request(
            rmcp_model::ClientRequest::CallToolRequest(rmcp_model::CallToolRequest::new(params)),
            PeerRequestOptions::no_options(),
        )
        .await
        .map_err(|error| anyhow::anyhow!("rmcp tools/call request failed: {error}"))?;
    let request_id = handle.id.clone();
    let response = handle
        .await_response()
        .await
        .map_err(|error| anyhow::anyhow!("rmcp tools/call response failed: {error}"))?;
    let result = match response {
        rmcp_model::ServerResult::CallToolResult(result) => result,
        other => {
            return Err(anyhow::anyhow!(
                "rmcp tools/call returned unexpected server result: {:?}",
                other
            ));
        }
    };
    Ok((result, request_id))
}

/// Bind one typed call result to exactly one raw request/response pair.
pub(crate) fn bind_call_tool_receipt(
    request_id: &rmcp_model::RequestId,
    tool_name: &str,
    arguments: Option<&Value>,
    result: &rmcp_model::CallToolResult,
    receipt: &RawMcpReceipt,
) -> anyhow::Result<McpOperationContext> {
    if receipt.overflow {
        return Err(anyhow::anyhow!(
            "raw MCP receipt overflow prevents operation binding"
        ));
    }
    let session_completeness =
        if receipt.request_partial.is_empty() && receipt.response_partial.is_empty() {
            McpSessionCompleteness::SnapshotFramed
        } else {
            McpSessionCompleteness::Unavailable
        };
    let expected_id = serde_json::to_value(request_id)
        .map_err(|error| anyhow::anyhow!("MCP request ID is not serializable: {error}"))?;
    let expected_arguments_sha256 = arguments.map(canonical_digest).transpose()?;
    let typed_result_sha256 = canonical_digest(result)?;

    let mut request_match: Option<(usize, &crate::raw_mcp_observer::RawMcpFrame)> = None;
    let mut response_match: Option<(usize, &crate::raw_mcp_observer::RawMcpFrame)> = None;
    for (ordinal, frame) in receipt.frames.iter().enumerate() {
        if frame.id.as_ref() != Some(&expected_id) {
            continue;
        }
        validate_frame_integrity(frame)?;
        match frame.direction {
            RawMcpDirection::Request => {
                if request_match.is_some() {
                    return Err(anyhow::anyhow!(
                        "duplicate raw MCP request frame for tools/call request ID"
                    ));
                }
                request_match = Some((ordinal, frame));
            }
            RawMcpDirection::Response => {
                if response_match.is_some() {
                    return Err(anyhow::anyhow!(
                        "duplicate raw MCP response frame for tools/call request ID"
                    ));
                }
                response_match = Some((ordinal, frame));
            }
        }
    }

    let (request_ordinal, request) = request_match.ok_or_else(|| {
        anyhow::anyhow!("raw MCP receipt has no request frame for rmcp request ID")
    })?;
    let (response_ordinal, response) = response_match.ok_or_else(|| {
        anyhow::anyhow!("raw MCP receipt has no response frame for rmcp request ID")
    })?;
    if response_ordinal <= request_ordinal {
        return Err(anyhow::anyhow!(
            "raw MCP response frame does not follow its request frame"
        ));
    }
    if request.method.as_deref() != Some(CALL_TOOL_METHOD) {
        return Err(anyhow::anyhow!("raw MCP request method is not tools/call"));
    }
    let request_json: Value = serde_json::from_slice(&request.bytes)
        .map_err(|error| anyhow::anyhow!("raw MCP request is invalid JSON: {error}"))?;
    let params = request_json
        .get("params")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("raw MCP tools/call request has no params object"))?;
    if params.get("name").and_then(Value::as_str) != Some(tool_name) {
        return Err(anyhow::anyhow!(
            "raw MCP tools/call name does not match the typed request"
        ));
    }
    if request.arguments_sha256 != expected_arguments_sha256 {
        return Err(anyhow::anyhow!(
            "raw MCP tools/call arguments digest does not match the typed request"
        ));
    }
    if response.result_sha256.as_deref() != Some(typed_result_sha256.as_str()) {
        return Err(anyhow::anyhow!(
            "raw MCP tools/call result digest does not match the typed response"
        ));
    }
    if response.method.is_some() || response.id.as_ref() != Some(&expected_id) {
        return Err(anyhow::anyhow!(
            "raw MCP response frame has invalid response identity"
        ));
    }

    Ok(McpOperationContext {
        json_rpc_request_id: expected_id.clone(),
        json_rpc_response_id: response.id.clone().unwrap_or(Value::Null),
        method: CALL_TOOL_METHOD.to_owned(),
        tool_name: tool_name.to_owned(),
        arguments_sha256: request.arguments_sha256.clone(),
        result_sha256: response
            .result_sha256
            .clone()
            .ok_or_else(|| anyhow::anyhow!("raw MCP response has no result digest"))?,
        request_frame_sha256: request.sha256.clone(),
        response_frame_sha256: response.sha256.clone(),
        request_frame_ordinal: request_ordinal,
        response_frame_ordinal: response_ordinal,
        session_completeness,
        query_identity: McpQueryOperationIdentity::default(),
        backend: None,
    })
}

fn validate_frame_integrity(frame: &crate::raw_mcp_observer::RawMcpFrame) -> anyhow::Result<()> {
    if digest_bytes(&frame.bytes) != frame.sha256 {
        return Err(anyhow::anyhow!(
            "raw MCP frame SHA-256 does not match captured bytes"
        ));
    }
    let parsed: Value = serde_json::from_slice(&frame.bytes)
        .map_err(|error| anyhow::anyhow!("raw MCP frame is invalid JSON: {error}"))?;
    let object = parsed
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("raw MCP frame envelope is not a JSON object"))?;
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(anyhow::anyhow!(
            "raw MCP frame does not declare JSON-RPC version 2.0"
        ));
    }
    let has_result = object.contains_key("result");
    let has_error = object.contains_key("error");
    if matches!(frame.direction, RawMcpDirection::Response) && has_result == has_error {
        return Err(anyhow::anyhow!(
            "raw MCP frame must contain exactly one result or error member"
        ));
    }
    if parsed.get("id") != frame.id.as_ref() {
        return Err(anyhow::anyhow!(
            "raw MCP frame ID metadata does not match captured bytes"
        ));
    }
    let method = parsed.get("method").and_then(Value::as_str);
    if method != frame.method.as_deref() {
        return Err(anyhow::anyhow!(
            "raw MCP frame method metadata does not match captured bytes"
        ));
    }
    let id = object
        .get("id")
        .ok_or_else(|| anyhow::anyhow!("raw MCP frame has no JSON-RPC ID"))?;
    if !(id.is_string() || id.is_number()) {
        return Err(anyhow::anyhow!(
            "raw MCP frame JSON-RPC ID must be a string or number"
        ));
    }
    match frame.direction {
        RawMcpDirection::Request => {
            if has_result || has_error {
                return Err(anyhow::anyhow!(
                    "raw MCP request must not contain result or error"
                ));
            }
            if method.is_none() {
                return Err(anyhow::anyhow!("raw MCP request has no method member"));
            }
            if !object.get("params").map(Value::is_object).unwrap_or(false) {
                return Err(anyhow::anyhow!("raw MCP request has no params object"));
            }
        }
        RawMcpDirection::Response => {
            if object.contains_key("method") || object.contains_key("params") {
                return Err(anyhow::anyhow!(
                    "raw MCP response contains request-only members"
                ));
            }
            if has_error {
                let error = object
                    .get("error")
                    .and_then(Value::as_object)
                    .ok_or_else(|| anyhow::anyhow!("raw MCP error member is not an object"))?;
                if !error.get("code").and_then(Value::as_i64).is_some()
                    || !error.get("message").map(Value::is_string).unwrap_or(false)
                {
                    return Err(anyhow::anyhow!(
                        "raw MCP error member lacks numeric code or string message"
                    ));
                }
            }
        }
    }
    let params = parsed.get("params");
    let arguments_sha256 = params
        .and_then(|value| value.get("arguments"))
        .map(canonical_digest)
        .transpose()?;
    if arguments_sha256 != frame.arguments_sha256 {
        return Err(anyhow::anyhow!(
            "raw MCP frame argument digest metadata does not match captured bytes"
        ));
    }
    let result_sha256 = parsed.get("result").map(canonical_digest).transpose()?;
    if result_sha256 != frame.result_sha256 {
        return Err(anyhow::anyhow!(
            "raw MCP frame result digest metadata does not match captured bytes"
        ));
    }
    Ok(())
}

fn canonical_digest<T: serde::Serialize>(value: &T) -> anyhow::Result<String> {
    let value = serde_json::to_value(value)
        .map_err(|error| anyhow::anyhow!("MCP receipt value is not serializable: {error}"))?;
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| anyhow::anyhow!("MCP receipt canonicalization failed: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn bounded_context_size(context: &McpOperationContext) -> anyhow::Result<usize> {
    let bytes = serde_json::to_vec(context)
        .map_err(|error| anyhow::anyhow!("MCP operation context is not serializable: {error}"))?;
    if bytes.len() > MAX_OPERATION_CONTEXT_BYTES {
        return Err(anyhow::anyhow!(
            "single MCP operation context exceeds bounded limit"
        ));
    }
    Ok(bytes.len())
}

fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
