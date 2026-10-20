//! Final provider-wire preparation and finite request accounting.
//!
//! Provider adapters must call this after all route-specific body mutations.
//! The returned bytes are immutable transport input: retries may change
//! headers, but they must reuse this body and its digest.

use crate::provider_types::ProviderRequest;
use claurst_core::types::{ContentBlock, Message, ToolResultContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{self, Write};
#[cfg(test)]
use std::{
    future::Future,
    sync::{Arc, Mutex},
};
use thiserror::Error;

const DEFAULT_FRAMING_TOKENS: u64 = 4_096;
const MAX_JSON_NODES: usize = 100_000;
const MAX_JSON_DEPTH: usize = 64;
const MAX_JSON_STRING_BYTES: usize = 64 * 1024 * 1024;

pub mod media;
pub use media::ImageAccounting;

/// Flatten the textual portion of a typed tool result for native provider
/// serializers. Text slots are already canonically delimited by the tools
/// assembler, so provider adapters must concatenate them without adding any
/// bytes of their own. Nested tool results are traversed in occurrence order;
/// media and other non-text blocks remain owned by the provider-specific
/// serializer.
pub(crate) fn tool_result_text(content: &ToolResultContent) -> String {
    fn append_content(content: &ToolResultContent, output: &mut String) {
        match content {
            ToolResultContent::Text(text) => output.push_str(text),
            ToolResultContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => output.push_str(text),
                        ContentBlock::ToolResult { content, .. } => {
                            append_content(content, output);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let mut output = String::new();
    append_content(content, &mut output);
    output
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaAccounting {
    None,
    Images(ImageAccounting),
}

impl MediaAccounting {
    /// Revalidate the content selected for this request. Persisted accounting
    /// metadata never authorizes a later payload or a different target.
    pub fn from_messages(
        provider: &str,
        model: &str,
        messages: &[Message],
    ) -> Result<Self, WireBudgetError> {
        media::account_messages(provider, model, messages)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireBudget {
    pub context_limit_tokens: u64,
    pub input_limit_tokens: u64,
    pub output_reserve_tokens: u64,
    pub max_wire_bytes: u64,
    pub framing_tokens: u64,
    pub media: MediaAccounting,
    pub policy_revision: String,
    pub budget_source: String,
}

impl WireBudget {
    pub fn finite(
        context_limit_tokens: u64,
        input_limit_tokens: u64,
        output_reserve_tokens: u64,
        max_wire_bytes: u64,
        media: MediaAccounting,
    ) -> Result<Self, WirePreparationError> {
        if context_limit_tokens == 0
            || input_limit_tokens == 0
            || output_reserve_tokens >= context_limit_tokens
            || input_limit_tokens > context_limit_tokens
            || input_limit_tokens
                .checked_add(output_reserve_tokens)
                .is_none_or(|total| total > context_limit_tokens)
            || max_wire_bytes == 0
        {
            return Err(WirePreparationError::InvalidBudget);
        }
        Ok(Self {
            context_limit_tokens,
            input_limit_tokens,
            output_reserve_tokens,
            max_wire_bytes,
            framing_tokens: DEFAULT_FRAMING_TOKENS,
            media,
            policy_revision: "wire-budget-v1".into(),
            budget_source: "fallback".into(),
        })
    }

    pub fn with_framing_tokens(
        mut self,
        framing_tokens: u64,
    ) -> Result<Self, WirePreparationError> {
        if framing_tokens == 0 {
            return Err(WirePreparationError::InvalidBudget);
        }
        self.framing_tokens = framing_tokens;
        Ok(self)
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Result<Self, WirePreparationError> {
        let source = source.into();
        if source.trim().is_empty() {
            return Err(WirePreparationError::InvalidBudget);
        }
        self.budget_source = source;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenAccounting {
    ConservativeEstimate { input_tokens: u64, basis: String },
}

impl TokenAccounting {
    pub fn input_tokens(&self) -> u64 {
        match self {
            Self::ConservativeEstimate { input_tokens, .. } => *input_tokens,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedWireRequest {
    pub provider: String,
    pub endpoint: String,
    pub stream: bool,
    body: Vec<u8>,
    pub wire_sha256: String,
    pub wire_bytes: u64,
    pub accounting: TokenAccounting,
    media: MediaAccounting,
}

impl PreparedWireRequest {
    pub fn bytes(&self) -> &[u8] {
        &self.body
    }

    pub fn into_body(self) -> Vec<u8> {
        self.body
    }

    pub fn media(&self) -> &MediaAccounting {
        &self.media
    }
}

/// Test-only observation of the exact immutable bytes handed to an adapter's
/// transport. The task-local scope prevents parallel tests from sharing
/// request content and keeps this recorder out of production builds.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecordedFinalRequest {
    pub provider: String,
    pub endpoint: String,
    pub stream: bool,
    pub body: Vec<u8>,
    pub wire_sha256: String,
    pub wire_bytes: u64,
    pub accounting: TokenAccounting,
    pub media: MediaAccounting,
}

#[cfg(test)]
tokio::task_local! {
    static FINAL_REQUEST_RECORDER: Arc<Mutex<Vec<RecordedFinalRequest>>>;
}

#[cfg(test)]
pub(crate) async fn record_final_requests<F>(future: F) -> (F::Output, Vec<RecordedFinalRequest>)
where
    F: Future,
{
    let records = Arc::new(Mutex::new(Vec::new()));
    let output = FINAL_REQUEST_RECORDER
        .scope(Arc::clone(&records), future)
        .await;
    let recorded = records.lock().expect("final request recorder lock").clone();
    (output, recorded)
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WireBudgetError {
    #[error("inference request is missing a finite wire budget")]
    MissingBudget,
    #[error("wire budget is invalid or non-finite")]
    InvalidBudget,
    #[error("request body serialization failed: {0}")]
    Serialization(String),
    #[error("request body exceeds the configured wire-byte limit")]
    WireOverflow {
        wire_bytes: u64,
        max_wire_bytes: u64,
        estimated_input_tokens: u64,
        input_limit_tokens: u64,
        context_limit_tokens: u64,
        output_reserve_tokens: u64,
        policy_revision: String,
        budget_source: String,
    },
    #[error("request body exceeds the configured finite input-token budget")]
    InputOverflow {
        estimated_input_tokens: u64,
        input_limit_tokens: u64,
        wire_bytes: u64,
        max_wire_bytes: u64,
        context_limit_tokens: u64,
        output_reserve_tokens: u64,
        policy_revision: String,
        budget_source: String,
    },
    #[error("request output reserve is smaller than final provider output options")]
    OutputReserveExceeded {
        estimated_input_tokens: u64,
        input_limit_tokens: u64,
        wire_bytes: u64,
        max_wire_bytes: u64,
        context_limit_tokens: u64,
        output_reserve_tokens: u64,
        requested_output_tokens: u64,
        policy_revision: String,
        budget_source: String,
    },
    #[error("request contains media without an implemented typed finite accounting policy")]
    UnsupportedMedia,
    #[error("invalid inline media: {0}")]
    InvalidMedia(#[from] claurst_core::media::MediaError),
    #[error("final provider media does not match the selected content and target policy")]
    MediaMismatch,
    #[error("request JSON exceeds the bounded traversal limit")]
    ComplexityLimit,
    #[error("request output option '{field}' must be a finite unsigned integer")]
    InvalidOutputOverride { field: String },
    #[error("request accounting overflowed")]
    ArithmeticOverflow,
}

impl WireBudgetError {
    /// Return the measured preflight fields needed by the caller's planner.
    /// Missing, malformed, media, and arithmetic failures intentionally have
    /// no misleading numeric estimate.
    pub fn request_budget_fields(&self) -> Option<(u64, u64, u64, u64, u64, u64, &str)> {
        match self {
            Self::WireOverflow {
                wire_bytes,
                max_wire_bytes,
                estimated_input_tokens,
                input_limit_tokens,
                context_limit_tokens,
                output_reserve_tokens,
                policy_revision: _,
                budget_source,
                ..
            } => Some((
                *estimated_input_tokens,
                *input_limit_tokens,
                *wire_bytes,
                *max_wire_bytes,
                *context_limit_tokens,
                *output_reserve_tokens,
                budget_source,
            )),
            Self::InputOverflow {
                estimated_input_tokens,
                input_limit_tokens,
                wire_bytes,
                max_wire_bytes,
                context_limit_tokens,
                output_reserve_tokens,
                policy_revision: _,
                budget_source,
            } => Some((
                *estimated_input_tokens,
                *input_limit_tokens,
                *wire_bytes,
                *max_wire_bytes,
                *context_limit_tokens,
                *output_reserve_tokens,
                budget_source,
            )),
            Self::OutputReserveExceeded {
                estimated_input_tokens,
                input_limit_tokens,
                wire_bytes,
                max_wire_bytes,
                context_limit_tokens,
                output_reserve_tokens,
                policy_revision: _,
                budget_source,
                requested_output_tokens: _,
            } => Some((
                *estimated_input_tokens,
                *input_limit_tokens,
                *wire_bytes,
                *max_wire_bytes,
                *context_limit_tokens,
                *output_reserve_tokens,
                budget_source,
            )),
            _ => None,
        }
    }
}

/// Backward-compatible name for callers that used the initial API draft.
pub type WirePreparationError = WireBudgetError;

fn read_output_path(body: &Value, path: &[&str]) -> Result<Option<u64>, WireBudgetError> {
    let mut current = body;
    for (index, key) in path.iter().enumerate() {
        let Some(object) = current.as_object() else {
            return Err(WireBudgetError::InvalidOutputOverride {
                field: path.join("."),
            });
        };
        let Some(next) = object.get(*key) else {
            return Ok(None);
        };
        if index == path.len() - 1 {
            return next
                .as_u64()
                .map(Some)
                .ok_or_else(|| WireBudgetError::InvalidOutputOverride {
                    field: path.join("."),
                });
        }
        current = next;
    }
    Ok(None)
}

fn body_requested_output_tokens(provider: &str, body: &Value) -> Result<u64, WireBudgetError> {
    let mut paths: Vec<Vec<&str>> = vec![
        vec!["max_tokens"],
        vec!["max_completion_tokens"],
        vec!["max_output_tokens"],
        vec!["thinking", "budget_tokens"],
    ];
    match provider {
        "google" => paths.extend([
            vec!["generationConfig", "maxOutputTokens"],
            vec!["generationConfig", "thinkingConfig", "thinkingBudget"],
        ]),
        "bedrock" | "amazon-bedrock" => paths.extend([
            vec!["inferenceConfig", "maxTokens"],
            vec!["reasoningConfig", "budgetTokens"],
        ]),
        _ => {}
    }
    paths.into_iter().try_fold(0, |current_max, path| {
        Ok(current_max.max(read_output_path(body, &path)?.unwrap_or(0)))
    })
}

pub fn prepare_json(
    provider: impl Into<String>,
    endpoint: impl Into<String>,
    stream: bool,
    body: &Value,
    budget: Option<&WireBudget>,
) -> Result<PreparedWireRequest, WirePreparationError> {
    let budget = budget.ok_or(WirePreparationError::MissingBudget)?;
    let provider = provider.into();
    prepare_bound_json(
        &provider,
        endpoint.into(),
        stream,
        body,
        budget,
        &budget.media,
        body.get("model").and_then(Value::as_str),
    )
}

fn prepare_bound_json(
    provider: &str,
    endpoint: String,
    stream: bool,
    body: &Value,
    budget: &WireBudget,
    expected_media: &MediaAccounting,
    model: Option<&str>,
) -> Result<PreparedWireRequest, WireBudgetError> {
    validate_budget(budget)?;
    validate_json_complexity(body)?;
    let verified_media = media::verify_wire(provider, model, body, expected_media)?;
    let mut writer = BoundedWireWriter {
        bytes: Vec::new(),
        total: 0,
        limit: budget.max_wire_bytes.min(MAX_JSON_STRING_BYTES as u64),
    };
    serde_json::to_writer(&mut writer, body)
        .map_err(|error| WireBudgetError::Serialization(error.to_string()))?;
    if writer.total > budget.max_wire_bytes {
        return Err(wire_overflow(
            writer.total,
            estimated_input_tokens(writer.total, budget, &verified_media)?,
            budget,
        ));
    }
    if writer.total > writer.limit {
        return Err(WireBudgetError::ComplexityLimit);
    }
    let prepared = prepare_validated_bytes(
        provider.to_owned(),
        endpoint,
        stream,
        writer.bytes,
        budget,
        verified_media,
    )?;
    let requested_output_tokens = body_requested_output_tokens(provider, body)?;
    if requested_output_tokens > budget.output_reserve_tokens {
        return Err(WireBudgetError::OutputReserveExceeded {
            estimated_input_tokens: prepared.accounting.input_tokens(),
            input_limit_tokens: budget.input_limit_tokens,
            wire_bytes: prepared.wire_bytes,
            max_wire_bytes: budget.max_wire_bytes,
            context_limit_tokens: budget.context_limit_tokens,
            output_reserve_tokens: budget.output_reserve_tokens,
            requested_output_tokens,
            policy_revision: budget.policy_revision.clone(),
            budget_source: budget.budget_source.clone(),
        });
    }
    Ok(prepared)
}

pub fn prepare_bytes(
    provider: impl Into<String>,
    endpoint: impl Into<String>,
    stream: bool,
    body: Vec<u8>,
    budget: &WireBudget,
) -> Result<PreparedWireRequest, WirePreparationError> {
    validate_budget(budget)?;
    if body.len() > MAX_JSON_STRING_BYTES {
        return Err(WireBudgetError::ComplexityLimit);
    }
    // Raw bytes cannot prove a caller-supplied media receipt. Media callers
    // must use the canonical-to-final JSON boundary above.
    if !matches!(budget.media, MediaAccounting::None) {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    let value: Value = serde_json::from_slice(&body)
        .map_err(|error| WireBudgetError::Serialization(error.to_string()))?;
    validate_json_complexity(&value)?;
    let provider = provider.into();
    let media = media::verify_wire(&provider, None, &value, &MediaAccounting::None)?;
    prepare_validated_bytes(provider, endpoint.into(), stream, body, budget, media)
}

fn prepare_validated_bytes(
    provider: String,
    endpoint: String,
    stream: bool,
    body: Vec<u8>,
    budget: &WireBudget,
    media: MediaAccounting,
) -> Result<PreparedWireRequest, WireBudgetError> {
    let wire_bytes =
        u64::try_from(body.len()).map_err(|_| WirePreparationError::ArithmeticOverflow)?;
    let input_tokens = estimated_input_tokens(wire_bytes, budget, &media)?;
    if wire_bytes > budget.max_wire_bytes {
        return Err(wire_overflow(wire_bytes, input_tokens, budget));
    }
    if input_tokens > budget.input_limit_tokens {
        return Err(WireBudgetError::InputOverflow {
            estimated_input_tokens: input_tokens,
            input_limit_tokens: budget.input_limit_tokens,
            wire_bytes,
            max_wire_bytes: budget.max_wire_bytes,
            context_limit_tokens: budget.context_limit_tokens,
            output_reserve_tokens: budget.output_reserve_tokens,
            policy_revision: budget.policy_revision.clone(),
            budget_source: budget.budget_source.clone(),
        });
    }
    let mut hasher = Sha256::new();
    hasher.update(&body);
    let wire_sha256 = hex::encode(hasher.finalize());
    let media_revision = match &media {
        MediaAccounting::None => "none",
        MediaAccounting::Images(images) => images.policy_revision(),
    };
    let prepared = PreparedWireRequest {
        provider,
        endpoint: redact_endpoint(&endpoint),
        stream,
        body,
        wire_sha256,
        wire_bytes,
        accounting: TokenAccounting::ConservativeEstimate {
            input_tokens,
            basis: format!(
                "one token per non-media wire byte + {} framing tokens + verified image policy ({media_revision}; {}; budget source {})",
                budget.framing_tokens, budget.policy_revision, budget.budget_source,
            ),
        },
        media,
    };
    #[cfg(test)]
    let _ = FINAL_REQUEST_RECORDER.try_with(|records| {
        records
            .lock()
            .expect("final request recorder lock")
            .push(RecordedFinalRequest {
                provider: prepared.provider.clone(),
                endpoint: prepared.endpoint.clone(),
                stream: prepared.stream,
                body: prepared.body.clone(),
                wire_sha256: prepared.wire_sha256.clone(),
                wire_bytes: prepared.wire_bytes,
                accounting: prepared.accounting.clone(),
                media: prepared.media.clone(),
            });
    });
    Ok(prepared)
}

fn validate_budget(budget: &WireBudget) -> Result<(), WireBudgetError> {
    if budget.context_limit_tokens == 0
        || budget.input_limit_tokens == 0
        || budget.output_reserve_tokens >= budget.context_limit_tokens
        || budget.input_limit_tokens > budget.context_limit_tokens
        || budget
            .input_limit_tokens
            .checked_add(budget.output_reserve_tokens)
            .is_none_or(|total| total > budget.context_limit_tokens)
        || budget.max_wire_bytes == 0
        || budget.framing_tokens == 0
        || budget.policy_revision.trim().is_empty()
        || budget.budget_source.trim().is_empty()
    {
        return Err(WireBudgetError::InvalidBudget);
    }
    Ok(())
}

fn estimated_input_tokens(
    wire_bytes: u64,
    budget: &WireBudget,
    media: &MediaAccounting,
) -> Result<u64, WireBudgetError> {
    let (media_tokens, encoded_bytes) = match media {
        MediaAccounting::None => (0, 0),
        MediaAccounting::Images(images) => (
            images.tokens(),
            u64::try_from(images.manifest().total_encoded_bytes())
                .map_err(|_| WireBudgetError::ArithmeticOverflow)?,
        ),
    };
    // STANDARD base64 contains only unescaped JSON string characters. The
    // fresh final-body manifest proves these exact occurrences before their
    // encoded bytes can be excluded from conservative text accounting.
    wire_bytes
        .checked_sub(encoded_bytes)
        .and_then(|value| value.checked_add(budget.framing_tokens))
        .and_then(|value| value.checked_add(media_tokens))
        .ok_or(WirePreparationError::ArithmeticOverflow)
}

fn wire_overflow(wire_bytes: u64, input_tokens: u64, budget: &WireBudget) -> WireBudgetError {
    WireBudgetError::WireOverflow {
        wire_bytes,
        max_wire_bytes: budget.max_wire_bytes,
        estimated_input_tokens: input_tokens,
        input_limit_tokens: budget.input_limit_tokens,
        context_limit_tokens: budget.context_limit_tokens,
        output_reserve_tokens: budget.output_reserve_tokens,
        policy_revision: budget.policy_revision.clone(),
        budget_source: budget.budget_source.clone(),
    }
}

/// Bound every JSON node, including schemas, before the media-only walker.
/// Schema contents remain ordinary text and are never treated as image input.
fn validate_json_complexity(body: &Value) -> Result<(), WireBudgetError> {
    let mut pending = vec![(body, 0usize)];
    let mut nodes = 0usize;
    let mut string_bytes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes = nodes
            .checked_add(1)
            .ok_or(WireBudgetError::ComplexityLimit)?;
        if nodes > MAX_JSON_NODES || depth > MAX_JSON_DEPTH {
            return Err(WireBudgetError::ComplexityLimit);
        }
        let children = match value {
            Value::Array(values) => values.len(),
            Value::Object(map) => map.len(),
            _ => 0,
        };
        if children > MAX_JSON_NODES.saturating_sub(nodes + pending.len()) {
            return Err(WireBudgetError::ComplexityLimit);
        }
        match value {
            Value::Array(values) => pending.extend(values.iter().map(|value| (value, depth + 1))),
            Value::Object(map) => {
                for (key, value) in map {
                    string_bytes = string_bytes
                        .checked_add(key.len())
                        .ok_or(WireBudgetError::ComplexityLimit)?;
                    pending.push((value, depth + 1));
                }
            }
            Value::String(value) => {
                string_bytes = string_bytes
                    .checked_add(value.len())
                    .ok_or(WireBudgetError::ComplexityLimit)?;
            }
            _ => {}
        }
        if string_bytes > MAX_JSON_STRING_BYTES {
            return Err(WireBudgetError::ComplexityLimit);
        }
    }
    Ok(())
}

/// Count the complete serialized request while retaining at most the finite
/// transport cap. Overflow still reports exact measured bytes to the planner.
struct BoundedWireWriter {
    bytes: Vec<u8>,
    total: u64,
    limit: u64,
}

impl Write for BoundedWireWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.total = self
            .total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("wire byte count overflow"))?;
        if self.total <= self.limit {
            self.bytes.extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn budget_from_request(request: &ProviderRequest) -> Result<&WireBudget, WirePreparationError> {
    request
        .wire_budget
        .as_ref()
        .ok_or(WirePreparationError::MissingBudget)
}

/// A private-construction binding to immutable selected conversation content.
/// Adapters create it before transformation/auth, then verify their final body.
/// It retains no copied image payloads and cannot be deserialized as authority.
pub struct CanonicalProviderRequest<'a> {
    provider: String,
    request: &'a ProviderRequest,
    media: MediaAccounting,
}

pub fn preflight_provider_request<'a>(
    provider: impl Into<String>,
    request: &'a ProviderRequest,
) -> Result<CanonicalProviderRequest<'a>, WireBudgetError> {
    let provider = provider.into();
    let budget = budget_from_request(request)?;
    validate_budget(budget)?;
    let media = MediaAccounting::from_messages(&provider, &request.model, &request.messages)?;
    if !matches!(budget.media, MediaAccounting::None) && budget.media != media {
        return Err(WireBudgetError::MediaMismatch);
    }
    Ok(CanonicalProviderRequest {
        provider,
        request,
        media,
    })
}

impl CanonicalProviderRequest<'_> {
    pub fn media(&self) -> &MediaAccounting {
        &self.media
    }

    pub fn prepare_json(
        &self,
        endpoint: impl Into<String>,
        stream: bool,
        body: &Value,
    ) -> Result<PreparedWireRequest, WireBudgetError> {
        prepare_bound_json(
            &self.provider,
            endpoint.into(),
            stream,
            body,
            budget_from_request(self.request)?,
            &self.media,
            Some(&self.request.model),
        )
    }
}

/// Prepare a provider request after the adapter has applied every route and
/// provider-option mutation. This is the shared boundary used immediately
/// before an inference transport is opened.
pub fn prepare_provider_json(
    provider: impl Into<String>,
    endpoint: impl Into<String>,
    stream: bool,
    body: &Value,
    request: &ProviderRequest,
) -> Result<PreparedWireRequest, WireBudgetError> {
    preflight_provider_request(provider, request)?.prepare_json(endpoint, stream, body)
}

fn contains_media(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_media),
        Value::Object(map) => {
            let typed_media = map.get("type").and_then(Value::as_str).is_some_and(|kind| {
                matches!(kind, "image" | "document" | "input_image" | "input_file")
                    && [
                        "source",
                        "image_url",
                        "url",
                        "data",
                        "inlineData",
                        "inline_data",
                    ]
                    .iter()
                    .any(|field| map.contains_key(*field))
            });
            let provider_media = [
                "inlineData",
                "inline_data",
                "image_url",
                "file_data",
                "fileData",
                "file_url",
            ]
            .iter()
            .any(|field| map.contains_key(*field));
            let bedrock_media = map
                .get("image")
                .and_then(Value::as_object)
                .and_then(|image| image.get("source"))
                .and_then(Value::as_object)
                .is_some_and(|source| source.contains_key("bytes"));
            if typed_media || provider_media || bedrock_media {
                return true;
            }
            [
                "messages", "contents", "input", "parts", "content", "system", "prompt",
            ]
            .iter()
            .filter_map(|field| map.get(*field))
            .any(contains_media)
        }
        _ => false,
    }
}

fn redact_endpoint(endpoint: &str) -> String {
    endpoint
        .split_once('?')
        .map(|(base, _)| format!("{base}?[redacted]"))
        .unwrap_or_else(|| endpoint.to_string())
}

pub fn prepare_json_checked(
    provider: impl Into<String>,
    endpoint: impl Into<String>,
    stream: bool,
    body: &Value,
    budget: Option<&WireBudget>,
) -> Result<PreparedWireRequest, WirePreparationError> {
    prepare_json(provider, endpoint, stream, body, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn budget() -> WireBudget {
        WireBudget::finite(30_000, 20_000, 1_000, 50_000, MediaAccounting::None).unwrap()
    }

    #[test]
    fn tool_result_text_preserves_typed_slot_order_without_provider_separators() {
        let content = ToolResultContent::Blocks(vec![
            ContentBlock::Text {
                text: "left".into(),
            },
            ContentBlock::Text {
                text: String::new(),
            },
            ContentBlock::ToolResult {
                tool_use_id: "nested".into(),
                content: ToolResultContent::Blocks(vec![
                    ContentBlock::Text { text: "é".into() },
                    ContentBlock::Text {
                        text: "\nright".into(),
                    },
                ]),
                is_error: None,
            },
        ]);

        assert_eq!(tool_result_text(&content), "lefté\nright");
    }

    #[test]
    fn tool_result_text_keeps_near_budget_payload_and_host_notice_exact() {
        let notice = "\n[EvidenceRead id=ev-1; next_offset=32768; done=true]";
        let payload = "x".repeat(32 * 1024 - notice.len());
        let content = ToolResultContent::Blocks(vec![
            ContentBlock::Text {
                text: payload.clone(),
            },
            ContentBlock::Text {
                text: String::new(),
            },
            ContentBlock::Text {
                text: notice.to_string(),
            },
        ]);

        let flattened = tool_result_text(&content);
        assert_eq!(flattened.len(), 32 * 1024);
        assert_eq!(flattened, format!("{payload}{notice}"));
    }

    #[test]
    fn immutable_body_hash_and_conservative_accounting() {
        let prepared = prepare_json(
            "test",
            "/inference",
            true,
            &json!({"x":"y"}),
            Some(&budget()),
        )
        .unwrap();
        assert_eq!(prepared.wire_bytes, prepared.bytes().len() as u64);
        assert_eq!(
            prepared.wire_sha256,
            hex::encode(Sha256::digest(prepared.bytes()))
        );
        assert!(prepared.accounting.input_tokens() >= prepared.wire_bytes + 4096);
    }

    #[test]
    fn missing_overflow_and_media_fail_before_transport() {
        let body = json!({"type":"image", "data":"x"});
        assert!(matches!(
            prepare_json_checked("test", "/inference", false, &body, None),
            Err(WirePreparationError::MissingBudget)
        ));
        let tiny = WireBudget::finite(10, 1, 1, 2, MediaAccounting::None).unwrap();
        assert!(matches!(
            prepare_json_checked(
                "test",
                "/inference",
                false,
                &json!({"x":"long"}),
                Some(&tiny)
            ),
            Err(WirePreparationError::WireOverflow { .. }
                | WirePreparationError::InputOverflow { .. })
        ));
        assert!(matches!(
            prepare_json_checked("test", "/inference", false, &body, Some(&budget())),
            Err(WirePreparationError::UnsupportedMedia)
        ));
    }

    #[test]
    fn overflow_exposes_measured_costs_for_context_planning() {
        let budget = WireBudget::finite(100, 20, 10, 50_000, MediaAccounting::None)
            .unwrap()
            .with_framing_tokens(4)
            .unwrap();
        let error = prepare_json_checked(
            "test",
            "/inference",
            false,
            &json!({"payload":"0123456789"}),
            Some(&budget),
        )
        .expect_err("the measured body must exceed the finite input budget");
        let fields = error
            .request_budget_fields()
            .expect("numeric overflow includes planner fields");
        assert_eq!(fields.1, 20);
        assert_eq!(fields.2, 24);
        assert_eq!(fields.3, 50_000);
        assert_eq!(fields.4, 100);
        assert_eq!(fields.5, 10);
        assert_eq!(fields.6, "fallback");
    }

    #[test]
    fn tool_schemas_are_not_misclassified_as_media() {
        let body = json!({
            "tools": [{
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "picture": {"type": "image"},
                        "image_url": {"type": "string"},
                        "inlineData": {"type": "string"}
                    }
                }
            }]
        });
        assert!(prepare_json_checked("test", "/inference", false, &body, Some(&budget())).is_ok());
    }

    #[test]
    fn provider_specific_output_caps_and_types_are_checked() {
        let budget =
            WireBudget::finite(30_000, 20_000, 1_000, 50_000, MediaAccounting::None).unwrap();
        let google = json!({
            "generationConfig": {
                "maxOutputTokens": 1_001,
                "thinkingConfig": {"thinkingBudget": 900}
            }
        });
        assert!(matches!(
            prepare_json_checked("google", "/generate", false, &google, Some(&budget)),
            Err(WireBudgetError::OutputReserveExceeded {
                requested_output_tokens: 1_001,
                ..
            })
        ));

        let bedrock = json!({
            "inferenceConfig": {"maxTokens": "1024"}
        });
        assert!(matches!(
            prepare_json_checked("bedrock", "/converse", false, &bedrock, Some(&budget)),
            Err(WireBudgetError::InvalidOutputOverride { field }) if field == "inferenceConfig.maxTokens"
        ));
    }

    #[test]
    fn media_shapes_are_rejected_without_typed_accounting() {
        let budget = budget();
        for body in [
            json!({"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "abc"}}]}]}),
            json!({"messages": [{"content": [{"type": "input_file", "file_url": "https://example.invalid/a"}]}]}),
            json!({"messages": [{"content": [{"image": {"source": {"bytes": "abc"}}}]}]}),
        ] {
            assert!(matches!(
                prepare_json_checked("test", "/inference", false, &body, Some(&budget)),
                Err(WireBudgetError::UnsupportedMedia)
            ));
        }
    }

    #[test]
    fn endpoint_query_is_redacted_without_changing_wire_bytes() {
        let body = json!({"prompt": "hello"});
        let prepared = prepare_json_checked(
            "google",
            "https://generativelanguage.googleapis.com/v1beta?key=secret",
            false,
            &body,
            Some(&budget()),
        )
        .unwrap();
        assert_eq!(
            prepared.endpoint,
            "https://generativelanguage.googleapis.com/v1beta?[redacted]"
        );
        assert_eq!(prepared.bytes(), serde_json::to_vec(&body).unwrap());
    }

    #[test]
    fn deep_or_wide_final_json_fails_before_media_walk_or_serialization() {
        let mut deep = json!({"type":"image","data":"untrusted"});
        for _ in 0..=MAX_JSON_DEPTH {
            deep = json!({"content":deep});
        }
        assert_eq!(
            prepare_json_checked("test", "/inference", false, &deep, Some(&budget())),
            Err(WireBudgetError::ComplexityLimit),
        );
        let wide = json!({"input": vec![Value::Null; MAX_JSON_NODES + 1]});
        assert_eq!(
            prepare_json_checked("test", "/inference", false, &wide, Some(&budget())),
            Err(WireBudgetError::ComplexityLimit),
        );
    }

    #[test]
    fn capped_serialization_reports_full_escaped_wire_length_without_buffering_it() {
        let body = json!({"text":"\n".repeat(20_000)});
        let budget = WireBudget::finite(100_000, 90_000, 1000, 100, MediaAccounting::None).unwrap();
        let error =
            prepare_json_checked("test", "/inference", false, &body, Some(&budget)).unwrap_err();
        assert_eq!(
            error.request_budget_fields().unwrap().2,
            serde_json::to_vec(&body).unwrap().len() as u64,
        );
        let mut writer = BoundedWireWriter {
            bytes: vec![],
            total: 0,
            limit: 4,
        };
        writer.write_all(b"abc").unwrap();
        writer.write_all(b"defghijkl").unwrap();
        assert_eq!(writer.total, 12);
        assert!(writer.bytes.len() <= 4);
    }

    #[test]
    fn mutated_invalid_budgets_fail_before_body_validation_and_serialization() {
        let mut invalid = budget();
        invalid.max_wire_bytes = 0;
        assert_eq!(
            prepare_json_checked(
                "test",
                "/inference",
                false,
                &json!({"x":"y"}),
                Some(&invalid)
            ),
            Err(WireBudgetError::InvalidBudget),
        );
        assert_eq!(
            prepare_bytes(
                "test",
                "/inference",
                false,
                b"malformed JSON".to_vec(),
                &invalid
            ),
            Err(WireBudgetError::InvalidBudget),
        );
        invalid = budget();
        invalid.framing_tokens = 0;
        assert_eq!(
            prepare_json_checked(
                "test",
                "/inference",
                false,
                &json!({"type":"image","data":"untrusted"}),
                Some(&invalid)
            ),
            Err(WireBudgetError::InvalidBudget),
        );
    }
}
