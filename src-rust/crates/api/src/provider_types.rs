// provider_types.rs — Unified request/response types shared across all
// provider implementations.
//
// These types form a provider-agnostic layer that every concrete provider
// adapter (Anthropic, OpenAI, Google, …) maps to/from.

use claurst_core::crypto_utils::sha256_hex;
use claurst_core::types::{ContentBlock, Message, ToolDefinition, UsageInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

// Re-export ThinkingConfig and SystemPrompt from the api types module so
// callers only need to import from this module.
use crate::provider_wire::WireBudget;
use crate::provider_witness::ProviderWireWitness;
pub use crate::types::{SystemPrompt, ThinkingConfig};

// ---------------------------------------------------------------------------
// StopReason
// ---------------------------------------------------------------------------

/// The reason a model stopped generating tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum StopReason {
    /// The model reached a natural stopping point.
    #[default]
    EndTurn,
    /// The model generated a stop sequence.
    StopSequence,
    /// The model hit the max_tokens limit.
    MaxTokens,
    /// The model made a tool/function call.
    ToolUse,
    /// Content was filtered by the provider's safety system.
    ContentFiltered,
    /// The provider returned an unknown or unrecognised stop reason.
    Other(String),
}

// ---------------------------------------------------------------------------
// ProviderRequest
// ---------------------------------------------------------------------------

const MAX_CACHE_SCOPE_COMPONENT_BYTES: usize = 4096;
const MAX_PROVIDER_RESPONSE_ID_BYTES: usize = 512;

/// Opaque provider-cache routing key bound to one workspace and conversation.
///
/// The raw workspace and session labels never reach a provider. Adapters may
/// map the bounded digest to their native cache hint when the route supports
/// one; content equality still decides whether a provider cache can hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCacheScope {
    key: String,
}

impl ProviderCacheScope {
    pub fn for_session(
        workspace_scope: &str,
        session_id: &str,
    ) -> Result<Self, ProviderContinuityError> {
        if workspace_scope.is_empty()
            || session_id.is_empty()
            || workspace_scope.len() > MAX_CACHE_SCOPE_COMPONENT_BYTES
            || session_id.len() > MAX_CACHE_SCOPE_COMPONENT_BYTES
        {
            return Err(ProviderContinuityError::InvalidCacheScope);
        }
        let mut framed = Vec::with_capacity(
            workspace_scope
                .len()
                .saturating_add(session_id.len())
                .saturating_add(48),
        );
        framed.extend_from_slice(b"forge-provider-cache-v1\0");
        framed.extend_from_slice(&(workspace_scope.len() as u64).to_be_bytes());
        framed.extend_from_slice(workspace_scope.as_bytes());
        framed.extend_from_slice(&(session_id.len() as u64).to_be_bytes());
        framed.extend_from_slice(session_id.as_bytes());
        Ok(Self {
            key: sha256_hex(&framed),
        })
    }

    pub fn key(&self) -> &str {
        &self.key
    }
}

/// Exact anchor for a provider-managed response continuation.
///
/// It is host metadata: provider adapters serialize only the prior response ID
/// and only after validating the retained message prefix and all stable request
/// fields against the current request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseContinuation {
    previous_response_id: String,
    message_prefix_len: usize,
    message_prefix_digest: String,
    request_fingerprint: String,
}

impl ResponseContinuation {
    pub fn new(
        previous_response_id: impl Into<String>,
        completed_message_prefix: &[Message],
        request_fingerprint: String,
    ) -> Result<Self, ProviderContinuityError> {
        let previous_response_id = previous_response_id.into();
        if previous_response_id.is_empty()
            || previous_response_id.len() > MAX_PROVIDER_RESPONSE_ID_BYTES
        {
            return Err(ProviderContinuityError::InvalidResponseId);
        }
        if completed_message_prefix.is_empty() || request_fingerprint.len() != 64 {
            return Err(ProviderContinuityError::InvalidAnchor);
        }
        Ok(Self {
            previous_response_id,
            message_prefix_len: completed_message_prefix.len(),
            message_prefix_digest: message_digest(completed_message_prefix)?,
            request_fingerprint,
        })
    }

    pub fn previous_response_id(&self) -> &str {
        &self.previous_response_id
    }

    pub fn message_prefix_len(&self) -> usize {
        self.message_prefix_len
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderContinuityError {
    InvalidCacheScope,
    InvalidResponseId,
    InvalidAnchor,
    Serialization(String),
}

impl fmt::Display for ProviderContinuityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCacheScope => formatter.write_str("provider cache scope is invalid"),
            Self::InvalidResponseId => formatter.write_str("provider response id is invalid"),
            Self::InvalidAnchor => formatter.write_str("provider continuation anchor is invalid"),
            Self::Serialization(_) => {
                formatter.write_str("provider continuity metadata could not be serialized")
            }
        }
    }
}

impl std::error::Error for ProviderContinuityError {}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        Value::Object(values) => {
            let mut names: Vec<&String> = values.keys().collect();
            names.sort();
            let mut canonical = serde_json::Map::new();
            for name in names {
                if let Some(value) = values.get(name) {
                    canonical.insert(name.clone(), canonical_json(value));
                }
            }
            Value::Object(canonical)
        }
        primitive => primitive.clone(),
    }
}

fn hash_canonical(value: &Value) -> Result<String, ProviderContinuityError> {
    serde_json::to_vec(&canonical_json(value))
        .map(|encoded| sha256_hex(&encoded))
        .map_err(|error| ProviderContinuityError::Serialization(error.to_string()))
}

fn message_digest(messages: &[Message]) -> Result<String, ProviderContinuityError> {
    serde_json::to_value(messages)
        .map_err(|error| ProviderContinuityError::Serialization(error.to_string()))
        .and_then(|value| hash_canonical(&value))
}

/// A normalised request that any provider adapter can consume.
///
/// Provider-specific parameters that cannot be expressed through the common
/// fields can be passed via `provider_options` as an arbitrary JSON object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRequest {
    /// The model identifier (e.g. `"claude-opus-4-5"`, `"gpt-4o"`).
    pub model: String,

    /// The conversation history to send to the model.
    pub messages: Vec<Message>,

    /// An optional system / developer prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<SystemPrompt>,

    /// Tool definitions available to the model for this turn.
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,

    /// Maximum number of tokens to generate.
    pub max_tokens: u32,

    /// Sampling temperature (provider-dependent range, typically 0.0–1.0 or 0.0–2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus sampling probability mass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling cutoff.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Sequences that cause the model to stop generating.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Extended thinking / chain-of-thought configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Arbitrary provider-specific options merged into the request body.
    /// Defaults to an empty JSON object `{}`.
    #[serde(default)]
    pub provider_options: Value,

    /// Opaque workspace/conversation cache scope. Adapters map it only to
    /// native cache controls supported by their active route.
    #[serde(skip)]
    pub cache_scope: Option<ProviderCacheScope>,

    /// Optional response-ID continuation anchor. It is ignored unless both the
    /// active provider route and the exact current request validate it.
    #[serde(skip)]
    pub response_continuation: Option<ResponseContinuation>,

    /// Finite preflight policy for this inference request. The policy is
    /// consumed by transports and deliberately omitted from provider JSON.
    #[serde(skip)]
    pub wire_budget: Option<WireBudget>,

    /// Host-owned dispatch authorization metadata. It is never serialized into
    /// provider JSON and present envelopes are rejected by uncertified routes.
    #[serde(skip)]
    pub dispatch: Option<crate::dispatch_request::DispatchRequest>,
}

impl ProviderRequest {
    /// Return tool definitions in a canonical order shared by provider
    /// adapters. Name is the primary identity; description and canonical
    /// schema break malformed duplicate-name ties deterministically.
    pub fn stable_tools(&self) -> Vec<&ToolDefinition> {
        let mut tools: Vec<(&ToolDefinition, String)> = self
            .tools
            .iter()
            .map(|tool| (tool, canonical_json(&tool.input_schema).to_string()))
            .collect();
        tools.sort_by(|(left, left_schema), (right, right_schema)| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.description.cmp(&right.description))
                .then_with(|| left_schema.cmp(right_schema))
        });
        tools.into_iter().map(|(tool, _)| tool).collect()
    }

    /// Hash every stable provider field except conversation messages and the
    /// response anchor itself. Tool order is canonicalized first.
    pub fn continuity_fingerprint(&self) -> Result<String, ProviderContinuityError> {
        let stable_tools: Vec<Value> = self
            .stable_tools()
            .into_iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": canonical_json(&tool.input_schema),
                })
            })
            .collect();
        let stable = serde_json::json!({
            "version": 1,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": stable_tools,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
            "thinking": self.thinking,
            "provider_options": canonical_json(&self.provider_options),
            "cache_scope": self.cache_scope.as_ref().map(ProviderCacheScope::key),
        });
        hash_canonical(&stable)
    }

    /// Validate a response anchor against the current retained prefix and all
    /// stable request fields. Appended messages form the eligible delta.
    pub fn validated_response_continuation(&self) -> Option<&ResponseContinuation> {
        let anchor = self.response_continuation.as_ref()?;
        if self.cache_scope.is_none()
            || anchor.message_prefix_len == 0
            || anchor.message_prefix_len >= self.messages.len()
        {
            return None;
        }
        let prefix = self.messages.get(..anchor.message_prefix_len)?;
        let digest = message_digest(prefix).ok()?;
        let fingerprint = self.continuity_fingerprint().ok()?;
        (digest == anchor.message_prefix_digest && fingerprint == anchor.request_fingerprint)
            .then_some(anchor)
    }
}

// ---------------------------------------------------------------------------
// ProviderResponse
// ---------------------------------------------------------------------------

/// A normalised response returned by any provider adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    /// Provider-assigned message / request identifier.
    pub id: String,

    /// The generated content blocks.
    pub content: Vec<ContentBlock>,

    /// Why the model stopped generating.
    pub stop_reason: StopReason,

    /// Token usage for billing / budget tracking.
    pub usage: UsageInfo,

    /// The model that produced this response (as reported by the provider).
    pub model: String,
}

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------

/// Events emitted by the provider-agnostic streaming layer.
///
/// Each provider's SSE/websocket parser maps its wire format onto these events
/// so that the rest of the application can consume a single unified stream.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Content-free evidence for the exact serialized provider request and its
    /// final native response binding.  This is emitted only by Codex.
    ProviderWireWitness(ProviderWireWitness),

    /// The message has started; carries the provider-assigned id and model.
    MessageStart {
        id: String,
        model: String,
        usage: UsageInfo,
    },

    /// A new content block is beginning.
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },

    /// Incremental text delta for an in-progress block.
    TextDelta { index: usize, text: String },

    /// Incremental thinking / reasoning delta.
    ThinkingDelta { index: usize, thinking: String },

    /// Incremental delta for tool-call JSON arguments.
    InputJsonDelta { index: usize, partial_json: String },

    /// Incremental delta for a cryptographic signature block.
    SignatureDelta { index: usize, signature: String },

    /// An in-progress content block is now complete.
    ContentBlockStop { index: usize },

    /// Final message-level delta carrying the stop reason and updated usage.
    MessageDelta {
        stop_reason: Option<StopReason>,
        usage: Option<UsageInfo>,
    },

    /// The message stream is fully complete.
    MessageStop,

    /// A provider-level error occurred mid-stream.
    Error { error_type: String, message: String },

    /// Incremental reasoning / scratchpad delta (alias used by some providers).
    ReasoningDelta { index: usize, reasoning: String },
}

// ---------------------------------------------------------------------------
// StreamBlockAccumulator
// ---------------------------------------------------------------------------

/// Partial content-block state accumulated while draining a `StreamEvent`
/// stream. Mirrors the streaming `StreamAccumulator` in `lib.rs`, but over the
/// provider-level [`StreamEvent`]/[`ContentBlock`] types.
enum PartialBlock {
    Text(String),
    Thinking {
        thinking_buf: String,
        signature_buf: String,
    },
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
        /// Opaque provider signature (e.g. Gemini's `thoughtSignature`) captured
        /// on `ContentBlockStart`; echoed back on the finalized block (#311).
        thought_signature: Option<String>,
    },
    /// Any block that arrives fully-formed on `ContentBlockStart` and needs no
    /// delta accumulation (e.g. `RedactedThinking`, images).
    Passthrough(ContentBlock),
}

impl PartialBlock {
    fn empty_thinking() -> Self {
        PartialBlock::Thinking {
            thinking_buf: String::new(),
            signature_buf: String::new(),
        }
    }

    fn from_start(block: ContentBlock) -> Self {
        match block {
            ContentBlock::Text { text } => PartialBlock::Text(text),
            ContentBlock::Thinking {
                thinking,
                signature,
            } => PartialBlock::Thinking {
                thinking_buf: thinking,
                signature_buf: signature,
            },
            ContentBlock::ToolUse {
                id,
                name,
                thought_signature,
                ..
            } => PartialBlock::ToolUse {
                id,
                name,
                json_buf: String::new(),
                thought_signature,
            },
            other => PartialBlock::Passthrough(other),
        }
    }

    fn finish(self) -> ContentBlock {
        match self {
            PartialBlock::Text(text) => ContentBlock::Text { text },
            PartialBlock::Thinking {
                thinking_buf,
                signature_buf,
            } => ContentBlock::Thinking {
                thinking: thinking_buf,
                signature: signature_buf,
            },
            PartialBlock::ToolUse {
                id,
                name,
                json_buf,
                thought_signature,
            } => {
                let input =
                    serde_json::from_str(&json_buf).unwrap_or(Value::Object(Default::default()));
                ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    thought_signature,
                }
            }
            PartialBlock::Passthrough(block) => block,
        }
    }
}

/// Aggregates provider-level [`StreamEvent`] content-block events into ordered,
/// finalized [`ContentBlock`]s. Shared by the non-streaming `create_message`
/// aggregators (Anthropic, MiniMax, …) so that `ThinkingDelta` / `SignatureDelta`
/// / `ReasoningDelta` are captured into their block (rather than dropped) and
/// every block — text, thinking, tool_use, and anything else — keeps the stream
/// index it was emitted at. `finish()` is a single index-ordered pass, so the
/// model's interleave order (thinking-first / signed-thinking replay) survives.
/// See issue #217.
#[derive(Default)]
pub(crate) struct StreamBlockAccumulator {
    partials: std::collections::BTreeMap<usize, PartialBlock>,
}

impl StreamBlockAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed one `StreamEvent`. Only content-block events mutate state; message
    /// lifecycle events (`MessageStart`, `MessageDelta`, `MessageStop`,
    /// `ContentBlockStop`, `Error`) are ignored so callers still handle those
    /// (id / model / usage / stop_reason) themselves.
    pub(crate) fn on_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                self.partials
                    .insert(*index, PartialBlock::from_start(content_block.clone()));
            }
            StreamEvent::TextDelta { index, text } => {
                if let Some(PartialBlock::Text(buf)) = self.partials.get_mut(index) {
                    buf.push_str(text);
                }
            }
            StreamEvent::ThinkingDelta { index, thinking } => {
                if let PartialBlock::Thinking { thinking_buf, .. } = self
                    .partials
                    .entry(*index)
                    .or_insert_with(PartialBlock::empty_thinking)
                {
                    thinking_buf.push_str(thinking);
                }
            }
            StreamEvent::ReasoningDelta { index, reasoning } => {
                // `ReasoningDelta` is an alias for thinking text used by some
                // providers; fold it into the same thinking block.
                if let PartialBlock::Thinking { thinking_buf, .. } = self
                    .partials
                    .entry(*index)
                    .or_insert_with(PartialBlock::empty_thinking)
                {
                    thinking_buf.push_str(reasoning);
                }
            }
            StreamEvent::SignatureDelta { index, signature } => {
                if let PartialBlock::Thinking { signature_buf, .. } = self
                    .partials
                    .entry(*index)
                    .or_insert_with(PartialBlock::empty_thinking)
                {
                    signature_buf.push_str(signature);
                }
            }
            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => {
                if let Some(PartialBlock::ToolUse { json_buf, .. }) = self.partials.get_mut(index) {
                    json_buf.push_str(partial_json);
                }
            }
            _ => {}
        }
    }

    /// Consume the accumulator, producing finalized content blocks in the
    /// stream-index order the model emitted them.
    pub(crate) fn finish(self) -> Vec<ContentBlock> {
        self.partials
            .into_values()
            .map(PartialBlock::finish)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ProviderCapabilities
// ---------------------------------------------------------------------------

/// Route-level response-ID continuation support.
///
/// `FullContextFallback` means the adapter may send a validated delta and must
/// retry at most once with the already-available full context when the route
/// rejects the continuation before a response stream opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponseContinuationSupport {
    #[default]
    Unsupported,
    FullContextFallback,
}

impl ResponseContinuationSupport {
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::FullContextFallback)
    }
}

/// Describes the features supported by a particular provider/model combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Provider supports streaming responses via SSE or websocket.
    pub streaming: bool,

    /// Provider supports function / tool calling.
    pub tool_calling: bool,

    /// Provider supports extended thinking / chain-of-thought tokens.
    pub thinking: bool,

    /// Provider accepts image inputs.
    pub image_input: bool,

    /// Provider accepts PDF document inputs.
    pub pdf_input: bool,

    /// Provider accepts audio inputs.
    pub audio_input: bool,

    /// Provider accepts video inputs.
    pub video_input: bool,

    /// Provider supports prompt caching.
    pub caching: bool,

    /// Provider supports JSON-schema-constrained structured output.
    pub structured_output: bool,

    /// How the provider expects the system prompt to be delivered.
    pub system_prompt_style: SystemPromptStyle,
}

/// Describes where/how a provider expects the system prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemPromptStyle {
    /// Delivered as a top-level `system` field in the request body (Anthropic style).
    TopLevel,
    /// Delivered as a `{"role": "system", "content": "…"}` message at index 0 (OpenAI style).
    SystemMessage,
    /// Delivered as a `system_instruction` field (Google Gemini style).
    SystemInstruction,
}

// ---------------------------------------------------------------------------
// ProviderStatus
// ---------------------------------------------------------------------------

/// The current health status of a provider endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum ProviderStatus {
    /// The provider is operating normally.
    Healthy,
    /// The provider is reachable but experiencing elevated errors or latency.
    Degraded { reason: String },
    /// The provider is unreachable or has been disabled.
    Unavailable { reason: String },
}

// ---------------------------------------------------------------------------
// AuthMethod
// ---------------------------------------------------------------------------

/// The authentication mechanism used to talk to a provider endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum AuthMethod {
    /// A static API key sent as an HTTP header.
    ApiKey { key: String, header: ApiKeyHeader },

    /// A bearer token sent in the `Authorization` header.
    Bearer { token: String },

    /// AWS Signature V4 credentials for Amazon Bedrock.
    AwsCredentials {
        #[serde(skip_serializing_if = "Option::is_none")]
        profile: Option<String>,
        region: String,
        /// Optional bearer token for cross-account or SSO scenarios.
        #[serde(skip_serializing_if = "Option::is_none")]
        bearer_token: Option<String>,
    },

    /// OAuth 2.0 access + refresh token pair.
    OAuth {
        access_token: String,
        refresh_token: String,
        /// Unix timestamp (seconds) when the access token expires.
        expires_at: u64,
    },

    /// No authentication required (e.g. local Ollama).
    None,
}

/// Which HTTP header carries the API key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeyHeader {
    /// `x-api-key: <key>` (Anthropic, Mistral, …)
    XApiKey,
    /// `Authorization: Bearer <key>` (OpenAI, Groq, …)
    Authorization,
    /// A custom header name.
    Custom(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::types::Message;
    use serde_json::json;

    fn continuity_request() -> ProviderRequest {
        ProviderRequest {
            model: "continuity-model".into(),
            messages: vec![Message::user("first")],
            system_prompt: Some(SystemPrompt::Text("stable system".into())),
            tools: vec![
                ToolDefinition {
                    name: "Zulu".into(),
                    description: "z".into(),
                    input_schema: json!({"type":"object","properties":{"z":{"type":"string"}}}),
                },
                ToolDefinition {
                    name: "Alpha".into(),
                    description: "a".into(),
                    input_schema: json!({"type":"object","properties":{"a":{"type":"string"}}}),
                },
            ],
            max_tokens: 128,
            temperature: Some(0.2),
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            thinking: None,
            provider_options: json!({"reasoningEffort":"low"}),
            cache_scope: Some(
                ProviderCacheScope::for_session("workspace-a", "session-a")
                    .expect("bounded cache scope"),
            ),
            response_continuation: None,
            wire_budget: None,
            dispatch: None,
        }
    }

    #[test]
    fn cache_scope_is_opaque_bounded_and_scope_sensitive() {
        let first =
            ProviderCacheScope::for_session("workspace-a", "session-a").expect("first scope");
        let same = ProviderCacheScope::for_session("workspace-a", "session-a").expect("same scope");
        let other_workspace =
            ProviderCacheScope::for_session("workspace-b", "session-a").expect("workspace scope");
        let other_session =
            ProviderCacheScope::for_session("workspace-a", "session-b").expect("session scope");

        assert_eq!(first.key(), same.key());
        assert_eq!(first.key().len(), 64);
        assert_ne!(first.key(), other_workspace.key());
        assert_ne!(first.key(), other_session.key());
        assert!(!first.key().contains("workspace"));
        assert!(ProviderCacheScope::for_session("", "session").is_err());
        assert!(ProviderCacheScope::for_session("workspace", "").is_err());
    }

    #[test]
    fn continuity_fingerprint_canonicalizes_tool_order() {
        let first = continuity_request();
        let mut reordered = first.clone();
        reordered.tools.reverse();

        assert_eq!(
            first.continuity_fingerprint().expect("first fingerprint"),
            reordered
                .continuity_fingerprint()
                .expect("reordered fingerprint")
        );
        assert_eq!(
            first
                .stable_tools()
                .into_iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Alpha", "Zulu"]
        );
    }

    #[test]
    fn response_continuation_requires_exact_prefix_and_static_request() {
        let mut request = continuity_request();
        let fingerprint = request
            .continuity_fingerprint()
            .expect("continuity fingerprint");
        let completed_prefix = vec![
            Message::user("first"),
            Message::assistant("provider response"),
        ];
        let anchor = ResponseContinuation::new("resp_1", &completed_prefix, fingerprint)
            .expect("continuation anchor");

        request.messages = completed_prefix.clone();
        request.messages.push(Message::user("second"));
        request.response_continuation = Some(anchor.clone());
        assert_eq!(
            request
                .validated_response_continuation()
                .expect("exact continuation")
                .message_prefix_len(),
            2
        );

        let mut changed_prefix = request.clone();
        changed_prefix.messages[0] = Message::user("changed");
        assert!(changed_prefix.validated_response_continuation().is_none());

        let mut changed_tools = request.clone();
        changed_tools.tools[0].description.push_str(" changed");
        assert!(changed_tools.validated_response_continuation().is_none());

        let mut changed_model = request.clone();
        changed_model.model = "different-model".into();
        assert!(changed_model.validated_response_continuation().is_none());

        let mut changed_effort = request.clone();
        changed_effort.provider_options = json!({"reasoningEffort":"high"});
        assert!(changed_effort.validated_response_continuation().is_none());

        let mut changed_sampling = request.clone();
        changed_sampling.temperature = Some(0.7);
        assert!(changed_sampling.validated_response_continuation().is_none());

        let mut changed_output_limit = request.clone();
        changed_output_limit.max_tokens += 1;
        assert!(changed_output_limit
            .validated_response_continuation()
            .is_none());

        let mut changed_scope = request.clone();
        changed_scope.cache_scope = Some(
            ProviderCacheScope::for_session("workspace-b", "session-a")
                .expect("changed cache scope"),
        );
        assert!(changed_scope.validated_response_continuation().is_none());

        let mut no_delta = request;
        no_delta.messages = completed_prefix;
        assert!(no_delta.validated_response_continuation().is_none());
    }

    /// Regression test for issue #217. The non-streaming aggregators used by
    /// the Anthropic and MiniMax providers delegate to `StreamBlockAccumulator`.
    /// It must (a) fold `ThinkingDelta`/`SignatureDelta` into the thinking block
    /// rather than dropping them, and (b) preserve the model's interleave order
    /// (thinking-first), instead of appending non-text blocks last.
    #[test]
    fn stream_block_accumulator_keeps_thinking_signature_and_order() {
        let mut acc = StreamBlockAccumulator::new();

        // Block 0: a signed thinking block, streamed as thinking-then-signature.
        acc.on_event(&StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
        });
        acc.on_event(&StreamEvent::ThinkingDelta {
            index: 0,
            thinking: "Let me ".into(),
        });
        acc.on_event(&StreamEvent::ThinkingDelta {
            index: 0,
            thinking: "reason.".into(),
        });
        acc.on_event(&StreamEvent::SignatureDelta {
            index: 0,
            signature: "sig-".into(),
        });
        acc.on_event(&StreamEvent::SignatureDelta {
            index: 0,
            signature: "abc123".into(),
        });
        acc.on_event(&StreamEvent::ContentBlockStop { index: 0 });

        // Block 1: visible text.
        acc.on_event(&StreamEvent::ContentBlockStart {
            index: 1,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        });
        acc.on_event(&StreamEvent::TextDelta {
            index: 1,
            text: "Hello world".into(),
        });
        acc.on_event(&StreamEvent::ContentBlockStop { index: 1 });

        // Block 2: a tool call, streamed as partial JSON.
        acc.on_event(&StreamEvent::ContentBlockStart {
            index: 2,
            content_block: ContentBlock::ToolUse {
                id: "tool_1".into(),
                name: "get_weather".into(),
                input: Value::Null,
                thought_signature: Some("sig-tool-xyz".into()),
            },
        });
        acc.on_event(&StreamEvent::InputJsonDelta {
            index: 2,
            partial_json: r#"{"city":"#.into(),
        });
        acc.on_event(&StreamEvent::InputJsonDelta {
            index: 2,
            partial_json: r#""Paris"}"#.into(),
        });
        acc.on_event(&StreamEvent::ContentBlockStop { index: 2 });

        let blocks = acc.finish();
        assert_eq!(blocks.len(), 3, "all three blocks survive");

        // Order preserved: thinking → text → tool_use. The old aggregator
        // appended non-text blocks last (usize::MAX), so thinking would have
        // landed *after* the text block.
        match &blocks[0] {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Let me reason.", "thinking text captured");
                assert_eq!(signature, "sig-abc123", "signature preserved verbatim");
            }
            other => panic!("expected Thinking block first, got {other:?}"),
        }
        match &blocks[1] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello world"),
            other => panic!("expected Text block second, got {other:?}"),
        }
        match &blocks[2] {
            ContentBlock::ToolUse {
                id,
                name,
                input,
                thought_signature,
            } => {
                assert_eq!(id, "tool_1");
                assert_eq!(name, "get_weather");
                assert_eq!(
                    input.get("city").and_then(Value::as_str),
                    Some("Paris"),
                    "tool_use JSON assembled from partial deltas"
                );
                assert_eq!(
                    thought_signature.as_deref(),
                    Some("sig-tool-xyz"),
                    "opaque tool-call signature survives stream assembly"
                );
            }
            other => panic!("expected ToolUse block third, got {other:?}"),
        }
    }

    /// The `ReasoningDelta` alias (used by some providers) must also land in the
    /// thinking block rather than being dropped.
    #[test]
    fn stream_block_accumulator_folds_reasoning_delta_into_thinking() {
        let mut acc = StreamBlockAccumulator::new();
        acc.on_event(&StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
        });
        acc.on_event(&StreamEvent::ReasoningDelta {
            index: 0,
            reasoning: "scratch pad".into(),
        });
        acc.on_event(&StreamEvent::ContentBlockStop { index: 0 });

        let blocks = acc.finish();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Thinking { thinking, .. } => assert_eq!(thinking, "scratch pad"),
            other => panic!("expected Thinking block, got {other:?}"),
        }
    }
}
