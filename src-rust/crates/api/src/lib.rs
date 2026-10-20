// claurst-api: Anthropic API client with streaming SSE support for Claurst
// Rust port.
//
// Handles:
// - POST /v1/messages with streaming
// - SSE event parsing (message_start, content_block_start, content_block_delta,
//   content_block_stop, message_delta, message_stop, error)
// - Delta types: text_delta, input_json_delta, thinking_delta, signature_delta
// - Rate-limit (429) and overloaded (529) retry with exponential back-off
// - Authentication via API key from env or config

use claurst_core::constants::{ANTHROPIC_API_VERSION, ANTHROPIC_BETA_HEADER};
use claurst_core::dispatch::{DispatchDenied, DispatchRoute};
use claurst_core::effort::EffortLevel;
use claurst_core::error::ClaudeError;
use claurst_core::provider_id::ProviderId;
use claurst_core::types::{ContentBlock, Message, MessageContent, Role, ToolDefinition, UsageInfo};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::dispatch_policy::{
    authorize_anthropic_dispatch, AuthorizedInferenceLease, DispatchIdentity, DispatchPolicy,
};
use crate::inference_admission::{AdmissionLease, InferenceAdmission};

fn wire_budget_claude_error(provider: &str, error: provider_wire::WireBudgetError) -> ClaudeError {
    ClaudeError::from(provider_error::ProviderError::budget(
        ProviderId::new(provider),
        error,
    ))
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------
pub mod bun_tls;
pub mod client_factory;
pub mod codex_adapter;
pub mod dispatch_policy;
pub mod dispatch_request;
#[cfg(test)]
mod dispatch_request_tests;
#[cfg(test)]
mod final_request_coverage_tests;
pub mod inference_admission;
#[cfg(test)]
mod inference_admission_tests;

// Provider-agnostic unified types (Phase 1A).
pub mod provider_error;
pub mod provider_types;
pub mod provider_wire;
pub mod provider_witness;

// Provider abstraction traits (Phase 1B).
pub mod auth;
pub mod provider;
pub mod stream_parser;
pub mod transform;

// Wire-format protocol layer (#228): request-building + stream decoding owned
// once per wire format, shared across the providers that speak it.
pub mod protocol;

// Provider registry (Phase 1C).
pub mod registry;

// Concrete provider adapters (Phase 1D).
pub mod providers;

// Model Registry (Phase 3).
pub mod model_registry;

// Model-adaptive effort ladders (#267).
pub mod effort_support;

// opencode variants() effort ladder port (Phase 2, #268).
pub mod variants;

// Provider-aware error handling (Phase 6).
pub mod error_handling;

// Message transform layer — concrete transformers (Phase 4).
pub mod transformers;

// ---------------------------------------------------------------------------
// Public re-exports
// ---------------------------------------------------------------------------
pub use client::AnthropicClient;
pub use client_factory::AnthropicClientFactory;
pub use streaming::{AnthropicStreamEvent, StreamHandler};
pub use types::*;

// Phase 1A re-exports — provider-agnostic layer.
pub use dispatch_request::DispatchRequest;
pub use provider_error::ProviderError;
pub use provider_types::*;
pub use provider_wire::*;
pub use provider_witness::*;

// Phase 1B re-exports — provider abstraction traits.
pub use auth::{AuthProvider, LoginFlow};
pub use provider::{LlmProvider, ModelInfo};
pub use stream_parser::{JsonLinesStreamParser, SseByteDecoder, SseStreamParser, StreamParser};
pub use transform::MessageTransformer;

// #228 protocol layer re-exports.
pub use protocol::{LineStreamDecoder, OpenAiChatDecoder};

// Phase 1C re-exports — provider registry.
pub use registry::ProviderRegistry;

// Phase 1D re-exports — concrete provider adapters.
pub use providers::AnthropicProvider;
pub use providers::GoogleProvider;
pub use providers::MinimaxProvider;
pub use providers::OpenAiProvider;

// Phase 3 re-exports — model registry.
pub use effort_support::{model_is_reasoning, supported_efforts, variant_ladder};
pub use model_registry::{
    effective_model_for_config, CostBreakdown, CostTier, CostTierCondition, ExperimentalMode,
    InterleavedReasoning, Modality, ModelEntry, ModelRegistry, ModelStatus, ProviderEntry,
    ProviderOverride,
};
pub use variants::{
    variant_efforts, OPENAI_NONE_EFFORT_RELEASE_DATE, OPENAI_XHIGH_EFFORT_RELEASE_DATE,
};

// Phase 6 re-exports — provider-aware error handling.
pub use error_handling::{is_context_overflow, parse_error_response, RetryConfig};

// Phase 2E re-exports — Azure, Bedrock, and GitHub Copilot providers.
pub use providers::AzureProvider;
pub use providers::BedrockProvider;
pub use providers::CopilotProvider;

// Phase 2B re-exports — OpenAI-compatible generic adapter + common factories.
pub use providers::{
    deepseek, groq, lm_studio, mistral, ollama, opencode_zen, openrouter, xai, OpenAiCompatProvider,
};

// ---------------------------------------------------------------------------
// Request timeout configuration (issue #175)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};

/// Default total request timeout in seconds when the user has not configured
/// one. Kept in sync with [`claurst_core::config::DEFAULT_REQUEST_TIMEOUT_SECS`].
pub use claurst_core::config::DEFAULT_REQUEST_TIMEOUT_SECS;

/// Process-wide total request timeout (seconds) applied to provider HTTP
/// clients that are constructed lazily without access to the user `Config`
/// (OpenAI, OpenAI-compatible, Cohere, MiniMax, Copilot, Azure, Bedrock, …).
///
/// Set once at startup from the resolved configuration via
/// [`set_request_timeout_secs`]; defaults to [`DEFAULT_REQUEST_TIMEOUT_SECS`].
/// A process-wide value is used because providers are built in many places via
/// builder chains that do not thread the user `Config` through.
static REQUEST_TIMEOUT_SECS: AtomicU64 = AtomicU64::new(DEFAULT_REQUEST_TIMEOUT_SECS);

/// Override the process-wide request timeout. A value of `0` resets to
/// [`DEFAULT_REQUEST_TIMEOUT_SECS`]. Idempotent; safe to call multiple times.
pub fn set_request_timeout_secs(secs: u64) {
    let value = if secs == 0 {
        DEFAULT_REQUEST_TIMEOUT_SECS
    } else {
        secs
    };
    REQUEST_TIMEOUT_SECS.store(value, Ordering::Relaxed);
}

/// Current process-wide request timeout in seconds.
pub fn request_timeout_secs() -> u64 {
    REQUEST_TIMEOUT_SECS.load(Ordering::Relaxed)
}

/// Current process-wide request timeout as a [`Duration`].
pub fn request_timeout() -> Duration {
    Duration::from_secs(request_timeout_secs())
}

/// Per-chunk idle timeout used to bound infinite mid-stream stalls (issue #185).
///
/// An OpenAI-compatible provider can begin a streamed tool call and then pause
/// indefinitely before sending the tool arguments. The streaming loops wrap
/// each `byte_stream.next()` in `tokio::time::timeout(stream_idle_timeout(), …)`
/// so such stalls surface as a stream error instead of hanging forever.
///
/// The value is GENEROUS and never smaller than the configured request timeout,
/// so legitimately slow-but-progressing local models (whose chunks reset the
/// timer) are never cut off — the goal is only to bound true infinite stalls.
pub fn stream_idle_timeout() -> Duration {
    request_timeout().max(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS))
}

// Composite "Free" provider — stacks many free-tier upstreams behind one
// `free/auto` model id.
pub use providers::{FreeEntry, FreeProvider, FreeUpstream, FREE_CATALOG};

// Phase 2D re-exports — Cohere native provider.
pub use providers::CohereProvider;

// Phase 4 re-exports — concrete message transformers.
pub use transformers::{AnthropicTransformer, OpenAiChatTransformer};

// ---------------------------------------------------------------------------
// request / response types
// ---------------------------------------------------------------------------
pub mod types {
    use super::*;

    /// The request body sent to `POST /v1/messages`.
    #[derive(Debug, Clone, Serialize)]
    pub struct CreateMessageRequest {
        pub model: String,
        pub max_tokens: u32,
        pub messages: Vec<ApiMessage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system: Option<SystemPrompt>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<ApiToolDefinition>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        pub stream: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking: Option<ThinkingConfig>,
        /// Anthropic native automatic prompt-caching control. Other adapters
        /// never serialize this legacy Messages request type.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cache_control: Option<CacheControl>,
        /// Finite preflight policy retained for the transport boundary.
        /// This metadata is never serialized into provider JSON.
        #[serde(skip)]
        pub wire_budget: Option<provider_wire::WireBudget>,
        /// Host-owned dispatch authorization metadata, omitted from provider JSON.
        #[serde(skip)]
        pub dispatch: Option<crate::DispatchRequest>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ThinkingConfig {
        #[serde(rename = "type")]
        pub thinking_type: String,
        #[serde(default, skip_serializing_if = "is_zero")]
        pub budget_tokens: u32,
    }

    fn is_zero(value: &u32) -> bool {
        *value == 0
    }

    impl ThinkingConfig {
        pub fn enabled(budget: u32) -> Self {
            Self {
                thinking_type: "enabled".to_string(),
                budget_tokens: budget,
            }
        }

        pub fn adaptive() -> Self {
            Self {
                thinking_type: "adaptive".to_string(),
                budget_tokens: 0,
            }
        }
    }

    /// System prompt - either a single string or structured blocks with cache.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum SystemPrompt {
        Text(String),
        Blocks(Vec<SystemBlock>),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SystemBlock {
        #[serde(rename = "type")]
        pub block_type: String,
        pub text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cache_control: Option<CacheControl>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CacheControl {
        #[serde(rename = "type")]
        pub control_type: String,
    }

    impl CacheControl {
        pub fn ephemeral() -> Self {
            Self {
                control_type: "ephemeral".to_string(),
            }
        }
    }

    /// Simplified message type for the API wire format.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ApiMessage {
        pub role: String,
        pub content: Value,
    }

    impl From<&Message> for ApiMessage {
        fn from(msg: &Message) -> Self {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            let content = match &msg.content {
                MessageContent::Text(t) => Value::String(t.clone()),
                MessageContent::Blocks(blocks) => {
                    serde_json::to_value(blocks).unwrap_or(Value::Null)
                }
            };
            Self {
                role: role.to_string(),
                content,
            }
        }
    }

    /// Tool definition in the API wire format.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ApiToolDefinition {
        pub name: String,
        pub description: String,
        pub input_schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cache_control: Option<CacheControl>,
    }

    impl From<&ToolDefinition> for ApiToolDefinition {
        fn from(td: &ToolDefinition) -> Self {
            Self {
                name: td.name.clone(),
                description: td.description.clone(),
                input_schema: td.input_schema.clone(),
                cache_control: None,
            }
        }
    }

    /// Non-streaming response from `POST /v1/messages`.
    #[derive(Debug, Clone, Deserialize)]
    pub struct CreateMessageResponse {
        pub id: String,
        #[serde(rename = "type")]
        pub response_type: String,
        pub role: String,
        pub content: Vec<Value>,
        pub model: String,
        pub stop_reason: Option<String>,
        pub stop_sequence: Option<String>,
        pub usage: UsageInfo,
    }

    /// Error body returned by the API.
    #[derive(Debug, Clone, Deserialize)]
    pub struct ApiErrorResponse {
        #[serde(rename = "type")]
        pub error_type: String,
        pub error: ApiErrorDetail,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct ApiErrorDetail {
        #[serde(rename = "type")]
        pub error_type: String,
        pub message: String,
    }
}

// ---------------------------------------------------------------------------
// SSE streaming types
// ---------------------------------------------------------------------------
pub mod streaming {
    use super::*;

    /// Events emitted by the Anthropic SSE streaming parser.
    #[derive(Debug, Clone)]
    pub enum AnthropicStreamEvent {
        /// The overall message has started; carries the message id and model.
        MessageStart {
            id: String,
            model: String,
            usage: UsageInfo,
        },
        /// A new content block has begun.
        ContentBlockStart {
            index: usize,
            content_block: ContentBlock,
        },
        /// Incremental delta for an existing content block.
        ContentBlockDelta { index: usize, delta: ContentDelta },
        /// A content block is finished.
        ContentBlockStop { index: usize },
        /// Final message-level delta (stop_reason, usage).
        MessageDelta {
            stop_reason: Option<String>,
            usage: Option<UsageInfo>,
        },
        /// The message is complete.
        MessageStop,
        /// An error occurred during streaming.
        Error { error_type: String, message: String },
        /// A ping/keep-alive event.
        Ping,
    }

    /// The delta payload inside a `content_block_delta` event.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ContentDelta {
        TextDelta { text: String },
        InputJsonDelta { partial_json: String },
        ThinkingDelta { thinking: String },
        SignatureDelta { signature: String },
    }

    /// Trait for anything that wants to consume streaming events in real time.
    pub trait StreamHandler: Send + Sync {
        fn on_event(&self, event: &AnthropicStreamEvent);
    }

    /// A no-op handler useful for non-interactive / batch mode.
    pub struct NullStreamHandler;
    impl StreamHandler for NullStreamHandler {
        fn on_event(&self, _event: &AnthropicStreamEvent) {}
    }
}

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------
mod sse_parser {
    /// Parsed SSE frame.
    #[derive(Debug)]
    pub struct SseFrame {
        pub event: Option<String>,
        pub data: String,
    }

    /// Incrementally accumulates raw bytes/lines and yields complete frames.
    pub struct SseLineParser {
        event_type: Option<String>,
        data_buf: String,
    }

    impl SseLineParser {
        pub fn new() -> Self {
            Self {
                event_type: None,
                data_buf: String::new(),
            }
        }

        /// Feed one line (without the trailing newline).  Returns `Some(frame)`
        /// when a blank line signals the end of an event.
        pub fn feed_line(&mut self, line: &str) -> Option<SseFrame> {
            if line.is_empty() {
                // Blank line = end of event
                if self.data_buf.is_empty() && self.event_type.is_none() {
                    return None; // spurious blank line
                }
                let frame = SseFrame {
                    event: self.event_type.take(),
                    data: std::mem::take(&mut self.data_buf),
                };
                return Some(frame);
            }

            if let Some(rest) = line.strip_prefix("event:") {
                self.event_type = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !self.data_buf.is_empty() {
                    self.data_buf.push('\n');
                }
                self.data_buf.push_str(rest.trim());
            } else if line.starts_with(':') {
                // SSE comment / keep-alive – ignore
            }

            None
        }
    }
}

// ---------------------------------------------------------------------------
// Models endpoint types (public)
// ---------------------------------------------------------------------------

/// A model entry returned by `GET /v1/models`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AvailableModel {
    pub id: String,
    pub display_name: Option<String>,
    /// Unix timestamp of when the model was created (seconds).
    pub created_at: Option<i64>,
}

// ---------------------------------------------------------------------------
// Anthropic client
// ---------------------------------------------------------------------------
pub mod client {
    use super::*;

    /// Provider selection for API calls.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum Provider {
        /// Use Anthropic's API
        #[default]
        Anthropic,
        /// Use OpenAI Codex via OAuth
        Codex,
    }

    /// Configuration for the HTTP client.
    #[derive(Debug, Clone)]
    pub struct ClientConfig {
        pub api_key: String,
        pub api_base: String,
        pub api_version: String,
        pub beta_features: String,
        pub max_retries: u32,
        pub initial_retry_delay: Duration,
        pub max_retry_delay: Duration,
        pub request_timeout: Duration,
        /// When true, send `Authorization: Bearer <api_key>` instead of `x-api-key`.
        /// Used for Claude.ai subscription (OAuth user:inference scope) tokens.
        pub use_bearer_auth: bool,
        /// Which provider to use for API calls.
        pub provider: Provider,
    }

    impl Default for ClientConfig {
        fn default() -> Self {
            Self {
                api_key: String::new(),
                api_base: claurst_core::constants::ANTHROPIC_API_BASE.to_string(),
                api_version: ANTHROPIC_API_VERSION.to_string(),
                beta_features: ANTHROPIC_BETA_HEADER.to_string(),
                max_retries: 5,
                initial_retry_delay: Duration::from_secs(1),
                max_retry_delay: Duration::from_secs(60),
                // Honour the process-wide configured timeout (issue #175);
                // falls back to DEFAULT_REQUEST_TIMEOUT_SECS when unset.
                request_timeout: crate::request_timeout(),
                use_bearer_auth: false,
                provider: Provider::Anthropic,
            }
        }
    }

    /// The main Anthropic API client.
    pub struct AnthropicClient {
        http: wreq::Client,
        config: ClientConfig,
        inference_admission: Arc<InferenceAdmission>,
        dispatch_identity: DispatchIdentity,
        dispatch_policy: Option<Arc<DispatchPolicy>>,
        /// Account UUID captured from the same stored token snapshot as
        /// `config.api_key`. It never follows later active-account switches.
        oauth_account_uuid: Option<String>,
        /// Per-client profile result. A process-wide cache can mix account
        /// metadata after a switch, so each credential owner keeps its own.
        oauth_account_meta: tokio::sync::OnceCell<Option<(String, bool)>>,
        /// Stable per-client session id; the official client reuses one id for
        /// the whole session, in both the header and `metadata.user_id`.
        session_id: String,
    }

    impl AnthropicClient {
        /// The selected legacy transport, independent of credential contents.
        /// This does not establish support for verified dispatch.
        pub fn provider_kind(&self) -> Provider {
            self.config.provider
        }

        /// Returns `true` when the client was constructed without an API key.
        ///
        /// The query loop checks this to know whether it should fall back to
        /// a runtime-built provider (e.g. from keys stored via `/connect`).
        pub fn api_key_is_empty(&self) -> bool {
            self.config.api_key.is_empty()
        }

        /// Returns `true` when this client is configured to use a Claude Code
        /// OAuth Bearer token (Claude.ai Pro/Max). The query path and the
        /// request builders check this to enable stealth-impersonation.
        pub fn is_oauth(&self) -> bool {
            self.config.use_bearer_auth
        }

        #[cfg(test)]
        pub(crate) fn captured_oauth_account_uuid(&self) -> Option<&str> {
            self.oauth_account_uuid.as_deref()
        }

        /// First user (non-meta) message text — input to the `cc_version` client
        /// hash. Mirrors the official `juA`: skips `<system-reminder>` blocks.
        fn first_user_message_text(messages: &[ApiMessage]) -> String {
            let is_meta = |s: &str| s.trim_start().starts_with("<system-reminder>");
            for m in messages {
                if m.role != "user" {
                    continue;
                }
                match &m.content {
                    serde_json::Value::String(s) if !is_meta(s) => return s.clone(),
                    serde_json::Value::Array(blocks) => {
                        for b in blocks {
                            if b.get("type").and_then(|v| v.as_str()) != Some("text") {
                                continue;
                            }
                            if let Some(text) = b.get("text").and_then(|v| v.as_str()) {
                                if !is_meta(text) {
                                    return text.to_string();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            String::new()
        }

        /// Make an OAuth request look like Claude Code: prepend the
        /// `x-anthropic-billing-header` block (`system[0]`) then the
        /// `"You are Claude Code…"` identity block (`system[1]`), and strip
        /// Claurst's own attribution so the official identity is the only one the
        /// server sees. No-op for API-key auth.
        pub(crate) fn apply_oauth_stealth(&self, request: &mut CreateMessageRequest) {
            if !self.config.use_bearer_auth {
                return;
            }

            let first_user_text = Self::first_user_message_text(&request.messages);
            let text_block = |text: String| SystemBlock {
                block_type: "text".to_string(),
                text,
                cache_control: None,
            };
            let billing_block = text_block(claurst_core::oauth_config::claude_code_billing_header(
                &first_user_text,
            ));
            let identity_block = text_block(
                claurst_core::oauth_config::CLAUDE_CODE_SYSTEM_PROMPT_PREFIX.to_string(),
            );

            // Drop a leading "You are Claurst…" / "You are a Claude agent…" line:
            // the injected official identity must be the only one the server sees.
            let strip_attr = |text: &str| -> String {
                let t = text.trim_start();
                if t.starts_with("You are Claurst") || t.starts_with("You are a Claude agent") {
                    if let Some(i) = t.find("\n\n") {
                        return t[i + 2..].to_string();
                    }
                    if let Some(i) = t.find('\n') {
                        return t[i + 1..].to_string();
                    }
                    return String::new();
                }
                text.to_string()
            };

            let mut preserved = match request.system.take() {
                None => Vec::new(),
                Some(SystemPrompt::Text(existing)) => {
                    let stripped = strip_attr(&existing);
                    (!stripped.is_empty())
                        .then(|| text_block(stripped))
                        .into_iter()
                        .collect()
                }
                Some(SystemPrompt::Blocks(blocks)) => blocks
                    .into_iter()
                    .filter(|block| {
                        !block.text.starts_with("x-anthropic-billing-header:")
                            && block.text
                                != claurst_core::oauth_config::CLAUDE_CODE_SYSTEM_PROMPT_PREFIX
                    })
                    .collect(),
            };
            if let Some(first) = preserved.first_mut() {
                first.text = strip_attr(&first.text);
            }
            preserved.retain(|block| !block.text.is_empty());

            let mut blocks = Vec::with_capacity(preserved.len() + 2);
            blocks.push(billing_block);
            blocks.push(identity_block);
            blocks.extend(preserved);
            request.system = Some(SystemPrompt::Blocks(blocks));
        }

        /// Build a new client. Uses a `wreq`/BoringSSL client whose TLS
        /// fingerprint matches Bun (the official client). An empty key is
        /// allowed; validation is deferred to the first call.
        pub fn new(config: ClientConfig) -> anyhow::Result<Self> {
            let http = crate::bun_tls::build_anthropic_client(config.request_timeout)?;
            let oauth_account_uuid = config
                .use_bearer_auth
                .then(|| {
                    claurst_core::oauth::OAuthTokens::active_account_uuid_for_credential(
                        &config.api_key,
                    )
                })
                .flatten();
            Ok(Self {
                http,
                config,
                inference_admission: InferenceAdmission::process(),
                dispatch_identity: DispatchIdentity::fresh(),
                dispatch_policy: None,
                oauth_account_uuid,
                oauth_account_meta: tokio::sync::OnceCell::const_new(),
                session_id: uuid::Uuid::new_v4().to_string(),
            })
        }

        /// Use a caller-owned process admission service. This is primarily
        /// useful for bounded tests and hosts that configure a shared limit.
        pub fn with_inference_admission(mut self, admission: Arc<InferenceAdmission>) -> Self {
            self.inference_admission = admission;
            self
        }

        /// Install a host-owned dispatch policy. Replacing the policy rotates
        /// the client identity, invalidating envelopes minted for the prior
        /// policy instance.
        pub fn with_dispatch_policy(mut self, policy: Arc<DispatchPolicy>) -> Self {
            self.dispatch_identity = DispatchIdentity::fresh();
            self.dispatch_policy = Some(policy);
            self
        }

        fn dispatch_endpoint(&self) -> String {
            format!("{}/v1/messages", self.config.api_base)
        }

        /// Construct a pure Anthropic API-key route. OAuth, Codex, missing
        /// credentials, and absent policy are unsupported by this bridge.
        pub fn dispatch_route(
            &self,
            model: &str,
            effort: EffortLevel,
        ) -> Result<DispatchRoute, DispatchDenied> {
            if self.config.provider != Provider::Anthropic
                || self.config.use_bearer_auth
                || self.config.api_key.is_empty()
            {
                return Err(DispatchDenied::UnsupportedRoute);
            }
            let policy = self
                .dispatch_policy
                .as_ref()
                .ok_or(DispatchDenied::UnsupportedRoute)?;
            policy.anthropic_route(
                &self.dispatch_identity,
                &self.dispatch_endpoint(),
                model,
                effort,
            )
        }

        fn validate_dispatch_request(
            &self,
            dispatch: Option<&crate::DispatchRequest>,
        ) -> Result<(), ClaudeError> {
            let Some(dispatch) = dispatch else {
                return Ok(());
            };
            if self.config.provider != Provider::Anthropic
                || self.config.use_bearer_auth
                || self.config.api_key.is_empty()
            {
                return Err(ClaudeError::InferenceDispatchDenied(
                    DispatchDenied::UnsupportedRoute,
                ));
            }
            let policy =
                self.dispatch_policy
                    .as_ref()
                    .ok_or(ClaudeError::InferenceDispatchDenied(
                        DispatchDenied::UnsupportedRoute,
                    ))?;
            policy
                .validate_anthropic_envelope(
                    &self.dispatch_identity,
                    &self.dispatch_endpoint(),
                    dispatch,
                )
                .map_err(ClaudeError::InferenceDispatchDenied)
        }

        /// Convenience constructor that resolves the key from config/env.
        pub fn from_config(cfg: &claurst_core::config::Config) -> anyhow::Result<Self> {
            let api_key = cfg
                .resolve_api_key()
                .ok_or_else(|| anyhow::anyhow!("No API key found"))?;
            let api_base = cfg.resolve_api_base();

            Self::new(ClientConfig {
                api_key,
                api_base,
                ..Default::default()
            })
        }

        // ---- Non-streaming create message --------------------------------

        /// Send a non-streaming `POST /v1/messages` and return the full response.
        pub async fn create_message(
            &self,
            mut request: CreateMessageRequest,
        ) -> Result<CreateMessageResponse, ClaudeError> {
            self.validate_dispatch_request(request.dispatch.as_ref())?;
            // Deferred key validation — fail here rather than at construction
            // so that non-Anthropic provider setups don't crash on startup.
            if self.config.api_key.is_empty() && self.config.provider != Provider::Codex {
                // Check if this model might belong to another provider, giving
                // the user a more actionable error message.
                let model = &request.model;
                let hint = if model.starts_with("gemini") || model.starts_with("gemma") {
                    format!(
                        "Model '{}' is a Google model. Use `--provider google` or set GOOGLE_API_KEY.",
                        model
                    )
                } else if model.starts_with("gpt-")
                    || model.starts_with("o1")
                    || model.starts_with("o3")
                    || model.starts_with("o4")
                {
                    format!(
                        "Model '{}' is an OpenAI model. Use `--provider openai` or set OPENAI_API_KEY.",
                        model
                    )
                } else if model.starts_with("deepseek") {
                    format!(
                        "Model '{}' is a DeepSeek model. Use `--provider deepseek` or set DEEPSEEK_API_KEY.",
                        model
                    )
                } else if model.starts_with("grok") {
                    format!(
                        "Model '{}' is an xAI model. Use `--provider xai` or set XAI_API_KEY.",
                        model
                    )
                } else if model.starts_with("mistral") || model.starts_with("codestral") {
                    format!(
                        "Model '{}' is a Mistral model. Use `--provider mistral` or set MISTRAL_API_KEY.",
                        model
                    )
                } else if model.starts_with("command-") {
                    format!(
                        "Model '{}' is a Cohere model. Use `--provider cohere` or set COHERE_API_KEY.",
                        model
                    )
                } else if model.starts_with("llama") {
                    format!(
                        "Model '{}' looks like a Llama model. Use `--provider groq` (set GROQ_API_KEY) or `--provider ollama` for local.",
                        model
                    )
                } else {
                    "Set ANTHROPIC_API_KEY, run `claurst auth login`, \
                     or use --provider to select a different provider (e.g. --provider openai)."
                        .to_string()
                };
                return Err(ClaudeError::Auth(format!(
                    "No API key for the selected model. {}",
                    hint
                )));
            }
            // Route to Codex if configured
            if self.config.provider == Provider::Codex {
                return self.create_message_codex(&request).await;
            }

            request.stream = false;
            self.apply_oauth_stealth(&mut request);
            let body = serde_json::to_value(&request).map_err(ClaudeError::Json)?;

            let (resp, _lease) = self
                .send_with_retry(
                    &body,
                    request.wire_budget.as_ref(),
                    request.dispatch.as_ref(),
                    false,
                )
                .await?;
            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|_| ClaudeError::Api("Failed to read provider response".into()))?;

            if !status.is_success() {
                return Err(Self::parse_api_error(status.as_u16(), &text));
            }

            serde_json::from_str(&text).map_err(ClaudeError::Json)
        }

        /// Send a request to OpenAI Codex API instead of Anthropic.
        async fn create_message_codex(
            &self,
            request: &CreateMessageRequest,
        ) -> Result<CreateMessageResponse, ClaudeError> {
            crate::dispatch_request::reject_unsupported_dispatch(
                &ProviderId::new(ProviderId::CODEX),
                request.dispatch.as_ref(),
            )
            .map_err(ClaudeError::from)?;
            Err(ClaudeError::Other(
                "The legacy Codex client cannot safely translate Chat messages to the Responses API; use the modern Codex provider"
                    .to_string(),
            ))
        }

        // ---- Streaming create message ------------------------------------

        /// Send a streaming `POST /v1/messages`.  Events are dispatched to the
        /// provided `handler` in real time, and also forwarded into the returned
        /// channel so the caller can drive a select loop.
        pub async fn create_message_stream(
            &self,
            mut request: CreateMessageRequest,
            handler: Arc<dyn StreamHandler>,
        ) -> Result<mpsc::Receiver<streaming::AnthropicStreamEvent>, ClaudeError> {
            self.validate_dispatch_request(request.dispatch.as_ref())?;
            // Deferred key validation
            if self.config.api_key.is_empty() && self.config.provider != Provider::Codex {
                let model = &request.model;
                let hint = if model.starts_with("gemini") || model.starts_with("gemma") {
                    format!(
                        "Model '{}' is a Google model. Use `--provider google` or set GOOGLE_API_KEY.",
                        model
                    )
                } else if model.starts_with("gpt-")
                    || model.starts_with("o1")
                    || model.starts_with("o3")
                    || model.starts_with("o4")
                {
                    format!(
                        "Model '{}' is an OpenAI model. Use `--provider openai` or set OPENAI_API_KEY.",
                        model
                    )
                } else if model.starts_with("deepseek") {
                    format!("Model '{}' is a DeepSeek model. Use `--provider deepseek` or set DEEPSEEK_API_KEY.", model)
                } else if model.starts_with("grok") {
                    format!(
                        "Model '{}' is an xAI model. Use `--provider xai` or set XAI_API_KEY.",
                        model
                    )
                } else if model.starts_with("mistral") || model.starts_with("codestral") {
                    format!("Model '{}' is a Mistral model. Use `--provider mistral` or set MISTRAL_API_KEY.", model)
                } else if model.starts_with("command-") {
                    format!("Model '{}' is a Cohere model. Use `--provider cohere` or set COHERE_API_KEY.", model)
                } else if model.starts_with("llama") {
                    format!("Model '{}' looks like a Llama model. Use `--provider groq` or `--provider ollama` for local.", model)
                } else {
                    "Set ANTHROPIC_API_KEY, run `claurst auth login`, \
                     or use --provider to select a different provider (e.g. --provider openai)."
                        .to_string()
                };
                return Err(ClaudeError::Auth(format!(
                    "No API key for the selected model. {}",
                    hint
                )));
            }
            // Codex provider doesn't support streaming yet
            if self.config.provider == Provider::Codex {
                return Err(ClaudeError::Other(
                    "Codex provider does not support streaming yet".to_string(),
                ));
            }

            request.stream = true;
            self.apply_oauth_stealth(&mut request);
            let body = serde_json::to_value(&request).map_err(ClaudeError::Json)?;

            let (resp, lease) = self
                .send_with_retry(
                    &body,
                    request.wire_budget.as_ref(),
                    request.dispatch.as_ref(),
                    true,
                )
                .await?;
            let status = resp.status();

            if !status.is_success() {
                let text = resp
                    .text()
                    .await
                    .map_err(|_| ClaudeError::Api("Failed to read provider response".into()))?;
                return Err(Self::parse_api_error(status.as_u16(), &text));
            }

            let (tx, rx) = mpsc::channel(256);

            // Spawn a task that reads the SSE byte stream and emits events.
            let process_handler = Arc::clone(&handler);
            tokio::spawn(async move {
                let lease = lease;
                let result = tokio::select! {
                    _ = tx.closed() => Ok(None),
                    result = Self::process_sse_stream(resp, process_handler, tx.clone()) => result,
                };
                // `process_sse_stream` owns the response body. Drop both it
                // and the authorization/admission lease before delivering a
                // terminal event to a retained receiver.
                drop(lease);
                match result {
                    Ok(Some(event)) => {
                        if !tx.is_closed() {
                            handler.on_event(&event);
                            let _ = tx.send(event).await;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        if !tx.is_closed() {
                            let _ = tx
                                .send(streaming::AnthropicStreamEvent::Error {
                                    error_type: "stream_error".into(),
                                    message: e.to_string(),
                                })
                                .await;
                        }
                    }
                }
            });

            Ok(rx)
        }

        // ---- Models list ------------------------------------------------

        /// Fetch available models from `GET /v1/models`.
        ///
        /// Returns a list of models the current API key has access to.
        /// Falls back gracefully: returns an empty `Vec` on any error so
        /// callers can fall back to the hardcoded default list instead of
        /// surfacing an error.
        pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<crate::AvailableModel>> {
            let url = format!("{}/v1/models", self.config.api_base);

            let mut req = self
                .http
                .get(&url)
                .header("anthropic-version", &self.config.api_version)
                .header("content-type", "application/json");
            if self.config.use_bearer_auth {
                req = req
                    .header(
                        "anthropic-beta",
                        claurst_core::oauth_config::OAUTH_BETA_FLAGS.join(","),
                    )
                    .header(
                        "user-agent",
                        claurst_core::oauth_config::claude_code_user_agent(),
                    )
                    .header("x-app", "cli")
                    .header("Authorization", format!("Bearer {}", self.config.api_key));
            } else {
                req = req.header("x-api-key", &self.config.api_key);
            }

            let resp = req.send().await?;

            if !resp.status().is_success() {
                anyhow::bail!("models endpoint returned {}", resp.status());
            }

            #[derive(serde::Deserialize)]
            struct ModelsResponse {
                data: Vec<crate::AvailableModel>,
            }

            let body: ModelsResponse = resp.json().await?;
            Ok(body.data)
        }

        // ---- Internal helpers --------------------------------------------

        /// Build the common request and execute with retry logic.
        async fn send_with_retry(
            &self,
            body: &Value,
            budget: Option<&provider_wire::WireBudget>,
            dispatch: Option<&crate::DispatchRequest>,
            stream: bool,
        ) -> Result<(wreq::Response, AuthorizedInferenceLease), ClaudeError> {
            self.validate_dispatch_request(dispatch)?;
            let url = self.dispatch_endpoint();
            let mut attempts = 0u32;
            let mut delay = self.config.initial_retry_delay;

            // Validate the unmodified request before OAuth account metadata
            // lookup. This keeps malformed, oversized, or unsupported-media
            // requests entirely local; the final body is prepared again after
            // metadata mutation below.
            provider_wire::prepare_json_checked("anthropic", &url, stream, body, budget)
                .map_err(|error| wire_budget_claude_error("anthropic", error))?;

            let use_oauth = self.config.use_bearer_auth;
            let session_id = self.session_id.clone();

            // Active OAuth account, fetched once for this exact client
            // credential. `account_uuid` -> `metadata.user_id`; `has_premium`
            // selects the account-tier `anthropic-beta` set.
            let (account_uuid, has_premium): (String, bool) = if use_oauth {
                self.oauth_account_meta
                    .get_or_init(|| {
                        claurst_core::oauth::anthropic_account_meta_for_credential(
                            &self.config.api_key,
                            self.oauth_account_uuid.clone(),
                        )
                    })
                    .await
                    .clone()
                    .unwrap_or_default()
            } else {
                (String::new(), false)
            };

            // On the OAuth path, inject `metadata.user_id`. There is no `cch`
            // step: the interactive CLI sends a literal `cch=00000` and its real
            // client hash rides in the `cc_version` suffix (set in
            // `apply_oauth_stealth`). See `claude-re/findings/CCH-NATIVE.md`.
            let body_val = if use_oauth {
                let mut body_val = body.clone();
                if let serde_json::Value::Object(map) = &mut body_val {
                    let device_id = {
                        use sha2::{Digest, Sha256};
                        let user = std::env::var("USER")
                            .or_else(|_| std::env::var("USERNAME"))
                            .unwrap_or_default();
                        let home = std::env::var("HOME")
                            .or_else(|_| std::env::var("USERPROFILE"))
                            .unwrap_or_default();
                        let mut h = Sha256::new();
                        h.update(user.as_bytes());
                        h.update(b":");
                        h.update(home.as_bytes());
                        format!("{:x}", h.finalize())
                    };
                    let user_id = serde_json::json!({
                        "device_id": device_id,
                        "account_uuid": account_uuid,
                        "session_id": session_id,
                    })
                    .to_string();
                    match map.get_mut("metadata") {
                        Some(serde_json::Value::Object(m)) => {
                            m.insert("user_id".to_string(), serde_json::Value::String(user_id));
                        }
                        _ => {
                            map.insert(
                                "metadata".to_string(),
                                serde_json::json!({ "user_id": user_id }),
                            );
                        }
                    }
                }
                body_val
            } else {
                body.clone()
            };
            let prepared =
                provider_wire::prepare_json_checked("anthropic", &url, stream, &body_val, budget)
                    .map_err(|error| wire_budget_claude_error("anthropic", error))?;
            let body_bytes = prepared.bytes().to_vec();

            // Account-tier `anthropic-beta` set (Pro vs Max), stable across retries.
            let anthropic_beta = if use_oauth {
                let mut s = claurst_core::oauth_config::oauth_beta_flags(has_premium).join(",");
                if !self.config.beta_features.is_empty() {
                    if !s.is_empty() {
                        s.push(',');
                    }
                    s.push_str(&self.config.beta_features);
                }
                s
            } else {
                self.config.beta_features.clone()
            };
            // Map Rust's target triple onto the Stainless SDK's OS/arch labels.
            let stainless_os = match std::env::consts::OS {
                "macos" => "MacOS",
                "linux" => "Linux",
                "windows" => "Windows",
                "freebsd" => "FreeBSD",
                "openbsd" => "OpenBSD",
                other => other,
            };
            let stainless_arch = match std::env::consts::ARCH {
                "aarch64" => "arm64",
                "x86_64" => "x64",
                "x86" => "x32",
                other => other,
            };

            let mut admission: Option<AdmissionLease> = None;

            loop {
                attempts += 1;

                let mut req = self
                    .http
                    .post(&url)
                    .header("anthropic-version", &self.config.api_version)
                    .header("anthropic-beta", &anthropic_beta)
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream");

                if use_oauth {
                    // Official UA, `x-app` and Stainless telemetry headers so the
                    // server treats this as a first-party request. The billing
                    // header already rides in `system[0]` of the body.
                    req = req
                        .header(
                            "user-agent",
                            claurst_core::oauth_config::claude_code_user_agent(),
                        )
                        .header("x-app", "cli")
                        .header("anthropic-dangerous-direct-browser-access", "true")
                        .header("x-stainless-lang", "js")
                        .header("x-stainless-runtime", "node")
                        .header("x-stainless-os", stainless_os)
                        .header("x-stainless-arch", stainless_arch)
                        .header("x-stainless-runtime-version", "v22.0.0")
                        .header("x-stainless-package-version", "0.94.0")
                        .header("x-stainless-retry-count", (attempts - 1).to_string())
                        .header(
                            "x-stainless-timeout",
                            self.config.request_timeout.as_secs().to_string(),
                        )
                        .header("x-claude-code-session-id", &session_id)
                        .header("x-client-request-id", uuid::Uuid::new_v4().to_string())
                        .header("Authorization", format!("Bearer {}", self.config.api_key));
                } else {
                    // API-key path: no `x-anthropic-billing-header` (it is a
                    // Claude Code / subscription artefact, not emitted by the
                    // direct-API SDK).
                    req = req.header("x-api-key", &self.config.api_key);
                }

                if dispatch.is_some() {
                    req = req.redirect(wreq::redirect::Policy::none());
                }
                let request = req
                    .body(body_bytes.clone())
                    .build()
                    .map_err(|_| ClaudeError::Api("HTTP request construction failed".into()))?;
                let endpoint = request.uri().to_string();
                if admission.is_none() {
                    admission =
                        Some(self.inference_admission.acquire().await.map_err(|error| {
                            ClaudeError::Api(format!("Inference admission failed: {error}"))
                        })?);
                }
                let authorization = authorize_anthropic_dispatch(
                    self.dispatch_policy.as_deref(),
                    &self.dispatch_identity,
                    &endpoint,
                    &prepared,
                    dispatch,
                )
                .await
                .map_err(ClaudeError::InferenceDispatchDenied)?;
                let resp = self
                    .http
                    .execute(request)
                    .await
                    .map_err(|_| ClaudeError::Api("Provider HTTP request failed".into()))?;
                let status = resp.status().as_u16();

                // 200-299: success
                if resp.status().is_success() {
                    let Some(admission) = admission.take() else {
                        return Err(ClaudeError::Api(
                            "inference admission lease unavailable".into(),
                        ));
                    };
                    return Ok((
                        resp,
                        AuthorizedInferenceLease::new(admission, authorization),
                    ));
                }

                // 429 (rate limit) or 529 (overloaded): retry
                if (status == 429 || status == 529) && attempts <= self.config.max_retries {
                    // Honour Retry-After header if present
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(Duration::from_secs);

                    let wait = retry_after.unwrap_or(delay);
                    warn!(
                        status,
                        attempt = attempts,
                        wait_secs = wait.as_secs(),
                        "Retryable API error, backing off"
                    );
                    // Fully consume the retry response before waiting. This
                    // keeps the response body lifetime inside the admission
                    // lease and lets the transport reuse the connection.
                    let _ = resp.text().await;
                    drop(authorization);
                    tokio::time::sleep(wait).await;
                    delay = (delay * 2).min(self.config.max_retry_delay);
                    continue;
                }

                // Non-retryable error – return immediately
                let text = resp.text().await.unwrap_or_default();
                drop(authorization);
                return Err(Self::parse_api_error(status, &text));
            }
        }

        /// Map a provider failure to a typed error without carrying response
        /// content into logs or user-facing output.
        pub(super) fn parse_api_error(status: u16, _body: &str) -> ClaudeError {
            match status {
                401 | 403 => {
                    ClaudeError::Auth("Provider rejected the configured credentials".into())
                }
                429 => ClaudeError::RateLimit,
                529 => ClaudeError::ApiStatus {
                    status,
                    message: "Provider overloaded".into(),
                },
                _ => ClaudeError::ApiStatus {
                    status,
                    message: "Provider request failed".into(),
                },
            }
        }

        // TODO(#228): this SSE loop + `frame_to_event` below are the decode half
        // of the **AnthropicMessages** wire protocol. They should be hoisted into
        // a sans-IO `protocol::anthropic_messages` decoder (mirroring
        // `protocol::openai_chat::OpenAiChatDecoder`) and shared with
        // `providers::anthropic::AnthropicProvider`, collapsing the two Anthropic
        // stacks. Remaining step / risk: this path decodes into the Anthropic-typed
        // `AnthropicStreamEvent` consumed by `StreamHandler`/`StreamAccumulator`
        // and the TUI, whereas the protocol decoders emit the provider-agnostic
        // `provider_types::StreamEvent`; unifying requires either a decoder generic
        // over its output event or migrating those consumers. Deferred to keep the
        // TUI and all tests green.
        /// Read an SSE byte stream, parse frames, and emit `AnthropicStreamEvent`s.
        async fn process_sse_stream(
            resp: wreq::Response,
            handler: Arc<dyn StreamHandler>,
            tx: mpsc::Sender<streaming::AnthropicStreamEvent>,
        ) -> Result<Option<streaming::AnthropicStreamEvent>, ClaudeError> {
            use sse_parser::SseLineParser;

            let mut parser = SseLineParser::new();
            // Shared byte-buffering decoder (#228): buffers raw bytes and only
            // decodes complete lines, so a multibyte codepoint split across a
            // network chunk boundary is never corrupted.
            let mut decoder = crate::SseByteDecoder::new();
            let mut byte_stream = resp.bytes_stream();

            loop {
                let chunk_result = tokio::select! {
                    _ = tx.closed() => return Ok(None),
                    chunk = byte_stream.next() => chunk,
                };
                let Some(chunk_result) = chunk_result else {
                    break;
                };
                let chunk = chunk_result
                    .map_err(|_| ClaudeError::Api("Provider stream read failed".into()))?;

                for line in decoder.push(&chunk) {
                    let line = line.trim_end_matches('\r');
                    if let Some(frame) = parser.feed_line(line) {
                        if let Some(event) = Self::frame_to_event(&frame.event, &frame.data) {
                            if tx.is_closed() {
                                return Ok(None);
                            }
                            let terminal = matches!(
                                &event,
                                streaming::AnthropicStreamEvent::MessageStop
                                    | streaming::AnthropicStreamEvent::Error { .. }
                            );
                            if terminal {
                                return Ok(Some(event));
                            }
                            handler.on_event(&event);
                            let send_result = tokio::select! {
                                _ = tx.closed() => return Ok(None),
                                result = tx.send(event) => result,
                            };
                            if send_result.is_err() {
                                return Ok(None);
                            }
                        }
                    }
                }
            }

            Ok(None)
        }

        /// Convert a parsed SSE frame into a typed `AnthropicStreamEvent`.
        fn frame_to_event(
            event_type: &Option<String>,
            data: &str,
        ) -> Option<streaming::AnthropicStreamEvent> {
            let event_name = event_type.as_deref().unwrap_or("");

            match event_name {
                "ping" => Some(streaming::AnthropicStreamEvent::Ping),

                "message_start" => {
                    let v: Value = serde_json::from_str(data).ok()?;
                    let msg = v.get("message")?;
                    let id = msg.get("id")?.as_str()?.to_string();
                    let model = msg.get("model")?.as_str()?.to_string();
                    let usage = msg
                        .get("usage")
                        .and_then(|u| serde_json::from_value::<UsageInfo>(u.clone()).ok())
                        .unwrap_or_default();

                    Some(streaming::AnthropicStreamEvent::MessageStart { id, model, usage })
                }

                "content_block_start" => {
                    let v: Value = serde_json::from_str(data).ok()?;
                    let index = v.get("index")?.as_u64()? as usize;
                    let block_value = v.get("content_block")?;
                    let content_block: ContentBlock =
                        serde_json::from_value(block_value.clone()).ok()?;
                    Some(streaming::AnthropicStreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    })
                }

                "content_block_delta" => {
                    let v: Value = serde_json::from_str(data).ok()?;
                    let index = v.get("index")?.as_u64()? as usize;
                    let delta_value = v.get("delta")?;
                    let delta: streaming::ContentDelta =
                        serde_json::from_value(delta_value.clone()).ok()?;
                    Some(streaming::AnthropicStreamEvent::ContentBlockDelta { index, delta })
                }

                "content_block_stop" => {
                    let v: Value = serde_json::from_str(data).ok()?;
                    let index = v.get("index")?.as_u64()? as usize;
                    Some(streaming::AnthropicStreamEvent::ContentBlockStop { index })
                }

                "message_delta" => {
                    let v: Value = serde_json::from_str(data).ok()?;
                    let delta = v.get("delta")?;
                    let stop_reason = delta
                        .get("stop_reason")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    let usage = v
                        .get("usage")
                        .and_then(|u| serde_json::from_value::<UsageInfo>(u.clone()).ok());
                    Some(streaming::AnthropicStreamEvent::MessageDelta { stop_reason, usage })
                }

                "message_stop" => Some(streaming::AnthropicStreamEvent::MessageStop),

                "error" => {
                    let v: Value = serde_json::from_str(data).ok()?;
                    let error = v.get("error")?;
                    let error_type = error
                        .get("type")
                        .and_then(|s| s.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let message = error
                        .get("message")
                        .and_then(|s| s.as_str())
                        .unwrap_or("Unknown error")
                        .to_string();
                    Some(streaming::AnthropicStreamEvent::Error {
                        error_type,
                        message,
                    })
                }

                _ => {
                    debug!(event = event_name, "Unhandled SSE event type");
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience builder for CreateMessageRequest
// ---------------------------------------------------------------------------

impl CreateMessageRequest {
    /// Create a minimal request builder.
    pub fn builder(model: impl Into<String>, max_tokens: u32) -> CreateMessageRequestBuilder {
        CreateMessageRequestBuilder {
            model: model.into(),
            max_tokens,
            messages: vec![],
            system: None,
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            cache_control: None,
            wire_budget: None,
            dispatch: None,
        }
    }
}

pub struct CreateMessageRequestBuilder {
    model: String,
    max_tokens: u32,
    messages: Vec<ApiMessage>,
    system: Option<SystemPrompt>,
    tools: Option<Vec<ApiToolDefinition>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    thinking: Option<ThinkingConfig>,
    cache_control: Option<CacheControl>,
    wire_budget: Option<provider_wire::WireBudget>,
    dispatch: Option<crate::DispatchRequest>,
}

impl CreateMessageRequestBuilder {
    pub fn messages(mut self, msgs: Vec<ApiMessage>) -> Self {
        self.messages = msgs;
        self
    }

    pub fn add_message(mut self, msg: ApiMessage) -> Self {
        self.messages.push(msg);
        self
    }

    pub fn system(mut self, s: SystemPrompt) -> Self {
        self.system = Some(s);
        self
    }

    pub fn system_text(mut self, text: impl Into<String>) -> Self {
        self.system = Some(SystemPrompt::Text(text.into()));
        self
    }

    pub fn tools(mut self, tools: Vec<ApiToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = Some(seqs);
        self
    }

    pub fn thinking(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    pub fn cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub fn wire_budget(mut self, budget: provider_wire::WireBudget) -> Self {
        self.wire_budget = Some(budget);
        self
    }

    pub fn dispatch(mut self, dispatch: crate::DispatchRequest) -> Self {
        self.dispatch = Some(dispatch);
        self
    }

    pub fn build(self) -> CreateMessageRequest {
        CreateMessageRequest {
            model: self.model,
            max_tokens: self.max_tokens,
            messages: self.messages,
            system: self.system,
            tools: self.tools,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences,
            stream: true,
            thinking: self.thinking,
            cache_control: self.cache_control,
            wire_budget: self.wire_budget,
            dispatch: self.dispatch,
        }
    }
}

// ---------------------------------------------------------------------------
// Accumulated message builder – reconstructs a full Message from stream events
// ---------------------------------------------------------------------------

/// Collects streaming events and produces a finished `Message` plus usage info.
pub struct StreamAccumulator {
    id: Option<String>,
    model: Option<String>,
    content_blocks: Vec<ContentBlock>,
    /// Partial accumulators keyed by block index.
    partials: std::collections::HashMap<usize, PartialBlock>,
    stop_reason: Option<String>,
    usage: UsageInfo,
}

#[derive(Debug)]
enum PartialBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
    },
    Thinking {
        thinking_buf: String,
        signature_buf: String,
    },
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self {
            id: None,
            model: None,
            content_blocks: vec![],
            partials: Default::default(),
            stop_reason: None,
            usage: UsageInfo::default(),
        }
    }

    /// Feed a stream event. Call this for every event received from the stream.
    pub fn on_event(&mut self, event: &streaming::AnthropicStreamEvent) {
        use streaming::AnthropicStreamEvent;
        match event {
            AnthropicStreamEvent::MessageStart { id, model, usage } => {
                self.id = Some(id.clone());
                self.model = Some(model.clone());
                self.usage = usage.clone();
            }

            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let partial = match content_block {
                    ContentBlock::Text { text } => PartialBlock::Text(text.clone()),
                    ContentBlock::ToolUse { id, name, .. } => PartialBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        json_buf: String::new(),
                    },
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                    } => PartialBlock::Thinking {
                        thinking_buf: thinking.clone(),
                        signature_buf: signature.clone(),
                    },
                    _ => return,
                };
                self.partials.insert(*index, partial);
            }

            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(partial) = self.partials.get_mut(index) {
                    match (partial, delta) {
                        (PartialBlock::Text(buf), streaming::ContentDelta::TextDelta { text }) => {
                            buf.push_str(text);
                        }
                        (
                            PartialBlock::ToolUse { json_buf, .. },
                            streaming::ContentDelta::InputJsonDelta { partial_json },
                        ) => {
                            json_buf.push_str(partial_json);
                        }
                        (
                            PartialBlock::Thinking { thinking_buf, .. },
                            streaming::ContentDelta::ThinkingDelta { thinking },
                        ) => {
                            thinking_buf.push_str(thinking);
                        }
                        (
                            PartialBlock::Thinking { signature_buf, .. },
                            streaming::ContentDelta::SignatureDelta { signature },
                        ) => {
                            signature_buf.push_str(signature);
                        }
                        _ => {}
                    }
                }
            }

            AnthropicStreamEvent::ContentBlockStop { index } => {
                if let Some(partial) = self.partials.remove(index) {
                    let block = match partial {
                        PartialBlock::Text(text) => ContentBlock::Text { text },
                        PartialBlock::ToolUse { id, name, json_buf } => {
                            let input = serde_json::from_str(&json_buf)
                                .unwrap_or(Value::Object(Default::default()));
                            ContentBlock::ToolUse {
                                id,
                                name,
                                input,
                                thought_signature: None,
                            }
                        }
                        PartialBlock::Thinking {
                            thinking_buf,
                            signature_buf,
                        } => ContentBlock::Thinking {
                            thinking: thinking_buf,
                            signature: signature_buf,
                        },
                    };
                    self.content_blocks.push(block);
                }
            }

            AnthropicStreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(sr) = stop_reason {
                    self.stop_reason = Some(sr.clone());
                }
                if let Some(u) = usage {
                    // The delta usage usually only has output_tokens;
                    // add them to the running total.
                    self.usage.output_tokens += u.output_tokens;
                }
            }

            AnthropicStreamEvent::MessageStop => {}
            AnthropicStreamEvent::Ping => {}
            AnthropicStreamEvent::Error { .. } => {}
        }
    }

    /// Finalize and produce the accumulated `Message`.
    pub fn finish(self) -> (Message, UsageInfo, Option<String>) {
        let msg = Message::assistant_blocks(self.content_blocks);
        (msg, self.usage, self.stop_reason)
    }

    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    pub fn usage(&self) -> &UsageInfo {
        &self.usage
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn oauth_clients_keep_coherent_account_and_identity_snapshots() {
        static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        struct EnvRestore(Option<std::ffi::OsString>);
        impl Drop for EnvRestore {
            fn drop(&mut self) {
                match self.0.take() {
                    Some(value) => std::env::set_var("CLAURST_HOME", value),
                    None => std::env::remove_var("CLAURST_HOME"),
                }
            }
        }
        struct TempRoot(std::path::PathBuf);
        impl Drop for TempRoot {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let _lock = ENV_LOCK.lock().await;
        let temp = TempRoot(
            std::env::temp_dir().join(format!("claurst-oauth-snapshot-{}", uuid::Uuid::new_v4())),
        );
        std::fs::create_dir_all(&temp.0).expect("temporary account root");
        let _restore = EnvRestore(std::env::var_os("CLAURST_HOME"));
        std::env::set_var("CLAURST_HOME", &temp.0);

        let tokens = |access_token: &str, account_uuid: &str, email: &str| {
            claurst_core::oauth::OAuthTokens {
                access_token: access_token.to_string(),
                scopes: vec![claurst_core::oauth::CLAUDE_AI_INFERENCE_SCOPE.to_string()],
                account_uuid: Some(account_uuid.to_string()),
                email: Some(email.to_string()),
                ..Default::default()
            }
        };

        tokens("first-token", "account-one", "one@example.com")
            .save_and_register(Some("one"))
            .await
            .expect("save first account");
        let first = client::AnthropicClient::new(client::ClientConfig {
            api_key: "first-token".into(),
            use_bearer_auth: true,
            ..Default::default()
        })
        .expect("first OAuth client");
        assert_eq!(first.captured_oauth_account_uuid(), Some("account-one"));

        tokens("second-token", "account-two", "two@example.com")
            .save_and_register(Some("two"))
            .await
            .expect("save second account");
        let second = client::AnthropicClient::new(client::ClientConfig {
            api_key: "second-token".into(),
            use_bearer_auth: true,
            ..Default::default()
        })
        .expect("second OAuth client");
        let stale_credential = client::AnthropicClient::new(client::ClientConfig {
            api_key: "first-token".into(),
            use_bearer_auth: true,
            ..Default::default()
        })
        .expect("stale-credential client");

        assert_eq!(first.captured_oauth_account_uuid(), Some("account-one"));
        assert_eq!(second.captured_oauth_account_uuid(), Some("account-two"));
        assert_eq!(stale_credential.captured_oauth_account_uuid(), None);

        let mut request = CreateMessageRequest::builder("fixture-model", 128)
            .messages(vec![ApiMessage {
                role: "user".into(),
                content: Value::String("hello".into()),
            }])
            .system(SystemPrompt::Blocks(vec![
                SystemBlock {
                    block_type: "text".into(),
                    text: "x-anthropic-billing-header: stale".into(),
                    cache_control: None,
                },
                SystemBlock {
                    block_type: "text".into(),
                    text: claurst_core::oauth_config::CLAUDE_CODE_SYSTEM_PROMPT_PREFIX.into(),
                    cache_control: None,
                },
                SystemBlock {
                    block_type: "text".into(),
                    text: "You are Claurst.\n\nKeep this instruction.".into(),
                    cache_control: None,
                },
            ]))
            .build();

        first.apply_oauth_stealth(&mut request);
        let once = serde_json::to_value(&request.system).expect("serialize identity blocks");
        first.apply_oauth_stealth(&mut request);
        assert_eq!(
            serde_json::to_value(&request.system).expect("serialize identity blocks"),
            once,
            "OAuth identity mutation must be idempotent"
        );

        let Some(SystemPrompt::Blocks(blocks)) = request.system else {
            panic!("OAuth identity must use structured blocks");
        };
        assert!(blocks[0].text.starts_with("x-anthropic-billing-header:"));
        assert_eq!(
            blocks[1].text,
            claurst_core::oauth_config::CLAUDE_CODE_SYSTEM_PROMPT_PREFIX
        );
        assert_eq!(blocks[2].text, "Keep this instruction.");
        assert_eq!(
            blocks
                .iter()
                .filter(|block| block.text.starts_with("x-anthropic-billing-header:"))
                .count(),
            1
        );
        assert_eq!(
            blocks
                .iter()
                .filter(|block| {
                    block.text == claurst_core::oauth_config::CLAUDE_CODE_SYSTEM_PROMPT_PREFIX
                })
                .count(),
            1
        );
    }

    #[test]
    fn legacy_provider_errors_exclude_response_content() {
        const PRIVATE: &str = "PRIVATE_PROVIDER_RESPONSE_AND_URL_CANARY";
        for (status, expected) in [
            (400, "Provider request failed"),
            (401, "Provider rejected the configured credentials"),
            (403, "Provider rejected the configured credentials"),
            (429, "Rate limit exceeded"),
            (529, "Provider overloaded"),
        ] {
            let body = format!(r#"{{"error":{{"type":"fixture","message":"{PRIVATE}"}}}}"#);
            let presented = client::AnthropicClient::parse_api_error(status, &body).to_string();
            assert!(presented.contains(expected));
            assert!(!presented.contains(PRIVATE));
        }
    }

    #[test]
    fn test_sse_parser_basic() {
        let mut parser = sse_parser::SseLineParser::new();
        assert!(parser.feed_line("event: message_start").is_none());
        assert!(parser
            .feed_line(r#"data: {"message":{"id":"m1","model":"claude","usage":{"input_tokens":0,"output_tokens":0}}}"#)
            .is_none());
        let frame = parser.feed_line("").expect("should produce frame");
        assert_eq!(frame.event.as_deref(), Some("message_start"));
        assert!(frame.data.contains("m1"));
    }

    #[test]
    fn test_create_message_request_builder() {
        let req = CreateMessageRequest::builder("claude-opus-4-6", 4096)
            .system_text("You are helpful.")
            .temperature(0.7)
            .build();
        assert_eq!(req.model, "claude-opus-4-6");
        assert_eq!(req.max_tokens, 4096);
        assert!(req.stream);
    }

    #[test]
    fn test_stream_accumulator_text() {
        let mut acc = StreamAccumulator::new();
        acc.on_event(&streaming::AnthropicStreamEvent::MessageStart {
            id: "m1".into(),
            model: "claude".into(),
            usage: UsageInfo::default(),
        });
        acc.on_event(&streaming::AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        });
        acc.on_event(&streaming::AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: streaming::ContentDelta::TextDelta {
                text: "Hello ".into(),
            },
        });
        acc.on_event(&streaming::AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: streaming::ContentDelta::TextDelta {
                text: "world!".into(),
            },
        });
        acc.on_event(&streaming::AnthropicStreamEvent::ContentBlockStop { index: 0 });
        acc.on_event(&streaming::AnthropicStreamEvent::MessageDelta {
            stop_reason: Some("end_turn".into()),
            usage: None,
        });
        acc.on_event(&streaming::AnthropicStreamEvent::MessageStop);

        let (msg, _usage, stop) = acc.finish();
        assert_eq!(msg.get_text(), Some("Hello world!"));
        assert_eq!(stop.as_deref(), Some("end_turn"));
    }

    // -----------------------------------------------------------------------
    // Request timeout (#175) + stream idle timeout (#185)
    // -----------------------------------------------------------------------

    #[test]
    fn request_timeout_threads_through_to_client_config() {
        // Reset to the default and confirm the generous 600s fallback.
        set_request_timeout_secs(0);
        assert_eq!(request_timeout_secs(), DEFAULT_REQUEST_TIMEOUT_SECS);
        assert_eq!(request_timeout(), Duration::from_secs(600));

        // An override threads through request_timeout() and ClientConfig::default.
        set_request_timeout_secs(1800);
        assert_eq!(request_timeout_secs(), 1800);
        assert_eq!(
            client::ClientConfig::default().request_timeout,
            Duration::from_secs(1800)
        );
        // Idle timeout is generous and never smaller than the request timeout.
        assert!(stream_idle_timeout() >= request_timeout());
        assert_eq!(stream_idle_timeout(), Duration::from_secs(1800));

        // A short request timeout still keeps a generous idle floor (#185:
        // bound stalls without cutting off slow-but-progressing streams).
        set_request_timeout_secs(60);
        assert_eq!(
            stream_idle_timeout(),
            Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)
        );

        // Restore the default so we do not leak state into other tests.
        set_request_timeout_secs(0);
    }

    #[tokio::test]
    async fn stalled_stream_elapses_instead_of_hanging() {
        use futures::StreamExt;

        // A byte stream that never yields models a provider that begins a
        // streamed response and then pauses indefinitely (issue #185).
        let mut stalled = futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>();

        // The exact construct used by the streaming loops: wrapping the chunk
        // read in tokio::time::timeout must elapse rather than hang forever.
        // A short duration keeps the test fast; production uses the generous
        // stream_idle_timeout() value asserted below.
        let result = tokio::time::timeout(Duration::from_millis(50), stalled.next()).await;
        assert!(
            result.is_err(),
            "a stalled stream must hit the idle timeout, not hang"
        );

        // Invariants that hold for any configured value: the production idle
        // timeout is finite, never below the request timeout, and never below
        // the generous default floor.
        assert!(stream_idle_timeout() >= request_timeout());
        assert!(stream_idle_timeout() >= Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS));
    }
}
