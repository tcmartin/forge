// providers/codex.rs — OpenAI Codex provider (OAuth-authenticated).
//
// Codex uses OpenAI's Responses API at:
//   https://chatgpt.com/backend-api/codex/responses
//
// Auth: Bearer token obtained via the Codex OAuth flow stored in
//   ~/.claurst/codex_tokens.json (`CodexTokens` struct).
//
// Token refresh: if `expires_at` is in the past we POST to the OpenAI token
//   endpoint with `grant_type=refresh_token` before making the request.
//
// Model list: static — the Codex endpoint does not expose a /models route,
//   so we use the `CODEX_MODELS` constant from `claurst-core`.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_stream::stream;
use async_trait::async_trait;
use claurst_core::codex_oauth::{
    CODEX_API_ENDPOINT, CODEX_MODELS, CODEX_TOKEN_URL, DEFAULT_CODEX_MODEL,
};
use claurst_core::oauth_config::{get_codex_tokens, save_codex_tokens, CodexTokens};
use claurst_core::provider_id::{ModelId, ProviderId};
use claurst_core::types::UsageInfo;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::error_handling::parse_error_response;
use crate::inference_admission::{AdmissionLease, InferenceAdmission};
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus,
    ResponseContinuationSupport, StopReason, StreamEvent, SystemPromptStyle,
};
use crate::provider_wire::prepare_provider_json;
use crate::provider_witness::{
    NativeResponseBinding, PreparedWitnessAttempt, ProviderWireWitnessSeed, WireAttemptOutcome,
};

// Re-use Copilot's message translation helpers via the public Copilot type.
use crate::providers::copilot::CopilotProvider;

const MAX_CODEX_SSE_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_CODEX_SSE_LINE_BYTES: usize = 256 * 1024;
const MAX_CODEX_SSE_EVENT_BYTES: usize = 1024 * 1024;
const MAX_CODEX_TOOL_ARGUMENT_BYTES: usize = 256 * 1024;
const MAX_CODEX_TOOL_CALLS: usize = 256;
const MAX_CODEX_OUTPUT_ITEMS: usize = 1024;

// ---------------------------------------------------------------------------
// CodexProvider
// ---------------------------------------------------------------------------

pub struct CodexProvider {
    id: ProviderId,
    http_client: reqwest::Client,
    api_endpoint: String,
    /// Mutable token cache: updated in-place when a refresh succeeds.
    tokens: Arc<Mutex<CodexTokens>>,
    inference_admission: Arc<InferenceAdmission>,
    response_continuation_enabled: bool,
    response_continuation_rejected: AtomicBool,
}

impl CodexProvider {
    pub fn new(tokens: CodexTokens) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(crate::request_timeout())
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build reqwest client");

        Self {
            id: ProviderId::new(ProviderId::CODEX),
            http_client,
            api_endpoint: CODEX_API_ENDPOINT.to_string(),
            tokens: Arc::new(Mutex::new(tokens)),
            inference_admission: InferenceAdmission::process(),
            response_continuation_enabled: false,
            response_continuation_rejected: AtomicBool::new(false),
        }
    }

    /// Override the Responses endpoint while retaining the Codex wire shape.
    /// This is used by bounded loopback fixtures and private compatible hosts.
    pub fn with_base_url(mut self, api_endpoint: impl Into<String>) -> Self {
        self.api_endpoint = api_endpoint.into();
        self
    }

    /// Enable response-ID continuation for an explicitly compatible Responses
    /// route. The stock Codex HTTP/SSE route remains disabled because its
    /// `store: false` contract does not provide WebSocket-scoped state.
    pub fn with_response_continuation(mut self) -> Self {
        self.response_continuation_enabled = true;
        self.response_continuation_rejected = AtomicBool::new(false);
        self
    }

    /// Inject a process admission service, primarily for bounded tests.
    pub fn with_inference_admission(mut self, admission: Arc<InferenceAdmission>) -> Self {
        self.inference_admission = admission;
        self
    }

    async fn acquire_inference(&self) -> Result<AdmissionLease, ProviderError> {
        self.inference_admission
            .acquire()
            .await
            .map_err(|error| ProviderError::AdmissionRejected {
                provider: self.id.clone(),
                error,
            })
    }

    /// Construct from stored tokens; returns `None` if no tokens are saved.
    pub fn from_stored() -> Option<Self> {
        let tokens = get_codex_tokens()?;
        if tokens.access_token.is_empty() {
            return None;
        }
        Some(Self::new(tokens))
    }

    // -----------------------------------------------------------------------
    // Token management
    // -----------------------------------------------------------------------

    fn is_expired(tokens: &CodexTokens) -> bool {
        let Some(expires_at) = tokens.expires_at else {
            return false; // No expiry info — assume still valid.
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Treat as expired 60 s early to avoid races.
        now + 60 >= expires_at
    }

    /// Return the current access token, refreshing first if it is expired.
    async fn access_token(&self) -> Result<String, ProviderError> {
        // Check expiry under the lock; clone what we need; release.
        let (token, needs_refresh, refresh_token) = {
            let guard = self.tokens.lock().unwrap();
            let expired = Self::is_expired(&guard);
            (
                guard.access_token.clone(),
                expired,
                guard.refresh_token.clone(),
            )
        };

        if !needs_refresh {
            return Ok(token);
        }

        let Some(refresh) = refresh_token else {
            // No refresh token — return what we have and hope for the best.
            warn!("Codex access token is expired and no refresh token is available");
            return Ok(token);
        };

        debug!("Codex access token expired — refreshing");
        self.refresh_token(&refresh).await
    }

    async fn refresh_token(&self, refresh_token: &str) -> Result<String, ProviderError> {
        let body = json!({
            "grant_type": "refresh_token",
            "client_id": claurst_core::codex_oauth::CODEX_CLIENT_ID,
            "refresh_token": refresh_token,
        });

        let resp = self
            .http_client
            .post(CODEX_TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Token refresh request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to read token refresh response: {}", e),
            status: Some(status),
            body: None,
        })?;

        if !(200..300).contains(&(status as usize)) {
            return Err(ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Token refresh failed (HTTP {})", status),
                status: Some(status),
                body: Some(text),
            });
        }

        let json_val: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to parse token refresh response: {}", e),
            status: Some(status),
            body: Some(text.clone()),
        })?;

        let new_access = json_val
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if new_access.is_empty() {
            return Err(ProviderError::Other {
                provider: self.id.clone(),
                message: "Token refresh response missing access_token".to_string(),
                status: Some(status),
                body: Some(text),
            });
        }

        let new_refresh = json_val
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let expires_in = json_val.get("expires_in").and_then(|v| v.as_u64());

        let new_expires_at = expires_in.map(|secs| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
                + secs
        });

        // Persist and cache the refreshed tokens.
        let mut updated = {
            let guard = self.tokens.lock().unwrap();
            guard.clone()
        };
        updated.access_token = new_access.clone();
        if let Some(r) = new_refresh {
            updated.refresh_token = Some(r);
        }
        updated.expires_at = new_expires_at;

        if save_codex_tokens(&updated).is_err() {
            warn!("Failed to persist refreshed Codex tokens");
        }

        {
            let mut guard = self.tokens.lock().unwrap();
            *guard = updated;
        }

        Ok(new_access)
    }

    // -----------------------------------------------------------------------
    // Request helpers
    // -----------------------------------------------------------------------

    fn codex_headers(
        &self,
        builder: reqwest::RequestBuilder,
        token: &str,
        account_id: Option<&str>,
    ) -> reqwest::RequestBuilder {
        let builder = builder
            .bearer_auth(token)
            .header("User-Agent", concat!("claurst/", env!("CARGO_PKG_VERSION")));

        if let Some(id) = account_id {
            builder.header("ChatGPT-Account-Id", id)
        } else {
            builder
        }
    }

    fn account_id(&self) -> Option<String> {
        self.tokens.lock().unwrap().account_id.clone()
    }

    fn system_prompt_to_text(request: &ProviderRequest) -> String {
        match request.system_prompt.as_ref() {
            Some(crate::provider_types::SystemPrompt::Text(text)) => text.clone(),
            Some(crate::provider_types::SystemPrompt::Blocks(blocks)) => blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        }
    }

    /// Build the Responses-API request body for Codex.
    fn build_responses_body(request: &ProviderRequest, allow_response_continuation: bool) -> Value {
        let response_continuation = allow_response_continuation
            .then(|| request.validated_response_continuation())
            .flatten();
        let messages = response_continuation.map_or(request.messages.as_slice(), |continuation| {
            &request.messages[continuation.message_prefix_len()..]
        });
        // Codex carries system text in top-level `instructions`; omit it from
        // Responses input to keep one stable prefix and avoid duplicate policy.
        let input = CopilotProvider::to_responses_messages_pub(messages);
        let instructions = Self::system_prompt_to_text(request);

        let tools: Vec<Value> = request
            .stable_tools()
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": false,
                })
            })
            .collect();

        let mut body = json!({
            "model": request.model,
            "input": input,
            "instructions": instructions,
            "max_output_tokens": request.max_tokens,
            "store": false,
        });

        if let Some(scope) = request.cache_scope.as_ref() {
            body["prompt_cache_key"] = json!(scope.key());
        }
        if let Some(continuation) = response_continuation {
            body["previous_response_id"] = json!(continuation.previous_response_id());
        }

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }

        // Apply reasoning effort / summary / verbosity exactly like the Copilot
        // Responses path. The query layer already populates `reasoningEffort`
        // (default "medium"), `reasoningSummary`, and `include` for gpt-5 Codex
        // models — without this they were silently dropped and every request ran
        // at the server default. Mirrors opencode's gpt-5 reasoning defaults.
        CopilotProvider::apply_responses_provider_options_pub(&mut body, &request.provider_options);

        body
    }

    fn continuation_rejected(status: u16, body: &str) -> bool {
        matches!(status, 400 | 404) && {
            let lower = body.to_ascii_lowercase();
            lower.contains("previous_response_id")
                || lower.contains("previous response")
                || lower.contains("previous_response_not_found")
        }
    }

    fn can_continue_response(&self) -> bool {
        self.response_continuation_enabled
            && !self.response_continuation_rejected.load(Ordering::Acquire)
    }

    fn stream_error(message: impl Into<String>) -> ProviderError {
        ProviderError::StreamError {
            provider: ProviderId::new(ProviderId::CODEX),
            message: message.into(),
            partial_response: None,
        }
    }

    fn required_output_index(value: &Value, event: &str) -> Result<usize, String> {
        value
            .get("output_index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| format!("Codex {event} lacks output_index"))
    }

    fn valid_function_arguments(arguments: &str) -> bool {
        serde_json::from_str::<Value>(arguments)
            .map(|value| value.is_object())
            .unwrap_or(false)
    }

    fn register_output_item_kind(
        output_item_kinds: &mut std::collections::HashMap<usize, &'static str>,
        output_index: usize,
        kind: &'static str,
    ) -> Result<(), String> {
        if let Some(existing) = output_item_kinds.get(&output_index) {
            if *existing != kind {
                return Err(format!(
                    "Codex output index changed kind from {existing} to {kind}"
                ));
            }
            return Ok(());
        }
        if output_item_kinds.len() >= MAX_CODEX_OUTPUT_ITEMS {
            return Err("Codex output item count exceeded the bound".to_string());
        }
        output_item_kinds.insert(output_index, kind);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // HTTP call
    // -----------------------------------------------------------------------

    async fn post_responses_body(
        &self,
        token: &str,
        account_id: Option<&str>,
        accept: &str,
        body: Vec<u8>,
    ) -> Result<reqwest::Response, ProviderError> {
        let builder = self
            .http_client
            .post(&self.api_endpoint)
            .header("Content-Type", "application/json")
            .header("Accept", accept);
        self.codex_headers(builder, token, account_id)
            .body(body)
            .send()
            .await
            .map_err(|error| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {error}"),
                status: None,
                body: None,
            })
    }

    async fn send_responses_request(
        &self,
        request: &ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let body = Self::build_responses_body(request, self.can_continue_response());
        let used_continuation = body.get("previous_response_id").is_some();
        // Run all local validation before checking or refreshing OAuth. A
        // rejected request must not cause remote token traffic.
        let prepared = prepare_provider_json("codex", &self.api_endpoint, false, &body, request)
            .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
        let full_context = if used_continuation {
            let full_body = Self::build_responses_body(request, false);
            Some(
                prepare_provider_json("codex", &self.api_endpoint, false, &full_body, request)
                    .map_err(|error| ProviderError::budget(self.id.clone(), error))?,
            )
        } else {
            None
        };
        let token = self.access_token().await?;
        let account_id = self.account_id();
        let _admission = self.acquire_inference().await?;

        let mut resp = self
            .post_responses_body(
                &token,
                account_id.as_deref(),
                "application/json",
                prepared.into_body(),
            )
            .await?;
        let mut status = resp.status().as_u16();
        let mut text = resp.text().await.map_err(|error| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to read response body: {error}"),
            status: Some(status),
            body: None,
        })?;

        if used_continuation
            && !(200..300).contains(&(status as usize))
            && Self::continuation_rejected(status, &text)
        {
            self.response_continuation_rejected
                .store(true, Ordering::Release);
            debug!(
                status,
                "Codex route rejected response continuation; retrying full context once"
            );
            let Some(full_context) = full_context else {
                return Err(parse_error_response(status, &text, &self.id));
            };
            resp = self
                .post_responses_body(
                    &token,
                    account_id.as_deref(),
                    "application/json",
                    full_context.into_body(),
                )
                .await?;
            status = resp.status().as_u16();
            text = resp.text().await.map_err(|error| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Failed to read response body: {error}"),
                status: Some(status),
                body: None,
            })?;
        }

        if !(200..300).contains(&(status as usize)) {
            return Err(parse_error_response(status, &text, &self.id));
        }

        let json_val: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to parse response JSON: {}", e),
            status: Some(status),
            body: Some(text.clone()),
        })?;

        Self::parse_responses_response(&self.id, &json_val)
    }

    async fn send_responses_streaming_request(
        &self,
        request: &ProviderRequest,
    ) -> Result<
        (
            reqwest::Response,
            AdmissionLease,
            Option<ProviderWireWitnessSeed>,
        ),
        ProviderError,
    > {
        let mut body = Self::build_responses_body(request, self.can_continue_response());
        let used_continuation = body.get("previous_response_id").is_some();
        body["stream"] = json!(true);
        let prepared = prepare_provider_json("codex", &self.api_endpoint, true, &body, request)
            .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
        let initial_witness =
            PreparedWitnessAttempt::from_prepared(&prepared, 0, used_continuation);
        let full_context = if used_continuation {
            let mut full_body = Self::build_responses_body(request, false);
            full_body["stream"] = json!(true);
            let prepared =
                prepare_provider_json("codex", &self.api_endpoint, true, &full_body, request)
                    .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
            Some((
                PreparedWitnessAttempt::from_prepared(&prepared, 1, false),
                prepared,
            ))
        } else {
            None
        };
        let token = self.access_token().await?;
        let account_id = self.account_id();
        let admission = self.acquire_inference().await?;

        let mut attempts = Vec::with_capacity(2);
        let mut resp = self
            .post_responses_body(
                &token,
                account_id.as_deref(),
                "text/event-stream",
                prepared.into_body(),
            )
            .await?;
        let mut status = resp.status().as_u16();
        if used_continuation && !(200..300).contains(&(status as usize)) {
            let text = resp.text().await.map_err(|error| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Failed to read response body: {error}"),
                status: Some(status),
                body: None,
            })?;
            if Self::continuation_rejected(status, &text) {
                attempts
                    .push(initial_witness.finish(status, WireAttemptOutcome::RejectedContinuation));
                self.response_continuation_rejected
                    .store(true, Ordering::Release);
                debug!(
                    status,
                    "Codex route rejected response continuation; retrying full context once"
                );
                let Some((full_witness, full_context)) = full_context else {
                    return Err(parse_error_response(status, &text, &self.id));
                };
                resp = self
                    .post_responses_body(
                        &token,
                        account_id.as_deref(),
                        "text/event-stream",
                        full_context.into_body(),
                    )
                    .await?;
                status = resp.status().as_u16();
                attempts.push(full_witness.finish(
                    status,
                    if (200..300).contains(&(status as usize)) {
                        WireAttemptOutcome::Accepted
                    } else {
                        WireAttemptOutcome::Failed
                    },
                ));
            } else {
                attempts.push(initial_witness.finish(status, WireAttemptOutcome::Failed));
                return Err(parse_error_response(status, &text, &self.id));
            }
        } else {
            attempts.push(initial_witness.finish(
                status,
                if (200..300).contains(&(status as usize)) {
                    WireAttemptOutcome::Accepted
                } else {
                    WireAttemptOutcome::Failed
                },
            ));
        }

        if !(200..300).contains(&(status as usize)) {
            let text = resp.text().await.map_err(|error| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Failed to read response body: {error}"),
                status: Some(status),
                body: None,
            })?;
            return Err(parse_error_response(status, &text, &self.id));
        }

        let witness_seed = ProviderWireWitnessSeed::try_new(attempts).ok();
        Ok((resp, admission, witness_seed))
    }

    // -----------------------------------------------------------------------
    // Response parsing  (mirrors CopilotProvider::parse_responses_response)
    // -----------------------------------------------------------------------

    fn parse_responses_response(
        provider_id: &ProviderId,
        json_val: &Value,
    ) -> Result<ProviderResponse, ProviderError> {
        use claurst_core::types::ContentBlock;

        let id = json_val
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let model = json_val
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_CODEX_MODEL)
            .to_string();

        let output = json_val
            .get("output")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ProviderError::Other {
                provider: provider_id.clone(),
                message: "No output in Codex Responses API response".to_string(),
                status: None,
                body: Some(json_val.to_string()),
            })?;

        let mut content: Vec<ContentBlock> = Vec::new();
        let mut has_tool_call = false;

        for item in output {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            match part.get("type").and_then(|v| v.as_str()) {
                                Some("output_text") | Some("text") => {
                                    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                        if !text.is_empty() {
                                            content.push(ContentBlock::Text {
                                                text: text.to_string(),
                                            });
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(summaries) = item.get("summary").and_then(|v| v.as_array()) {
                        let reasoning: String = summaries
                            .iter()
                            .filter_map(|s| s.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("");
                        if !reasoning.is_empty() {
                            content.push(ContentBlock::Thinking {
                                thinking: reasoning,
                                signature: String::new(),
                            });
                        }
                    }
                }
                Some("function_call") => {
                    has_tool_call = true;
                    let id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let input = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                    content.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input,
                        thought_signature: None,
                    });
                }
                _ => {}
            }
        }

        let stop_reason = if has_tool_call {
            StopReason::ToolUse
        } else {
            match json_val
                .get("incomplete_details")
                .and_then(|v| v.get("reason"))
                .and_then(|v| v.as_str())
            {
                Some("max_output_tokens") => StopReason::MaxTokens,
                Some("content_filter") => StopReason::ContentFiltered,
                Some(other) if !other.is_empty() => StopReason::Other(other.to_string()),
                _ => StopReason::EndTurn,
            }
        };

        let usage = {
            let u = json_val.get("usage");
            UsageInfo {
                input_tokens: u
                    .and_then(|v| v.get("input_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                output_tokens: u
                    .and_then(|v| v.get("output_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }
        };

        Ok(ProviderResponse {
            id,
            content,
            stop_reason,
            usage,
            model,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for CodexProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "OpenAI Codex"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        crate::dispatch_request::reject_unsupported_dispatch(&self.id, request.dispatch.as_ref())?;
        self.send_responses_request(&request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        crate::dispatch_request::reject_unsupported_dispatch(&self.id, request.dispatch.as_ref())?;
        let (resp, admission, witness_seed) =
            self.send_responses_streaming_request(&request).await?;
        let provider_id = self.id.clone();

        let s = stream! {
                    let mut byte_stream = resp.bytes_stream();
                    let mut decoder = crate::SseByteDecoder::new();
                    let mut current_event = String::new();
                    let mut current_data: Vec<String> = Vec::new();
                    let mut current_data_bytes = 0_usize;
                    let mut message_started = false;
                    let mut message_id = String::from("unknown");
                    let mut model_name = String::from(DEFAULT_CODEX_MODEL);
                    let mut native_binding = NativeResponseBinding::default();
                    let mut saw_tool_call = false;
                    let mut open_blocks: std::collections::HashSet<usize> = std::collections::HashSet::new();
                    // Every output index has one immutable semantic kind.  Keeping
                    // text and function-call indices in the same map prevents a
                    // later event from reusing an index across kinds.
                    let mut output_item_kinds: std::collections::HashMap<usize, &'static str> =
                        std::collections::HashMap::new();
                    let mut tool_call_indices_by_call_id: std::collections::HashMap<String, usize> =
                        std::collections::HashMap::new();
                    // Responses can send arguments in deltas, in a completion-only
                    // `response.function_call_arguments.done`, or both. Keep one
                    // bounded buffer per output index and bind finalization to the
                    // function-call item's stable `id`.
                    let mut tool_call_item_ids: std::collections::HashMap<usize, String> =
                        std::collections::HashMap::new();
                    let mut tool_call_call_ids: std::collections::HashMap<usize, String> =
                        std::collections::HashMap::new();
                    let mut tool_call_names: std::collections::HashMap<usize, String> =
                        std::collections::HashMap::new();
                    let mut tool_call_indices_by_item_id: std::collections::HashMap<String, usize> =
                        std::collections::HashMap::new();
                    let mut tool_call_argument_buffers: std::collections::HashMap<usize, String> =
                        std::collections::HashMap::new();
                    let mut finalized_tool_calls: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    let mut finalized_tool_call_arguments: std::collections::HashMap<usize, String> =
                        std::collections::HashMap::new();
                    let mut completed_tool_call_items: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    let mut completed_output_items: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    let mut tool_argument_bytes = 0_usize;

                    while let Some(chunk_result) = byte_stream.next().await {
                        let chunk = match chunk_result {
                            Ok(chunk) => chunk,
                            Err(e) => {
                                yield Err(ProviderError::StreamError {
                                    provider: provider_id.clone(),
                                    message: format!("Stream read error: {}", e),
                                    partial_response: None,
                                });
                                return;
                            }
                        };

                        if chunk.len() > MAX_CODEX_SSE_CHUNK_BYTES {
                            yield Err(Self::stream_error("Codex stream chunk exceeded the 1 MiB limit"));
                            return;
                        }

                        for raw_line in decoder.push(&chunk) {
                            if raw_line.len() > MAX_CODEX_SSE_LINE_BYTES {
                                yield Err(Self::stream_error("Codex stream line exceeded the 256 KiB limit"));
                                return;
                            }
                            let line = raw_line.trim_end_matches('\r');

                            if line.is_empty() {
                                if current_data.is_empty() {
                                    current_event.clear();
                                    continue;
                                }

                                let data = current_data.join("\n");
                                current_data.clear();
                                current_data_bytes = 0;
                                let trimmed = data.trim();
                                if trimmed.is_empty() || trimmed == "[DONE]" {
                                    current_event.clear();
                                    continue;
                                }

                                let json_val: Value = match serde_json::from_str(trimmed) {
                                    Ok(value) => value,
                                    Err(_) => {
                                        yield Err(Self::stream_error("Codex stream contained malformed JSON"));
                                        return;
                                    }
                                };

                                let event_name = if current_event.is_empty() {
                                    json_val
                                        .get("type")
                                        .and_then(|value| value.as_str())
                                        .unwrap_or("")
                                        .to_string()
                                } else {
                                    current_event.clone()
                                };

                                match event_name.as_str() {
                                    "response.created" | "response.in_progress" => {
                                        native_binding.observe(json_val.get("response"), false);
                                        if let Some(response) = json_val.get("response") {
                                            if let Some(id) = response.get("id").and_then(|value| value.as_str()) {
                                                message_id = id.to_string();
                                            }
                                            if let Some(model) = response.get("model").and_then(|value| value.as_str()) {
                                                model_name = model.to_string();
                                            }
                                            if !message_started {
                                                yield Ok(StreamEvent::MessageStart {
                                                    id: message_id.clone(),
                                                    model: model_name.clone(),
                                                    usage: UsageInfo::default(),
                                                });
                                                message_started = true;
                                            }
                                        }
                                    }
                                    "response.output_item.added" => {
                                        let output_index = match Self::required_output_index(
                                            &json_val,
                                            "response.output_item.added",
                                        ) {
                                            Ok(index) => index,
                                            Err(message) => {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        };
                                        let Some(item) = json_val.get("item") else {
                                            yield Err(Self::stream_error(
                                                "Codex output item event lacks an item object",
                                            ));
                                            return;
                                        };
                                        let Some(item_type) = item.get("type").and_then(|value| value.as_str()) else {
                                            yield Err(Self::stream_error(
                                                "Codex output item event lacks a string type",
                                            ));
                                            return;
                                        };
                                        if output_item_kinds.contains_key(&output_index) {
                                            let message = if completed_output_items.contains(&output_index) {
                                                "Codex output item arrived after completion"
                                            } else {
                                                "Codex output index was reused across output items"
                                            };
                                            yield Err(Self::stream_error(message));
                                            return;
                                        }
                                        match item_type {
                                                "message" => {
                                                    if let Err(message) = Self::register_output_item_kind(
                                                        &mut output_item_kinds,
                                                        output_index,
                                                        "text",
                                                    ) {
                                                        yield Err(Self::stream_error(message));
                                                        return;
                                                    }
                                                    if let Some(id) = item.get("id").and_then(|value| value.as_str()) {
                                                        message_id = id.to_string();
                                                    }
                                                    if !message_started {
                                                        yield Ok(StreamEvent::MessageStart {
                                                            id: message_id.clone(),
                                                            model: model_name.clone(),
                                                            usage: UsageInfo::default(),
                                                        });
                                                        message_started = true;
                                                    }
                                                }
                                                "function_call" => {
                                                    saw_tool_call = true;
                                                    if tool_call_item_ids.len() >= MAX_CODEX_TOOL_CALLS {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call count exceeded the bound",
                                                        ));
                                                        return;
                                                    }
                                                    let Some(item_id) = item
                                                        .get("id")
                                                        .and_then(|value| value.as_str())
                                                        .filter(|value| !value.is_empty())
                                                    else {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item lacks a stable id",
                                                        ));
                                                        return;
                                                    };
                                                    if tool_call_item_ids.contains_key(&output_index) {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call output index was duplicated",
                                                        ));
                                                        return;
                                                    }
                                                    if tool_call_indices_by_item_id
                                                        .insert(item_id.to_string(), output_index)
                                                        .is_some()
                                                    {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item id was reused across output indices",
                                                        ));
                                                        return;
                                                    }
                                                    let call_id = item
                                                        .get("call_id")
                                                        .and_then(|value| value.as_str())
                                                        .filter(|value| !value.is_empty())
                                                        .map(str::to_string);
                                                    let Some(call_id) = call_id else {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item lacks a stable call_id",
                                                        ));
                                                        return;
                                                    };
                                                    if let Some(existing) = tool_call_indices_by_call_id
                                                        .insert(call_id.clone(), output_index)
                                                    {
                                                        if existing == output_index {
                                                            yield Err(Self::stream_error(
                                                                "Codex function_call call_id was duplicated",
                                                            ));
                                                            return;
                                                        }
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call call_id was reused across output indices",
                                                        ));
                                                        return;
                                                    }
                                                    let Some(name) = item
                                                        .get("name")
                                                        .and_then(|value| value.as_str())
                                                        .filter(|value| !value.is_empty())
                                                        .map(str::to_string)
                                                    else {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item lacks a stable name",
                                                        ));
                                                        return;
                                                    };
                                                    tool_call_item_ids.insert(output_index, item_id.to_string());
                                                    if let Err(message) = Self::register_output_item_kind(
                                                        &mut output_item_kinds,
                                                        output_index,
                                                        "function_call",
                                                    ) {
                                                        yield Err(Self::stream_error(message));
                                                        return;
                                                    }
                                                    tool_call_call_ids.insert(output_index, call_id.clone());
                                                    tool_call_names.insert(output_index, name.clone());
                                                    tool_call_argument_buffers.entry(output_index).or_default();
                                                    if open_blocks.insert(output_index) {
                                                        yield Ok(StreamEvent::ContentBlockStart {
                                                            index: output_index,
                                                            content_block: claurst_core::types::ContentBlock::ToolUse {
                                                                id: call_id,
                                                                name,
                                                                input: json!({}),
                                                                thought_signature: None,
                                                            },
                                                        });
                                                    }
                                                }
                                                _ => {
                                                    if let Err(message) = Self::register_output_item_kind(
                                                        &mut output_item_kinds,
                                                        output_index,
                                                        "other",
                                                    ) {
                                                        yield Err(Self::stream_error(message));
                                                        return;
                                                    }
                                                }
                                            }
                                    }
                                    "response.content_part.added" => {
                                        let output_index = match Self::required_output_index(
                                            &json_val,
                                            "response.content_part.added",
                                        ) {
                                            Ok(index) => index,
                                            Err(message) => {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        };
                                        let Some(part) = json_val.get("part") else {
                                            yield Err(Self::stream_error(
                                                "Codex content part event lacks a part object",
                                            ));
                                            return;
                                        };
                                        let Some(part_type) = part.get("type").and_then(|value| value.as_str()) else {
                                            yield Err(Self::stream_error(
                                                "Codex content part event lacks a string type",
                                            ));
                                            return;
                                        };
                                        if matches!(part_type, "output_text" | "text") {
                                            if completed_output_items.contains(&output_index) {
                                                yield Err(Self::stream_error(
                                                    "Codex content part arrived after output item completion",
                                                ));
                                                return;
                                            }
                                            if output_item_kinds.get(&output_index) == Some(&"function_call") {
                                                yield Err(Self::stream_error(
                                                    "Codex text content part reused a function_call output index",
                                                ));
                                                return;
                                            }
                                            if let Err(message) = Self::register_output_item_kind(
                                                &mut output_item_kinds,
                                                output_index,
                                                "text",
                                            ) {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                            if !message_started {
                                                    yield Ok(StreamEvent::MessageStart {
                                                        id: message_id.clone(),
                                                        model: model_name.clone(),
                                                        usage: UsageInfo::default(),
                                                    });
                                                    message_started = true;
                                                }
                                                if open_blocks.insert(output_index) {
                                                    yield Ok(StreamEvent::ContentBlockStart {
                                                        index: output_index,
                                                        content_block: claurst_core::types::ContentBlock::Text {
                                                            text: String::new(),
                                                        },
                                                    });
                                                }
                                        }
                                    }
                                    "response.output_text.delta" => {
                                        let output_index = match Self::required_output_index(
                                            &json_val,
                                            "response.output_text.delta",
                                        ) {
                                            Ok(index) => index,
                                            Err(message) => {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        };
                                        let Some(delta) = json_val.get("delta").and_then(|value| value.as_str()) else {
                                            yield Err(Self::stream_error(
                                                "Codex text delta event lacks a string delta",
                                            ));
                                            return;
                                        };
                                        if completed_output_items.contains(&output_index) {
                                            yield Err(Self::stream_error(
                                                "Codex text delta arrived after output item completion",
                                            ));
                                            return;
                                        }
                                        if output_item_kinds.get(&output_index) == Some(&"function_call") {
                                            yield Err(Self::stream_error(
                                                "Codex text delta reused a function_call output index",
                                            ));
                                            return;
                                        }
                                        if let Err(message) = Self::register_output_item_kind(
                                            &mut output_item_kinds,
                                            output_index,
                                            "text",
                                        ) {
                                            yield Err(Self::stream_error(message));
                                            return;
                                        }
                                        if !message_started {
                                            yield Ok(StreamEvent::MessageStart {
                                                id: message_id.clone(),
                                                model: model_name.clone(),
                                                usage: UsageInfo::default(),
                                            });
                                            message_started = true;
                                        }
                                        if open_blocks.insert(output_index) {
                                            yield Ok(StreamEvent::ContentBlockStart {
                                                index: output_index,
                                                content_block: claurst_core::types::ContentBlock::Text {
                                                    text: String::new(),
                                                },
                                            });
                                        }
                                        if !delta.is_empty() {
                                            yield Ok(StreamEvent::TextDelta {
                                                index: output_index,
                                                text: delta.to_string(),
                                            });
                                        }
                                    }
                                    "response.function_call_arguments.delta" => {
                                        let output_index = match Self::required_output_index(
                                            &json_val,
                                            "function_call_arguments.delta",
                                        ) {
                                            Ok(index) => index,
                                            Err(message) => {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        };
                                        if output_item_kinds.get(&output_index) != Some(&"function_call") {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta lacks a prior function_call item",
                                            ));
                                            return;
                                        }
                                        if completed_output_items.contains(&output_index) {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta arrived after output item completion",
                                            ));
                                            return;
                                        }
                                        let Some(item_id) = json_val
                                            .get("item_id")
                                            .and_then(|value| value.as_str())
                                            .filter(|value| !value.is_empty())
                                        else {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta lacks item_id",
                                            ));
                                            return;
                                        };
                                        let Some(expected_item_id) = tool_call_item_ids.get(&output_index) else {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta lacks a prior item",
                                            ));
                                            return;
                                        };
                                        if expected_item_id != item_id {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta changed item identity",
                                            ));
                                            return;
                                        }
                                        let Some(delta) = json_val.get("delta").and_then(|value| value.as_str()) else {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta lacks a string delta",
                                            ));
                                            return;
                                        };
                                        if finalized_tool_calls.contains(&output_index) {
                                            yield Err(Self::stream_error(
                                                "Codex function_call arguments arrived after finalization",
                                            ));
                                            return;
                                        }
                                        let Some(buffer) = tool_call_argument_buffers.get_mut(&output_index) else {
                                            yield Err(Self::stream_error(
                                                "Codex function_call argument delta lacks a prior item",
                                            ));
                                            return;
                                        };
                                        let Some(next_len) = buffer.len().checked_add(delta.len()) else {
                                            yield Err(Self::stream_error(
                                                "Codex function_call arguments exceeded the bound",
                                            ));
                                            return;
                                        };
                                        let Some(next_total) = tool_argument_bytes.checked_add(delta.len()) else {
                                            yield Err(Self::stream_error(
                                                "Codex function_call arguments exceeded the bound",
                                            ));
                                            return;
                                        };
                                        if next_len > MAX_CODEX_TOOL_ARGUMENT_BYTES
                                            || next_total > MAX_CODEX_TOOL_ARGUMENT_BYTES
                                        {
                                            yield Err(Self::stream_error(
                                                "Codex function_call arguments exceeded the bound",
                                            ));
                                            return;
                                        }
                                        buffer.push_str(delta);
                                        tool_argument_bytes = next_total;
                                        if !delta.is_empty() {
                                            yield Ok(StreamEvent::InputJsonDelta {
                                                index: output_index,
                                                partial_json: delta.to_string(),
                                            });
                                        }
                                    }
                                    "response.function_call_arguments.done" => {
                                        let output_index = match Self::required_output_index(
                                            &json_val,
                                            "function_call_arguments.done",
                                        ) {
                                            Ok(index) => index,
                                            Err(message) => {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        };
                                        let Some(item_id) = json_val.get("item_id").and_then(|value| value.as_str()) else {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments lack item_id",
                                            ));
                                            return;
                                        };
                                        let Some(expected_item_id) = tool_call_item_ids.get(&output_index) else {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments lack a prior item",
                                            ));
                                            return;
                                        };
                                        if expected_item_id != item_id {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments changed item identity",
                                            ));
                                            return;
                                        }
                                        let Some(arguments) = json_val.get("arguments").and_then(|value| value.as_str()) else {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments lack arguments",
                                            ));
                                            return;
                                        };
                                        if !Self::valid_function_arguments(arguments) {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments were not a JSON object",
                                            ));
                                            return;
                                        }
                                        if !finalized_tool_calls.insert(output_index) {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments were duplicated",
                                            ));
                                            return;
                                        }
                                        let Some(buffer) = tool_call_argument_buffers.get(&output_index) else {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments lack a buffer",
                                            ));
                                            return;
                                        };
                                        if !buffer.is_empty() && buffer != arguments {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments disagree with deltas",
                                            ));
                                            return;
                                        }
                                        if buffer.is_empty()
                                            && (arguments.len() > MAX_CODEX_TOOL_ARGUMENT_BYTES
                                                || tool_argument_bytes
                                                    .checked_add(arguments.len())
                                                    .is_none_or(|total| total > MAX_CODEX_TOOL_ARGUMENT_BYTES))
                                        {
                                            yield Err(Self::stream_error(
                                                "Codex finalized function_call arguments exceeded the bound",
                                            ));
                                            return;
                                        }
                                        if buffer.is_empty() && !arguments.is_empty() {
                                            tool_call_argument_buffers.insert(output_index, arguments.to_string());
                                            tool_argument_bytes += arguments.len();
                                            yield Ok(StreamEvent::InputJsonDelta {
                                                index: output_index,
                                                partial_json: arguments.to_string(),
                                            });
                                        }
                                        finalized_tool_call_arguments.insert(output_index, arguments.to_string());
                                    }
                                    "response.output_item.done" => {
                                        let output_index = match Self::required_output_index(
                                            &json_val,
                                            "output_item.done",
                                        ) {
                                            Ok(index) => index,
                                            Err(message) => {
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        };
                                        if completed_output_items.contains(&output_index) {
                                            let message = if output_item_kinds.get(&output_index) == Some(&"function_call") {
                                                "Codex function_call item completion was duplicated"
                                            } else {
                                                "Codex output item completion was duplicated"
                                            };
                                            yield Err(Self::stream_error(message));
                                            return;
                                        }
                                        let item_type = match json_val.get("item") {
                                            None => None,
                                            Some(item) => {
                                                let Some(item_object) = item.as_object() else {
                                                    yield Err(Self::stream_error(
                                                        "Codex output item completion item is not an object",
                                                    ));
                                                    return;
                                                };
                                                let Some(item_type) = item_object.get("type").and_then(|value| value.as_str()) else {
                                                    yield Err(Self::stream_error(
                                                        "Codex output item completion lacks a string type",
                                                    ));
                                                    return;
                                                };
                                                Some(item_type)
                                            }
                                        };
                                        match output_item_kinds.get(&output_index).copied() {
                                            Some("text") => {
                                                if item_type == Some("function_call") {
                                                    yield Err(Self::stream_error(
                                                        "Codex function_call item completion changed output index type",
                                                    ));
                                                    return;
                                                }
                                            }
                                            Some("function_call") => {
        if !completed_tool_call_items.insert(output_index) {
                                            yield Err(Self::stream_error(
                                                "Codex function_call item completion was duplicated",
                                            ));
                                            return;
                                        }
                                        if let Some(expected_item_id) = tool_call_item_ids.get(&output_index) {
                                            if let Some(item) = json_val.get("item") {
                                                if item.get("type").and_then(|value| value.as_str()) != Some("function_call") {
                                                    yield Err(Self::stream_error(
                                                        "Codex function_call item completion changed item type",
                                                    ));
                                                    return;
                                                }
                                                let Some(item_id) = item.get("id").and_then(|value| value.as_str()) else {
                                                    yield Err(Self::stream_error(
                                                        "Codex function_call item completion lacks a stable id",
                                                    ));
                                                    return;
                                                };
                                                if item_id != expected_item_id {
                                                    yield Err(Self::stream_error(
                                                        "Codex function_call item completion changed item identity",
                                                    ));
                                                    return;
                                                }
                                                if let Some(expected_call_id) = tool_call_call_ids.get(&output_index) {
                                                    if item.get("call_id").and_then(|value| value.as_str()) != Some(expected_call_id) {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item completion changed call identity",
                                                        ));
                                                        return;
                                                    }
                                                }
                                                if let Some(expected_name) = tool_call_names.get(&output_index) {
                                                    if item.get("name").and_then(|value| value.as_str()) != Some(expected_name) {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item completion changed tool name",
                                                        ));
                                                        return;
                                                    }
                                                }
                                                if let Some(arguments) = item.get("arguments").and_then(|value| value.as_str()) {
                                                    if !Self::valid_function_arguments(arguments) {
                                                        yield Err(Self::stream_error(
                                                            "Codex function_call item arguments were not a JSON object",
                                                        ));
                                                        return;
                                                    }
                                                    if finalized_tool_calls.contains(&output_index) {
                                                        if finalized_tool_call_arguments.get(&output_index).map(String::as_str) != Some(arguments) {
                                                            yield Err(Self::stream_error(
                                                                "Codex function_call item completion changed finalized arguments",
                                                            ));
                                                            return;
                                                        }
                                                    } else {
                                                        let Some(buffer) = tool_call_argument_buffers.get(&output_index) else {
                                                            yield Err(Self::stream_error(
                                                                "Codex function_call item completion lacks a buffer",
                                                            ));
                                                            return;
                                                        };
                                                        if !buffer.is_empty() && buffer != arguments {
                                                            yield Err(Self::stream_error(
                                                                "Codex function_call item completion disagrees with deltas",
                                                            ));
                                                            return;
                                                        }
                                                        if buffer.is_empty()
                                                            && (arguments.len() > MAX_CODEX_TOOL_ARGUMENT_BYTES
                                                                || tool_argument_bytes
                                                                    .checked_add(arguments.len())
                                                                    .is_none_or(|total| total > MAX_CODEX_TOOL_ARGUMENT_BYTES))
                                                        {
                                                            yield Err(Self::stream_error(
                                                                "Codex function_call item completion exceeded the bound",
                                                            ));
                                                            return;
                                                        }
                                                        if buffer.is_empty() && !arguments.is_empty() {
                                                            tool_call_argument_buffers.insert(output_index, arguments.to_string());
                                                            tool_argument_bytes += arguments.len();
                                                            yield Ok(StreamEvent::InputJsonDelta {
                                                                index: output_index,
                                                                partial_json: arguments.to_string(),
                                                            });
                                                        }
                                                        finalized_tool_calls.insert(output_index);
                                                        finalized_tool_call_arguments.insert(output_index, arguments.to_string());
                                                    }
                                                } else if !finalized_tool_calls.contains(&output_index) {
                                                    yield Err(Self::stream_error(
                                                        "Codex function_call item ended before finalized arguments",
                                                    ));
                                                    return;
                                                }
                                            } else if !finalized_tool_calls.contains(&output_index) {
                                                yield Err(Self::stream_error(
                                                    "Codex function_call item ended before finalized arguments",
                                                ));
                                                return;
                                            }
                                            }
                                            }
                                            Some("other") if item_type.is_some() && !matches!(item_type, Some("function_call") | Some("message")) => {}
                                            _ => {
                                                let message = if item_type == Some("function_call") {
                                                    "Codex function_call item completion lacks a prior item"
                                                } else {
                                                    "Codex output item completion lacks a prior output item"
                                                };
                                                yield Err(Self::stream_error(message));
                                                return;
                                            }
                                        }
                                        completed_output_items.insert(output_index);
                                        if open_blocks.remove(&output_index) {
                                            yield Ok(StreamEvent::ContentBlockStop { index: output_index });
                                        }
                                    }
                                    "response.completed" => {
                                        if tool_call_item_ids
                                            .keys()
                                            .any(|index| !finalized_tool_calls.contains(index))
                                        {
                                            yield Err(Self::stream_error(
                                                "Codex response completed before finalized function_call arguments",
                                            ));
                                            return;
                                        }
                                        if let Some(response) = json_val.get("response") {
                                            native_binding.observe(Some(response), true);
                                            if let Some(id) = response.get("id").and_then(|value| value.as_str()) {
                                                message_id = id.to_string();
                                            }
                                            if let Some(model) = response.get("model").and_then(|value| value.as_str()) {
                                                model_name = model.to_string();
                                            }

                                            if !message_started {
                                                yield Ok(StreamEvent::MessageStart {
                                                    id: message_id.clone(),
                                                    model: model_name.clone(),
                                                    usage: UsageInfo::default(),
                                                });
                                                // No reassignment: this branch is the
                                                // stream-completion handler — execution
                                                // never re-enters the guard below.
                                            }

                                            let mut remaining: Vec<usize> = open_blocks.drain().collect();
                                            remaining.sort_unstable();
                                            for index in remaining {
                                                yield Ok(StreamEvent::ContentBlockStop { index });
                                            }

                                            let usage_json = response.get("usage");
                                            let usage = UsageInfo {
                                                input_tokens: usage_json
                                                    .and_then(|value| value.get("input_tokens"))
                                                    .and_then(|value| value.as_u64())
                                                    .unwrap_or(0),
                                                output_tokens: usage_json
                                                    .and_then(|value| value.get("output_tokens"))
                                                    .and_then(|value| value.as_u64())
                                                    .unwrap_or(0),
                                                cache_creation_input_tokens: 0,
                                                cache_read_input_tokens: usage_json
                                                    .and_then(|value| value.get("input_tokens_details"))
                                                    .and_then(|value| value.get("cached_tokens"))
                                                    .and_then(|value| value.as_u64())
                                                    .unwrap_or(0),
                                            };

                                            let stop_reason = if saw_tool_call {
                                                StopReason::ToolUse
                                            } else {
                                                match response
                                                    .get("incomplete_details")
                                                    .and_then(|value| value.get("reason"))
                                                    .and_then(|value| value.as_str())
                                                {
                                                    Some("max_output_tokens") => StopReason::MaxTokens,
                                                    Some("content_filter") => StopReason::ContentFiltered,
                                                    Some(other) if !other.is_empty() => StopReason::Other(other.to_string()),
                                                    _ => StopReason::EndTurn,
                                                }
                                            };

                                            if let Some(seed) = witness_seed.as_ref() {
                                                if let Some(witness) = native_binding.finalize(seed) {
                                                    yield Ok(StreamEvent::ProviderWireWitness(witness));
                                                }
                                            }

                                            yield Ok(StreamEvent::MessageDelta {
                                                stop_reason: Some(stop_reason),
                                                usage: Some(usage),
                                            });
                                            yield Ok(StreamEvent::MessageStop);
                                            return;
                                        }
                                    }
                                    "response.incomplete" => {
                                        if saw_tool_call {
                                            yield Err(Self::stream_error(
                                                "Codex response became incomplete with a function_call",
                                            ));
                                            return;
                                        }
                                        if tool_call_item_ids
                                            .keys()
                                            .any(|index| !finalized_tool_calls.contains(index))
                                        {
                                            yield Err(Self::stream_error(
                                                "Codex response became incomplete before finalized function_call arguments",
                                            ));
                                            return;
                                        }
                                        if let Some(response) = json_val.get("response") {
                                            if let Some(id) = response.get("id").and_then(|value| value.as_str()) {
                                                message_id = id.to_string();
                                            }
                                            if let Some(model) = response.get("model").and_then(|value| value.as_str()) {
                                                model_name = model.to_string();
                                            }
                                            if !message_started {
                                                yield Ok(StreamEvent::MessageStart {
                                                    id: message_id.clone(),
                                                    model: model_name.clone(),
                                                    usage: UsageInfo::default(),
                                                });
                                            }

                                            let mut remaining: Vec<usize> = open_blocks.drain().collect();
                                            remaining.sort_unstable();
                                            for index in remaining {
                                                yield Ok(StreamEvent::ContentBlockStop { index });
                                            }

                                            let usage_json = response.get("usage");
                                            let usage = UsageInfo {
                                                input_tokens: usage_json
                                                    .and_then(|value| value.get("input_tokens"))
                                                    .and_then(|value| value.as_u64())
                                                    .unwrap_or(0),
                                                output_tokens: usage_json
                                                    .and_then(|value| value.get("output_tokens"))
                                                    .and_then(|value| value.as_u64())
                                                    .unwrap_or(0),
                                                cache_creation_input_tokens: 0,
                                                cache_read_input_tokens: usage_json
                                                    .and_then(|value| value.get("input_tokens_details"))
                                                    .and_then(|value| value.get("cached_tokens"))
                                                    .and_then(|value| value.as_u64())
                                                    .unwrap_or(0),
                                            };
                                            let stop_reason = if saw_tool_call {
                                                StopReason::ToolUse
                                            } else {
                                                match response
                                                    .get("incomplete_details")
                                                    .and_then(|value| value.get("reason"))
                                                    .and_then(|value| value.as_str())
                                                {
                                                    Some("max_output_tokens") => StopReason::MaxTokens,
                                                    Some("content_filter") => StopReason::ContentFiltered,
                                                    Some(other) if !other.is_empty() => StopReason::Other(other.to_string()),
                                                    _ => StopReason::EndTurn,
                                                }
                                            };
                                            yield Ok(StreamEvent::MessageDelta {
                                                stop_reason: Some(stop_reason),
                                                usage: Some(usage),
                                            });
                                            yield Ok(StreamEvent::MessageStop);
                                            return;
                                        }
                                    }
                                    "response.failed" | "response.error" | "error" => {
                                        yield Err(Self::stream_error("Codex stream reported a terminal error"));
                                        return;
                                    }
                                    _ => {}
                                }

                                current_event.clear();
                                continue;
                            }

                            if let Some(rest) = line.strip_prefix("event:") {
                                current_event = rest.trim().to_string();
                                continue;
                            }

                            if let Some(rest) = line.strip_prefix("data:") {
                                let data_line = rest.trim_start();
                                current_data_bytes = match current_data_bytes
                                    .checked_add(data_line.len())
                                    .and_then(|bytes| bytes.checked_add(1))
                                {
                                    Some(bytes) if bytes <= MAX_CODEX_SSE_EVENT_BYTES => bytes,
                                    _ => {
                                        yield Err(Self::stream_error("Codex stream event exceeded the 1 MiB limit"));
                                        return;
                                    }
                                };
                                current_data.push(data_line.to_string());
                            }
                        }

                        if decoder.buffered_len() > MAX_CODEX_SSE_LINE_BYTES {
                            yield Err(Self::stream_error("Codex stream line exceeded the 256 KiB limit"));
                            return;
                        }
                    }

                    yield Err(ProviderError::StreamError {
                        provider: provider_id,
                        message: "Codex stream ended before response.completed".to_string(),
                        partial_response: None,
                    });
                };

        Ok(crate::inference_admission::hold_stream(s, admission))
    }

    async fn discover_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        use claurst_core::codex_oauth::codex_limit_override;
        let models = CODEX_MODELS
            .iter()
            .map(|(id, name)| {
                let (context_window, max_output_tokens) = codex_limit_override(id)
                    .map(|(ctx, _, out)| (ctx, out))
                    .unwrap_or((400_000, 128_000));
                ModelInfo {
                    id: ModelId::new(*id),
                    provider_id: self.id.clone(),
                    name: name.to_string(),
                    context_window,
                    max_output_tokens,
                    ..Default::default()
                }
            })
            .collect();
        Ok(models)
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        // Validate that a non-expired token exists without making a network call.
        let guard = self.tokens.lock().unwrap();
        if guard.access_token.is_empty() {
            return Ok(ProviderStatus::Unavailable {
                reason: "no Codex access token — run /connect to authenticate".to_string(),
            });
        }
        if Self::is_expired(&guard) && guard.refresh_token.is_none() {
            return Ok(ProviderStatus::Unavailable {
                reason: "Codex access token expired and no refresh token — re-run /connect"
                    .to_string(),
            });
        }
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: SystemPromptStyle::SystemMessage,
        }
    }

    fn response_continuation_support(&self) -> ResponseContinuationSupport {
        if self.can_continue_response() {
            ResponseContinuationSupport::FullContextFallback
        } else {
            ResponseContinuationSupport::Unsupported
        }
    }

    fn response_continuation_route_identity(&self, model: &str) -> Option<String> {
        if !self.can_continue_response() {
            return None;
        }
        let account_id = self
            .tokens
            .lock()
            .ok()?
            .account_id
            .clone()
            .unwrap_or_default();
        let encoded = serde_json::to_vec(&(
            "forge-codex-continuation-route-v1",
            &self.api_endpoint,
            account_id,
            model,
        ))
        .ok()?;
        Some(claurst_core::crypto_utils::sha256_hex(&encoded))
    }
}

#[cfg(test)]
#[path = "codex_stream_tests.rs"]
mod stream_regressions;
