// providers/anthropic.rs — AnthropicProvider: wraps AnthropicClient in the
// unified LlmProvider trait.
//
// Phase 2A: create_message and create_message_stream are fully implemented by
// mapping ProviderRequest → CreateMessageRequest and mapping
// AnthropicStreamEvent → provider_types::StreamEvent.

use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use claurst_core::dispatch::{DispatchDenied, DispatchRoute};
use claurst_core::effort::EffortLevel;
use claurst_core::error::ClaudeError;
use claurst_core::provider_id::{ModelId, ProviderId};
use claurst_core::types::{ContentBlock, UsageInfo};
use futures::Stream;

use crate::client::{AnthropicClient, ClientConfig};
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StopReason,
    StreamBlockAccumulator, StreamEvent, SystemPromptStyle,
};
use crate::provider_wire::{preflight_provider_request, WireBudgetError};
use crate::streaming::{AnthropicStreamEvent, ContentDelta, NullStreamHandler};
use crate::types::{ApiMessage, ApiToolDefinition, CreateMessageRequest};

use super::message_normalization::normalize_anthropic_messages;

// ---------------------------------------------------------------------------
// AnthropicProvider
// ---------------------------------------------------------------------------

/// Wraps [`AnthropicClient`] so it can be held in a [`ProviderRegistry`] behind
/// `Arc<dyn LlmProvider>`.
pub struct AnthropicProvider {
    client: Arc<AnthropicClient>,
    id: ProviderId,
}

impl AnthropicProvider {
    /// Wrap an already-constructed (and Arc-wrapped) [`AnthropicClient`].
    pub fn new(client: Arc<AnthropicClient>) -> Self {
        Self {
            client,
            id: ProviderId::new(ProviderId::ANTHROPIC),
        }
    }

    /// Construct directly from a [`ClientConfig`], creating the inner client.
    pub fn from_config(config: ClientConfig) -> Self {
        let client = AnthropicClient::new(config)
            .expect("AnthropicProvider::from_config: failed to create AnthropicClient");
        Self {
            client: Arc::new(client),
            id: ProviderId::new(ProviderId::ANTHROPIC),
        }
    }

    /// Build a [`CreateMessageRequest`] from a [`ProviderRequest`].
    fn build_request(request: &ProviderRequest) -> CreateMessageRequest {
        let normalized_messages = normalize_anthropic_messages(&request.messages);
        let api_messages: Vec<ApiMessage> =
            normalized_messages.iter().map(ApiMessage::from).collect();

        let api_tools: Option<Vec<ApiToolDefinition>> = if request.tools.is_empty() {
            None
        } else {
            Some(
                request
                    .stable_tools()
                    .into_iter()
                    .map(ApiToolDefinition::from)
                    .collect(),
            )
        };

        let system = request.system_prompt.clone();

        let mut builder = CreateMessageRequest::builder(&request.model, request.max_tokens)
            .messages(api_messages);

        if let Some(sys) = system {
            builder = builder.system(sys);
        }
        if let Some(tools) = api_tools {
            builder = builder.tools(tools);
        }
        if let Some(t) = request.temperature {
            builder = builder.temperature(t as f32);
        }
        if let Some(p) = request.top_p {
            builder = builder.top_p(p as f32);
        }
        if let Some(k) = request.top_k {
            builder = builder.top_k(k);
        }
        if !request.stop_sequences.is_empty() {
            builder = builder.stop_sequences(request.stop_sequences.clone());
        }
        if let Some(tc) = request.thinking.clone() {
            builder = builder.thinking(tc);
        }
        if request.cache_scope.is_some() {
            builder = builder.cache_control(crate::types::CacheControl::ephemeral());
        }
        if let Some(budget) = request.wire_budget.clone() {
            builder = builder.wire_budget(budget);
        }
        if let Some(dispatch) = request.dispatch.clone() {
            builder = builder.dispatch(dispatch);
        }

        builder.build()
    }

    /// Bind freshly validated canonical media to the legacy request before it
    /// reaches OAuth/account lookup or transport. The legacy client performs a
    /// second final-wire gate after OAuth mutation; this bound budget is the
    /// authority that gate receives.
    fn prepare_request(request: &ProviderRequest) -> Result<CreateMessageRequest, ProviderError> {
        let selected =
            preflight_provider_request(ProviderId::ANTHROPIC, request).map_err(|error| {
                ProviderError::budget(ProviderId::new(ProviderId::ANTHROPIC), error)
            })?;

        let mut api_request = Self::build_request(request);
        if let Some(budget) = api_request.wire_budget.as_mut() {
            budget.media = selected.media().clone();
        }
        let body = serde_json::to_value(&api_request).map_err(|error| {
            ProviderError::budget(
                ProviderId::new(ProviderId::ANTHROPIC),
                WireBudgetError::Serialization(error.to_string()),
            )
        })?;
        selected
            .prepare_json("/v1/messages", true, &body)
            .map_err(|error| {
                ProviderError::budget(ProviderId::new(ProviderId::ANTHROPIC), error)
            })?;
        Ok(api_request)
    }

    /// Map a string stop_reason from Anthropic wire format to [`StopReason`].
    fn map_stop_reason(s: &str) -> StopReason {
        match s {
            "end_turn" => StopReason::EndTurn,
            "stop_sequence" => StopReason::StopSequence,
            "max_tokens" => StopReason::MaxTokens,
            "tool_use" => StopReason::ToolUse,
            other => StopReason::Other(other.to_string()),
        }
    }

    /// Map an [`AnthropicStreamEvent`] to the provider-agnostic [`StreamEvent`].
    fn map_stream_event(evt: AnthropicStreamEvent) -> Option<StreamEvent> {
        match evt {
            AnthropicStreamEvent::MessageStart { id, model, usage } => {
                Some(StreamEvent::MessageStart { id, model, usage })
            }
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => Some(StreamEvent::ContentBlockStart {
                index,
                content_block,
            }),
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
                ContentDelta::TextDelta { text } => Some(StreamEvent::TextDelta { index, text }),
                ContentDelta::ThinkingDelta { thinking } => {
                    Some(StreamEvent::ThinkingDelta { index, thinking })
                }
                ContentDelta::SignatureDelta { signature } => {
                    Some(StreamEvent::SignatureDelta { index, signature })
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    Some(StreamEvent::InputJsonDelta {
                        index,
                        partial_json,
                    })
                }
            },
            AnthropicStreamEvent::ContentBlockStop { index } => {
                Some(StreamEvent::ContentBlockStop { index })
            }
            AnthropicStreamEvent::MessageDelta { stop_reason, usage } => {
                let mapped_stop = stop_reason.as_deref().map(Self::map_stop_reason);
                Some(StreamEvent::MessageDelta {
                    stop_reason: mapped_stop,
                    usage,
                })
            }
            AnthropicStreamEvent::MessageStop => Some(StreamEvent::MessageStop),
            AnthropicStreamEvent::Error {
                error_type,
                message,
            } => Some(StreamEvent::Error {
                error_type,
                message,
            }),
            AnthropicStreamEvent::Ping => None,
        }
    }

    fn map_client_error(provider: &ProviderId, error: ClaudeError) -> ProviderError {
        match error {
            ClaudeError::InferenceDispatchDenied(error) => ProviderError::DispatchRejected {
                provider: provider.clone(),
                error,
            },
            ClaudeError::RequestBudgetExceeded {
                input_tokens,
                input_limit_tokens,
                wire_bytes,
                wire_byte_limit,
                context_limit_tokens,
                output_reserve_tokens,
                budget_source,
            } => ProviderError::budget(
                provider.clone(),
                // ProviderError currently exposes only the typed wire-budget
                // envelope. Preserve every planner field through its numeric
                // overflow variant until the dedicated RequestBudgetExceeded
                // provider variant is available.
                WireBudgetError::InputOverflow {
                    estimated_input_tokens: input_tokens,
                    input_limit_tokens,
                    wire_bytes,
                    max_wire_bytes: wire_byte_limit,
                    context_limit_tokens,
                    output_reserve_tokens,
                    policy_revision: "legacy-request-budget".into(),
                    budget_source,
                },
            ),
            error => ProviderError::Other {
                provider: provider.clone(),
                message: error.to_string(),
                status: None,
                body: None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "Anthropic"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        // The legacy client owns route validation and dispatch authorization.
        // Wrappers must preserve the opaque envelope through request building.
        // Collect stream events to build a complete response.
        let mut stream = self.create_message_stream(request).await?;

        let mut id = String::from("unknown");
        let mut model = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut usage = UsageInfo::default();

        // Accumulate every content block (text, thinking, tool_use, …) keyed by
        // its stream index. Captures thinking/signature/reasoning deltas (which
        // the previous `_ => {}` arm silently dropped) and preserves interleave
        // order via a single ordered pass. See issue #217.
        let mut blocks = StreamBlockAccumulator::new();

        use futures::StreamExt;
        while let Some(result) = stream.next().await {
            match result {
                Err(e) => return Err(e),
                Ok(evt) => {
                    // Content-block events feed the accumulator; message
                    // lifecycle events update the local id/model/usage/stop.
                    blocks.on_event(&evt);
                    match evt {
                        StreamEvent::MessageStart {
                            id: msg_id,
                            model: msg_model,
                            usage: msg_usage,
                        } => {
                            id = msg_id;
                            model = msg_model;
                            usage = msg_usage;
                        }
                        StreamEvent::MessageDelta {
                            stop_reason: sr,
                            usage: delta_usage,
                        } => {
                            if let Some(r) = sr {
                                stop_reason = r;
                            }
                            if let Some(u) = delta_usage {
                                usage.output_tokens += u.output_tokens;
                            }
                        }
                        StreamEvent::MessageStop => break,
                        StreamEvent::Error {
                            error_type,
                            message,
                        } => {
                            return Err(ProviderError::StreamError {
                                provider: self.id.clone(),
                                message: format!("[{}] {}", error_type, message),
                                partial_response: None,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        // Finalize every block in stream-index order — a single ordered pass
        // that preserves interleave (thinking → text → tool_use, …). No block is
        // appended out of band, so signed-thinking replay order is intact. #217.
        let final_content: Vec<ContentBlock> = blocks.finish();

        Ok(ProviderResponse {
            id,
            content: final_content,
            stop_reason,
            usage,
            model,
        })
    }

    // TODO(#228): `AnthropicProvider` is a thin adapter over the legacy
    // `AnthropicClient` (crate::client, in lib.rs): it delegates SSE decoding to
    // `client.create_message_stream` and then maps the Anthropic-typed
    // `AnthropicStreamEvent`s to `StreamEvent` via `map_stream_event`. These are
    // the two Anthropic stacks #228 wants collapsed under one `AnthropicMessages`
    // protocol (see the matching TODO on `AnthropicClient::process_sse_stream`).
    // Not done in this pass because `AnthropicClient` is a public type the TUI and
    // other crates depend on directly — deferring keeps everything green.
    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let api_request = Self::prepare_request(&request)?;
        let handler = Arc::new(NullStreamHandler);

        let provider_id = self.id.clone();

        let mut rx = self
            .client
            .create_message_stream(api_request, handler)
            .await
            .map_err(|e| Self::map_client_error(&provider_id, e))?;

        let s = stream! {
            while let Some(anthropic_evt) = rx.recv().await {
                if let Some(unified_evt) = AnthropicProvider::map_stream_event(anthropic_evt) {
                    yield Ok(unified_evt);
                }
            }
        };

        Ok(Box::pin(s))
    }

    /// Live model discovery via `GET /v1/models`, returning exactly the models
    /// the active credential can use. For a Claude Pro/Max OAuth token this is
    /// the curated subscription set (no legacy claude-3.x); for an API key it is
    /// the org's accessible models. On any failure (offline, expired/absent
    /// token) returns an empty vec so the picker keeps the models.dev catalog
    /// projection instead of surfacing an error.
    async fn discover_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        let available = match self.client.fetch_available_models().await {
            Ok(m) => m,
            Err(_) => return Ok(Vec::new()),
        };
        let models = available
            .into_iter()
            .map(|m| {
                let name = m.display_name.unwrap_or_else(|| m.id.clone());
                ModelInfo {
                    id: ModelId::new(&m.id),
                    provider_id: self.id.clone(),
                    name,
                    // Sensible Claude defaults; the picker overlays richer
                    // context/cost from the models.dev catalog for known ids.
                    context_window: 200_000,
                    max_output_tokens: 64_000,
                    ..Default::default()
                }
            })
            .collect();
        Ok(models)
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        // Client was successfully constructed with a non-empty API key.
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: true,
            image_input: true,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: true,
            structured_output: true,
            system_prompt_style: SystemPromptStyle::TopLevel,
        }
    }

    fn dispatch_route(
        &self,
        model: &str,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, DispatchDenied> {
        self.client.dispatch_route(model, effort)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::types::{ImageSource, Message};

    const TINY_PNG: &str =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

    fn image_message() -> Message {
        Message::user_blocks(vec![ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".into(),
                media_type: Some("image/png".into()),
                data: Some(TINY_PNG.into()),
                url: None,
            },
        }])
    }

    fn request() -> ProviderRequest {
        ProviderRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![image_message()],
            system_prompt: None,
            tools: Vec::new(),
            max_tokens: 128,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            thinking: None,
            provider_options: serde_json::json!({}),
            cache_scope: None,
            response_continuation: None,
            wire_budget: Some(
                crate::provider_wire::WireBudget::finite(
                    30_000,
                    20_000,
                    1_000,
                    50_000,
                    crate::provider_wire::MediaAccounting::None,
                )
                .unwrap(),
            ),
            dispatch: None,
        }
    }

    #[test]
    fn legacy_request_binds_canonical_media_before_transport() {
        let prepared = AnthropicProvider::prepare_request(&request()).unwrap();
        let budget = prepared
            .wire_budget
            .as_ref()
            .expect("preflight budget should be retained");
        assert!(matches!(
            budget.media,
            crate::provider_wire::MediaAccounting::Images(_)
        ));
        let body = serde_json::to_value(&prepared).unwrap();
        assert_eq!(body["messages"][0]["content"][0]["type"], "image");
        assert_eq!(
            body["messages"][0]["content"][0]["source"]["type"],
            "base64"
        );
    }

    #[test]
    fn canonical_media_preflight_does_not_change_text_messages() {
        let mut request = request();
        request.messages = vec![Message::user("plain text")];
        let prepared = AnthropicProvider::prepare_request(&request).unwrap();
        assert!(matches!(
            &prepared.messages[0].content,
            serde_json::Value::String(text) if text == "plain text"
        ));
        assert!(matches!(
            prepared.wire_budget.expect("budget").media,
            crate::provider_wire::MediaAccounting::None
        ));
    }

    #[test]
    fn scoped_request_enables_native_cache_control_and_stable_tool_order() {
        let mut first = request();
        first.messages = vec![Message::user("first")];
        first.system_prompt = Some(crate::provider_types::SystemPrompt::Text(
            "stable system".into(),
        ));
        first.tools = vec![
            claurst_core::types::ToolDefinition {
                name: "Zulu".into(),
                description: "z".into(),
                input_schema: serde_json::json!({"type":"object"}),
            },
            claurst_core::types::ToolDefinition {
                name: "Alpha".into(),
                description: "a".into(),
                input_schema: serde_json::json!({"type":"object"}),
            },
        ];
        first.cache_scope = Some(
            crate::provider_types::ProviderCacheScope::for_session("workspace", "session")
                .expect("cache scope"),
        );

        let first_wire =
            serde_json::to_value(AnthropicProvider::build_request(&first)).expect("first wire");
        let mut second = first.clone();
        second.messages.push(Message::assistant("answer"));
        second.messages.push(Message::user("second"));
        let second_wire =
            serde_json::to_value(AnthropicProvider::build_request(&second)).expect("second wire");

        assert_eq!(
            first_wire["cache_control"],
            serde_json::json!({"type":"ephemeral"})
        );
        assert_eq!(first_wire["system"], second_wire["system"]);
        assert_eq!(first_wire["tools"], second_wire["tools"]);
        assert_eq!(first_wire["tools"][0]["name"], "Alpha");
        assert_eq!(first_wire["tools"][1]["name"], "Zulu");
    }

    #[test]
    fn unscoped_request_omits_native_cache_control() {
        let wire =
            serde_json::to_value(AnthropicProvider::build_request(&request())).expect("wire");
        assert!(wire.get("cache_control").is_none());
    }
}
