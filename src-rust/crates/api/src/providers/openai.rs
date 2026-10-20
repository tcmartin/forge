// providers/openai.rs — OpenAI Chat Completions provider adapter.
//
// Implements LlmProvider for the OpenAI Chat Completions API (POST
// /v1/chat/completions).  Works equally well for any OpenAI-compatible
// endpoint (e.g. Azure OpenAI, local Ollama, Together AI) by configuring
// `base_url`.
//
// Phase 2A implementation covers:
//  - Request transformation (Anthropic internal types → OpenAI wire format)
//  - Streaming via Server-Sent Events (data: {...}\n\n lines)
//  - Non-streaming JSON response parsing
//  - Tool-call support (request and response)
//  - Model listing via GET /v1/models
//  - Health check
//  - ProviderCapabilities

use async_stream::stream;
use async_trait::async_trait;
use claurst_core::dispatch::{DispatchDenied, DispatchRoute};
use claurst_core::effort::EffortLevel;
use claurst_core::provider_id::ProviderId;
use claurst_core::types::{
    ContentBlock, ImageSource, MessageContent, Role, ToolResultContent, UsageInfo,
};
use futures::Stream;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::pin::Pin;
use std::sync::Arc;
use tracing::debug;

use crate::dispatch_policy::{
    authorize_openai_dispatch, AuthorizedInferenceLease, DispatchIdentity, DispatchPolicy,
};
use crate::error_handling::parse_error_response;
use crate::inference_admission::{AdmissionLease, InferenceAdmission};
use crate::provider::LlmProvider;
use crate::provider_error::ProviderError;
use crate::provider_types::SystemPrompt;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StopReason,
    StreamEvent, SystemPromptStyle,
};
use crate::provider_wire::{preflight_provider_request, tool_result_text};
use crate::DispatchRequest;

use super::request_options::merge_openai_compatible_options;

// OpenAI SSE framing and retention limits. These are intentionally generous
// enough for normal Chat Completions tool streams while keeping every parser
// buffer and cumulative response finite.
const MAX_OPENAI_SSE_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_OPENAI_SSE_LINE_BYTES: usize = 256 * 1024;
const MAX_OPENAI_SSE_EVENT_BYTES: usize = 1024 * 1024;
const MAX_OPENAI_SSE_TOTAL_BYTES: usize = 64 * 1024 * 1024;
const MAX_OPENAI_SSE_TOOL_ARGUMENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_OPENAI_SSE_TOOL_CALLS: usize = 256;

// ---------------------------------------------------------------------------
// OpenAiProvider
// ---------------------------------------------------------------------------

pub struct OpenAiProvider {
    id: ProviderId,
    name: String,
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
    inference_admission: Arc<InferenceAdmission>,
    dispatch_identity: DispatchIdentity,
    dispatch_policy: Option<Arc<DispatchPolicy>>,
    #[cfg(feature = "a016-openai-loopback")]
    test_transport_base_url: Option<String>,
}

impl OpenAiProvider {
    pub fn new(api_key: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(crate::request_timeout())
            .build()
            .expect("failed to build reqwest client");

        Self {
            id: ProviderId::new(ProviderId::OPENAI),
            name: "OpenAI".to_string(),
            base_url: "https://api.openai.com".to_string(),
            api_key,
            http_client,
            inference_admission: InferenceAdmission::process(),
            dispatch_identity: DispatchIdentity::fresh(),
            dispatch_policy: None,
            #[cfg(feature = "a016-openai-loopback")]
            test_transport_base_url: None,
        }
    }

    fn stream_error(message: impl Into<String>) -> ProviderError {
        ProviderError::StreamError {
            provider: ProviderId::new(ProviderId::OPENAI),
            message: message.into(),
            partial_response: None,
        }
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

    /// Override the API base URL (e.g. for Azure, Ollama, or other compatible
    /// endpoints).
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Install a host-owned dispatch policy. Replacing the policy rotates the
    /// client identity, invalidating envelopes minted for the prior policy.
    pub fn with_dispatch_policy(mut self, policy: Arc<DispatchPolicy>) -> Self {
        self.dispatch_identity = DispatchIdentity::fresh();
        self.dispatch_policy = Some(policy);
        self
    }

    fn bound_with_dispatch_policy(&self, policy: Arc<DispatchPolicy>) -> Self {
        Self {
            id: self.id.clone(),
            name: self.name.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            http_client: self.http_client.clone(),
            inference_admission: Arc::clone(&self.inference_admission),
            dispatch_identity: DispatchIdentity::fresh(),
            dispatch_policy: Some(policy),
            #[cfg(feature = "a016-openai-loopback")]
            test_transport_base_url: self.test_transport_base_url.clone(),
        }
    }

    #[cfg(feature = "a016-openai-loopback")]
    pub fn with_test_loopback_base_url(mut self, base_url: String) -> Self {
        self.test_transport_base_url = Some(base_url);
        self
    }

    fn dispatch_endpoint(&self) -> String {
        format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        )
    }

    fn transport_endpoint(&self) -> String {
        #[cfg(feature = "a016-openai-loopback")]
        if let Some(base_url) = &self.test_transport_base_url {
            return format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        }
        self.dispatch_endpoint()
    }

    fn validate_dispatch(
        &self,
        model: &str,
        dispatch: Option<&DispatchRequest>,
    ) -> Result<Option<&'static str>, ProviderError> {
        let Some(dispatch) = dispatch else {
            return Ok(None);
        };
        if self.api_key.is_empty() {
            return Err(ProviderError::DispatchRejected {
                provider: self.id.clone(),
                error: DispatchDenied::UnsupportedRoute,
            });
        }
        if dispatch.expected_route().model_id() != model {
            return Err(ProviderError::DispatchRejected {
                provider: self.id.clone(),
                error: DispatchDenied::RouteMismatch,
            });
        }
        let policy = self
            .dispatch_policy
            .as_deref()
            .ok_or(ProviderError::DispatchRejected {
                provider: self.id.clone(),
                error: DispatchDenied::UnsupportedRoute,
            })?;
        policy
            .openai_reasoning_for_envelope(
                &self.dispatch_identity,
                &self.dispatch_endpoint(),
                dispatch,
            )
            .map(Some)
            .map_err(|error| ProviderError::DispatchRejected {
                provider: self.id.clone(),
                error,
            })
    }

    fn prepared_dispatch_error(error: DispatchDenied) -> ProviderError {
        ProviderError::DispatchRejected {
            provider: ProviderId::new(ProviderId::OPENAI),
            error,
        }
    }

    fn validate_dispatch_options(
        dispatch: Option<&DispatchRequest>,
        provider_options: &Value,
        reasoning_effort: Option<&str>,
    ) -> Result<Value, ProviderError> {
        if dispatch.is_none() {
            return Ok(provider_options.clone());
        }
        let protected = [
            "model",
            "stream",
            "reasoning_effort",
            "reasoningEffort",
            "max_completion_tokens",
            "max_tokens",
        ];
        let mut options = provider_options.clone();
        if let Some(object) = options.as_object_mut() {
            if object.contains_key("reasoningEffort") {
                if object.get("reasoningEffort").and_then(Value::as_str) != reasoning_effort {
                    return Err(Self::prepared_dispatch_error(DispatchDenied::RouteMismatch));
                }
                // The query builder emits this compatibility spelling for all
                // OpenAI-family requests. The adapter owns the final
                // snake-case wire field, so remove the duplicate before
                // preparing the immutable dispatched body.
                object.remove("reasoningEffort");
            }
            if object.keys().any(|key| protected.contains(&key.as_str())) {
                return Err(Self::prepared_dispatch_error(DispatchDenied::RouteMismatch));
            }
        }
        Ok(options)
    }

    fn request_build_error(error: reqwest::Error) -> ProviderError {
        ProviderError::Other {
            provider: ProviderId::new(ProviderId::OPENAI),
            message: format!("failed to build HTTP request: {error}"),
            status: None,
            body: None,
        }
    }

    /// Returns `true` if the model should use the Responses API instead of
    /// Chat Completions for an unbound request.
    fn use_responses_api(model: &str) -> bool {
        // gpt-5.5 is explicitly certified on the canonical Chat Completions
        // route for this adapter; retain Responses routing for the other
        // reasoning families until their wire contracts are implemented.
        (model.starts_with("gpt-5") && model != "gpt-5.5")
            || model.starts_with("o3")
            || model.starts_with("o4")
    }

    fn uses_chat_reasoning_model(model: &str) -> bool {
        model == "gpt-5.5"
    }

    // -----------------------------------------------------------------------
    // Request transformation helpers
    // -----------------------------------------------------------------------

    /// Public wrapper for Azure/Copilot providers that share the OpenAI wire format.
    pub fn to_openai_messages_pub(
        messages: &[claurst_core::types::Message],
        system_prompt: Option<&SystemPrompt>,
    ) -> Vec<Value> {
        Self::to_openai_messages(messages, system_prompt)
    }

    /// Public wrapper for tool conversion used by Azure/Copilot providers.
    pub fn to_openai_tools_pub(tools: &[claurst_core::types::ToolDefinition]) -> Vec<Value> {
        Self::to_openai_tools(tools)
    }

    /// Public wrapper for finish-reason mapping.
    pub fn map_finish_reason_pub(reason: &str) -> StopReason {
        Self::map_finish_reason(reason)
    }

    /// Public wrapper for usage parsing.
    pub fn parse_usage_pub(usage: Option<&Value>) -> UsageInfo {
        Self::parse_usage(usage)
    }

    /// Public wrapper for non-streaming response parsing.
    pub fn parse_non_streaming_response_pub(
        json: &Value,
        provider_id: &claurst_core::provider_id::ProviderId,
    ) -> Result<crate::provider_types::ProviderResponse, crate::provider_error::ProviderError> {
        Self::parse_non_streaming_response(json, provider_id)
    }

    /// Convert a provider-agnostic [`ProviderRequest`] into the OpenAI Chat
    /// Completions `messages` array.
    fn to_openai_messages(
        messages: &[claurst_core::types::Message],
        system_prompt: Option<&SystemPrompt>,
    ) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();

        // System prompt goes first as a `system` role message.
        if let Some(sys) = system_prompt {
            let sys_text = match sys {
                SystemPrompt::Text(t) => t.clone(),
                SystemPrompt::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            result.push(json!({ "role": "system", "content": sys_text }));
        }

        for msg in messages {
            match msg.role {
                Role::User => {
                    Self::append_user_messages(&mut result, &msg.content);
                }
                Role::Assistant => {
                    let (text_content, tool_calls) =
                        Self::assistant_content_to_openai(&msg.content);
                    let mut obj = serde_json::Map::new();
                    obj.insert("role".into(), json!("assistant"));
                    if let Some(tc) = text_content {
                        obj.insert("content".into(), json!(tc));
                    } else {
                        obj.insert("content".into(), Value::Null);
                    }
                    if !tool_calls.is_empty() {
                        obj.insert("tool_calls".into(), json!(tool_calls));
                    }
                    result.push(Value::Object(obj));

                    // ToolResult blocks in an assistant message need to be
                    // emitted as separate `role: tool` messages.
                    let tool_results = Self::extract_tool_results(&msg.content);
                    result.extend(tool_results);
                }
            }
        }

        result
    }

    fn append_user_messages(result: &mut Vec<Value>, content: &MessageContent) {
        match content {
            MessageContent::Text(text) => {
                result.push(json!({ "role": "user", "content": text }));
            }
            MessageContent::Blocks(blocks) => {
                let has_result_images = blocks.iter().any(|block| {
                    matches!(block, ContentBlock::ToolResult { content, .. }
                        if Self::result_has_images(content, 0))
                });
                if has_result_images {
                    // Chat tool messages cannot contain image parts. Keep all
                    // tool responses contiguous, then send associated images
                    // in one user message, in canonical occurrence order.
                    let mut parts = Vec::new();
                    for block in blocks {
                        if let Some(tool_result) = Self::tool_result_to_openai_message(block) {
                            result.push(tool_result);
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } = block
                            {
                                let before = parts.len();
                                Self::append_result_images(content, &mut parts, 0);
                                if parts.len() > before {
                                    parts.insert(before, json!({
                                        "type": "text",
                                        "text": format!("Images returned by tool call {tool_use_id}:"),
                                    }));
                                }
                            }
                        } else if let Some(part) = Self::user_block_to_openai_part(block) {
                            parts.push(part);
                        }
                    }
                    result.push(json!({"role":"user","content":parts}));
                    return;
                }
                let mut user_parts: Vec<Value> = Vec::new();
                let flush_user_parts = |result: &mut Vec<Value>, parts: &mut Vec<Value>| {
                    if !parts.is_empty() {
                        result.push(json!({
                            "role": "user",
                            "content": std::mem::take(parts),
                        }));
                    }
                };

                for block in blocks {
                    if let Some(tool_result) = Self::tool_result_to_openai_message(block) {
                        flush_user_parts(result, &mut user_parts);
                        result.push(tool_result);
                    } else if let Some(part) = Self::user_block_to_openai_part(block) {
                        user_parts.push(part);
                    }
                }

                flush_user_parts(result, &mut user_parts);
            }
        }
    }

    fn result_has_images(content: &ToolResultContent, depth: usize) -> bool {
        if depth > claurst_core::media::MediaLimits::default().max_nesting {
            return false;
        }
        match content {
            ToolResultContent::Text(_) => false,
            ToolResultContent::Blocks(blocks) => blocks.iter().any(|block| match block {
                ContentBlock::Image { .. } => true,
                ContentBlock::ToolResult { content, .. } => {
                    Self::result_has_images(content, depth + 1)
                }
                _ => false,
            }),
        }
    }

    fn append_result_images(content: &ToolResultContent, parts: &mut Vec<Value>, depth: usize) {
        if depth > claurst_core::media::MediaLimits::default().max_nesting {
            return;
        }
        if let ToolResultContent::Blocks(blocks) = content {
            for block in blocks {
                match block {
                    ContentBlock::Image { .. } => {
                        if let Some(part) = Self::user_block_to_openai_part(block) {
                            parts.push(part);
                        }
                    }
                    ContentBlock::ToolResult { content, .. } => {
                        Self::append_result_images(content, parts, depth + 1);
                    }
                    _ => {}
                }
            }
        }
    }

    fn user_block_to_openai_part(block: &ContentBlock) -> Option<Value> {
        match block {
            ContentBlock::Text { text } => Some(json!({ "type": "text", "text": text })),
            ContentBlock::Image { source } => {
                let url = Self::image_source_to_url(source);
                Some(json!({
                    "type": "image_url",
                    "image_url": { "url": url }
                }))
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                // Tool results become separate `role: tool` messages at the
                // conversation level — handled in append_user_messages.
                let _ = (tool_use_id, content, is_error);
                None
            }
            // Thinking, RedactedThinking, etc. are not supported by OpenAI.
            _ => None,
        }
    }

    fn image_source_to_url(source: &ImageSource) -> String {
        if let Some(url) = &source.url {
            return url.clone();
        }
        // base64-encoded image
        let media_type = source.media_type.as_deref().unwrap_or("image/png");
        let data = source.data.as_deref().unwrap_or("");
        format!("data:{};base64,{}", media_type, data)
    }

    /// Split assistant content blocks into (text_string, tool_calls_array).
    fn assistant_content_to_openai(content: &MessageContent) -> (Option<String>, Vec<Value>) {
        let blocks = match content {
            MessageContent::Text(t) => return (Some(t.clone()), vec![]),
            MessageContent::Blocks(b) => b,
        };

        let mut text_parts: Vec<&str> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();

        for block in blocks {
            match block {
                ContentBlock::Text { text } => {
                    text_parts.push(text.as_str());
                }
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    let args = serde_json::to_string(input).unwrap_or_default();
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args
                        }
                    }));
                }
                // Thinking is dropped — not supported by OpenAI.
                _ => {}
            }
        }

        let text_content = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        (text_content, tool_calls)
    }

    /// Collect any ToolResult blocks and emit them as `role: tool` messages.
    fn extract_tool_results(content: &MessageContent) -> Vec<Value> {
        let blocks = match content {
            MessageContent::Text(_) => return vec![],
            MessageContent::Blocks(b) => b,
        };

        blocks
            .iter()
            .filter_map(Self::tool_result_to_openai_message)
            .collect()
    }

    fn tool_result_to_openai_message(block: &ContentBlock) -> Option<Value> {
        let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        else {
            return None;
        };

        let text = tool_result_text(content);

        Some(json!({
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": text,
        }))
    }

    /// Convert tool definitions to the OpenAI `tools` array format.
    fn to_openai_tools(tools: &[claurst_core::types::ToolDefinition]) -> Vec<Value> {
        tools
            .iter()
            .map(|td| {
                json!({
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.input_schema
                    }
                })
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // HTTP helpers
    // -----------------------------------------------------------------------

    fn auth_header(&self) -> (&'static str, String) {
        ("Authorization", format!("Bearer {}", self.api_key))
    }

    fn map_http_error(&self, status: u16, body: &str) -> ProviderError {
        parse_error_response(status, body, &self.id)
    }

    // -----------------------------------------------------------------------
    // Non-streaming create_message
    // -----------------------------------------------------------------------

    async fn create_message_non_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let selected = preflight_provider_request("openai", request)
            .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
        let messages = Self::to_openai_messages(&request.messages, request.system_prompt.as_ref());
        let tools = Self::to_openai_tools(&request.tools);
        let url = self.dispatch_endpoint();
        let transport_url = self.transport_endpoint();
        let reasoning_effort = self.validate_dispatch(&request.model, request.dispatch.as_ref())?;
        let provider_options = Self::validate_dispatch_options(
            request.dispatch.as_ref(),
            &request.provider_options,
            reasoning_effort,
        )?;

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": false,
            "store": false,
        });
        if reasoning_effort.is_some() || Self::uses_chat_reasoning_model(&request.model) {
            body["max_completion_tokens"] = json!(request.max_tokens);
            if let Some(reasoning_effort) = reasoning_effort {
                body["reasoning_effort"] = json!(reasoning_effort);
            }
        } else {
            body["max_tokens"] = json!(request.max_tokens);
        }

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(t) = request.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(p) = request.top_p {
            body["top_p"] = json!(p);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }
        merge_openai_compatible_options(&mut body, &provider_options);

        let prepared = selected
            .prepare_json(&url, false, &body)
            .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
        let (auth_key, auth_val) = self.auth_header();
        let outbound = self
            .http_client
            .post(&transport_url)
            .header(auth_key, auth_val)
            .header("Content-Type", "application/json")
            .body(prepared.bytes().to_vec())
            .build()
            .map_err(Self::request_build_error)?;

        let admission = self.acquire_inference().await?;
        let authorization = authorize_openai_dispatch(
            self.dispatch_policy.as_deref(),
            &self.dispatch_identity,
            &url,
            &prepared,
            request.dispatch.as_ref(),
            request.max_tokens,
        )
        .await
        .map_err(Self::prepared_dispatch_error)?;
        let _lease = AuthorizedInferenceLease::new(admission, authorization);
        let resp = self
            .http_client
            .execute(outbound)
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to read response body: {}", e),
            status: Some(status),
            body: None,
        })?;

        if !(200..300).contains(&(status as usize)) {
            return Err(self.map_http_error(status, &text));
        }

        let json: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to parse response JSON: {}", e),
            status: Some(status),
            body: Some(text.clone()),
        })?;

        Self::parse_non_streaming_response(&json, &self.id)
    }

    fn parse_non_streaming_response(
        json: &Value,
        provider_id: &ProviderId,
    ) -> Result<ProviderResponse, ProviderError> {
        let id = json
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let choice = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| ProviderError::Other {
                provider: provider_id.clone(),
                message: "No choices in response".to_string(),
                status: None,
                body: None,
            })?;

        let message = choice.get("message").ok_or_else(|| ProviderError::Other {
            provider: provider_id.clone(),
            message: "No message in choice".to_string(),
            status: None,
            body: None,
        })?;

        let mut content_blocks: Vec<ContentBlock> = Vec::new();

        // Text content
        if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                content_blocks.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }

        // Tool calls
        if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args_str = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                content_blocks.push(ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    thought_signature: None,
                });
            }
        }

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("stop");
        let stop_reason = Self::map_finish_reason(finish_reason);

        let usage = Self::parse_usage(json.get("usage"));

        Ok(ProviderResponse {
            id,
            content: content_blocks,
            stop_reason,
            usage,
            model,
        })
    }

    // -----------------------------------------------------------------------
    // Streaming helpers
    // -----------------------------------------------------------------------

    fn map_finish_reason(reason: &str) -> StopReason {
        match reason {
            "stop" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" | "function_call" => StopReason::ToolUse,
            "content_filter" => StopReason::ContentFiltered,
            other => StopReason::Other(other.to_string()),
        }
    }

    fn parse_usage(usage: Option<&Value>) -> UsageInfo {
        let u = match usage {
            Some(v) => v,
            None => return UsageInfo::default(),
        };
        let prompt_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        // OpenAI-compatible servers report prompt-cache hits under
        // `prompt_tokens_details.cached_tokens` (OpenAI spec; llama.cpp >= b4600
        // mirrors it when prompt caching is enabled). Unlike Anthropic, whose
        // `input_tokens` excludes cached tokens, OpenAI's `prompt_tokens` is
        // *inclusive* of the cached count, so we split it back out to keep the
        // additive convention (`input_tokens + cache_read_input_tokens ==
        // prompt_tokens`) used by the cost tracker and context accounting.
        let cache_read = u
            .get("prompt_tokens_details")
            .and_then(|v| v.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            .min(prompt_tokens);
        UsageInfo {
            input_tokens: prompt_tokens.saturating_sub(cache_read),
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: cache_read,
        }
    }

    // -----------------------------------------------------------------------
    // Streaming create_message_stream
    // -----------------------------------------------------------------------

    async fn do_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<(reqwest::Response, AuthorizedInferenceLease), ProviderError> {
        let selected = preflight_provider_request("openai", request)
            .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
        let messages = Self::to_openai_messages(&request.messages, request.system_prompt.as_ref());
        let tools = Self::to_openai_tools(&request.tools);
        let url = self.dispatch_endpoint();
        let transport_url = self.transport_endpoint();
        let reasoning_effort = self.validate_dispatch(&request.model, request.dispatch.as_ref())?;
        let provider_options = Self::validate_dispatch_options(
            request.dispatch.as_ref(),
            &request.provider_options,
            reasoning_effort,
        )?;

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": true,
            "stream_options": { "include_usage": true },
            "store": false,
        });
        if reasoning_effort.is_some() || Self::uses_chat_reasoning_model(&request.model) {
            body["max_completion_tokens"] = json!(request.max_tokens);
            if let Some(reasoning_effort) = reasoning_effort {
                body["reasoning_effort"] = json!(reasoning_effort);
            }
        } else {
            body["max_tokens"] = json!(request.max_tokens);
        }

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(t) = request.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(p) = request.top_p {
            body["top_p"] = json!(p);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }
        merge_openai_compatible_options(&mut body, &provider_options);

        let prepared = selected
            .prepare_json(&url, true, &body)
            .map_err(|error| ProviderError::budget(self.id.clone(), error))?;
        let (auth_key, auth_val) = self.auth_header();
        let outbound = self
            .http_client
            .post(&transport_url)
            .header(auth_key, auth_val)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .body(prepared.bytes().to_vec())
            .build()
            .map_err(Self::request_build_error)?;

        let admission = self.acquire_inference().await?;
        let authorization = authorize_openai_dispatch(
            self.dispatch_policy.as_deref(),
            &self.dispatch_identity,
            &url,
            &prepared,
            request.dispatch.as_ref(),
            request.max_tokens,
        )
        .await
        .map_err(Self::prepared_dispatch_error)?;
        let lease = AuthorizedInferenceLease::new(admission, authorization);
        let resp = self
            .http_client
            .execute(outbound)
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&(status as usize)) {
            let text = resp.text().await.unwrap_or_default();
            return Err(self.map_http_error(status, &text));
        }

        Ok((resp, lease))
    }
}

// The `LlmProvider` impl and capability helpers below are declared after this
// module; keeping these wire-format tests next to the helpers they exercise is
// intentional, so allow the item-ordering lint here.
#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use claurst_core::types::Message;
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    fn stream_request() -> ProviderRequest {
        ProviderRequest {
            model: "fixture-model".to_string(),
            messages: Vec::new(),
            system_prompt: None,
            tools: Vec::new(),
            max_tokens: 128,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            thinking: None,
            provider_options: json!({}),
            cache_scope: None,
            response_continuation: None,
            wire_budget: Some(
                crate::WireBudget::finite(
                    1_000_000,
                    900_000,
                    100_000,
                    4_000_000,
                    crate::MediaAccounting::None,
                )
                .expect("test wire budget"),
            ),
            dispatch: None,
        }
    }

    #[test]
    fn user_tool_results_become_tool_messages() {
        let messages = vec![
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "search".to_string(),
                input: json!({ "q": "test" }),
                thought_signature: None,
            }]),
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: ToolResultContent::Text("done".to_string()),
                is_error: Some(false),
            }]),
        ];

        let wire = OpenAiProvider::to_openai_messages(&messages, None);
        assert_eq!(wire.len(), 2);
        assert_eq!(
            wire[0].get("role").and_then(|v| v.as_str()),
            Some("assistant")
        );
        assert_eq!(wire[1].get("role").and_then(|v| v.as_str()), Some("tool"));
        assert_eq!(
            wire[1].get("tool_call_id").and_then(|v| v.as_str()),
            Some("call_1")
        );
        assert_eq!(
            wire[1].get("content").and_then(|v| v.as_str()),
            Some("done")
        );
    }

    #[test]
    fn mixed_user_content_flushes_before_tool_result() {
        let messages = vec![Message::user_blocks(vec![
            ContentBlock::Text {
                text: "preface".to_string(),
            },
            ContentBlock::ToolResult {
                tool_use_id: "call_2".to_string(),
                content: ToolResultContent::Text("ok".to_string()),
                is_error: Some(false),
            },
        ])];

        let wire = OpenAiProvider::to_openai_messages(&messages, None);
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[0].get("role").and_then(|v| v.as_str()), Some("user"));
        assert_eq!(wire[1].get("role").and_then(|v| v.as_str()), Some("tool"));
        assert_eq!(
            wire[1].get("tool_call_id").and_then(|v| v.as_str()),
            Some("call_2")
        );
    }

    #[test]
    fn tool_result_text_keeps_empty_and_nested_slots_without_newlines() {
        let messages = vec![Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "call_nested".into(),
            content: ToolResultContent::Blocks(vec![
                ContentBlock::Text { text: "a".into() },
                ContentBlock::Text {
                    text: String::new(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "inner".into(),
                    content: ToolResultContent::Blocks(vec![
                        ContentBlock::Text { text: "é".into() },
                        ContentBlock::Text { text: "b".into() },
                    ]),
                    is_error: None,
                },
            ]),
            is_error: None,
        }])];

        let wire = OpenAiProvider::to_openai_messages(&messages, None);
        assert_eq!(wire[0]["role"], "tool");
        assert_eq!(wire[0]["content"], "aéb");
    }

    #[test]
    fn tool_images_follow_all_tool_responses_and_preserve_media_selection() {
        const RED: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
        const BLUE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC";
        let image = |data: &str| ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".into(),
                media_type: Some("image/png".into()),
                data: Some(data.into()),
                url: None,
            },
        };
        let mut request = stream_request();
        request.model = "gpt-4o".into();
        request.messages = vec![
            Message::assistant_blocks(vec![
                ContentBlock::ToolUse {
                    id: "read1".into(),
                    name: "Read".into(),
                    input: json!({}),
                    thought_signature: None,
                },
                ContentBlock::ToolUse {
                    id: "read2".into(),
                    name: "Read".into(),
                    input: json!({}),
                    thought_signature: None,
                },
            ]),
            Message::user_blocks(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "read1".into(),
                    content: ToolResultContent::Blocks(vec![image(RED), image(BLUE)]),
                    is_error: None,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "read2".into(),
                    content: ToolResultContent::Blocks(vec![image(RED)]),
                    is_error: None,
                },
            ]),
        ];
        let selected = preflight_provider_request("openai", &request).unwrap();
        let messages = OpenAiProvider::to_openai_messages(&request.messages, None);
        assert_eq!(
            messages
                .iter()
                .map(|m| m["role"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["assistant", "tool", "tool", "user"]
        );
        assert_eq!(messages[1]["tool_call_id"], "read1");
        assert_eq!(messages[2]["tool_call_id"], "read2");
        let body = json!({"model":request.model,"messages":messages});
        let prepared = selected
            .prepare_json("/chat/completions", true, &body)
            .unwrap();
        assert!(
            matches!(prepared.media(), crate::MediaAccounting::Images(images) if images.manifest().items().len()==3)
        );
        let mut dropped = body.clone();
        dropped["messages"].as_array_mut().unwrap().pop();
        assert!(matches!(
            selected.prepare_json("/chat/completions", true, &dropped),
            Err(crate::WireBudgetError::MediaMismatch)
        ));
    }

    #[test]
    fn parse_usage_none_is_all_zero() {
        let usage = OpenAiProvider::parse_usage(None);
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn parse_usage_without_cache_details_reports_zero_cache() {
        let usage = OpenAiProvider::parse_usage(Some(&json!({
            "prompt_tokens": 1000,
            "completion_tokens": 42,
        })));
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.output_tokens, 42);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn parse_usage_reads_cached_tokens_from_prompt_details() {
        // OpenAI spec / llama.cpp: prompt_tokens is inclusive of cached_tokens.
        let usage = OpenAiProvider::parse_usage(Some(&json!({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": { "cached_tokens": 800 },
        })));
        assert_eq!(usage.cache_read_input_tokens, 800);
        // input_tokens is the non-cached remainder so the two never double-count.
        assert_eq!(usage.input_tokens, 200);
        assert_eq!(
            usage.input_tokens + usage.cache_read_input_tokens,
            1000,
            "input + cache_read must reconstruct the reported prompt_tokens"
        );
        assert_eq!(usage.output_tokens, 50);
    }

    #[test]
    fn parse_usage_cache_details_present_but_zero() {
        let usage = OpenAiProvider::parse_usage(Some(&json!({
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_tokens_details": { "cached_tokens": 0 },
        })));
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn parse_usage_cached_tokens_clamped_to_prompt_tokens() {
        // Defensive: a malformed server that reports cached > prompt must not
        // underflow input_tokens.
        let usage = OpenAiProvider::parse_usage(Some(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": { "cached_tokens": 999 },
        })));
        assert_eq!(usage.cache_read_input_tokens, 100);
        assert_eq!(usage.input_tokens, 0);
    }

    #[tokio::test]
    async fn loopback_stream_consumes_empty_choices_usage_chunk() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        let body = concat!(
            r#"data: {"id":"fixture","model":"fixture-model","choices":[{"delta":{"content":"answer"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "\n\n",
            r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("request should connect");
            socket
                .write_all(response.as_bytes())
                .await
                .expect("response should write");
        });

        let provider = OpenAiProvider::new("fixture-key".to_string())
            .with_base_url(format!("http://{address}"));
        let mut stream = provider
            .create_message_stream(stream_request())
            .await
            .expect("loopback stream should start");
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.expect("loopback stream should decode"));
        }
        server.await.expect("server should finish");

        assert!(events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::MessageDelta {
                    stop_reason: None,
                    usage: Some(usage),
                } if usage.input_tokens == 5 && usage.output_tokens == 2
            )
        }));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::MessageStop))
                .count(),
            1
        );
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
    }
}

// ---------------------------------------------------------------------------
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.validate_dispatch(&request.model, request.dispatch.as_ref())?;
        if request.dispatch.is_none() && Self::use_responses_api(&request.model) {
            return Err(ProviderError::InvalidRequest {
                provider: self.id.clone(),
                message: format!(
                    "Model '{}' requires the OpenAI Responses API which is not yet fully \
                     implemented. Use gpt-4o or gpt-4o-mini for now, or set \
                     OPENAI_BASE_URL to a compatible endpoint.",
                    request.model
                ),
            });
        }
        self.create_message_non_streaming(&request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        self.validate_dispatch(&request.model, request.dispatch.as_ref())?;
        if request.dispatch.is_none() && Self::use_responses_api(&request.model) {
            return Err(ProviderError::InvalidRequest {
                provider: self.id.clone(),
                message: format!(
                    "Model '{}' requires the OpenAI Responses API which is not yet fully \
                     implemented. Use gpt-4o or gpt-4o-mini for now, or set \
                     OPENAI_BASE_URL to a compatible endpoint.",
                    request.model
                ),
            });
        }
        let (resp, lease) = self.do_streaming(&request).await?;
        let provider_id = self.id.clone();

        // We need the message ID to emit MessageStart.  We'll generate one on
        // the first chunk that carries it.
        let s = stream! {
            use futures::StreamExt;

            let mut byte_stream = resp.bytes_stream();
            let mut decoder = crate::SseByteDecoder::new();
            let mut total_stream_bytes = 0_usize;
            let mut current_event_bytes = 0_usize;
            let mut tool_argument_bytes = 0_usize;
            let mut tool_call_ids = std::collections::HashSet::new();
            let mut tool_call_id_slots: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();

            // State carried across chunks
            let mut message_started = false;
            let mut message_id = String::from("unknown");
            let mut model_name = String::new();
            // Track accumulating tool call argument buffers: index -> (id, name, buf)
            let mut tool_call_buffers: std::collections::HashMap<
                usize,
                (String, String, String),
            > = std::collections::HashMap::new();

            // Bound infinite mid-stream stalls (issue #185): wrap each chunk
            // read in a generous idle timeout so a provider that pauses
            // indefinitely surfaces an error instead of hanging. Each chunk
            // resets the timer, so slow-but-progressing models are never cut off.
            let idle_timeout = crate::stream_idle_timeout();
            loop {
                let chunk_result = match tokio::time::timeout(
                    idle_timeout,
                    byte_stream.next(),
                )
                .await
                {
                    Ok(Some(chunk_result)) => chunk_result,
                    // Stream ended normally.
                    Ok(None) => break,
                    // No bytes for `idle_timeout` — provider stalled mid-stream.
                    Err(_) => {
                        yield Err(ProviderError::StreamError {
                            provider: provider_id.clone(),
                            message: format!(
                                "Stream stalled: no data received for {}s; aborting to avoid hanging",
                                idle_timeout.as_secs()
                            ),
                            partial_response: None,
                        });
                        return;
                    }
                };
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(ProviderError::StreamError {
                            provider: provider_id.clone(),
                            message: format!("Stream read error: {}", e),
                            partial_response: None,
                        });
                        return;
                    }
                };

                if chunk.len() > MAX_OPENAI_SSE_CHUNK_BYTES {
                    yield Err(Self::stream_error("OpenAI stream chunk exceeded the 1 MiB limit"));
                    return;
                }
                total_stream_bytes = match total_stream_bytes.checked_add(chunk.len()) {
                    Some(bytes) if bytes <= MAX_OPENAI_SSE_TOTAL_BYTES => bytes,
                    _ => {
                        yield Err(Self::stream_error(
                            "OpenAI stream exceeded the 64 MiB cumulative limit",
                        ));
                        return;
                    }
                };

                for raw_line in decoder.push(&chunk) {
                    if raw_line.len() > MAX_OPENAI_SSE_LINE_BYTES {
                        yield Err(Self::stream_error("OpenAI stream line exceeded the 256 KiB limit"));
                        return;
                    }
                    let line = raw_line.trim_end_matches('\r').trim();

                    // A blank line terminates the current SSE event. Only data
                    // lines are retained as payload, matching the existing
                    // OpenAI parser while bounding multiline events.
                    if line.is_empty() {
                        current_event_bytes = 0;
                        continue;
                    }
                    if line.starts_with(':') {
                        continue;
                    }

                    let data = if let Some(rest) = line.strip_prefix("data:") {
                        rest.trim()
                    } else {
                        continue;
                    };
                    current_event_bytes = match current_event_bytes
                        .checked_add(data.len())
                        .and_then(|bytes| bytes.checked_add(1))
                    {
                        Some(bytes) if bytes <= MAX_OPENAI_SSE_EVENT_BYTES => bytes,
                        _ => {
                            yield Err(Self::stream_error(
                                "OpenAI stream event exceeded the 1 MiB limit",
                            ));
                            return;
                        }
                    };

                    if data == "[DONE]" {
                        yield Ok(StreamEvent::MessageStop);
                        return;
                    }

                    let chunk_json: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            debug!(error_kind = ?e.classify(), "OpenAI SSE: invalid JSON event");
                            continue;
                        }
                    };

                    // Extract message id and model on first chunk.
                    if !message_started {
                        if let Some(id) = chunk_json.get("id").and_then(|v| v.as_str()) {
                            message_id = id.to_string();
                        }
                        if let Some(m) = chunk_json.get("model").and_then(|v| v.as_str()) {
                            model_name = m.to_string();
                        }
                        // Emit MessageStart — usage will be filled in later from
                        // the final chunk; emit zeros for now.
                        yield Ok(StreamEvent::MessageStart {
                            id: message_id.clone(),
                            model: model_name.clone(),
                            usage: UsageInfo::default(),
                        });
                        // Emit ContentBlockStart for the text block (index 0).
                        yield Ok(StreamEvent::ContentBlockStart {
                            index: 0,
                            content_block: ContentBlock::Text { text: String::new() },
                        });
                        message_started = true;
                    }

                    let choices = match chunk_json.get("choices").and_then(|c| c.as_array()) {
                        Some(c) => c,
                        None => {
                            // May be a usage-only chunk (the final one).
                            if let Some(usage_val) = chunk_json.get("usage") {
                                let usage = OpenAiProvider::parse_usage(Some(usage_val));
                                yield Ok(StreamEvent::MessageDelta {
                                    stop_reason: None,
                                    usage: Some(usage),
                                });
                            }
                            continue;
                        }
                    };

                    let choice = match choices.first() {
                        Some(c) => c,
                        None => {
                            // OpenAI's final usage chunk carries
                            // `choices: []`; consume it before [DONE] closes
                            // the message instead of silently dropping usage.
                            if let Some(usage_val) = chunk_json.get("usage") {
                                let usage = OpenAiProvider::parse_usage(Some(usage_val));
                                yield Ok(StreamEvent::MessageDelta {
                                    stop_reason: None,
                                    usage: Some(usage),
                                });
                            }
                            continue;
                        }
                    };

                    let delta = match choice.get("delta") {
                        Some(d) => d,
                        None => continue,
                    };

                    // Text content delta
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            yield Ok(StreamEvent::TextDelta {
                                index: 0,
                                text: content.to_string(),
                            });
                        }
                    }

                    // Tool call deltas
                    if let Some(tool_calls) =
                        delta.get("tool_calls").and_then(|t| t.as_array())
                    {
                        for tc in tool_calls {
                            let tc_index_value = match tc.get("index") {
                                None => 0_u64,
                                Some(value) => match value.as_u64() {
                                    Some(index) => index,
                                    None => {
                                        yield Err(Self::stream_error(
                                            "OpenAI stream tool-call index was not a nonnegative integer",
                                        ));
                                        return;
                                    }
                                },
                            };
                            if tc_index_value >= MAX_OPENAI_SSE_TOOL_CALLS as u64 {
                                yield Err(Self::stream_error(
                                    "OpenAI stream tool-call index exceeded the 256-slot limit",
                                ));
                                return;
                            }
                            let tc_index = tc_index_value as usize;
                            let block_index = match tc_index.checked_add(1) {
                                Some(index) => index,
                                None => {
                                    yield Err(Self::stream_error(
                                        "OpenAI stream tool-call index overflowed",
                                    ));
                                    return;
                                }
                            };

                            // OpenAI sends id/name only on the first chunk for
                            // each tool call. Repeated metadata for the same
                            // id and slot must preserve the accumulated buffer.
                            if let Some(tc_id) = tc.get("id").and_then(|v| v.as_str()) {
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let is_new_block = match tool_call_buffers.get(&block_index) {
                                    Some((existing_id, _, _)) if existing_id != tc_id => {
                                        yield Err(Self::stream_error(
                                            "OpenAI stream reused a tool-call slot for another id",
                                        ));
                                        return;
                                    }
                                    Some(_) => false,
                                    None => true,
                                };
                                if let Some(existing_block) = tool_call_id_slots.get(tc_id) {
                                    if *existing_block != block_index {
                                        yield Err(Self::stream_error(
                                            "OpenAI stream reused a tool-call id at another index",
                                        ));
                                        return;
                                    }
                                }
                                if is_new_block {
                                    if tool_call_buffers.len() >= MAX_OPENAI_SSE_TOOL_CALLS {
                                        yield Err(Self::stream_error(
                                            "OpenAI stream exceeded the 256 retained tool-call limit",
                                        ));
                                        return;
                                    }
                                    if tool_call_ids.len() >= MAX_OPENAI_SSE_TOOL_CALLS {
                                        yield Err(Self::stream_error(
                                            "OpenAI stream exceeded the 256 tool-call limit",
                                        ));
                                        return;
                                    }
                                    tool_call_ids.insert(tc_id.to_string());
                                    tool_call_id_slots.insert(tc_id.to_string(), block_index);
                                    tool_call_buffers.insert(
                                        block_index,
                                        (tc_id.to_string(), name.clone(), String::new()),
                                    );
                                    yield Ok(StreamEvent::ContentBlockStart {
                                        index: block_index,
                                        content_block: ContentBlock::ToolUse {
                                            id: tc_id.to_string(),
                                            name,
                                            input: json!({}),
                                            thought_signature: None,
                                        },
                                    });
                                }
                            }
                            // Argument fragment
                            if let Some(args_frag) = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                            {
                                if !args_frag.is_empty() {
                                    tool_argument_bytes = match tool_argument_bytes
                                        .checked_add(args_frag.len())
                                    {
                                        Some(bytes) if bytes <= MAX_OPENAI_SSE_TOOL_ARGUMENT_BYTES => bytes,
                                        _ => {
                                            yield Err(Self::stream_error(
                                                "OpenAI stream exceeded the 8 MiB tool-argument limit",
                                            ));
                                            return;
                                        }
                                    };
                                    if let Some((_, _, buf)) =
                                        tool_call_buffers.get_mut(&block_index)
                                    {
                                        buf.push_str(args_frag);
                                    }
                                    yield Ok(StreamEvent::InputJsonDelta {
                                        index: block_index,
                                        partial_json: args_frag.to_string(),
                                    });
                                }
                            }
                        }
                    }

                    // finish_reason signals end of message.
                    if let Some(finish_reason) =
                        choice.get("finish_reason").and_then(|v| v.as_str())
                    {
                        if !finish_reason.is_empty() && finish_reason != "null" {
                            // Close the text content block.
                            yield Ok(StreamEvent::ContentBlockStop { index: 0 });
                            // Close any open tool call blocks.
                            let mut tc_indices: Vec<usize> =
                                tool_call_buffers.keys().cloned().collect();
                            tc_indices.sort();
                            for idx in tc_indices {
                                yield Ok(StreamEvent::ContentBlockStop { index: idx });
                            }

                            let stop_reason =
                                OpenAiProvider::map_finish_reason(finish_reason);

                            // Usage might come in the same chunk or a later one.
                            let usage_val = chunk_json.get("usage");
                            let usage = usage_val.map(|u| OpenAiProvider::parse_usage(Some(u)));

                            yield Ok(StreamEvent::MessageDelta {
                                stop_reason: Some(stop_reason),
                                usage,
                            });
                        }
                    }
                }

                if decoder.buffered_len() > MAX_OPENAI_SSE_LINE_BYTES {
                    yield Err(Self::stream_error("OpenAI stream line exceeded the 256 KiB limit"));
                    return;
                }
            }

            // If we consumed all bytes without seeing [DONE], emit stop.
            if message_started {
                yield Ok(StreamEvent::MessageStop);
            }
        };

        Ok(crate::inference_admission::hold_stream_guard(s, lease))
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        let (auth_key, auth_val) = self.auth_header();
        let url = format!("{}/v1/models", self.base_url);

        let resp = self
            .http_client
            .get(&url)
            .header(auth_key, auth_val)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => Ok(ProviderStatus::Healthy),
            Ok(r) => Ok(ProviderStatus::Unavailable {
                reason: format!("models endpoint returned {}", r.status()),
            }),
            Err(e) => Ok(ProviderStatus::Unavailable {
                reason: e.to_string(),
            }),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: true,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: true,
            system_prompt_style: SystemPromptStyle::SystemMessage,
        }
    }

    fn host_configuration_fingerprint(&self) -> Option<String> {
        let test_transport_base_url = {
            #[cfg(feature = "a016-openai-loopback")]
            {
                self.test_transport_base_url.as_deref().unwrap_or("")
            }
            #[cfg(not(feature = "a016-openai-loopback"))]
            {
                ""
            }
        };
        let mut hasher = Sha256::new();
        for value in [
            self.api_key.as_bytes(),
            self.base_url.as_bytes(),
            test_transport_base_url.as_bytes(),
        ] {
            hasher.update((value.len() as u64).to_be_bytes());
            hasher.update(value);
        }
        Some(hex::encode(hasher.finalize()))
    }

    fn dispatch_route(
        &self,
        model: &str,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, DispatchDenied> {
        if self.api_key.is_empty() {
            return Err(DispatchDenied::UnsupportedRoute);
        }
        let policy = self
            .dispatch_policy
            .as_deref()
            .ok_or(DispatchDenied::UnsupportedRoute)?;
        policy.openai_route(
            &self.dispatch_identity,
            &self.dispatch_endpoint(),
            model,
            effort,
        )
    }

    fn bind_dispatch_policy(
        &self,
        policy: Arc<DispatchPolicy>,
    ) -> Result<Arc<dyn LlmProvider>, DispatchDenied> {
        if self.api_key.is_empty() {
            return Err(DispatchDenied::UnsupportedRoute);
        }
        Ok(Arc::new(self.bound_with_dispatch_policy(policy)))
    }
}

#[cfg(test)]
#[path = "openai_stream_tests.rs"]
mod stream_regressions;
