use super::*;
use crate::client::{AnthropicClient, ClientConfig, Provider};
use crate::provider::LlmProvider;
use crate::providers::{
    AnthropicProvider, AzureProvider, BedrockProvider, CodexProvider, CohereProvider,
    CopilotProvider, GoogleProvider, MinimaxProvider, OpenAiCompatProvider,
};
use crate::streaming::NullStreamHandler;
use claurst_core::dispatch::{
    DispatchAuthorization, DispatchAuthorizer, DispatchDenied, DispatchPermit, DispatchRoute,
    DispatchRouteSpec, LogicalRequestId,
};
use claurst_core::oauth_config::CodexTokens;
use claurst_core::provider_id::ProviderId;
use std::sync::Arc;
use std::time::Duration;

struct NeverCalledAuthorizer;

impl DispatchAuthorizer for NeverCalledAuthorizer {
    fn authorize<'a>(
        &'a self,
        _attempt: &'a claurst_core::dispatch::DispatchAttempt,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<DispatchAuthorization, DispatchDenied>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { panic!("dispatch authorizer must not run on unsupported route") })
    }
}

fn route() -> DispatchRoute {
    DispatchRoute::new(DispatchRouteSpec {
        provider_id: "openai".into(),
        provider_instance_id: "fixture-instance".into(),
        account_identity: "fixture-account".into(),
        model_id: "fixture-model".into(),
        endpoint_identity: "fixture-endpoint".into(),
        effort: "default".into(),
        policy_revision: 1,
    })
    .expect("valid fixture route")
}

fn dispatch() -> DispatchRequest {
    DispatchRequest::new(
        DispatchPermit::new(Arc::new(NeverCalledAuthorizer)),
        LogicalRequestId::fresh(),
        route(),
    )
}

fn request() -> ProviderRequest {
    ProviderRequest {
        model: "fixture-model".into(),
        messages: vec![],
        system_prompt: None,
        tools: vec![],
        max_tokens: 32,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: serde_json::json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            WireBudget::finite(8_192, 7_680, 512, 1024 * 1024, MediaAccounting::None)
                .expect("fixture budget"),
        ),
        dispatch: Some(dispatch()),
    }
}

async fn assert_unsupported_adapter(provider: &dyn LlmProvider) {
    let nonstream = provider.create_message(request()).await;
    assert!(matches!(
        nonstream,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        })
    ));

    let stream = provider.create_message_stream(request()).await;
    assert!(matches!(
        stream,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        })
    ));
}

#[test]
fn dispatch_envelope_is_opaque_cloneable_and_omitted_from_json() {
    let envelope = dispatch();
    let cloned = envelope.clone();
    assert_eq!(cloned.logical_request_id(), envelope.logical_request_id());
    assert_eq!(cloned.expected_route(), envelope.expected_route());
    assert_eq!(format!("{envelope:?}"), "DispatchRequest(..)");

    let serialized = serde_json::to_value(request()).expect("request serializes");
    assert!(serialized.get("dispatch").is_none());
    let mut forged = serialized;
    forged["dispatch"] = serde_json::json!({"permit": "forged", "route": "forged"});
    let decoded: ProviderRequest = serde_json::from_value(forged).expect("request decodes");
    assert!(decoded.dispatch.is_none());

    let legacy = CreateMessageRequest::builder("fixture-model", 32)
        .dispatch(envelope)
        .build();
    let legacy_json = serde_json::to_value(legacy).expect("legacy request serializes");
    assert!(legacy_json.get("dispatch").is_none());
}

#[test]
fn unsupported_dispatch_is_categorical_and_non_retryable() {
    let error = crate::dispatch_request::reject_unsupported_dispatch(
        &ProviderId::new("fixture"),
        Some(&dispatch()),
    )
    .expect_err("present dispatch must be rejected");
    assert!(matches!(
        error,
        ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        }
    ));
    assert!(!error.is_retryable());
    assert_eq!(error.provider_id(), "fixture");
}

#[tokio::test]
async fn openai_unsupported_dispatch_makes_zero_loopback_requests() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let address = listener.local_addr().expect("listener address");
    let provider =
        OpenAiProvider::new("fixture-key".into()).with_base_url(format!("http://{address}"));

    let result = provider.create_message(request()).await;
    assert!(matches!(
        result,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        })
    ));
    let accepted = tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
    assert!(
        accepted.is_err(),
        "unsupported dispatch opened loopback HTTP"
    );

    let invalid_url =
        OpenAiProvider::new("fixture-key".into()).with_base_url("not a URL".to_string());
    let mut malformed = request();
    malformed.wire_budget = None;
    malformed.provider_options = serde_json::json!("malformed provider options");
    let result = invalid_url.create_message(malformed).await;
    assert!(matches!(
        result,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        })
    ));
}

#[tokio::test]
async fn free_unsupported_dispatch_rejects_before_upstream_selection() {
    let provider = FreeProvider::new(Vec::new());

    let nonstream = provider.create_message(request()).await;
    assert!(matches!(
        nonstream,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        })
    ));

    let stream = provider.create_message_stream(request()).await;
    assert!(matches!(
        stream,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::UnsupportedRoute,
            ..
        })
    ));
}

#[tokio::test]
async fn every_adapter_rejects_dispatch_before_auth_or_transport() {
    let anthropic = AnthropicProvider::from_config(ClientConfig {
        api_key: String::new(),
        api_base: "http://127.0.0.1:9".into(),
        ..Default::default()
    });
    assert_unsupported_adapter(&anthropic).await;

    let azure = AzureProvider::new(String::new(), String::new());
    assert_unsupported_adapter(&azure).await;

    let bedrock = BedrockProvider::test_fixture();
    assert_unsupported_adapter(&bedrock).await;

    let cohere = CohereProvider::new(String::new());
    assert_unsupported_adapter(&cohere).await;

    let codex = CodexProvider::new(CodexTokens {
        access_token: "expired-fixture-token".into(),
        refresh_token: None,
        account_id: None,
        expires_at: Some(0),
    });
    assert_unsupported_adapter(&codex).await;

    let copilot = CopilotProvider::new(String::new());
    assert_unsupported_adapter(&copilot).await;

    let google = GoogleProvider::new(String::new());
    assert_unsupported_adapter(&google).await;

    let minimax = MinimaxProvider::new(String::new()).with_base_url("http://127.0.0.1:9");
    assert_unsupported_adapter(&minimax).await;

    let compat = OpenAiCompatProvider::new(
        "compat-fixture",
        "Compatibility fixture",
        "http://127.0.0.1:9/v1",
    );
    assert_unsupported_adapter(&compat).await;
}

#[tokio::test]
async fn legacy_anthropic_rejects_dispatch_before_nonstream_or_stream_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let address = listener.local_addr().expect("listener address");
    let client = AnthropicClient::new(ClientConfig {
        api_key: String::new(),
        api_base: format!("http://{address}"),
        ..Default::default()
    })
    .expect("client configuration");
    let request = || {
        CreateMessageRequest::builder("fixture-model", 32)
            .dispatch(dispatch())
            .build()
    };

    let nonstream = client.create_message(request()).await;
    assert!(matches!(
        nonstream,
        Err(claurst_core::error::ClaudeError::InferenceDispatchDenied(
            DispatchDenied::UnsupportedRoute
        ))
    ));
    let stream = client
        .create_message_stream(request(), Arc::new(NullStreamHandler))
        .await;
    assert!(matches!(
        stream,
        Err(claurst_core::error::ClaudeError::InferenceDispatchDenied(
            DispatchDenied::UnsupportedRoute
        ))
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "legacy unsupported dispatch opened loopback HTTP"
    );
}

#[tokio::test]
async fn legacy_codex_rejects_dispatch_before_token_refresh_or_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let address = listener.local_addr().expect("listener address");
    let client = AnthropicClient::new(ClientConfig {
        api_key: String::new(),
        api_base: format!("http://{address}"),
        provider: Provider::Codex,
        ..Default::default()
    })
    .expect("client configuration");
    let request = CreateMessageRequest::builder("fixture-model", 32)
        .dispatch(dispatch())
        .build();

    let result = client.create_message(request).await;
    assert!(matches!(
        result,
        Err(claurst_core::error::ClaudeError::InferenceDispatchDenied(
            DispatchDenied::UnsupportedRoute
        ))
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "legacy Codex unsupported dispatch opened loopback HTTP"
    );
}

#[tokio::test]
async fn legacy_codex_without_dispatch_fails_closed_before_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let address = listener.local_addr().expect("listener address");
    let client = AnthropicClient::new(ClientConfig {
        api_key: "synthetic-codex-token".into(),
        api_base: format!("http://{address}"),
        provider: Provider::Codex,
        ..Default::default()
    })
    .expect("client configuration");
    let request = CreateMessageRequest::builder("fixture-model", 32)
        .wire_budget(
            WireBudget::finite(8_192, 7_680, 512, 1024 * 1024, MediaAccounting::None)
                .expect("fixture budget"),
        )
        .build();

    let error = client
        .create_message(request)
        .await
        .expect_err("legacy Codex route must fail closed");
    assert!(
        error.to_string().contains("legacy Codex client"),
        "unexpected error: {error}"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "legacy Codex route opened loopback HTTP"
    );
}
