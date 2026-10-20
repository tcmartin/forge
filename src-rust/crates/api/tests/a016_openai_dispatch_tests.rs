#![cfg(feature = "a016-openai-loopback")]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use claurst_api::dispatch_policy::{DispatchModelSpec, DispatchPolicy, EffortWireBinding};
use claurst_api::provider::LlmProvider;
use claurst_api::provider_types::ProviderRequest;
use claurst_api::providers::OpenAiProvider;
use claurst_api::{DispatchRequest, MediaAccounting, ProviderError, WireBudget};
use claurst_core::dispatch::{
    DispatchAttempt, DispatchAuthorization, DispatchAuthorizer, DispatchDecision, DispatchDenied,
    DispatchPermit, DispatchReservation, LogicalRequestId,
};
use claurst_core::effort::EffortLevel;
use claurst_core::types::Message;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MODEL: &str = "gpt-5.5";

fn budget() -> WireBudget {
    WireBudget::finite(32_768, 16_384, 8_192, 1_048_576, MediaAccounting::None)
        .expect("fixture budget")
}

fn policy() -> Arc<DispatchPolicy> {
    policy_with_revision(19)
}

fn policy_with_revision(revision: u64) -> Arc<DispatchPolicy> {
    Arc::new(
        DispatchPolicy::new(
            revision,
            vec![DispatchModelSpec {
                model_id: MODEL.into(),
                efforts: vec![
                    EffortWireBinding::openai_reasoning(EffortLevel::Medium),
                    EffortWireBinding::openai_reasoning(EffortLevel::High),
                ],
            }],
        )
        .expect("fixture policy"),
    )
}

#[derive(Default)]
struct AuthorizerState {
    attempts: Vec<DispatchAttempt>,
}

struct FixtureAuthorizer {
    state: Arc<Mutex<AuthorizerState>>,
}

struct FixtureReservation;

impl DispatchReservation for FixtureReservation {}

impl DispatchAuthorizer for FixtureAuthorizer {
    fn authorize<'a>(
        &'a self,
        attempt: &'a DispatchAttempt,
    ) -> Pin<Box<dyn Future<Output = Result<DispatchAuthorization, DispatchDenied>> + Send + 'a>>
    {
        Box::pin(async move {
            self.state
                .lock()
                .expect("authorizer lock")
                .attempts
                .push(attempt.clone());
            Ok(DispatchAuthorization::new(
                DispatchDecision::AuthorizedFresh,
                Box::new(FixtureReservation),
            ))
        })
    }
}

fn provider(base_url: String) -> OpenAiProvider {
    provider_with_policy(base_url, policy())
}

fn provider_with_policy(base_url: String, dispatch_policy: Arc<DispatchPolicy>) -> OpenAiProvider {
    OpenAiProvider::new("fixture-openai-key".into())
        .with_test_loopback_base_url(base_url)
        .with_dispatch_policy(dispatch_policy)
}

fn request(dispatch: Option<DispatchRequest>, provider_options: Value) -> ProviderRequest {
    ProviderRequest {
        model: MODEL.into(),
        messages: vec![Message::user("hello")],
        system_prompt: None,
        tools: Vec::new(),
        max_tokens: 1_024,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: Vec::new(),
        thinking: None,
        provider_options,
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(budget()),
        dispatch,
    }
}

fn dispatch_for(
    provider: &OpenAiProvider,
    authorizer: Arc<FixtureAuthorizer>,
    effort: EffortLevel,
) -> DispatchRequest {
    let route = provider
        .dispatch_route(MODEL, effort)
        .expect("registered OpenAI route");
    DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        route,
    )
}

async fn read_request_body(socket: &mut TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 4096];
    let body_start = loop {
        let read = socket.read(&mut buffer).await.expect("request read");
        assert!(read > 0, "request closed before headers");
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..body_start]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.strip_prefix("content-length:")
                .or_else(|| line.strip_prefix("Content-Length:"))
        })
        .and_then(|value| value.trim().parse::<usize>().ok())
        .expect("content length");
    while bytes.len() < body_start + content_length {
        let read = socket.read(&mut buffer).await.expect("request body read");
        assert!(read > 0, "request closed before body");
        bytes.extend_from_slice(&buffer[..read]);
    }
    bytes[body_start..body_start + content_length].to_vec()
}

async fn serve_error(listener: TcpListener) -> Vec<u8> {
    let (mut socket, _) = listener.accept().await.expect("loopback accept");
    let body = read_request_body(&mut socket).await;
    socket
        .write_all(
            b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 31\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"fixture\"}}",
        )
        .await
        .expect("loopback response");
    body
}

#[tokio::test]
async fn openai_medium_route_authorizes_exact_nonstream_body() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let endpoint = format!("http://{}", listener.local_addr().expect("address"));
    let authorizer_state = Arc::new(Mutex::new(AuthorizerState::default()));
    let authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::clone(&authorizer_state),
    });
    let provider = provider(endpoint);
    let dispatch = dispatch_for(&provider, Arc::clone(&authorizer), EffortLevel::Medium);
    let server = tokio::spawn(serve_error(listener));

    assert!(provider
        .create_message(request(Some(dispatch), json!({})))
        .await
        .is_err());
    let body = server.await.expect("server task");
    let body_value: Value = serde_json::from_slice(&body).expect("JSON body");
    assert_eq!(body_value["model"], MODEL);
    assert_eq!(body_value["reasoning_effort"], "medium");
    assert_eq!(body_value["max_completion_tokens"], 1_024);
    assert!(body_value.get("max_tokens").is_none());
    assert_eq!(body_value["stream"], false);

    let attempts = &authorizer_state.lock().expect("authorizer lock").attempts;
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].route().provider_id(), "openai");
    assert_eq!(attempts[0].route().model_id(), MODEL);
    assert_eq!(
        attempts[0].body_sha256(),
        hex::encode(Sha256::digest(&body))
    );
}

#[tokio::test]
async fn openai_high_route_authorizes_exact_stream_body() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let endpoint = format!("http://{}", listener.local_addr().expect("address"));
    let authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::new(Mutex::new(AuthorizerState::default())),
    });
    let provider = provider(endpoint);
    let dispatch = dispatch_for(&provider, Arc::clone(&authorizer), EffortLevel::High);
    let server = tokio::spawn(serve_error(listener));

    let stream = provider
        .create_message_stream(request(Some(dispatch), json!({})))
        .await;
    assert!(stream.is_err());
    let body = server.await.expect("server task");
    let body_value: Value = serde_json::from_slice(&body).expect("JSON body");
    assert_eq!(body_value["model"], MODEL);
    assert_eq!(body_value["reasoning_effort"], "high");
    assert_eq!(body_value["max_completion_tokens"], 1_024);
    assert!(body_value.get("max_tokens").is_none());
    assert_eq!(body_value["stream"], true);
    let state = authorizer.state.lock().expect("authorizer lock");
    assert_eq!(state.attempts.len(), 1);
    assert_eq!(state.attempts[0].route().provider_id(), "openai");
    assert_eq!(state.attempts[0].route().model_id(), MODEL);
    assert_eq!(
        state.attempts[0].body_sha256(),
        hex::encode(Sha256::digest(&body))
    );
}

#[tokio::test]
async fn openai_route_mismatches_reject_and_user_pins_keep_fixed_wire() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let endpoint = format!("http://{}", listener.local_addr().expect("address"));
    let authorizer_state = Arc::new(Mutex::new(AuthorizerState::default()));
    let authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::clone(&authorizer_state),
    });
    let provider = provider(endpoint);

    assert_eq!(
        provider.dispatch_route("gpt-5.4", EffortLevel::Medium),
        Err(DispatchDenied::UnsupportedRoute)
    );

    // A user-pinned/fixed request remains eligible for its normal base wire;
    // it simply receives no adaptive authorization attempt.
    let server = tokio::spawn(serve_error(listener));
    assert!(provider
        .create_message(request(None, json!({"reasoningEffort": "high"})))
        .await
        .is_err());
    let body = server.await.expect("normal request server");
    let body_value: Value = serde_json::from_slice(&body).expect("normal JSON body");
    assert_eq!(body_value["model"], MODEL);
    assert_eq!(body_value["stream"], false);
    assert_eq!(body_value["max_completion_tokens"], 1_024);
    assert!(body_value.get("max_tokens").is_none());
    assert_eq!(body_value["reasoning_effort"], "high");
    assert_eq!(
        authorizer_state
            .lock()
            .expect("authorizer lock")
            .attempts
            .len(),
        0
    );

    // A model/body mismatch still fails before opening a transport.
    let mismatch_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mismatch loopback");
    let mismatch_endpoint = format!(
        "http://{}",
        mismatch_listener.local_addr().expect("mismatch address")
    );
    let mismatch_provider = self::provider(mismatch_endpoint);
    let dispatch = dispatch_for(&mismatch_provider, authorizer, EffortLevel::Medium);
    let mut mismatched = request(Some(dispatch), json!({}));
    mismatched.model = "gpt-5.4".into();
    assert!(matches!(
        mismatch_provider.create_message(mismatched).await,
        Err(ProviderError::DispatchRejected { .. })
    ));

    assert!(
        tokio::time::timeout(Duration::from_millis(100), mismatch_listener.accept())
            .await
            .is_err(),
        "mismatched routes must open no transport"
    );
    assert!(
        authorizer_state
            .lock()
            .expect("authorizer lock")
            .attempts
            .is_empty(),
        "fixed and mismatched requests must consume no adaptive authority"
    );
}

#[tokio::test]
async fn openai_final_body_mismatch_is_rejected_before_transport() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let endpoint = format!("http://{}", listener.local_addr().expect("address"));
    let authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::new(Mutex::new(AuthorizerState::default())),
    });
    let provider = provider(endpoint);
    let dispatch = dispatch_for(&provider, authorizer, EffortLevel::Medium);

    let result = provider
        .create_message(request(Some(dispatch), json!({"reasoningEffort": "high"})))
        .await;
    assert!(matches!(
        result,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::RouteMismatch,
            ..
        })
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "final body mismatch must open no transport"
    );
}

#[tokio::test]
async fn openai_negative_authority_matrix_has_zero_transport_and_authorization() {
    // A route minted by a different provider instance must fail before the
    // test transport or the opaque authorizer can observe an attempt.
    let foreign_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("foreign loopback");
    let foreign_endpoint = format!(
        "http://{}",
        foreign_listener.local_addr().expect("foreign address")
    );
    let foreign_state = Arc::new(Mutex::new(AuthorizerState::default()));
    let foreign_authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::clone(&foreign_state),
    });
    let minted_by = provider(foreign_endpoint.clone());
    let foreign_provider = provider(foreign_endpoint);
    let foreign_dispatch = dispatch_for(
        &minted_by,
        Arc::clone(&foreign_authorizer),
        EffortLevel::Medium,
    );
    assert!(matches!(
        foreign_provider
            .create_message(request(Some(foreign_dispatch), json!({})))
            .await,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::RouteMismatch,
            ..
        })
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), foreign_listener.accept())
            .await
            .is_err(),
        "foreign provider instance opened transport"
    );
    assert!(foreign_state
        .lock()
        .expect("foreign authorizer lock")
        .attempts
        .is_empty());

    // Revalidation against a replacement policy invalidates an old route even
    // when model, endpoint, and credentials are otherwise unchanged.
    let stale_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("stale loopback");
    let stale_endpoint = format!(
        "http://{}",
        stale_listener.local_addr().expect("stale address")
    );
    let stale_state = Arc::new(Mutex::new(AuthorizerState::default()));
    let stale_authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::clone(&stale_state),
    });
    let stale_provider = provider(stale_endpoint.clone());
    let revalidated_provider = provider_with_policy(stale_endpoint, policy_with_revision(20));
    let stale_dispatch = dispatch_for(
        &stale_provider,
        Arc::clone(&stale_authorizer),
        EffortLevel::Medium,
    );
    assert!(matches!(
        revalidated_provider
            .create_message(request(Some(stale_dispatch), json!({})))
            .await,
        Err(ProviderError::DispatchRejected {
            error: DispatchDenied::RouteMismatch,
            ..
        })
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), stale_listener.accept())
            .await
            .is_err(),
        "revalidated stale proof opened transport"
    );
    assert!(stale_state
        .lock()
        .expect("stale authorizer lock")
        .attempts
        .is_empty());

    // Every route-critical mutation is rejected before request execution and
    // before the one-shot authorizer is called.
    let mutations = [
        ("output cap", json!({"max_completion_tokens": 1_025})),
        ("reasoning effort", json!({"reasoningEffort": "high"})),
        ("stream flag", json!({"stream": true})),
    ];
    for (label, options) in mutations {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mutation loopback");
        let endpoint = format!(
            "http://{}",
            listener.local_addr().expect("mutation address")
        );
        let state = Arc::new(Mutex::new(AuthorizerState::default()));
        let authorizer = Arc::new(FixtureAuthorizer {
            state: Arc::clone(&state),
        });
        let provider = provider(endpoint);
        let dispatch = dispatch_for(&provider, Arc::clone(&authorizer), EffortLevel::Medium);
        let changed = request(Some(dispatch), options);
        let result = tokio::time::timeout(Duration::from_secs(2), provider.create_message(changed))
            .await
            .expect("negative dispatch must reject without waiting for network I/O");
        assert!(
            matches!(result, Err(ProviderError::DispatchRejected { .. })),
            "{label} mutation unexpectedly reached provider"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err(),
            "{label} mutation opened transport"
        );
        assert!(state
            .lock()
            .expect("mutation authorizer lock")
            .attempts
            .is_empty());
    }

    // Invalid authentication header construction must fail before admission or
    // dispatch authorization. The local listener proves no request escaped.
    let malformed_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("malformed loopback");
    let malformed_endpoint = format!(
        "http://{}",
        malformed_listener.local_addr().expect("malformed address")
    );
    let malformed_state = Arc::new(Mutex::new(AuthorizerState::default()));
    let malformed_authorizer = Arc::new(FixtureAuthorizer {
        state: Arc::clone(&malformed_state),
    });
    let malformed_provider = OpenAiProvider::new("bad\nheader".into())
        .with_test_loopback_base_url(malformed_endpoint)
        .with_dispatch_policy(policy());
    let malformed_dispatch = dispatch_for(
        &malformed_provider,
        Arc::clone(&malformed_authorizer),
        EffortLevel::Medium,
    );
    assert!(matches!(
        malformed_provider
            .create_message(request(Some(malformed_dispatch), json!({})))
            .await,
        Err(ProviderError::Other { .. })
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), malformed_listener.accept())
            .await
            .is_err(),
        "malformed request opened transport"
    );
    assert!(malformed_state
        .lock()
        .expect("malformed authorizer lock")
        .attempts
        .is_empty());
}
