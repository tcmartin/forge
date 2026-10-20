use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use claurst_api::client::{AnthropicClient, ClientConfig, Provider};
use claurst_api::dispatch_policy::{DispatchModelSpec, DispatchPolicy, EffortWireBinding};
use claurst_api::inference_admission::InferenceAdmission;
use claurst_api::provider::LlmProvider;
use claurst_api::provider_types::StreamEvent;
use claurst_api::providers::AnthropicProvider;
use claurst_api::streaming::{AnthropicStreamEvent, NullStreamHandler};
use claurst_api::types::{ApiMessage, ThinkingConfig};
use claurst_api::{
    CreateMessageRequest, DispatchRequest, MediaAccounting, ProviderRequest, WireBudget,
};
use claurst_core::dispatch::{
    DispatchAttempt, DispatchAuthorization, DispatchAuthorizer, DispatchDecision, DispatchDenied,
    DispatchPermit, DispatchReservation, DispatchRoute, DispatchRouteSpec, LogicalRequestId,
};
use claurst_core::effort::EffortLevel;
use claurst_core::types::Message;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Notify};

const MODEL: &str = "claude-dispatch-fixture";

fn budget() -> WireBudget {
    WireBudget::finite(32_768, 12_000, 16_384, 1_048_576, MediaAccounting::None)
        .expect("fixture budget")
}

fn policy(revision: u64) -> Arc<DispatchPolicy> {
    Arc::new(
        DispatchPolicy::new(
            revision,
            vec![DispatchModelSpec {
                model_id: MODEL.to_string(),
                efforts: vec![EffortWireBinding::anthropic_thinking(EffortLevel::High)],
            }],
        )
        .expect("fixture policy"),
    )
}

fn json_response() -> &'static str {
    r#"{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-dispatch-fixture","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#
}

fn stream_response() -> &'static str {
    concat!(
        "event: message_start\n",
        "data: {\"message\":{\"id\":\"m1\",\"model\":\"claude-dispatch-fixture\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: message_stop\n\n"
    )
}

#[derive(Clone, Debug)]
struct AttemptRecord {
    route: DispatchRoute,
    logical_request_id: LogicalRequestId,
    body_sha256: String,
    decision: DispatchDecision,
}

#[derive(Clone, Debug)]
struct AttemptBinding {
    route: DispatchRoute,
    logical_request_id: LogicalRequestId,
    body_sha256: String,
}

#[derive(Default)]
struct RecorderState {
    attempts: Vec<AttemptRecord>,
    calls: usize,
    active: usize,
    drops: usize,
    binding: Option<AttemptBinding>,
    expected_route: Option<DispatchRoute>,
    expected_logical_request_id: Option<LogicalRequestId>,
    expected_body_sha256: Option<String>,
    reject: Option<DispatchDenied>,
    reject_while_active: bool,
    drop_notify: Option<Arc<Notify>>,
}

struct HostFakeAuthorizer {
    state: Arc<Mutex<RecorderState>>,
}

struct DropReservation {
    state: Arc<Mutex<RecorderState>>,
}

impl DispatchReservation for DropReservation {}

impl Drop for DropReservation {
    fn drop(&mut self) {
        let (notify, should_notify) = {
            let mut state = self.state.lock().expect("recorder lock");
            state.active = state.active.saturating_sub(1);
            state.drops += 1;
            (state.drop_notify.clone(), state.active == 0)
        };
        if should_notify {
            if let Some(notify) = notify {
                notify.notify_one();
            }
        }
    }
}

impl DispatchAuthorizer for HostFakeAuthorizer {
    fn authorize<'a>(
        &'a self,
        attempt: &'a DispatchAttempt,
    ) -> Pin<Box<dyn Future<Output = Result<DispatchAuthorization, DispatchDenied>> + Send + 'a>>
    {
        Box::pin(async move {
            let mut state = self.state.lock().expect("recorder lock");
            state.calls += 1;
            if let Some(error) = state.reject {
                return Err(error);
            }
            if state.reject_while_active && state.active != 0 {
                return Err(DispatchDenied::InFlight);
            }
            if let Some(expected) = &state.expected_route {
                if expected != attempt.route() {
                    return Err(DispatchDenied::RouteMismatch);
                }
            }
            if let Some(expected) = &state.expected_logical_request_id {
                if expected != attempt.logical_request_id() {
                    return Err(DispatchDenied::InvalidAttempt);
                }
            }
            if let Some(expected) = &state.expected_body_sha256 {
                if expected != attempt.body_sha256() {
                    return Err(DispatchDenied::InvalidAttempt);
                }
            }
            if let Some(binding) = &state.binding {
                if binding.route != *attempt.route()
                    || binding.logical_request_id != *attempt.logical_request_id()
                    || binding.body_sha256 != attempt.body_sha256()
                {
                    return Err(DispatchDenied::InvalidAttempt);
                }
            }
            let decision = if state.attempts.is_empty() {
                DispatchDecision::AuthorizedFresh
            } else {
                DispatchDecision::AuthorizedRetry
            };
            if state.binding.is_none() {
                state.binding = Some(AttemptBinding {
                    route: attempt.route().clone(),
                    logical_request_id: attempt.logical_request_id().clone(),
                    body_sha256: attempt.body_sha256().to_string(),
                });
            }
            state.attempts.push(AttemptRecord {
                route: attempt.route().clone(),
                logical_request_id: attempt.logical_request_id().clone(),
                body_sha256: attempt.body_sha256().to_string(),
                decision,
            });
            state.active += 1;
            Ok(DispatchAuthorization::new(
                decision,
                Box::new(DropReservation {
                    state: Arc::clone(&self.state),
                }),
            ))
        })
    }
}

fn recorder() -> (Arc<HostFakeAuthorizer>, Arc<Mutex<RecorderState>>) {
    let state = Arc::new(Mutex::new(RecorderState::default()));
    (
        Arc::new(HostFakeAuthorizer {
            state: Arc::clone(&state),
        }),
        state,
    )
}

fn client(base_url: String, policy: Arc<DispatchPolicy>) -> AnthropicClient {
    AnthropicClient::new(ClientConfig {
        api_key: "fixture-api-key".into(),
        api_base: base_url,
        max_retries: 1,
        initial_retry_delay: Duration::from_millis(1),
        max_retry_delay: Duration::from_millis(1),
        request_timeout: Duration::from_secs(5),
        ..Default::default()
    })
    .expect("client")
    .with_inference_admission(InferenceAdmission::new(1).expect("admission"))
    .with_dispatch_policy(policy)
}

fn dispatch_for(
    client: &AnthropicClient,
    authorizer: Arc<HostFakeAuthorizer>,
    model: &str,
    effort: EffortLevel,
) -> (DispatchRequest, DispatchRoute) {
    let route = client
        .dispatch_route(model, effort)
        .expect("registered dispatch route");
    let request = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        route.clone(),
    );
    (request, route)
}

fn request(dispatch: Option<DispatchRequest>) -> CreateMessageRequest {
    let mut request = CreateMessageRequest::builder(MODEL, 10_001)
        .messages(vec![ApiMessage {
            role: "user".into(),
            content: serde_json::Value::String("hello".into()),
        }])
        .thinking(ThinkingConfig::enabled(10_000))
        .wire_budget(budget())
        .build();
    request.dispatch = dispatch;
    request
}

fn provider_request(dispatch: Option<DispatchRequest>) -> ProviderRequest {
    ProviderRequest {
        model: MODEL.into(),
        messages: vec![Message::user("hello")],
        system_prompt: None,
        tools: vec![],
        max_tokens: 10_001,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: Some(ThinkingConfig::enabled(10_000)),
        provider_options: serde_json::json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(budget()),
        dispatch,
    }
}

async fn read_http_body(socket: &mut TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    let headers_end;
    let content_length;
    loop {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.expect("read request");
        assert!(count > 0, "client closed before request body");
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(position) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            headers_end = position;
            let headers = String::from_utf8_lossy(&bytes[..position]);
            assert!(headers
                .lines()
                .next()
                .is_some_and(|line| line == "POST /v1/messages HTTP/1.1"));
            assert!(
                headers.lines().any(|line| {
                    line.split_once(':').is_some_and(|(name, value)| {
                        name.eq_ignore_ascii_case("x-api-key") && value.trim() == "fixture-api-key"
                    })
                }),
                "missing fixture API-key header"
            );
            content_length = headers.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("content length"))
            });
            break;
        }
    }
    let content_length = content_length.unwrap_or(0);
    let body_start = headers_end + 4;
    while bytes.len() < body_start + content_length {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.expect("read request body");
        assert!(count > 0, "client closed before complete body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    bytes[body_start..body_start + content_length].to_vec()
}

async fn write_response(socket: &mut TcpStream, content_type: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("response");
}

async fn wait_for_drop(state: &Arc<Mutex<RecorderState>>) {
    let notify = {
        let mut guard = state.lock().expect("recorder lock");
        if guard.active == 0 {
            return;
        }
        let notify = Arc::new(Notify::new());
        guard.drop_notify = Some(Arc::clone(&notify));
        notify
    };
    tokio::time::timeout(Duration::from_secs(1), notify.notified())
        .await
        .expect("dispatch reservation should drop");
    assert_eq!(state.lock().expect("recorder lock").active, 0);
}

fn assert_drop_count(state: &Arc<Mutex<RecorderState>>, expected: usize) {
    let snapshot = state.lock().expect("recorder lock");
    assert_eq!(snapshot.active, 0);
    assert_eq!(snapshot.drops, expected);
}

#[tokio::test]
async fn api_key_anthropic_binds_final_body_hash_and_drops_authorization() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    let client = client(format!("http://{address}"), policy(7));
    let (dispatch, route) = dispatch_for(&client, authorizer.clone(), MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route.clone());

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let body = read_http_body(&mut socket).await;
        write_response(&mut socket, "application/json", json_response()).await;
        body
    });

    let response = client
        .create_message(request(Some(dispatch)))
        .await
        .expect("response");
    assert_eq!(response.model, MODEL);
    let body = server.await.expect("server");
    let hash = format!("{:x}", Sha256::digest(body));
    {
        let snapshot = state.lock().expect("recorder lock");
        assert_eq!(snapshot.attempts.len(), 1);
        assert_eq!(snapshot.calls, 1);
        assert_eq!(snapshot.attempts[0].route, route);
        assert_eq!(snapshot.attempts[0].body_sha256, hash);
        assert_eq!(
            snapshot.attempts[0].decision,
            DispatchDecision::AuthorizedFresh
        );
    }
    wait_for_drop(&state).await;
    assert_drop_count(&state, 1);
}

#[tokio::test]
async fn retry_reauthorizes_same_logical_body_after_partial_429_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    state.lock().expect("recorder lock").reject_while_active = true;
    let client = client(format!("http://{address}"), policy(8));
    let (dispatch, route) = dispatch_for(&client, authorizer, MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route);
    let (partial_tx, partial_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.expect("first accept");
        let first_body = read_http_body(&mut first).await;
        first
            .write_all(b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 10\r\nConnection: keep-alive\r\n\r\nretry")
            .await
            .expect("partial retry response");
        let _ = partial_tx.send(first_body);
        release_rx.await.expect("release partial body");
        first.write_all(b" body").await.expect("finish retry body");
        drop(first);
        let (mut second, _) = listener.accept().await.expect("second accept");
        let second_body = read_http_body(&mut second).await;
        write_response(&mut second, "application/json", json_response()).await;
        second_body
    });

    let call = tokio::spawn(async move { client.create_message(request(Some(dispatch))).await });
    let first_body = tokio::time::timeout(Duration::from_secs(1), partial_rx)
        .await
        .expect("partial response")
        .expect("partial body channel");
    {
        let snapshot = state.lock().expect("recorder lock");
        assert_eq!(snapshot.attempts.len(), 1);
        assert_eq!(snapshot.calls, 1);
        assert_eq!(snapshot.active, 1);
        assert_eq!(snapshot.drops, 0);
    }
    release_tx.send(()).expect("release retry body");
    tokio::time::timeout(Duration::from_secs(2), call)
        .await
        .expect("retry call should complete")
        .expect("call task")
        .expect("retry response");
    let second_body = tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("retry server should complete")
        .expect("server");
    {
        let snapshot = state.lock().expect("recorder lock");
        assert_eq!(snapshot.attempts.len(), 2);
        assert_eq!(snapshot.calls, 2);
        assert_eq!(
            snapshot.attempts[0].logical_request_id,
            snapshot.attempts[1].logical_request_id
        );
        assert_eq!(
            snapshot.attempts[0].body_sha256,
            snapshot.attempts[1].body_sha256
        );
        assert_eq!(
            snapshot.attempts[0].decision,
            DispatchDecision::AuthorizedFresh
        );
        assert_eq!(
            snapshot.attempts[1].decision,
            DispatchDecision::AuthorizedRetry
        );
        assert_eq!(
            snapshot.attempts[0].body_sha256,
            format!("{:x}", Sha256::digest(first_body))
        );
        assert_eq!(
            snapshot.attempts[1].body_sha256,
            format!("{:x}", Sha256::digest(second_body))
        );
    }
    wait_for_drop(&state).await;
    assert_drop_count(&state, 2);
}

#[tokio::test]
async fn nonstream_partial_body_keeps_authorization_until_caller_abort() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    let partial_client = client(format!("http://{address}"), policy(81));
    let (dispatch, route) = dispatch_for(&partial_client, authorizer, MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route);
    let (partial_tx, partial_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = read_http_body(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\nConnection: keep-alive\r\n\r\npartial",
            )
            .await
            .expect("partial body");
        let _ = partial_tx.send(());
        let _ = release_rx.await;
    });
    let call =
        tokio::spawn(async move { partial_client.create_message(request(Some(dispatch))).await });
    tokio::time::timeout(Duration::from_secs(1), partial_rx)
        .await
        .expect("partial body should arrive")
        .expect("partial body signal");
    {
        let snapshot = state.lock().expect("recorder lock");
        assert_eq!(snapshot.attempts.len(), 1);
        assert_eq!(snapshot.calls, 1);
        assert_eq!(snapshot.active, 1);
        assert_eq!(snapshot.drops, 0);
    }
    call.abort();
    let _ = call.await;
    wait_for_drop(&state).await;
    assert_drop_count(&state, 1);
    release_tx.send(()).expect("release partial server");
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("partial server should complete")
        .expect("server");
}

#[tokio::test]
async fn stream_eof_and_receiver_drop_release_dispatch_reservation() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    let eof_client = client(format!("http://{address}"), policy(9));
    let (dispatch, route) = dispatch_for(&eof_client, authorizer, MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route);
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = read_http_body(&mut socket).await;
        write_response(&mut socket, "text/event-stream", stream_response()).await;
    });
    let mut stream = eof_client
        .create_message_stream(request(Some(dispatch)), Arc::new(NullStreamHandler))
        .await
        .expect("stream");
    while stream.recv().await.is_some() {}
    server.await.expect("server");
    wait_for_drop(&state).await;
    assert_drop_count(&state, 1);

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    let silent_client = client(format!("http://{address}"), policy(10));
    let (dispatch, route) = dispatch_for(&silent_client, authorizer, MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route);
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = read_http_body(&mut socket).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n")
            .await
            .expect("silent headers");
        std::future::pending::<()>().await;
    });
    let stream = silent_client
        .create_message_stream(request(Some(dispatch)), Arc::new(NullStreamHandler))
        .await
        .expect("silent stream");
    drop(stream);
    wait_for_drop(&state).await;
    assert_drop_count(&state, 1);
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn stream_terminal_error_releases_dispatch_reservation() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    let client = client(format!("http://{address}"), policy(12));
    let (dispatch, route) = dispatch_for(&client, authorizer, MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route);
    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = read_http_body(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\nevent: error\ndata: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"fixture\"}}\n\n",
            )
            .await
            .expect("error event");
        let _ = release_rx.await;
    });
    let mut stream = client
        .create_message_stream(request(Some(dispatch)), Arc::new(NullStreamHandler))
        .await
        .expect("stream");
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), stream.recv())
            .await
            .expect("error event should arrive"),
        Some(AnthropicStreamEvent::Error { .. })
    ));
    wait_for_drop(&state).await;
    assert_drop_count(&state, 1);
    assert_eq!(state.lock().expect("recorder lock").calls, 1);
    assert!(
        stream.recv().await.is_none(),
        "terminal error should close receiver"
    );
    drop(stream);
    release_tx.send(()).expect("release error server");
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("error server should complete")
        .expect("server");
}

#[tokio::test]
async fn wrapper_preserves_dispatch_and_authorizes_once() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (authorizer, state) = recorder();
    let client = Arc::new(client(format!("http://{address}"), policy(11)));
    let (dispatch, route) = dispatch_for(&client, authorizer, MODEL, EffortLevel::High);
    state.lock().expect("recorder lock").expected_route = Some(route.clone());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let body = read_http_body(&mut socket).await;
        write_response(&mut socket, "text/event-stream", stream_response()).await;
        body
    });
    let provider = AnthropicProvider::new(Arc::clone(&client));
    let response = provider
        .create_message(provider_request(Some(dispatch)))
        .await
        .expect("wrapped response");
    assert_eq!(response.model, MODEL);
    let body = server.await.expect("server");
    {
        let snapshot = state.lock().expect("recorder lock");
        assert_eq!(snapshot.attempts.len(), 1);
        assert_eq!(snapshot.calls, 1);
        assert_eq!(snapshot.attempts[0].route, route);
        assert_eq!(
            snapshot.attempts[0].body_sha256,
            format!("{:x}", Sha256::digest(body))
        );
    }
    wait_for_drop(&state).await;
    assert_drop_count(&state, 1);
}

#[tokio::test]
async fn malformed_preflight_and_unsupported_auth_open_zero_http_and_authorizations() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let route_client = client(format!("http://{address}"), policy(19));
    let route = route_client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("fixture route");
    let cases = [
        ("missing-budget", false, false, false),
        ("overflow", false, true, false),
        ("malformed-header", true, false, false),
        ("oauth", false, false, true),
    ];
    for (name, malformed_header, overflow, oauth) in cases {
        let (authorizer, state) = recorder();
        let base = format!("http://{address}");
        let mut config = ClientConfig {
            api_key: "fixture-api-key".into(),
            api_base: base,
            use_bearer_auth: oauth,
            ..Default::default()
        };
        if malformed_header {
            config.api_version = "bad\nheader".into();
        }
        let client = AnthropicClient::new(config)
            .expect("client")
            .with_dispatch_policy(policy(20));
        let case_route = if oauth {
            route.clone()
        } else {
            client
                .dispatch_route(MODEL, EffortLevel::High)
                .expect("local fixture route")
        };
        let dispatch = DispatchRequest::new(
            DispatchPermit::new(authorizer),
            LogicalRequestId::fresh(),
            case_route,
        );
        let mut req = request(Some(dispatch));
        if name == "missing-budget" {
            req.wire_budget = None;
        }
        if overflow {
            req.wire_budget = Some(
                WireBudget::finite(128, 64, 32, 1, MediaAccounting::None).expect("overflow budget"),
            );
        }
        let result = client.create_message(req).await;
        assert!(result.is_err(), "{name} should fail closed");
        assert!(
            state.lock().expect("recorder lock").attempts.is_empty(),
            "{name} authorized"
        );
        assert_eq!(
            state.lock().expect("recorder lock").calls,
            0,
            "{name} called authorizer"
        );
        assert_drop_count(&state, 0);
    }

    let (authorizer, state) = recorder();
    let invalid = AnthropicClient::new(ClientConfig {
        api_key: "fixture-api-key".into(),
        api_base: "not a URL".into(),
        ..Default::default()
    })
    .expect("client")
    .with_dispatch_policy(policy(21));
    let dispatch = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        route.clone(),
    );
    assert!(invalid
        .create_message(request(Some(dispatch)))
        .await
        .is_err());
    assert!(state.lock().expect("recorder lock").attempts.is_empty());
    assert_eq!(state.lock().expect("recorder lock").calls, 0);
    assert_drop_count(&state, 0);

    let (authorizer, state) = recorder();
    let codex = AnthropicClient::new(ClientConfig {
        api_key: "fixture-api-key".into(),
        api_base: format!("http://{address}"),
        provider: Provider::Codex,
        ..Default::default()
    })
    .expect("client")
    .with_dispatch_policy(policy(22));
    let dispatch = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        route,
    );
    assert!(codex.create_message(request(Some(dispatch))).await.is_err());
    assert!(state.lock().expect("recorder lock").attempts.is_empty());
    assert_eq!(state.lock().expect("recorder lock").calls, 0);
    assert_drop_count(&state, 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn certified_redirect_is_not_followed_or_reauthorized() {
    for (status, reason, revision) in [
        (307_u16, "Temporary Redirect", 25_u64),
        (308_u16, "Permanent Redirect", 26_u64),
    ] {
        let first_listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let first_address = first_listener.local_addr().expect("address");
        let second_listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let second_address = second_listener.local_addr().expect("address");
        let (authorizer, state) = recorder();
        let client = client(format!("http://{first_address}"), policy(revision));
        let (dispatch, route) = dispatch_for(&client, authorizer, MODEL, EffortLevel::High);
        state.lock().expect("recorder lock").expected_route = Some(route);
        let server = tokio::spawn(async move {
            let (mut socket, _) = first_listener.accept().await.expect("first accept");
            let _ = read_http_body(&mut socket).await;
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nLocation: http://{second_address}/v1/messages\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("redirect");
        });
        let result = client.create_message(request(Some(dispatch))).await;
        assert!(result.is_err());
        server.await.expect("first server");
        assert_eq!(state.lock().expect("recorder lock").attempts.len(), 1);
        wait_for_drop(&state).await;
        assert_drop_count(&state, 1);
        assert!(
            tokio::time::timeout(Duration::from_millis(100), second_listener.accept())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn consuming_policy_replacement_rotates_same_client_identity() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let original_client = client(format!("http://{address}"), policy(29));
    let old_route = original_client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("old route");
    let replaced_client = original_client.with_dispatch_policy(policy(29));
    let new_route = replaced_client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("new route");
    assert_ne!(
        old_route.provider_instance_id(),
        new_route.provider_instance_id()
    );
    assert_ne!(old_route.account_identity(), new_route.account_identity());
    let (authorizer, state) = recorder();
    let dispatch = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        old_route,
    );
    let result = replaced_client
        .create_message(request(Some(dispatch)))
        .await;
    assert!(matches!(
        result,
        Err(claurst_core::error::ClaudeError::InferenceDispatchDenied(
            DispatchDenied::RouteMismatch
        ))
    ));
    assert_eq!(state.lock().expect("recorder lock").calls, 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn policy_replacement_and_route_identity_changes_fail_closed() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let original = client(format!("http://{address}"), policy(30));
    let original_route = original
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("route");
    let replaced = client(format!("http://{address}"), policy(31));
    let (authorizer, state) = recorder();
    state.lock().expect("recorder lock").expected_route = Some(
        replaced
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("replacement route"),
    );
    let dispatch = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        original_route,
    );
    assert!(replaced
        .create_message(request(Some(dispatch)))
        .await
        .is_err());
    assert!(state.lock().expect("recorder lock").attempts.is_empty());

    let original = replaced
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("route");
    let altered_routes = [
        DispatchRouteSpec {
            provider_id: "other-provider".into(),
            provider_instance_id: original.provider_instance_id().into(),
            account_identity: original.account_identity().into(),
            model_id: original.model_id().into(),
            endpoint_identity: original.endpoint_identity().into(),
            effort: original.effort().into(),
            policy_revision: original.policy_revision(),
        },
        DispatchRouteSpec {
            provider_id: original.provider_id().into(),
            provider_instance_id: "other-instance".into(),
            account_identity: original.account_identity().into(),
            model_id: original.model_id().into(),
            endpoint_identity: original.endpoint_identity().into(),
            effort: original.effort().into(),
            policy_revision: original.policy_revision(),
        },
        DispatchRouteSpec {
            provider_id: original.provider_id().into(),
            provider_instance_id: original.provider_instance_id().into(),
            account_identity: "other-account".into(),
            model_id: original.model_id().into(),
            endpoint_identity: original.endpoint_identity().into(),
            effort: original.effort().into(),
            policy_revision: original.policy_revision(),
        },
        DispatchRouteSpec {
            provider_id: original.provider_id().into(),
            provider_instance_id: original.provider_instance_id().into(),
            account_identity: original.account_identity().into(),
            model_id: "other-model".into(),
            endpoint_identity: original.endpoint_identity().into(),
            effort: original.effort().into(),
            policy_revision: original.policy_revision(),
        },
        DispatchRouteSpec {
            provider_id: original.provider_id().into(),
            provider_instance_id: original.provider_instance_id().into(),
            account_identity: original.account_identity().into(),
            model_id: original.model_id().into(),
            endpoint_identity: "other-endpoint".into(),
            effort: original.effort().into(),
            policy_revision: original.policy_revision(),
        },
        DispatchRouteSpec {
            provider_id: original.provider_id().into(),
            provider_instance_id: original.provider_instance_id().into(),
            account_identity: original.account_identity().into(),
            model_id: original.model_id().into(),
            endpoint_identity: original.endpoint_identity().into(),
            effort: "other-effort".into(),
            policy_revision: original.policy_revision(),
        },
        DispatchRouteSpec {
            provider_id: original.provider_id().into(),
            provider_instance_id: original.provider_instance_id().into(),
            account_identity: original.account_identity().into(),
            model_id: original.model_id().into(),
            endpoint_identity: original.endpoint_identity().into(),
            effort: original.effort().into(),
            policy_revision: original.policy_revision() + 1,
        },
    ];
    for spec in altered_routes {
        let (authorizer, state) = recorder();
        state.lock().expect("recorder lock").expected_route = Some(original.clone());
        let dispatch = DispatchRequest::new(
            DispatchPermit::new(authorizer),
            LogicalRequestId::fresh(),
            DispatchRoute::new(spec).expect("altered route"),
        );
        assert!(replaced
            .create_message(request(Some(dispatch)))
            .await
            .is_err());
        assert!(state.lock().expect("recorder lock").attempts.is_empty());
    }

    let (authorizer, state) = recorder();
    state.lock().expect("recorder lock").expected_route = Some(original.clone());
    state.lock().expect("recorder lock").expected_body_sha256 = Some("0".repeat(64));
    let dispatch = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        original.clone(),
    );
    assert!(replaced
        .create_message(request(Some(dispatch)))
        .await
        .is_err());
    assert!(state.lock().expect("recorder lock").attempts.is_empty());

    let (authorizer, state) = recorder();
    let expected_id = LogicalRequestId::fresh();
    state.lock().expect("recorder lock").expected_route = Some(original.clone());
    state
        .lock()
        .expect("recorder lock")
        .expected_logical_request_id = Some(expected_id);
    let dispatch = DispatchRequest::new(
        DispatchPermit::new(authorizer),
        LogicalRequestId::fresh(),
        original,
    );
    assert!(replaced
        .create_message(request(Some(dispatch)))
        .await
        .is_err());
    assert!(state.lock().expect("recorder lock").attempts.is_empty());
}

#[tokio::test]
async fn no_dispatch_keeps_normal_api_key_send() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = read_http_body(&mut socket).await;
        write_response(&mut socket, "application/json", json_response()).await;
    });
    let client = AnthropicClient::new(ClientConfig {
        api_key: "fixture-api-key".into(),
        api_base: format!("http://{address}"),
        ..Default::default()
    })
    .expect("client");
    let response = client
        .create_message(request(None))
        .await
        .expect("response");
    assert_eq!(response.model, MODEL);
    server.await.expect("server");
}

#[tokio::test]
async fn message_stop_releases_legacy_lease_before_body_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let admission = InferenceAdmission::new(1).expect("admission");
    let client = AnthropicClient::new(ClientConfig {
        api_key: "fixture-api-key".into(),
        api_base: format!("http://{address}"),
        request_timeout: Duration::from_secs(5),
        ..Default::default()
    })
    .expect("client")
    .with_inference_admission(Arc::clone(&admission))
    .with_dispatch_policy(policy(41));

    let (first_authorizer, first_state) = recorder();
    let (first_dispatch, first_route) =
        dispatch_for(&client, first_authorizer, MODEL, EffortLevel::High);
    first_state.lock().expect("recorder lock").expected_route = Some(first_route);
    let (second_authorizer, second_state) = recorder();
    let (second_dispatch, second_route) =
        dispatch_for(&client, second_authorizer, MODEL, EffortLevel::High);
    second_state.lock().expect("recorder lock").expected_route = Some(second_route);

    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.expect("first accept");
        let _ = read_http_body(&mut first).await;
        first
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n",
            )
            .await
            .expect("stream headers");
        first
            .write_all(stream_response().as_bytes())
            .await
            .expect("terminal stream events");

        // Keep the first response body open while the client attempts the
        // second request. This makes admission release observable without EOF.
        let (mut second, _) = listener.accept().await.expect("second accept");
        let _ = read_http_body(&mut second).await;
        write_response(&mut second, "text/event-stream", stream_response()).await;
        let _ = release_rx.await;
        drop(first);
    });

    let mut first_stream = client
        .create_message_stream(request(Some(first_dispatch)), Arc::new(NullStreamHandler))
        .await
        .expect("first stream");
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), first_stream.recv())
            .await
            .expect("message_start should arrive"),
        Some(AnthropicStreamEvent::MessageStart { .. })
    ));
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), first_stream.recv())
            .await
            .expect("message_stop should arrive"),
        Some(AnthropicStreamEvent::MessageStop)
    ));

    let second_result = tokio::time::timeout(
        Duration::from_secs(1),
        client.create_message_stream(request(Some(second_dispatch)), Arc::new(NullStreamHandler)),
    )
    .await;
    let second_ok = matches!(&second_result, Ok(Ok(_)));
    let mut second_stream = match second_result {
        Ok(Ok(stream)) => Some(stream),
        Ok(Err(_)) | Err(_) => None,
    };
    let first_released_before_eof = {
        let snapshot = first_state.lock().expect("recorder lock");
        snapshot.active == 0 && snapshot.drops == 1
    };

    if let Some(stream) = second_stream.as_mut() {
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), stream.recv())
                .await
                .expect("second message_start should arrive"),
            Some(AnthropicStreamEvent::MessageStart { .. })
        ));
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), stream.recv())
                .await
                .expect("second message_stop should arrive"),
            Some(AnthropicStreamEvent::MessageStop)
        ));
    }

    let _ = release_tx.send(());
    drop(second_stream);
    drop(first_stream);
    let mut server = server;
    if tokio::time::timeout(Duration::from_secs(1), &mut server)
        .await
        .is_err()
    {
        server.abort();
        let _ = server.await;
    }

    assert!(
        first_released_before_eof,
        "first dispatch lease remained held after message_stop"
    );
    assert!(
        second_ok,
        "second request could not acquire the shared admission after message_stop"
    );
    wait_for_drop(&second_state).await;
    assert_drop_count(&second_state, 1);
}

#[tokio::test]
async fn message_stop_releases_provider_wrapper_lease_before_body_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let admission = InferenceAdmission::new(1).expect("admission");
    let client = Arc::new(
        AnthropicClient::new(ClientConfig {
            api_key: "fixture-api-key".into(),
            api_base: format!("http://{address}"),
            request_timeout: Duration::from_secs(5),
            ..Default::default()
        })
        .expect("client")
        .with_inference_admission(Arc::clone(&admission))
        .with_dispatch_policy(policy(42)),
    );
    let provider = AnthropicProvider::new(Arc::clone(&client));

    let (first_authorizer, first_state) = recorder();
    let (first_dispatch, first_route) =
        dispatch_for(&client, first_authorizer, MODEL, EffortLevel::High);
    first_state.lock().expect("recorder lock").expected_route = Some(first_route);
    let (second_authorizer, second_state) = recorder();
    let (second_dispatch, second_route) =
        dispatch_for(&client, second_authorizer, MODEL, EffortLevel::High);
    second_state.lock().expect("recorder lock").expected_route = Some(second_route);

    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.expect("first accept");
        let _ = read_http_body(&mut first).await;
        first
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n",
            )
            .await
            .expect("stream headers");
        first
            .write_all(stream_response().as_bytes())
            .await
            .expect("terminal stream events");

        let (mut second, _) = listener.accept().await.expect("second accept");
        let _ = read_http_body(&mut second).await;
        write_response(&mut second, "text/event-stream", stream_response()).await;
        let _ = release_rx.await;
        drop(first);
    });

    let mut first_stream = provider
        .create_message_stream(provider_request(Some(first_dispatch)))
        .await
        .expect("first provider stream");
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), first_stream.next())
            .await
            .expect("provider message_start should arrive"),
        Some(Ok(StreamEvent::MessageStart { .. }))
    ));
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), first_stream.next())
            .await
            .expect("provider message_stop should arrive"),
        Some(Ok(StreamEvent::MessageStop))
    ));

    let second_result = tokio::time::timeout(
        Duration::from_secs(1),
        provider.create_message_stream(provider_request(Some(second_dispatch))),
    )
    .await;
    let second_ok = matches!(&second_result, Ok(Ok(_)));
    let mut second_stream = match second_result {
        Ok(Ok(stream)) => Some(stream),
        Ok(Err(_)) | Err(_) => None,
    };
    let first_released_before_eof = {
        let snapshot = first_state.lock().expect("recorder lock");
        snapshot.active == 0 && snapshot.drops == 1
    };

    if let Some(stream) = second_stream.as_mut() {
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .expect("second provider message_start should arrive"),
            Some(Ok(StreamEvent::MessageStart { .. }))
        ));
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .expect("second provider message_stop should arrive"),
            Some(Ok(StreamEvent::MessageStop))
        ));
    }

    let _ = release_tx.send(());
    drop(second_stream);
    drop(first_stream);
    let mut server = server;
    if tokio::time::timeout(Duration::from_secs(1), &mut server)
        .await
        .is_err()
    {
        server.abort();
        let _ = server.await;
    }

    assert!(
        first_released_before_eof,
        "provider wrapper retained dispatch lease after message_stop"
    );
    assert!(
        second_ok,
        "provider wrapper could not acquire shared admission after message_stop"
    );
    wait_for_drop(&second_state).await;
    assert_drop_count(&second_state, 1);
}
