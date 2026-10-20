#![cfg(unix)]

use crate::test_support::allow_all_context;
use crate::{execute_authorized, finalize_proof, EditCheckTool, FileWriteTool, ToolContext};
use claurst_api::client::{AnthropicClient, ClientConfig};
use claurst_api::dispatch_policy::{DispatchModelSpec, DispatchPolicy, EffortWireBinding};
use claurst_api::inference_admission::InferenceAdmission;
use claurst_api::streaming::{AnthropicStreamEvent, NullStreamHandler};
use claurst_api::types::{ApiMessage, ThinkingConfig};
use claurst_api::{CreateMessageRequest, DispatchRequest, MediaAccounting, WireBudget};
use claurst_core::crypto_utils::sha256_hex;
use claurst_core::dispatch::{DispatchDenied, LogicalRequestId};
use claurst_core::effort::EffortLevel;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

const MODEL: &str = "claude-proof-fixture";
const CHECKER_COMMAND: &str = "test \"$(cat target)\" = after; printf x >> checker-runs";

fn budget() -> WireBudget {
    WireBudget::finite(32_768, 12_000, 16_384, 1_048_576, MediaAccounting::None)
        .expect("fixture budget")
}

fn client(base_url: String) -> AnthropicClient {
    let policy = Arc::new(
        DispatchPolicy::new(
            1,
            vec![DispatchModelSpec {
                model_id: MODEL.into(),
                efforts: vec![EffortWireBinding::anthropic_thinking(EffortLevel::High)],
            }],
        )
        .expect("fixture policy"),
    );
    AnthropicClient::new(ClientConfig {
        api_key: "fixture-api-key".into(),
        api_base: base_url,
        max_retries: 0,
        request_timeout: Duration::from_secs(3),
        ..Default::default()
    })
    .expect("client")
    .with_inference_admission(InferenceAdmission::new(1).expect("admission"))
    .with_dispatch_policy(policy)
}

async fn proof_context(root: &Path) -> (ToolContext, crate::TrustedVerificationProof) {
    let mut ctx = allow_all_context(root.to_path_buf());
    let registration = crate::TrustedCheckerRegistration::new(crate::RegistrationSpec {
        id: "anthropic-proof-fixture".into(),
        revision: 1,
        source: crate::TrustedCheckerSource::Application,
        workspace: root.to_path_buf(),
        session_id: ctx.session_id.clone(),
        target: root.join("target"),
        operation: crate::TrustedCheckerOperation::Write,
        invocation: crate::TrustedCheckerInvocation::foreground(
            crate::TrustedCheckerShell::Bash,
            CHECKER_COMMAND,
            root,
            Duration::from_secs(5),
        )
        .expect("checker invocation"),
    })
    .expect("checker registration");
    let mut catalog = crate::TrustedCheckerCatalogBuilder::new();
    catalog
        .register(registration)
        .expect("catalog registration");
    ctx.trusted_checker_catalog = Arc::new(catalog.build());

    let mut result = execute_authorized(
        &EditCheckTool,
        json!({
            "operation": "Write",
            "edit": {"file_path": "target", "content": "after"},
            "check": {
                "tool": "Bash",
                "input": {"command": CHECKER_COMMAND, "timeout": 5000}
            }
        }),
        &ctx,
    )
    .await;
    assert!(!result.is_error, "{}", result.content);
    let receipts = result
        .take_application_receipts()
        .expect("edit/check receipts");
    let proof = finalize_proof(&ctx, &receipts)
        .await
        .expect("trusted proof");
    (ctx, proof)
}

fn request(dispatch: DispatchRequest) -> CreateMessageRequest {
    CreateMessageRequest::builder(MODEL, 10_001)
        .messages(vec![ApiMessage {
            role: "user".into(),
            content: Value::String("proof transport fixture".into()),
        }])
        .thinking(ThinkingConfig::enabled(10_000))
        .wire_budget(budget())
        .dispatch(dispatch)
        .build()
}

fn wire_body(mut request: CreateMessageRequest, stream: bool) -> Vec<u8> {
    request.stream = stream;
    let body = serde_json::to_value(request).expect("fixture request value");
    serde_json::to_vec(&body).expect("fixture final wire serialization")
}

fn response_body() -> String {
    format!(
        "{{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"ok\"}}],\"model\":\"{MODEL}\",\"stop_reason\":\"end_turn\",\"usage\":{{\"input_tokens\":1,\"output_tokens\":1}}}}"
    )
}

fn checker_runs(root: &Path) -> Vec<u8> {
    std::fs::read(root.join("checker-runs")).expect("checker run marker")
}

async fn read_request_body(socket: &mut TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    let headers_end;
    let content_length;
    loop {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.expect("request headers");
        assert!(count > 0, "client closed before request headers");
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(position) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            headers_end = position;
            let headers = String::from_utf8_lossy(&bytes[..position]);
            content_length = headers.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("content length"))
            });
            break;
        }
    }
    let content_length = content_length.expect("content length");
    let body_start = headers_end + 4;
    while bytes.len() < body_start + content_length {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.expect("request body");
        assert!(count > 0, "client closed before request body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    bytes[body_start..body_start + content_length].to_vec()
}

async fn write_json(socket: &mut TcpStream) {
    let body = response_body();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("response");
}

async fn write_stream_headers(socket: &mut TcpStream) {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n",
        )
        .await
        .expect("stream headers");
}

async fn write_stream_before_message_stop(socket: &mut TcpStream) {
    socket
        .write_all(
            br#"event: message_start
data: {"type":"message_start","message":{"id":"msg_fixture","model":"claude-proof-fixture","usage":{"input_tokens":1,"output_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}

"#,
        )
        .await
        .expect("pre-terminal SSE events");
}

async fn write_stream_message_stop(socket: &mut TcpStream) {
    socket
        .write_all(b"event: message_stop\ndata: {}\n\n")
        .await
        .expect("terminal SSE event");
}

async fn write_until_success(ctx: &ToolContext, path: &str, content: &str) -> crate::ToolResult {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let result = execute_authorized(
            &FileWriteTool,
            json!({"file_path": path, "content": content}),
            ctx,
        )
        .await;
        if !result.is_error || Instant::now() >= deadline {
            return result;
        }
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
async fn proof_stream_reservation_blocks_same_workspace_writers_until_drop() {
    let root = tempfile::tempdir().expect("tempdir");
    let (ctx, proof) = proof_context(root.path()).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let client = client(format!("http://{address}"));
    let route = client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("route");
    let dispatch = DispatchRequest::new(
        proof.dispatch_permit(&ctx, route.clone()),
        LogicalRequestId::fresh(),
        route,
    );
    let (accepted_tx, accepted_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let body = read_request_body(&mut socket).await;
        accepted_tx.send(body).expect("accepted body");
        write_stream_headers(&mut socket).await;
        let _ = release_rx.await;
    });

    let stream_request = request(dispatch);
    let expected_body = wire_body(stream_request.clone(), true);
    let stream = client
        .create_message_stream(stream_request, Arc::new(NullStreamHandler))
        .await
        .expect("stream request");
    let body = tokio::time::timeout(Duration::from_secs(1), accepted_rx)
        .await
        .expect("accepted request")
        .expect("accepted body");
    assert_eq!(body, expected_body, "wire body changed before transport");
    assert_eq!(sha256_hex(&body), sha256_hex(&expected_body));

    let before = crate::mutation::current_stamp(&ctx);
    let checker_before = checker_runs(root.path());
    let denied_same = execute_authorized(
        &FileWriteTool,
        json!({"file_path": "target", "content": "blocked-same"}),
        &ctx,
    )
    .await;
    assert!(
        denied_same.is_error,
        "same workspace writer escaped reservation"
    );
    let mut foreign = ctx.clone();
    foreign.session_id = "foreign-proof-session".into();
    let foreign_before = crate::mutation::current_stamp(&foreign);
    let denied_foreign = execute_authorized(
        &FileWriteTool,
        json!({"file_path": "foreign", "content": "blocked-foreign"}),
        &foreign,
    )
    .await;
    assert!(
        denied_foreign.is_error,
        "foreign session writer escaped workspace reservation"
    );
    assert_eq!(crate::mutation::current_stamp(&ctx), before);
    assert_eq!(crate::mutation::current_stamp(&foreign), foreign_before);
    assert_eq!(
        std::fs::read(root.path().join("target")).expect("target"),
        b"after"
    );
    assert_eq!(checker_runs(root.path()), checker_before);
    assert!(!root.path().join("foreign").exists());

    drop(stream);
    let released = write_until_success(&ctx, "target", "after-stream").await;
    assert!(
        !released.is_error,
        "writer did not resume: {}",
        released.content
    );
    assert_eq!(
        std::fs::read(root.path().join("target")).expect("target"),
        b"after-stream"
    );
    release_tx.send(()).expect("release stream server");
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("stream server")
        .expect("server task");
}

#[tokio::test]
async fn proof_stream_releases_reservation_at_message_stop_before_eof() {
    let root = tempfile::tempdir().expect("tempdir");
    let (ctx, proof) = proof_context(root.path()).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let client = client(format!("http://{address}"));
    let route = client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("route");
    let dispatch = DispatchRequest::new(
        proof.dispatch_permit(&ctx, route.clone()),
        LogicalRequestId::fresh(),
        route,
    );
    let (accepted_tx, accepted_rx) = oneshot::channel();
    let (preterminal_tx, preterminal_rx) = oneshot::channel();
    let (terminal_tx, terminal_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let body = read_request_body(&mut socket).await;
        accepted_tx.send(body).expect("accepted body");
        write_stream_headers(&mut socket).await;
        write_stream_before_message_stop(&mut socket).await;
        preterminal_tx.send(()).expect("pre-terminal readiness");
        terminal_rx.await.expect("terminal signal");
        write_stream_message_stop(&mut socket).await;
        release_rx.await.expect("release signal");
    });

    let stream_request = request(dispatch);
    let expected_body = wire_body(stream_request.clone(), true);
    let mut receiver = client
        .create_message_stream(stream_request, Arc::new(NullStreamHandler))
        .await
        .expect("stream request");
    let body = tokio::time::timeout(Duration::from_secs(1), accepted_rx)
        .await
        .expect("accepted request")
        .expect("accepted body");
    assert_eq!(body, expected_body, "wire body changed before transport");
    preterminal_rx.await.expect("pre-terminal stream readiness");

    let before = crate::mutation::current_stamp(&ctx);
    let checker_before = checker_runs(root.path());
    let denied_same = execute_authorized(
        &FileWriteTool,
        json!({"file_path": "target", "content": "before-terminal"}),
        &ctx,
    )
    .await;
    let mut foreign = ctx.clone();
    foreign.session_id = "terminal-foreign-session".into();
    let foreign_before = crate::mutation::current_stamp(&foreign);
    let denied_foreign = execute_authorized(
        &FileWriteTool,
        json!({"file_path": "foreign-before-terminal", "content": "blocked"}),
        &foreign,
    )
    .await;
    assert!(
        denied_same.is_error,
        "same-session write escaped before stop"
    );
    assert!(
        denied_foreign.is_error,
        "foreign-session write escaped before stop"
    );
    assert_eq!(crate::mutation::current_stamp(&ctx), before);
    assert_eq!(crate::mutation::current_stamp(&foreign), foreign_before);
    assert_eq!(checker_runs(root.path()), checker_before);

    terminal_tx.send(()).expect("terminal signal");
    let terminal_received = tokio::time::timeout(Duration::from_secs(1), async {
        while let Some(event) = receiver.recv().await {
            if matches!(event, AnthropicStreamEvent::MessageStop) {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    if !terminal_received {
        drop(receiver);
        let _ = release_tx.send(());
        let _ = tokio::time::timeout(Duration::from_secs(1), server).await;
        panic!("message_stop was not observed");
    }

    let same_after = tokio::time::timeout(
        Duration::from_secs(3),
        execute_authorized(
            &FileWriteTool,
            json!({"file_path": "target", "content": "after-terminal"}),
            &ctx,
        ),
    )
    .await
    .expect("same-session write deadline");
    let foreign_after = tokio::time::timeout(
        Duration::from_secs(3),
        execute_authorized(
            &FileWriteTool,
            json!({"file_path": "foreign-after-terminal", "content": "foreign"}),
            &foreign,
        ),
    )
    .await
    .expect("foreign-session write deadline");

    drop(receiver);
    release_tx.send(()).expect("release signal");
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("stream server")
        .expect("server task");

    assert!(
        !same_after.is_error,
        "same-session write remained blocked after message_stop"
    );
    assert!(
        !foreign_after.is_error,
        "foreign-session write remained blocked after message_stop"
    );
    assert_eq!(
        std::fs::read(root.path().join("target")).expect("target"),
        b"after-terminal"
    );
    assert_eq!(
        std::fs::read(root.path().join("foreign-after-terminal")).expect("foreign target"),
        b"foreign"
    );
}

#[tokio::test]
async fn finite_preflight_does_not_consume_proof_dispatch() {
    let root = tempfile::tempdir().expect("tempdir");
    let (ctx, proof) = proof_context(root.path()).await;
    let checker_before = checker_runs(root.path());
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let client = client(format!("http://{address}"));
    let route = client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("route");
    let dispatch = DispatchRequest::new(
        proof.dispatch_permit(&ctx, route.clone()),
        LogicalRequestId::fresh(),
        route,
    );
    for overflow in [false, true] {
        let mut invalid = request(dispatch.clone());
        if overflow {
            invalid.wire_budget = Some(
                WireBudget::finite(16_384, 12_000, 1_024, 1_048_576, MediaAccounting::None)
                    .expect("bounded overflow fixture"),
            );
        } else {
            invalid.wire_budget = None;
        }
        assert!(client.create_message(invalid).await.is_err());
        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err(),
            "preflight failure reached HTTP"
        );
    }

    // A different logical ID must still be usable after rejected preflight.
    // Reusing the old ID alone would also pass for an accidentally consumed retry.
    let valid_dispatch = DispatchRequest::new(
        dispatch.permit().clone(),
        LogicalRequestId::fresh(),
        dispatch.expected_route().clone(),
    );
    let valid_request = request(valid_dispatch);
    let expected_body = wire_body(valid_request.clone(), false);
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let body = read_request_body(&mut socket).await;
        write_json(&mut socket).await;
        body
    });
    let response = client
        .create_message(valid_request)
        .await
        .expect("valid request after preflight");
    assert_eq!(response.model, MODEL);
    let body = tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("server")
        .expect("server task");
    assert_eq!(body, expected_body, "valid request body changed");
    assert_eq!(sha256_hex(&body), sha256_hex(&expected_body));
    assert_eq!(checker_runs(root.path()), checker_before);
}

#[tokio::test]
async fn later_target_mutation_invalidates_exact_retry_before_http() {
    let root = tempfile::tempdir().expect("tempdir");
    let (ctx, proof) = proof_context(root.path()).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let client = client(format!("http://{address}"));
    let route = client
        .dispatch_route(MODEL, EffortLevel::High)
        .expect("route");
    let dispatch = DispatchRequest::new(
        proof.dispatch_permit(&ctx, route.clone()),
        LogicalRequestId::fresh(),
        route,
    );
    let retry_dispatch = dispatch.clone();
    let first_request = request(dispatch);
    let expected_body = wire_body(first_request.clone(), false);
    let (retry_done_tx, retry_done_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("first accept");
        let first_body = read_request_body(&mut socket).await;
        write_json(&mut socket).await;
        // Keep the listener armed for the entire retry. A queued connection is
        // also checked when the client reports that its local denial is done.
        let second = tokio::select! {
            result = listener.accept() => Some(result),
            _ = retry_done_rx => {
                tokio::time::timeout(Duration::from_millis(50), listener.accept()).await.ok()
            }
        };
        let second_http = match second {
            Some(Ok((mut second_socket, _))) => {
                let _ = read_request_body(&mut second_socket).await;
                write_json(&mut second_socket).await;
                true
            }
            Some(Err(error)) => panic!("second accept failed: {error}"),
            None => false,
        };
        (first_body, second_http)
    });
    client
        .create_message(first_request)
        .await
        .expect("first request");
    let mutated = execute_authorized(
        &FileWriteTool,
        json!({"file_path": "target", "content": "later-mutation"}),
        &ctx,
    )
    .await;
    assert!(
        !mutated.is_error,
        "mutation setup failed: {}",
        mutated.content
    );
    let checker_before_retry = checker_runs(root.path());
    let retry = tokio::time::timeout(
        Duration::from_secs(3),
        client.create_message(request(retry_dispatch)),
    )
    .await
    .expect("invalidated retry must return promptly");
    let _ = retry_done_tx.send(());
    assert!(
        matches!(
            retry,
            Err(claurst_core::error::ClaudeError::InferenceDispatchDenied(
                DispatchDenied::ProofInvalidated
            ))
        ),
        "retry did not return typed proof invalidation: {retry:?}"
    );
    assert_eq!(checker_runs(root.path()), checker_before_retry);
    let (body, second_http) = tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("server")
        .expect("server task");
    assert_eq!(body, expected_body, "first request body changed");
    assert_eq!(sha256_hex(&body), sha256_hex(&expected_body));
    assert!(
        !second_http,
        "invalidated retry reached a second HTTP request"
    );
}
