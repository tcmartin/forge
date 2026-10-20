//! Loopback coverage for inference admission at real provider transports.
//!
//! These tests deliberately use two separately constructed leaf providers and
//! a local listener.  They verify that queueing happens before HTTP and that a
//! lease remains held for the complete non-streaming or streaming response.

use std::sync::Arc;
use std::time::Duration;

use crate::client::ClientConfig;
use crate::inference_admission::{AdmissionOptions, InferenceAdmission};
use crate::provider::LlmProvider;
use crate::provider_types::{ProviderRequest, StreamEvent};
use crate::providers::{
    catalog_entry, AnthropicProvider, FreeEntry, FreeProvider, OpenAiCompatProvider, OpenAiProvider,
};
use crate::WireBudget;
use claurst_core::types::ContentBlock;
use futures::StreamExt;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

fn request() -> ProviderRequest {
    ProviderRequest {
        model: "fixture-model".to_string(),
        messages: Vec::new(),
        system_prompt: None,
        tools: Vec::new(),
        max_tokens: 32,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: Vec::new(),
        thinking: None,
        provider_options: json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            WireBudget::finite(8_192, 7_680, 512, 1024 * 1024, crate::MediaAccounting::None)
                .expect("fixture budget"),
        ),
        dispatch: None,
    }
}

fn provider(base_url: String, admission: Arc<InferenceAdmission>) -> OpenAiProvider {
    OpenAiProvider::new("fixture-key".to_string())
        .with_base_url(base_url)
        .with_inference_admission(admission)
}

async fn accept_and_reply(listener: &TcpListener, body: String, content_type: &'static str) {
    let (mut socket, _) = listener.accept().await.expect("fixture request");
    let mut request_bytes = vec![0; 4096];
    let _ = tokio::time::timeout(
        Duration::from_secs(1),
        tokio::io::AsyncReadExt::read(&mut socket, &mut request_bytes),
    )
    .await;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("fixture response");
}

async fn accept_with_status(
    listener: &TcpListener,
    status: &'static str,
    body: String,
    content_type: &'static str,
) {
    let (mut socket, _) = listener.accept().await.expect("fixture request");
    let mut request_bytes = vec![0; 4096];
    let _ = tokio::time::timeout(
        Duration::from_secs(1),
        tokio::io::AsyncReadExt::read(&mut socket, &mut request_bytes),
    )
    .await;
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("fixture response");
}

#[tokio::test]
async fn independently_constructed_providers_share_one_admission_slot() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let first = provider(format!("http://{address}"), Arc::clone(&admission));
    let second = provider(format!("http://{address}"), Arc::clone(&admission));
    let body = json!({
        "id": "fixture",
        "object": "chat.completion",
        "model": "fixture-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
    .to_string();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("first request");
        let mut request_bytes = vec![0; 4096];
        let _ = tokio::time::timeout(
            Duration::from_secs(1),
            tokio::io::AsyncReadExt::read(&mut socket, &mut request_bytes),
        )
        .await;
        let headers =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        socket.write_all(headers.as_bytes()).await.expect("headers");
        let (mut second_socket, _) = listener.accept().await.expect("second request");
        let mut request_bytes = vec![0; 4096];
        let _ = tokio::time::timeout(
            Duration::from_secs(1),
            tokio::io::AsyncReadExt::read(&mut second_socket, &mut request_bytes),
        )
        .await;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        second_socket
            .write_all(response.as_bytes())
            .await
            .expect("second response");
    });

    // Return the first response only after the provider has received headers;
    // the lease is then held by the lazy stream while this body is open.
    let first_stream = tokio::time::timeout(
        Duration::from_secs(1),
        first.create_message_stream(request()),
    )
    .await
    .expect("first stream deadline")
    .expect("first stream");
    let second_task = tokio::spawn(async move { second.create_message(request()).await });
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(
        !second_task.is_finished(),
        "second provider should queue behind the first stream"
    );

    drop(first_stream);
    let response = tokio::time::timeout(Duration::from_secs(1), second_task)
        .await
        .expect("second provider deadline")
        .expect("second task")
        .expect("second response");
    server.await.expect("admission server");
    assert!(matches!(
        response.content.as_slice(),
        [ContentBlock::Text { text }] if text == "ok"
    ));
}

#[tokio::test]
async fn free_fallback_releases_then_reacquires_shared_admission() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let first = OpenAiCompatProvider::new("groq", "Groq fixture", format!("http://{address}"))
        .with_inference_admission(Arc::clone(&admission));
    let second =
        OpenAiCompatProvider::new("cerebras", "Cerebras fixture", format!("http://{address}"))
            .with_inference_admission(Arc::clone(&admission));
    let free = FreeProvider::new(vec![
        FreeEntry {
            upstream: *catalog_entry("groq").expect("groq catalog entry"),
            provider: Arc::new(first),
        },
        FreeEntry {
            upstream: *catalog_entry("cerebras").expect("cerebras catalog entry"),
            provider: Arc::new(second),
        },
    ]);
    let success_body = json!({
        "id": "fixture",
        "object": "chat.completion",
        "model": "qwen-3-235b-a22b-instruct-2507",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "fallback ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
    .to_string();
    let server = tokio::spawn(async move {
        let failure = r#"{"error":{"message":"first upstream failed","type":"server_error"}}"#;
        accept_with_status(
            &listener,
            "500 Internal Server Error",
            failure.to_string(),
            "application/json",
        )
        .await;
        accept_and_reply(&listener, success_body, "application/json").await;
    });
    let mut free_request = request();
    free_request.model = "free/auto".to_string();
    let response = free
        .create_message(free_request)
        .await
        .expect("fallback upstream should succeed");
    server.await.expect("fallback server");
    assert!(matches!(
        response.content.as_slice(),
        [ContentBlock::Text { text }] if text == "fallback ok"
    ));
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("fallback attempts release admission");
    drop(lease);
}

#[tokio::test]
async fn anthropic_wrapper_uses_injected_legacy_client_admission() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let body = concat!(
        "event: message_start\n",
        "data: {\"message\":{\"id\":\"msg_fixture\",\"model\":\"fixture-model\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"anthropic ok\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {}\n\n"
    )
    .to_string();
    let server = tokio::spawn(async move {
        accept_and_reply(&listener, body, "text/event-stream").await;
    });
    let config = ClientConfig {
        api_key: "fixture-key".to_string(),
        api_base: format!("http://{address}"),
        request_timeout: Duration::from_secs(2),
        ..Default::default()
    };
    let client = crate::AnthropicClient::new(config)
        .expect("legacy client")
        .with_inference_admission(Arc::clone(&admission));
    let provider = AnthropicProvider::new(Arc::new(client));
    let response = provider
        .create_message(request())
        .await
        .expect("anthropic wrapper response");
    server.await.expect("anthropic server");
    assert!(matches!(
        response.content.as_slice(),
        [ContentBlock::Text { text }] if text == "anthropic ok"
    ));
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("legacy client releases admission");
    drop(lease);
}

#[tokio::test]
async fn anthropic_terminal_error_releases_admission() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let body = "event: error\ndata: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"fixture overloaded\"}}\n\n".to_string();
    let server = tokio::spawn(async move {
        accept_and_reply(&listener, body, "text/event-stream").await;
    });
    let config = ClientConfig {
        api_key: "fixture-key".to_string(),
        api_base: format!("http://{address}"),
        request_timeout: Duration::from_secs(2),
        ..Default::default()
    };
    let client = crate::AnthropicClient::new(config)
        .expect("legacy client")
        .with_inference_admission(Arc::clone(&admission));
    let provider = AnthropicProvider::new(Arc::new(client));
    let mut stream = provider
        .create_message_stream(request())
        .await
        .expect("anthropic error stream response");
    assert!(matches!(
        stream.next().await,
        Some(Ok(StreamEvent::Error { .. }))
    ));
    server.await.expect("anthropic error server");
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("terminal error releases lease");
    drop(lease);
}

#[tokio::test]
async fn queued_cancellation_sends_zero_http_requests() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let held = admission.acquire().await.expect("held lease");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));

    let queued = tokio::spawn(async move { provider_a.create_message(request()).await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    queued.abort();
    let _ = queued.await;
    drop(held);

    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "a cancelled admission must not reach HTTP"
    );
}

#[tokio::test]
async fn non_streaming_success_and_error_release_the_slot() {
    let success_body = json!({
        "id": "fixture",
        "object": "chat.completion",
        "model": "fixture-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
    .to_string();
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let admission = InferenceAdmission::new(1).expect("capacity");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));
    let server = tokio::spawn(async move {
        accept_and_reply(&listener, success_body, "application/json").await;
    });
    let response = provider_a
        .create_message(request())
        .await
        .expect("success response");
    server.await.expect("success server");
    assert!(matches!(
        response.content.as_slice(),
        [ContentBlock::Text { text }] if text == "ok"
    ));
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("success releases lease");
    drop(lease);

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind error");
    let address = listener.local_addr().expect("address error");
    let provider_b = provider(format!("http://{address}"), Arc::clone(&admission));
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("error request");
        let error = r#"{"error":{"message":"fixture failure","type":"server_error"}}"#;
        let response = format!(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{error}",
            error.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("error response");
    });
    assert!(provider_b.create_message(request()).await.is_err());
    server.await.expect("error server");
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("error releases lease");
    drop(lease);
}

#[tokio::test]
async fn partial_non_streaming_body_keeps_lease_until_body_completion() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));
    let body = json!({
        "id": "fixture",
        "object": "chat.completion",
        "model": "fixture-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "complete"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
    .to_string();
    let split = body.len() / 2;
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let (continue_tx, continue_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("partial request");
        let mut request_bytes = vec![0; 4096];
        let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut request_bytes).await;
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(headers.as_bytes()).await.expect("headers");
        socket
            .write_all(&body.as_bytes()[..split])
            .await
            .expect("partial body");
        socket.flush().await.expect("partial flush");
        ready_tx.send(()).expect("ready receiver");
        continue_rx.await.expect("continue sender");
        socket
            .write_all(&body.as_bytes()[split..])
            .await
            .expect("remainder body");
    });
    let request_task = tokio::spawn(async move { provider_a.create_message(request()).await });
    ready_rx.await.expect("partial response ready");
    assert!(
        admission
            .acquire_with_options(
                AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
            )
            .await
            .is_err(),
        "partial body must retain the active lease"
    );
    continue_tx.send(()).expect("continue receiver");
    let response = request_task
        .await
        .expect("request task")
        .expect("completed response");
    server.await.expect("partial server");
    assert!(matches!(
        response.content.as_slice(),
        [ContentBlock::Text { text }] if text == "complete"
    ));
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("completed body releases lease");
    drop(lease);
}

#[tokio::test]
async fn malformed_non_streaming_body_releases_lease() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));
    let server = tokio::spawn(async move {
        accept_with_status(
            &listener,
            "200 OK",
            "{ malformed".to_string(),
            "application/json",
        )
        .await;
    });
    assert!(provider_a.create_message(request()).await.is_err());
    server.await.expect("malformed server");
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("malformed body releases lease");
    drop(lease);
}

#[tokio::test]
async fn dropping_a_live_stream_releases_admission() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("stream request");
        let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        socket
            .write_all(headers.as_bytes())
            .await
            .expect("stream headers");
        tokio::time::sleep(Duration::from_secs(2)).await;
    });
    let stream = provider_a
        .create_message_stream(request())
        .await
        .expect("stream response");
    drop(stream);
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(100)).expect("options"),
        )
        .await
        .expect("drop releases lease");
    drop(lease);
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn stream_eof_releases_admission_after_transport_close() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("EOF request");
        let mut request_bytes = vec![0; 4096];
        let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut request_bytes).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("EOF headers");
    });
    let mut stream = provider_a
        .create_message_stream(request())
        .await
        .expect("EOF stream response");
    assert!(stream.next().await.is_none(), "stream should reach EOF");
    server.await.expect("EOF server");
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("EOF releases lease");
    drop(lease);
}

#[tokio::test]
async fn stream_transport_error_releases_admission() {
    let admission = InferenceAdmission::new(1).expect("capacity");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("address");
    let provider_a = provider(format!("http://{address}"), Arc::clone(&admission));
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("error request");
        let mut request_bytes = vec![0; 4096];
        let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut request_bytes).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\nx",
            )
            .await
            .expect("error prefix");
    });
    let mut stream = provider_a
        .create_message_stream(request())
        .await
        .expect("error stream response");
    assert!(
        matches!(stream.next().await, Some(Err(_))),
        "truncated body must be an error"
    );
    server.await.expect("error server");
    let lease = admission
        .acquire_with_options(
            AdmissionOptions::with_queue_timeout(Duration::from_millis(50)).expect("options"),
        )
        .await
        .expect("stream error releases lease");
    drop(lease);
}
