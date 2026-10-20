use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use claurst_api::client::{AnthropicClient, ClientConfig};
use claurst_api::inference_admission::InferenceAdmission;
use claurst_api::streaming::{AnthropicStreamEvent, NullStreamHandler};
use claurst_api::types::ApiMessage;
use claurst_api::{CreateMessageRequest, MediaAccounting, WireBudget};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Notify};

fn budget() -> WireBudget {
    WireBudget::finite(16_384, 12_000, 1_024, 1_048_576, MediaAccounting::None).unwrap()
}

fn request() -> CreateMessageRequest {
    CreateMessageRequest::builder("claude-test", 64)
        .messages(vec![ApiMessage {
            role: "user".into(),
            content: serde_json::Value::String("hello".into()),
        }])
        .wire_budget(budget())
        .build()
}

async fn read_request(socket: &mut TcpStream) {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0_u8; 4096];
        let size = socket.read(&mut chunk).await.unwrap();
        assert!(size > 0, "client closed before sending request");
        bytes.extend_from_slice(&chunk[..size]);
        let Some(headers_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&bytes[..headers_end]);
        let content_length = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        });
        if bytes.len() >= headers_end + 4 + content_length.unwrap_or(0) {
            return;
        }
    }
}

fn client(base_url: String, admission: Arc<InferenceAdmission>) -> AnthropicClient {
    AnthropicClient::new(ClientConfig {
        api_key: "test-key".into(),
        api_base: base_url,
        max_retries: 1,
        initial_retry_delay: Duration::from_millis(5),
        max_retry_delay: Duration::from_millis(5),
        request_timeout: Duration::from_secs(5),
        ..Default::default()
    })
    .unwrap()
    .with_inference_admission(admission)
}

const STREAM_RESPONSE: &str = concat!(
    "event: message_start\n",
    "data: {\"message\":{\"id\":\"m1\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
    "event: message_stop\n\n"
);

#[tokio::test]
async fn queued_stream_cancellation_does_not_open_transport() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let accepted = Arc::new(AtomicUsize::new(0));
    let accepted_server = Arc::clone(&accepted);
    let accepted_notify = Arc::new(Notify::new());
    let accepted_notify_server = Arc::clone(&accepted_notify);
    let additional_notify = Arc::new(Notify::new());
    let additional_notify_server = Arc::clone(&additional_notify);
    let (finish_tx, finish_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        accepted_server.fetch_add(1, Ordering::SeqCst);
        accepted_notify_server.notify_one();
        read_request(&mut socket).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n")
            .await
            .unwrap();
        tokio::select! {
            _ = finish_rx => drop(socket),
            result = listener.accept() => {
                let _ = result.unwrap();
                additional_notify_server.notify_one();
            }
        }
    });

    let admission = InferenceAdmission::new(1).unwrap();
    let client = Arc::new(client(format!("http://{address}"), Arc::clone(&admission)));
    let first = tokio::spawn({
        let client = Arc::clone(&client);
        async move {
            client
                .create_message_stream(request(), Arc::new(NullStreamHandler))
                .await
                .unwrap()
        }
    });
    tokio::time::timeout(Duration::from_secs(1), accepted_notify.notified())
        .await
        .expect("first request should reach the loopback server");

    let queued = tokio::spawn({
        let client = Arc::clone(&client);
        async move {
            client
                .create_message_stream(request(), Arc::new(NullStreamHandler))
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    let first_rx = first.await.unwrap();
    drop(first_rx);
    assert!(
        tokio::time::timeout(Duration::from_millis(200), additional_notify.notified())
            .await
            .is_err(),
        "cancelled queued request must not reach HTTP"
    );

    assert_eq!(accepted.load(Ordering::SeqCst), 1);
    let _ = finish_tx.send(());
    server.abort();
}

#[tokio::test]
async fn receiver_drop_during_silent_body_releases_legacy_lease() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let accepted = Arc::new(AtomicUsize::new(0));
    let accepted_server = Arc::clone(&accepted);
    let second_notify = Arc::new(Notify::new());
    let second_notify_server = Arc::clone(&second_notify);
    let server = tokio::spawn(async move {
        let (mut first_socket, _) = listener.accept().await.unwrap();
        accepted_server.fetch_add(1, Ordering::SeqCst);
        read_request(&mut first_socket).await;
        first_socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n")
            .await
            .unwrap();

        let (mut second_socket, _) = listener.accept().await.unwrap();
        accepted_server.fetch_add(1, Ordering::SeqCst);
        second_notify_server.notify_one();
        read_request(&mut second_socket).await;
        second_socket
            .write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", STREAM_RESPONSE.len(), STREAM_RESPONSE).as_bytes())
            .await
            .unwrap();
        drop(first_socket);
    });

    let admission = InferenceAdmission::new(1).unwrap();
    let client = client(format!("http://{address}"), Arc::clone(&admission));
    let first = client
        .create_message_stream(request(), Arc::new(NullStreamHandler))
        .await
        .unwrap();
    drop(first);

    let mut replacement = tokio::time::timeout(
        Duration::from_secs(1),
        client.create_message_stream(request(), Arc::new(NullStreamHandler)),
    )
    .await
    .expect("receiver drop should release the lease")
    .unwrap();
    tokio::time::timeout(Duration::from_secs(1), second_notify.notified())
        .await
        .expect("replacement request should reach the server while first body is open");
    assert!(matches!(
        replacement.recv().await,
        Some(AnthropicStreamEvent::MessageStart { .. })
    ));
    assert!(matches!(
        replacement.recv().await,
        Some(AnthropicStreamEvent::MessageStop)
    ));
    assert_eq!(accepted.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn retry_consumes_error_body_before_one_final_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let requests_server = Arc::clone(&requests);
    let partial_sent = Arc::new(Notify::new());
    let partial_sent_server = Arc::clone(&partial_sent);
    let second_before_release = Arc::new(Notify::new());
    let second_before_release_server = Arc::clone(&second_before_release);
    let (release_tx, release_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first_socket, _) = listener.accept().await.unwrap();
        requests_server.fetch_add(1, Ordering::SeqCst);
        read_request(&mut first_socket).await;
        first_socket
            .write_all(b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 17\r\nConnection: keep-alive\r\n\r\nretry")
            .await
            .unwrap();
        partial_sent_server.notify_one();

        let mut early_second = None;
        tokio::select! {
            _ = release_rx => {}
            result = listener.accept() => {
                let (socket, _) = result.unwrap();
                second_before_release_server.notify_one();
                early_second = Some(socket);
            }
        }
        first_socket.write_all(b" body marker").await.unwrap();
        drop(first_socket);

        let mut second_socket = match early_second {
            Some(socket) => socket,
            None => listener.accept().await.unwrap().0,
        };
        requests_server.fetch_add(1, Ordering::SeqCst);
        read_request(&mut second_socket).await;
        second_socket
            .write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", STREAM_RESPONSE.len(), STREAM_RESPONSE).as_bytes())
            .await
            .unwrap();
    });

    let admission = InferenceAdmission::new(1).unwrap();
    let client = client(format!("http://{address}"), Arc::clone(&admission));
    let request_task = tokio::spawn(async move {
        client
            .create_message_stream(request(), Arc::new(NullStreamHandler))
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(1), partial_sent.notified())
        .await
        .expect("partial retry response should be sent");
    let queued = tokio::time::timeout(
        Duration::from_millis(50),
        admission.acquire_with_options(
            claurst_api::inference_admission::AdmissionOptions::with_queue_timeout(
                Duration::from_millis(40),
            )
            .unwrap(),
        ),
    )
    .await
    .expect("admission queue check should finish");
    assert!(
        matches!(
            queued,
            Err(claurst_api::inference_admission::AdmissionError::QueueTimeout)
        ),
        "retry body must retain the only admission slot"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(50), second_before_release.notified())
            .await
            .is_err(),
        "retry must not open a second request before body release"
    );
    let _ = release_tx.send(());

    let mut stream = tokio::time::timeout(Duration::from_secs(1), request_task)
        .await
        .expect("retry should finish after body release")
        .unwrap();
    let mut seen = Vec::new();
    while let Some(event) = stream.recv().await {
        seen.push(event);
        if matches!(seen.last(), Some(AnthropicStreamEvent::MessageStop)) {
            break;
        }
    }
    assert!(seen
        .iter()
        .any(|event| matches!(event, AnthropicStreamEvent::MessageStart { .. })));
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    server.await.unwrap();
}

#[tokio::test]
async fn terminal_sse_error_releases_legacy_lease_before_receiver_drop() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let second_notify = Arc::new(Notify::new());
    let second_notify_server = Arc::clone(&second_notify);
    let server = tokio::spawn(async move {
        let (mut first_socket, _) = listener.accept().await.unwrap();
        read_request(&mut first_socket).await;
        first_socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\nevent: error\ndata: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"fixture\"}}\n\n",
            )
            .await
            .unwrap();

        // Keep the first response body open while the client retains its
        // receiver. A replacement request proves the terminal error released
        // the admission lease rather than relying on EOF or receiver drop.
        let (mut second_socket, _) = listener.accept().await.unwrap();
        second_notify_server.notify_one();
        read_request(&mut second_socket).await;
        second_socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    STREAM_RESPONSE.len(),
                    STREAM_RESPONSE
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        drop(first_socket);
    });

    let admission = InferenceAdmission::new(1).unwrap();
    let client = client(format!("http://{address}"), Arc::clone(&admission));
    let mut first = client
        .create_message_stream(request(), Arc::new(NullStreamHandler))
        .await
        .unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), first.recv())
            .await
            .expect("terminal SSE error should be delivered"),
        Some(AnthropicStreamEvent::Error { .. })
    ));

    let mut replacement = tokio::time::timeout(
        Duration::from_millis(250),
        client.create_message_stream(request(), Arc::new(NullStreamHandler)),
    )
    .await
    .expect("terminal SSE error must release admission while receiver remains alive")
    .unwrap();
    tokio::time::timeout(Duration::from_secs(1), second_notify.notified())
        .await
        .expect("replacement request should reach the server");
    assert!(matches!(
        replacement.recv().await,
        Some(AnthropicStreamEvent::MessageStart { .. })
    ));
    assert!(matches!(
        replacement.recv().await,
        Some(AnthropicStreamEvent::MessageStop)
    ));
    drop(first);
    server.await.unwrap();
}
