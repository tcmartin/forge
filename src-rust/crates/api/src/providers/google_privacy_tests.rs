use super::*;
use crate::provider_wire::{MediaAccounting, WireBudget};
use std::fmt::Write as _;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::instrument::WithSubscriber;
use tracing::{span, Event, Metadata, Subscriber};

const KEY: &str = "SYNTHETIC_GOOGLE_KEY_CANARY";
const RESPONSE: &str = "PRIVATE_GOOGLE_SSE_CANARY";

#[derive(Clone)]
struct Diagnostics(Arc<parking_lot::Mutex<String>>);

struct Fields(String);
impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, _: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, "{value:?} ");
    }
}

impl Subscriber for Diagnostics {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields(String::new());
        event.record(&mut fields);
        self.0.lock().push_str(&fields.0);
    }
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
}

fn request() -> ProviderRequest {
    ProviderRequest {
        model: "gemini-2.5-flash".into(),
        messages: vec![Message::user("PRIVATE_PROMPT_CANARY")],
        system_prompt: None,
        tools: vec![],
        max_tokens: 512,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            WireBudget::finite(8192, 7680, 512, 1024 * 1024, MediaAccounting::None).unwrap(),
        ),
        dispatch: None,
    }
}

#[tokio::test]
async fn stream_diagnostics_exclude_key_and_response_content() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut received = Vec::new();
        loop {
            let mut chunk = [0_u8; 4096];
            let size = socket.read(&mut chunk).await.unwrap();
            assert!(size > 0);
            received.extend_from_slice(&chunk[..size]);
            assert!(received.len() <= 64 * 1024);
            if let Some(start) = received.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&received[..start]);
                assert!(
                    !headers.lines().next().unwrap().contains(KEY),
                    "credential in HTTP request target"
                );
                assert!(
                    headers.lines().any(|line| {
                        line.split_once(':').is_some_and(|(name, value)| {
                            name.eq_ignore_ascii_case("x-goog-api-key") && value.trim() == KEY
                        })
                    }),
                    "authentication header missing"
                );
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                if received.len() >= start + 4 + length {
                    break;
                }
            }
        }
        let event = json!({"candidates":[{"content":{"parts":[{"text":"GOOGLE_OK"}]},"finishReason":"STOP"}], "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}});
        let body = format!("data: {RESPONSE} not-json\n\ndata: {event}\n\n");
        let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let provider = GoogleProvider::new(KEY.into()).with_base_url(format!("http://{address}/"));
    let captured = Arc::new(parking_lot::Mutex::new(String::new()));
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        async {
            let stream = provider.create_message_stream(request()).await.unwrap();
            stream.collect::<Vec<_>>().await
        }
        .with_subscriber(Diagnostics(Arc::clone(&captured))),
    )
    .await
    .unwrap();
    server.await.unwrap();
    assert!(events.iter().any(
        |event| matches!(event, Ok(StreamEvent::TextDelta { text, .. }) if text == "GOOGLE_OK")
    ));
    let diagnostics = captured.lock();
    assert!(diagnostics.contains("Google SSE: invalid JSON event"));
    assert!(
        !diagnostics.contains(KEY),
        "credential appeared in diagnostics: {diagnostics}"
    );
    assert!(
        !diagnostics.contains(RESPONSE),
        "response appeared in diagnostics: {diagnostics}"
    );
    assert!(!diagnostics.contains("PRIVATE_PROMPT_CANARY"));
}

#[tokio::test]
async fn transport_errors_exclude_credential_url() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let provider = GoogleProvider::new(KEY.into()).with_base_url(format!("http://{address}/"));
    let error = provider.create_message(request()).await.unwrap_err();
    assert!(
        !format!("{error:?}").contains(KEY),
        "credential appeared in error"
    );
    let error = match provider.create_message_stream(request()).await {
        Ok(_) => panic!("closed endpoint accepted"),
        Err(error) => error,
    };
    assert!(
        !format!("{error:?}").contains(KEY),
        "credential appeared in stream error"
    );
    let status = provider.health_check().await.unwrap();
    assert!(
        !format!("{status:?}").contains(KEY),
        "credential appeared in health status"
    );
}
