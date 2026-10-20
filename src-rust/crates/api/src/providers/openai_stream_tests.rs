use super::*;
use claurst_core::types::Message;
use futures::StreamExt;
use std::fmt::Write as _;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::instrument::WithSubscriber;
use tracing::{span, Event, Metadata, Subscriber};

const PRIVATE_RESPONSE: &str = "PRIVATE_OPENAI_SSE_CANARY";

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

async fn collect_results(
    body: Vec<u8>,
    split_points: Vec<usize>,
) -> (Vec<Result<StreamEvent, ProviderError>>, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        loop {
            let mut bytes = [0_u8; 4096];
            let size = socket.read(&mut bytes).await.unwrap();
            assert!(size > 0);
            request.extend_from_slice(&bytes[..size]);
            assert!(request.len() < 64 * 1024);
            if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await.unwrap();
        let mut start = 0;
        for end in split_points.into_iter().chain(std::iter::once(body.len())) {
            assert!(end > start && end <= body.len());
            socket
                .write_all(format!("{:X}\r\n", end - start).as_bytes())
                .await?;
            socket.write_all(&body[start..end]).await?;
            socket.write_all(b"\r\n").await?;
            socket.flush().await?;
            // Separate HTTP body frames, including incomplete UTF-8 scalars.
            tokio::task::yield_now().await;
            start = end;
        }
        socket.write_all(b"0\r\n\r\n").await
    });
    let admission = InferenceAdmission::new(1).unwrap();
    let provider = OpenAiProvider::new("synthetic-key".into())
        .with_base_url(format!("http://{address}"))
        .with_inference_admission(Arc::clone(&admission));
    let request = ProviderRequest {
        model: "fixture-model".into(),
        messages: vec![Message::user("PRIVATE_OPENAI_PROMPT")],
        system_prompt: None,
        tools: vec![],
        max_tokens: 128,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            crate::WireBudget::finite(8192, 7680, 512, 1024 * 1024, crate::MediaAccounting::None)
                .unwrap(),
        ),
        dispatch: None,
    };
    let diagnostics = Arc::new(parking_lot::Mutex::new(String::new()));
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        async {
            let mut stream = provider.create_message_stream(request).await.unwrap();
            let mut events = Vec::new();
            while let Some(event) = stream.next().await {
                let terminal_error = event.is_err();
                events.push(event);
                if terminal_error {
                    // Keep the stream alive: the error itself must release
                    // admission before the consumer drops its stream handle.
                    let lease = tokio::time::timeout(
                        std::time::Duration::from_secs(1),
                        admission.acquire(),
                    )
                    .await
                    .expect("terminal stream error retained admission")
                    .unwrap();
                    drop(lease);
                    break;
                }
            }
            events
        }
        .with_subscriber(Diagnostics(Arc::clone(&diagnostics))),
    )
    .await
    .unwrap();
    let server_result = tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("loopback server deadline")
        .unwrap();
    assert!(
        server_result.is_ok() || events.iter().any(Result::is_err),
        "loopback server failed during a successful stream: {server_result:?}"
    );
    let captured = diagnostics.lock().clone();
    (events, captured)
}

async fn collect(body: Vec<u8>, split_points: Vec<usize>) -> (Vec<StreamEvent>, String) {
    let (events, captured) = collect_results(body, split_points).await;
    (events.into_iter().map(Result::unwrap).collect(), captured)
}

#[tokio::test]
async fn split_utf8_preserves_text_tool_arguments_usage_and_terminal() {
    let text = "答🙂tail";
    let arguments = json!({"text":"工具🙂value"}).to_string();
    let first = json!({"id":"fixture","model":"fixture-model","choices":[{"delta":{"content":text},"finish_reason":null}]});
    let second = json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_unicode","function":{"name":"Write","arguments":arguments}}]},"finish_reason":"tool_calls"}]});
    let usage = json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}});
    let body = format!("data: {first}\n\ndata: {second}\n\ndata: {usage}\n\ndata: [DONE]\n\n")
        .into_bytes();
    let positions: Vec<usize> = body
        .windows("🙂".len())
        .enumerate()
        .filter_map(|(index, bytes)| (bytes == "🙂".as_bytes()).then_some(index + 2))
        .collect();
    assert_eq!(positions.len(), 2);
    let (events, _) = collect(body, positions).await;
    assert!(
        events.iter().any(
            |event| matches!(event, StreamEvent::TextDelta { text: actual, .. } if actual == text)
        ),
        "text corrupted: {events:?}"
    );
    assert!(events.iter().any(|event| matches!(event, StreamEvent::InputJsonDelta { partial_json, .. } if partial_json == &arguments)), "tool arguments corrupted: {events:?}");
    assert!(events.iter().any(|event| matches!(event, StreamEvent::MessageDelta { usage: Some(usage), .. } if usage.input_tokens == 5 && usage.output_tokens == 2)));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::MessageStop))
            .count(),
        1
    );
}

#[tokio::test]
async fn malformed_sse_diagnostics_exclude_raw_response() {
    let event =
        json!({"id":"fixture","choices":[{"delta":{"content":"SAFE_OK"},"finish_reason":"stop"}]});
    let body =
        format!("data: {PRIVATE_RESPONSE} invalid-json\n\ndata: {event}\n\ndata: [DONE]\n\n")
            .into_bytes();
    let (events, diagnostics) = collect(body, vec![]).await;
    assert!(events
        .iter()
        .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "SAFE_OK")));
    assert!(
        !diagnostics.contains(PRIVATE_RESPONSE),
        "raw response in diagnostics: {diagnostics}"
    );
    assert!(!diagnostics.contains("PRIVATE_OPENAI_PROMPT"));
    assert!(diagnostics.contains("OpenAI SSE"));
}

#[tokio::test]
async fn oversized_partial_line_fails_without_retaining_response_content() {
    const PRIVATE_LINE: &str = "PRIVATE_OPENAI_OVERSIZE_CANARY";
    let mut body = b"data: ".to_vec();
    body.extend_from_slice(PRIVATE_LINE.as_bytes());
    body.resize(MAX_OPENAI_SSE_LINE_BYTES + 1, b'x');
    let split_points: Vec<usize> = (32 * 1024..body.len()).step_by(32 * 1024).collect();
    let (events, _) = collect_results(body, split_points).await;
    let error = events
        .into_iter()
        .find_map(Result::err)
        .expect("oversized line error");
    match error {
        ProviderError::StreamError {
            message,
            partial_response,
            ..
        } => {
            assert!(
                message.contains("line exceeded"),
                "unexpected error: {message}"
            );
            assert!(!message.contains(PRIVATE_LINE));
            assert!(partial_response.is_none());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn oversized_multiline_event_fails_before_json_parse() {
    const PRIVATE_EVENT: &str = "PRIVATE_OPENAI_EVENT_CANARY";
    let mut body = Vec::new();
    let data_line_bytes = PRIVATE_EVENT.len() + 4096 + 1;
    let line_count = MAX_OPENAI_SSE_EVENT_BYTES / data_line_bytes + 2;
    for _ in 0..line_count {
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(PRIVATE_EVENT.as_bytes());
        body.extend(std::iter::repeat_n(b'x', 4096));
        body.push(b'\n');
    }
    body.push(b'\n');
    let split_points: Vec<usize> = (64 * 1024..body.len()).step_by(64 * 1024).collect();
    let (events, _) = collect_results(body, split_points).await;
    let error = events
        .into_iter()
        .find_map(Result::err)
        .expect("oversized event error");
    match error {
        ProviderError::StreamError {
            message,
            partial_response,
            ..
        } => {
            assert!(
                message.contains("event exceeded"),
                "unexpected error: {message}"
            );
            assert!(!message.contains(PRIVATE_EVENT));
            assert!(partial_response.is_none());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn cumulative_body_limit_rejects_many_small_valid_comment_frames() {
    let mut comment = vec![b'x'; 4096];
    comment[0] = b':';
    comment[4094..].copy_from_slice(b"\n\n");
    let mut body = Vec::with_capacity(MAX_OPENAI_SSE_TOTAL_BYTES + comment.len());
    for _ in 0..=MAX_OPENAI_SSE_TOTAL_BYTES / comment.len() {
        body.extend_from_slice(&comment);
    }
    let splits = (64 * 1024..body.len()).step_by(64 * 1024).collect();
    let (events, _) = collect_results(body, splits).await;
    expect_stream_error(events, "64 MiB cumulative limit");
}

fn expect_stream_error(events: Vec<Result<StreamEvent, ProviderError>>, needle: &str) {
    let error = events
        .into_iter()
        .find_map(Result::err)
        .expect("expected OpenAI stream error");
    match error {
        ProviderError::StreamError {
            message,
            partial_response,
            ..
        } => {
            assert!(message.contains(needle), "unexpected error: {message}");
            assert!(partial_response.is_none());
        }
        other => panic!("unexpected provider error: {other:?}"),
    }
}

#[tokio::test]
async fn malformed_tool_call_indices_and_reused_ids_fail_without_panic() {
    let mut reused_id_body = String::new();
    for index in 0..=256_u64 {
        let event = json!({
            "id": "fixture",
            "model": "fixture-model",
            "choices": [{
                "delta": {"tool_calls": [{
                    "index": index,
                    "id": "same-tool-id",
                    "function": {"name": "Write", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        });
        writeln!(&mut reused_id_body, "data: {event}\n").unwrap();
    }
    reused_id_body.push_str("data: [DONE]\n\n");
    let (events, _) = collect_results(reused_id_body.into_bytes(), vec![]).await;
    expect_stream_error(events, "reused a tool-call id");

    let huge_index = json!({
        "id": "fixture",
        "model": "fixture-model",
        "choices": [{
            "delta": {"tool_calls": [{
                "index": u64::MAX,
                "id": "huge-index",
                "function": {"name": "Write", "arguments": "{}"}
            }]},
            "finish_reason": null
        }]
    });
    let body = format!("data: {huge_index}\n\ndata: [DONE]\n\n").into_bytes();
    let (events, _) = collect_results(body, vec![]).await;
    expect_stream_error(events, "index exceeded");

    for index in [json!(-1), json!(0.5), json!("0")] {
        let invalid_index = json!({"choices": [{"delta": {"tool_calls": [{
            "index": index,
            "id": "invalid-index",
            "function": {"name": "Write", "arguments": "{}"}
        }]}}]});
        let body = format!("data: {invalid_index}\n\ndata: [DONE]\n\n").into_bytes();
        let (events, _) = collect_results(body, vec![]).await;
        expect_stream_error(events, "index was not a nonnegative integer");
    }
}

#[tokio::test]
async fn repeated_tool_call_same_index_preserves_fragments_and_single_start() {
    let first = json!({
        "id": "fixture",
        "model": "fixture-model",
        "choices": [{
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "same-index",
                "function": {"name": "Write", "arguments": "{\"a\":"}
            }]},
            "finish_reason": null
        }]
    });
    let second = json!({
        "choices": [{
            "delta": {"tool_calls": [{
                "index": 0,
                "id": "same-index",
                "function": {"name": "Write", "arguments": "1}"}
            }]},
            "finish_reason": "tool_calls"
        }]
    });
    let body = format!("data: {first}\n\ndata: {second}\n\ndata: [DONE]\n\n").into_bytes();
    let (events, _) = collect(body, vec![]).await;
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::ContentBlockStart { index: 1, .. }))
            .count(),
        1,
        "repeated metadata must not reset or restart the retained block"
    );
    let fragments: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::InputJsonDelta { partial_json, .. } => Some(partial_json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(fragments, vec!["{\"a\":", "1}"]);
}

#[tokio::test]
async fn retained_tool_call_slots_have_independent_bound() {
    let mut body = String::new();
    for index in 0..=256_u64 {
        let event = json!({
            "id": "fixture",
            "model": "fixture-model",
            "choices": [{
                "delta": {"tool_calls": [{
                    "index": index,
                    "id": format!("tool-{index}"),
                    "function": {"name": "Write", "arguments": "{}"}
                }]},
                "finish_reason": null
            }]
        });
        writeln!(&mut body, "data: {event}\n").unwrap();
    }
    body.push_str("data: [DONE]\n\n");
    let (events, _) = collect_results(body.into_bytes(), vec![]).await;
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Ok(StreamEvent::ContentBlockStart { index, .. }) if *index > 0))
            .count(),
        256,
        "every valid tool-call slot must be admitted before the invalid 257th index"
    );
    expect_stream_error(events, "index exceeded the 256-slot limit");
}

#[tokio::test]
async fn cumulative_tool_arguments_fail_before_retention() {
    let fragment = "x".repeat(200_000);
    let mut body = String::new();
    for sequence in 0..43 {
        let event = if sequence == 0 {
            json!({
                "id": "fixture",
                "model": "fixture-model",
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "large-arguments",
                        "function": {"name": "Write", "arguments": fragment}
                    }]},
                    "finish_reason": null
                }]
            })
        } else {
            json!({
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": fragment}
                    }]},
                    "finish_reason": null
                }]
            })
        };
        writeln!(&mut body, "data: {event}\n").unwrap();
    }
    body.push_str("data: [DONE]\n\n");
    let split_points: Vec<usize> = (64 * 1024..body.len()).step_by(64 * 1024).collect();
    let (events, _) = collect_results(body.into_bytes(), split_points).await;
    expect_stream_error(events, "8 MiB tool-argument limit");
}
