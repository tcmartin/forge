use super::*;
use crate::provider_witness::{
    NativeResponseBinding, PreparedWitnessAttempt, ProviderWireWitnessSeed, WireAttemptOutcome,
    WireEffort,
};
use claurst_core::crypto_utils::sha256_hex;
use claurst_core::types::Message;
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn read_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    loop {
        let mut bytes = [0_u8; 4096];
        let size = socket.read(&mut bytes).await.expect("request read");
        assert!(size > 0, "request closed before its body completed");
        request.extend_from_slice(&bytes[..size]);
        assert!(
            request.len() <= 256 * 1024,
            "request exceeded fixture bound"
        );
        let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("content length"))
            })
            .expect("content-length header");
        let body_start = header_end + 4;
        if request.len() >= body_start + content_length {
            return request[body_start..body_start + content_length].to_vec();
        }
    }
}

async fn collect(
    body: Vec<u8>,
    split_points: Vec<usize>,
) -> (Vec<Result<StreamEvent, ProviderError>>, Value) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let address = listener.local_addr().expect("listener address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("loopback accept");
        let request = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("response headers");
        let mut start = 0;
        for end in split_points.into_iter().chain(std::iter::once(body.len())) {
            assert!(end > start && end <= body.len());
            if socket
                .write_all(format!("{:X}\r\n", end - start).as_bytes())
                .await
                .is_err()
            {
                break;
            }
            if socket.write_all(&body[start..end]).await.is_err()
                || socket.write_all(b"\r\n").await.is_err()
                || socket.flush().await.is_err()
            {
                break;
            }
            tokio::task::yield_now().await;
            start = end;
        }
        let _ = socket.write_all(b"0\r\n\r\n").await;
        serde_json::from_slice(&request).expect("recorded request JSON")
    });

    let provider = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("synthetic-account".into()),
        expires_at: None,
    })
    .with_base_url(format!("http://{address}/responses"));
    let request = ProviderRequest {
        model: "fixture-codex-model".into(),
        messages: vec![Message::user("PRIVATE_CODEX_PROMPT")],
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
                .expect("fixture budget"),
        ),
        dispatch: None,
    };
    let events = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        provider
            .create_message_stream(request)
            .await
            .expect("Codex stream")
            .collect::<Vec<_>>()
            .await
    })
    .await
    .expect("Codex fixture timeout");
    let recorded = server.await.expect("loopback server");
    (events, recorded)
}

fn assert_exact_stream_error(events: &[Result<StreamEvent, ProviderError>], expected: &str) {
    assert!(
        events.iter().any(|event| matches!(
            event,
            Err(ProviderError::StreamError { message, .. }) if message == expected
        )),
        "expected exact stream error {expected:?}, events: {events:?}",
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::MessageStop))),
        "rejected streams must not emit MessageStop: {events:?}",
    );
}

fn sse(events: &[Value]) -> Vec<u8> {
    let mut body = Vec::new();
    for event in events {
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(event.to_string().as_bytes());
        body.extend_from_slice(b"\n\n");
    }
    body
}

async fn redirect_probe(allow_redirect: bool) -> (bool, bool) {
    use std::sync::atomic::{AtomicBool, Ordering};

    let origin = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("origin listener");
    let target = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("redirect target listener");
    let origin_address = origin.local_addr().expect("origin address");
    let target_address = target.local_addr().expect("target address");
    let origin_server = tokio::spawn(async move {
        let (mut socket, _) = origin.accept().await.expect("origin accept");
        read_request(&mut socket).await;
        socket
            .write_all(
                format!(
                    "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{target_address}/redirected\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .expect("redirect response");
    });
    let followed = Arc::new(AtomicBool::new(false));
    let captured = followed.clone();
    let target_server = tokio::spawn(async move {
        let (mut socket, _) = target.accept().await.expect("target accept");
        read_request(&mut socket).await;
        captured.store(true, Ordering::Release);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await
            .expect("target response");
    });
    let mut provider = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("synthetic-account".into()),
        expires_at: None,
    })
    .with_base_url(format!("http://{origin_address}/responses"));
    if allow_redirect {
        provider.http_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(1))
            .build()
            .expect("negative-control client");
    }
    let request = ProviderRequest {
        model: "fixture-codex-model".into(),
        messages: vec![Message::user("PRIVATE_CODEX_PROMPT")],
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
                .expect("fixture budget"),
        ),
        dispatch: None,
    };
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.create_message_stream(request),
    )
    .await;
    origin_server.await.expect("origin server");
    target_server.abort();
    let _ = target_server.await;
    (
        result.expect("bounded redirect probe").is_ok(),
        followed.load(Ordering::Acquire),
    )
}

#[tokio::test]
async fn codex_transport_rejects_unrecorded_redirect_sends() {
    assert_eq!(redirect_probe(false).await, (false, false));
}

#[tokio::test]
async fn redirect_fixture_detects_a_transport_followup() {
    assert_eq!(redirect_probe(true).await, (true, true));
}

#[test]
fn witness_extracts_model_effort_and_digest_from_exact_prepared_bytes() {
    let request = ProviderRequest {
        model: "fixture-codex-model".into(),
        messages: vec![Message::user("PRIVATE_CODEX_PROMPT")],
        system_prompt: None,
        tools: vec![],
        max_tokens: 512,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: json!({"reasoningEffort":"xhigh"}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            crate::WireBudget::finite(8192, 7680, 512, 1024 * 1024, crate::MediaAccounting::None)
                .expect("fixture budget"),
        ),
        dispatch: None,
    };
    let body = CodexProvider::build_responses_body(&request, false);
    let body = prepare_provider_json(
        "codex",
        "https://fixture.invalid/responses",
        true,
        &body,
        &request,
    )
    .expect("prepared request");
    let exact = body.bytes().to_vec();
    let attempt = PreparedWitnessAttempt::from_prepared(&body, 0, false)
        .finish(200, WireAttemptOutcome::Accepted);
    let seed = ProviderWireWitnessSeed::try_new(vec![attempt]).expect("valid witness seed");
    let accepted = seed.accepted_attempt().expect("accepted attempt");
    assert_eq!(accepted.wire_model.as_deref(), Some("fixture-codex-model"));
    assert_eq!(accepted.wire_effort, Some(WireEffort::XHigh));
    assert_eq!(accepted.wire_sha256, sha256_hex(&exact));
    assert_eq!(accepted.wire_bytes, exact.len() as u64);
    assert!(!serde_json::to_string(accepted)
        .expect("witness JSON")
        .contains("PRIVATE_CODEX_PROMPT"));
    let mut response = NativeResponseBinding::default();
    response.observe(
        Some(&json!({
            "id": "response_fixture",
            "model": "fixture-codex-model",
            "reasoning": {"effort": "xhigh"}
        })),
        true,
    );
    let witness = response.finalize(&seed).expect("provider witness");
    let witness = serde_json::to_value(&witness).expect("witness JSON");
    assert_eq!(witness["request_max_output_tokens"], 512);
    assert_eq!(witness["attempts"][0]["wire_max_output_tokens"], 512);
}

#[test]
fn witness_output_cap_rejects_invalid_and_inconsistent_serialized_caps() {
    fn prepared_body(cap: Value, missing: bool) -> crate::provider_wire::PreparedWireRequest {
        let request = ProviderRequest {
            model: "fixture-codex-model".into(),
            messages: vec![Message::user("PRIVATE_CODEX_PROMPT")],
            system_prompt: None,
            tools: vec![],
            max_tokens: 512,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: vec![],
            thinking: None,
            provider_options: json!({"reasoningEffort":"xhigh"}),
            cache_scope: None,
            response_continuation: None,
            wire_budget: Some(
                crate::WireBudget::finite(
                    8192,
                    7680,
                    512,
                    1024 * 1024,
                    crate::MediaAccounting::None,
                )
                .expect("fixture budget"),
            ),
            dispatch: None,
        };
        let mut body = CodexProvider::build_responses_body(&request, false);
        if missing {
            body.as_object_mut()
                .expect("responses body object")
                .remove("max_output_tokens");
        } else {
            body["max_output_tokens"] = cap;
        }
        // Exercise the witness boundary with immutable raw preparation. The
        // provider JSON boundary rejects malformed output options earlier.
        crate::provider_wire::prepare_bytes(
            "codex",
            "https://fixture.invalid/responses",
            true,
            serde_json::to_vec(&body).expect("serialized witness fixture"),
            request.wire_budget.as_ref().expect("fixture budget"),
        )
        .expect("prepared witness fixture")
    }

    for (label, cap, missing) in [
        ("missing", Value::Null, true),
        ("zero", json!(0), false),
        ("fractional", json!(512.5), false),
        ("oversized", json!(u64::from(u32::MAX) + 1), false),
    ] {
        let body = prepared_body(cap, missing);
        let attempt = PreparedWitnessAttempt::from_prepared(&body, 0, false)
            .finish(200, WireAttemptOutcome::Accepted);
        assert!(
            ProviderWireWitnessSeed::try_new(vec![attempt]).is_err(),
            "invalid serialized cap must be rejected: {label}"
        );
    }

    let body = prepared_body(json!(512), false);
    let accepted_continuation = PreparedWitnessAttempt::from_prepared(&body, 0, true)
        .finish(200, WireAttemptOutcome::Accepted);
    assert!(ProviderWireWitnessSeed::try_new(vec![accepted_continuation]).is_ok());

    let mut body = body;
    body.wire_sha256 = "0".repeat(64);
    let mutated = PreparedWitnessAttempt::from_prepared(&body, 0, false)
        .finish(200, WireAttemptOutcome::Accepted);
    assert!(ProviderWireWitnessSeed::try_new(vec![mutated]).is_err());

    let rejected =
        PreparedWitnessAttempt::from_prepared(&prepared_body(json!(512), false), 0, true)
            .finish(400, WireAttemptOutcome::RejectedContinuation);
    let accepted =
        PreparedWitnessAttempt::from_prepared(&prepared_body(json!(512), false), 1, false)
            .finish(200, WireAttemptOutcome::Accepted);
    assert!(ProviderWireWitnessSeed::try_new(vec![rejected, accepted]).is_ok());

    let rejected =
        PreparedWitnessAttempt::from_prepared(&prepared_body(json!(511), false), 0, true)
            .finish(400, WireAttemptOutcome::RejectedContinuation);
    let accepted =
        PreparedWitnessAttempt::from_prepared(&prepared_body(json!(512), false), 1, false)
            .finish(200, WireAttemptOutcome::Accepted);
    assert!(ProviderWireWitnessSeed::try_new(vec![rejected, accepted]).is_err());
}

#[tokio::test]
async fn split_utf8_preserves_text_tool_arguments_usage_and_terminal() {
    let text = "答🙂tail";
    let arguments = json!({"text":"工具🙂value"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_fixture","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"msg_fixture","type":"message"}}),
        json!({"type":"response.content_part.added","output_index":0,"part":{"type":"output_text"}}),
        json!({"type":"response.output_text.delta","output_index":0,"delta":text}),
        json!({"type":"response.output_item.done","output_index":0}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"fc_fixture","call_id":"call_fixture","type":"function_call","name":"Write"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_fixture","delta":arguments}),
        json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"fc_fixture","arguments":arguments}),
        json!({"type":"response.output_item.done","output_index":1}),
        json!({"type":"response.completed","response":{"id":"resp_fixture","model":"fixture-codex-model","usage":{"input_tokens":5,"output_tokens":2,"input_tokens_details":{"cached_tokens":1}}}}),
    ]);
    let splits: Vec<usize> = body
        .windows("🙂".len())
        .enumerate()
        .filter_map(|(index, bytes)| (bytes == "🙂".as_bytes()).then_some(index + 2))
        .collect();
    // The finalized argument payload repeats the emoji in the event stream;
    // retain all three UTF-8 split points in this regression fixture.
    assert_eq!(splits.len(), 3);

    let (events, request) = collect(body, splits).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    let events: Vec<StreamEvent> = events.into_iter().map(Result::unwrap).collect();
    assert!(events.iter().any(
        |event| matches!(event, StreamEvent::TextDelta { text: actual, .. } if actual == text)
    ));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::InputJsonDelta { partial_json, .. } if partial_json == &arguments))
            .count(),
        1,
        "matching finalized arguments must not be emitted a second time",
    );
    assert!(events.iter().any(|event| {
        matches!(event, StreamEvent::MessageDelta { usage: Some(usage), .. } if usage.input_tokens == 5 && usage.output_tokens == 2 && usage.cache_read_input_tokens == 1)
    }));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::MessageStop))
            .count(),
        1
    );
    assert!(request.get("input").and_then(Value::as_array).is_some());
    assert!(request.get("messages").is_none());
    assert_eq!(request["stream"], true);
    assert_eq!(request["max_output_tokens"], 128);
}

#[tokio::test]
async fn completion_only_arguments_are_emitted_for_interleaved_tool_indices() {
    let first = json!({"file_path":"first"}).to_string();
    let second = json!({"command":"printf second"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_completion_only","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":3,"item":{"id":"item_first","call_id":"call_first","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.added","output_index":7,"item":{"id":"item_second","call_id":"call_second","type":"function_call","name":"Bash"}}),
        json!({"type":"response.function_call_arguments.done","output_index":7,"item_id":"item_second","arguments":second}),
        json!({"type":"response.function_call_arguments.done","output_index":3,"item_id":"item_first","arguments":first}),
        json!({"type":"response.output_item.done","output_index":3}),
        json!({"type":"response.output_item.done","output_index":7}),
        json!({"type":"response.completed","response":{"id":"resp_completion_only","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    let events: Vec<StreamEvent> = events.into_iter().map(Result::unwrap).collect();
    let deltas: Vec<(usize, String)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => Some((*index, partial_json.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec![(7, second), (3, first)]);
}

#[tokio::test]
async fn finalized_arguments_must_match_prior_deltas_and_item_identity() {
    let delta = json!({"file_path":"first"}).to_string();
    let mismatch = json!({"file_path":"second"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_mismatch","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"id":"item_read","call_id":"call_read","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":2,"item_id":"item_read","delta":delta}),
        json!({"type":"response.function_call_arguments.done","output_index":2,"item_id":"item_read","arguments":mismatch}),
        json!({"type":"response.completed","response":{"id":"resp_mismatch","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex finalized function_call arguments disagree with deltas",
    );
}

#[tokio::test]
async fn finalized_arguments_reject_foreign_item_identity() {
    let arguments = json!({"file_path":"first"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_foreign_item","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":4,"item":{"id":"item_read","call_id":"call_read","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.done","output_index":4,"item_id":"item_other","arguments":arguments}),
        json!({"type":"response.completed","response":{"id":"resp_foreign_item","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex finalized function_call arguments changed item identity",
    );
}

#[tokio::test]
async fn function_call_without_final_arguments_fails_closed_at_output_completion() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_missing_final","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_missing_final","call_id":"call_missing_final","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"item_missing_final","delta":"{}"}),
        json!({"type":"response.output_item.done","output_index":1}),
        json!({"type":"response.completed","response":{"id":"resp_missing_final","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex function_call item ended before finalized arguments",
    );
}

#[tokio::test]
async fn split_matching_deltas_finalize_once_without_reemitting_arguments() {
    let first = "{".to_string();
    let second = "\"command\":\"split\"}".to_string();
    let arguments = format!("{first}{second}");
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_split_args","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":5,"item":{"id":"item_split","call_id":"call_split","type":"function_call","name":"Bash"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":5,"item_id":"item_split","delta":first}),
        json!({"type":"response.function_call_arguments.delta","output_index":5,"item_id":"item_split","delta":second}),
        json!({"type":"response.function_call_arguments.done","output_index":5,"item_id":"item_split","arguments":arguments}),
        json!({"type":"response.output_item.done","output_index":5,"item":{"id":"item_split","call_id":"call_split","type":"function_call","name":"Bash","arguments":arguments}}),
        json!({"type":"response.completed","response":{"id":"resp_split_args","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    let events: Vec<StreamEvent> = events.into_iter().map(Result::unwrap).collect();
    let deltas: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::InputJsonDelta { partial_json, .. } => Some(partial_json.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec![first, second]);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::MessageStop))
            .count(),
        1
    );
}

#[tokio::test]
async fn output_item_done_full_function_call_is_a_bounded_finalization_fallback() {
    let arguments = json!({"file_path":"empty-or-real"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_item_done","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"id":"item_done","call_id":"call_done","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.done","output_index":2,"item":{"id":"item_done","call_id":"call_done","type":"function_call","name":"Read","arguments":arguments}}),
        json!({"type":"response.completed","response":{"id":"resp_item_done","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    let events: Vec<StreamEvent> = events.into_iter().map(Result::unwrap).collect();
    assert_eq!(
        events.iter().filter(|event| matches!(event, StreamEvent::InputJsonDelta { partial_json, .. } if partial_json == &arguments)).count(),
        1,
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::MessageStop))
            .count(),
        1
    );
}

#[tokio::test]
async fn output_item_done_empty_object_arguments_emit_once_without_double_finalization() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_empty_args","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"item_empty","call_id":"call_empty","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"id":"item_empty","call_id":"call_empty","type":"function_call","name":"Read","arguments":"{}"}}),
        json!({"type":"response.completed","response":{"id":"resp_empty_args","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    let events: Vec<StreamEvent> = events.into_iter().map(Result::unwrap).collect();
    assert_eq!(events.iter().filter(|event| matches!(event, StreamEvent::InputJsonDelta { partial_json, .. } if partial_json == "{}")).count(), 1);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, StreamEvent::MessageStop))
            .count(),
        1
    );
}

#[tokio::test]
async fn output_item_done_rejects_untracked_function_call_with_terminal_completed_fixture() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_untracked_done","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.done","output_index":4,"item":{"id":"item_untracked","call_id":"call_untracked","type":"function_call","name":"Read","arguments":"{}"}}),
        json!({"type":"response.completed","response":{"id":"resp_untracked_done","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex function_call item completion lacks a prior item",
    );
}

#[tokio::test]
async fn output_item_done_rejects_duplicate_completion_with_terminal_completed_fixture() {
    let item = json!({"id":"item_duplicate_done","call_id":"call_duplicate_done","type":"function_call","name":"Read","arguments":"{}"});
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_duplicate_item_done","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":4,"item":{"id":"item_duplicate_done","call_id":"call_duplicate_done","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.done","output_index":4,"item":item}),
        json!({"type":"response.output_item.done","output_index":4,"item":{"id":"item_duplicate_done","call_id":"call_duplicate_done","type":"function_call","name":"Read","arguments":"{}"}}),
        json!({"type":"response.completed","response":{"id":"resp_duplicate_item_done","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex function_call item completion was duplicated",
    );
}

#[tokio::test]
async fn function_call_added_requires_explicit_call_id_with_terminal_completed_fixture() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_missing_call_id","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":4,"item":{"id":"item_missing_call_id","type":"function_call","name":"Read"}}),
        json!({"type":"response.completed","response":{"id":"resp_missing_call_id","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(&events, "Codex function_call item lacks a stable call_id");
}

#[tokio::test]
async fn repeated_identical_arguments_done_is_rejected_even_when_terminal_completed_follows() {
    let arguments = json!({"file_path":"repeat"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_duplicate_done","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_duplicate","call_id":"call_duplicate","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"item_duplicate","arguments":arguments}),
        json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"item_duplicate","arguments":arguments}),
        json!({"type":"response.completed","response":{"id":"resp_duplicate_done","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex finalized function_call arguments were duplicated",
    );
}

#[tokio::test]
async fn missing_output_index_is_rejected_instead_of_defaulting_to_zero() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_missing_index","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":3,"item":{"id":"item_index","call_id":"call_index","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.delta","item_id":"item_index","delta":"{}"}),
        json!({"type":"response.completed","response":{"id":"resp_missing_index","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex function_call_arguments.delta lacks output_index",
    );
}

#[tokio::test]
async fn delta_item_identity_is_bound_and_foreign_delta_is_rejected() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_delta_identity","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":3,"item":{"id":"item_index","call_id":"call_index","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":3,"item_id":"other_item","delta":"{}"}),
        json!({"type":"response.completed","response":{"id":"resp_delta_identity","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex function_call argument delta changed item identity",
    );
}

#[tokio::test]
async fn item_ids_cannot_be_reused_at_different_output_indices() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_reused_item","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"same_item","call_id":"call_one","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"id":"same_item","call_id":"call_two","type":"function_call","name":"Bash"}}),
        json!({"type":"response.completed","response":{"id":"resp_reused_item","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex function_call item id was reused across output indices",
    );
}

#[tokio::test]
async fn function_call_count_bound_is_enforced_before_terminal_completion() {
    let mut events = vec![
        json!({"type":"response.created","response":{"id":"resp_call_bound","model":"fixture-codex-model"}}),
    ];
    for index in 0..=MAX_CODEX_TOOL_CALLS {
        events.push(json!({
            "type":"response.output_item.added",
            "output_index": index,
            "item":{"id":format!("item_bound_{index}"),"call_id":format!("call_bound_{index}"),"type":"function_call","name":"Read"}
        }));
    }
    events.push(json!({"type":"response.completed","response":{"id":"resp_call_bound","model":"fixture-codex-model"}}));
    let (events, _) = collect(sse(&events), vec![]).await;
    assert_exact_stream_error(&events, "Codex function_call count exceeded the bound");
}

#[tokio::test]
async fn argument_bounds_reject_per_call_overflow_before_terminal_stop() {
    let chunk = "x".repeat(MAX_CODEX_TOOL_ARGUMENT_BYTES / 2 + 1);
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_argument_bounds","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_bound_one","call_id":"call_bound_one","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"item_bound_one","delta":chunk}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"item_bound_one","delta":chunk}),
        json!({"type":"response.completed","response":{"id":"resp_argument_bounds","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(&events, "Codex function_call arguments exceeded the bound");
}

#[tokio::test]
async fn argument_bounds_reject_total_overflow_across_indices_before_terminal_stop() {
    let chunk = "x".repeat(MAX_CODEX_TOOL_ARGUMENT_BYTES / 2 + 1);
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_total_bound","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_total_one","call_id":"call_total_one","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"id":"item_total_two","call_id":"call_total_two","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"item_total_one","delta":chunk}),
        json!({"type":"response.function_call_arguments.delta","output_index":2,"item_id":"item_total_two","delta":chunk}),
        json!({"type":"response.completed","response":{"id":"resp_total_bound","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(&events, "Codex function_call arguments exceeded the bound");
}

#[tokio::test]
async fn response_incomplete_is_terminal_and_preserves_usage() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_incomplete","model":"fixture-codex-model"}}),
        json!({"type":"response.incomplete","response":{"id":"resp_incomplete","model":"fixture-codex-model","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":8,"output_tokens":128}}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    assert!(events.iter().any(|event| {
        matches!(event, Ok(StreamEvent::MessageDelta { stop_reason: Some(StopReason::MaxTokens), usage: Some(usage) }) if usage.input_tokens == 8 && usage.output_tokens == 128)
    }));
    assert!(events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::MessageStop))));
}

#[tokio::test]
async fn oversized_partial_line_fails_without_retaining_response_content() {
    const PRIVATE_LINE: &str = "PRIVATE_CODEX_OVERSIZE_CANARY";
    let mut body = b"data: ".to_vec();
    body.extend_from_slice(PRIVATE_LINE.as_bytes());
    body.resize(MAX_CODEX_SSE_LINE_BYTES + 1, b'x');
    let split_points: Vec<usize> = (32 * 1024..body.len()).step_by(32 * 1024).collect();
    let (events, _) = collect(body, split_points).await;
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
    const PRIVATE_EVENT: &str = "PRIVATE_CODEX_EVENT_CANARY";
    let mut body = Vec::new();
    let data_line_bytes = PRIVATE_EVENT.len() + 4096 + 1;
    let line_count = MAX_CODEX_SSE_EVENT_BYTES / data_line_bytes + 2;
    for _ in 0..line_count {
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(PRIVATE_EVENT.as_bytes());
        body.extend(std::iter::repeat_n(b'x', 4096));
        body.push(b'\n');
    }
    body.push(b'\n');
    let split_points: Vec<usize> = (64 * 1024..body.len()).step_by(64 * 1024).collect();
    let (events, _) = collect(body, split_points).await;
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

fn continuity_request() -> ProviderRequest {
    let completed_prefix = vec![
        Message::user("first"),
        Message::assistant("provider response"),
    ];
    let mut request = ProviderRequest {
        model: "fixture-codex-model".into(),
        messages: completed_prefix.clone(),
        system_prompt: Some(crate::provider_types::SystemPrompt::Text(
            "stable system".into(),
        )),
        tools: vec![
            claurst_core::types::ToolDefinition {
                name: "Zulu".into(),
                description: "z".into(),
                input_schema: json!({"type":"object"}),
            },
            claurst_core::types::ToolDefinition {
                name: "Alpha".into(),
                description: "a".into(),
                input_schema: json!({"type":"object"}),
            },
        ],
        max_tokens: 128,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: json!({"reasoningEffort":"low"}),
        cache_scope: Some(
            crate::provider_types::ProviderCacheScope::for_session("workspace", "session")
                .expect("cache scope"),
        ),
        response_continuation: None,
        wire_budget: Some(
            crate::WireBudget::finite(
                32_000,
                31_000,
                1_000,
                1024 * 1024,
                crate::MediaAccounting::None,
            )
            .expect("continuity wire budget"),
        ),
        dispatch: None,
    };
    let fingerprint = request
        .continuity_fingerprint()
        .expect("continuity fingerprint");
    request.response_continuation = Some(
        crate::provider_types::ResponseContinuation::new("resp_1", &completed_prefix, fingerprint)
            .expect("continuation anchor"),
    );
    request.messages.push(Message::user("second"));
    request
}

#[test]
fn stock_codex_body_uses_cache_key_and_full_context_without_response_chaining() {
    let request = continuity_request();
    let body = CodexProvider::build_responses_body(&request, false);

    assert_eq!(body["store"], false);
    assert_eq!(body["prompt_cache_key"].as_str().map(str::len), Some(64));
    assert!(body.get("previous_response_id").is_none());
    assert_eq!(body["tools"][0]["name"], "Alpha");
    assert_eq!(body["tools"][1]["name"], "Zulu");
    assert_eq!(
        body["input"].as_array().expect("full input").len(),
        3,
        "system instructions are not duplicated in the Responses input"
    );
}

#[test]
fn compatible_codex_body_sends_only_validated_delta() {
    let request = continuity_request();
    let body = CodexProvider::build_responses_body(&request, true);

    assert_eq!(body["previous_response_id"], "resp_1");
    assert_eq!(
        body["input"],
        json!([{"role":"user","content":[{"type":"input_text","text":"second"}]}])
    );

    let mut changed = request;
    changed.messages[0] = Message::user("trimmed or changed");
    let full = CodexProvider::build_responses_body(&changed, true);
    assert!(full.get("previous_response_id").is_none());
    assert_eq!(full["input"].as_array().expect("full input").len(), 3);
}

#[test]
fn compatible_route_identity_is_opaque_and_account_endpoint_model_scoped() {
    let provider = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("account-a".into()),
        expires_at: None,
    })
    .with_base_url("https://compatible-a.example/responses")
    .with_response_continuation();
    let same = provider
        .response_continuation_route_identity("model-a")
        .expect("compatible route identity");
    assert_eq!(same.len(), 64);
    assert!(!same.contains("account-a"));
    assert_eq!(
        provider.response_continuation_route_identity("model-a"),
        Some(same.clone())
    );

    let other_account = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("account-b".into()),
        expires_at: None,
    })
    .with_base_url("https://compatible-a.example/responses")
    .with_response_continuation();
    assert_ne!(
        other_account.response_continuation_route_identity("model-a"),
        Some(same.clone())
    );

    let other_endpoint = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("account-a".into()),
        expires_at: None,
    })
    .with_base_url("https://compatible-b.example/responses")
    .with_response_continuation();
    assert_ne!(
        other_endpoint.response_continuation_route_identity("model-a"),
        Some(same.clone())
    );
    assert_ne!(
        provider.response_continuation_route_identity("model-b"),
        Some(same)
    );

    let stock = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("account-a".into()),
        expires_at: None,
    });
    assert_eq!(stock.response_continuation_route_identity("model-a"), None);
}

#[tokio::test]
async fn rejected_continuation_retries_full_context_once_and_disables_route_probe() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener");
    let address = listener.local_addr().expect("listener address");
    let success_body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_recovered","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"msg_recovered","type":"message"}}),
        json!({"type":"response.content_part.added","output_index":0,"part":{"type":"output_text"}}),
        json!({"type":"response.output_text.delta","output_index":0,"delta":"recovered"}),
        json!({"type":"response.output_item.done","output_index":0}),
        json!({"type":"response.completed","response":{"id":"resp_recovered","model":"fixture-codex-model","usage":{"input_tokens":3,"output_tokens":1}}}),
    ]);
    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        for index in 0..3 {
            let (mut socket, _) = listener.accept().await.expect("loopback accept");
            requests.push(
                serde_json::from_slice::<Value>(&read_request(&mut socket).await)
                    .expect("recorded request JSON"),
            );
            if index == 0 {
                let error = br#"{"error":{"code":"previous_response_not_found","param":"previous_response_id"}}"#;
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            error.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .expect("error headers");
                socket.write_all(error).await.expect("error body");
            } else {
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            success_body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .expect("success headers");
                socket.write_all(&success_body).await.expect("success body");
            }
        }
        requests
    });

    let provider = CodexProvider::new(CodexTokens {
        access_token: "synthetic-codex-token".into(),
        refresh_token: None,
        account_id: Some("synthetic-account".into()),
        expires_at: None,
    })
    .with_base_url(format!("http://{address}/responses"))
    .with_response_continuation();

    for query_index in 0..2 {
        let events = provider
            .create_message_stream(continuity_request())
            .await
            .expect("bounded continuation fallback")
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "events: {events:?}");
        assert!(events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::MessageStop))));
        let witnesses: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Ok(StreamEvent::ProviderWireWitness(witness)) => Some(witness),
                _ => None,
            })
            .collect();
        assert_eq!(witnesses.len(), 1);
        let attempts = &witnesses[0].attempts;
        if query_index == 0 {
            assert_eq!(attempts.len(), 2);
            assert!(matches!(
                attempts[0].outcome,
                WireAttemptOutcome::RejectedContinuation
            ));
            assert!(attempts[0].continuation);
        } else {
            assert_eq!(attempts.len(), 1);
        }
        let accepted = attempts.last().expect("accepted attempt");
        assert!(matches!(accepted.outcome, WireAttemptOutcome::Accepted));
        assert!(!accepted.continuation);
    }

    let requests = server.await.expect("loopback server");
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["previous_response_id"], "resp_1");
    assert!(requests[1].get("previous_response_id").is_none());
    assert!(requests[2].get("previous_response_id").is_none());
    assert_eq!(requests[1]["input"], requests[2]["input"]);
}

#[tokio::test]
async fn completion_only_arguments_are_emitted_for_interleaved_tool_indices_root_red() {
    let first = json!({"file_path":"first"}).to_string();
    let second = json!({"command":"printf second"}).to_string();
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_completion_only","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":3,"item":{"id":"item_first","call_id":"call_first","type":"function_call","name":"Read"}}),
        json!({"type":"response.output_item.added","output_index":7,"item":{"id":"item_second","call_id":"call_second","type":"function_call","name":"Bash"}}),
        json!({"type":"response.function_call_arguments.done","output_index":7,"item_id":"item_second","arguments":second}),
        json!({"type":"response.function_call_arguments.done","output_index":3,"item_id":"item_first","arguments":first}),
        json!({"type":"response.output_item.done","output_index":3}),
        json!({"type":"response.output_item.done","output_index":7}),
        json!({"type":"response.completed","response":{"id":"resp_completion_only","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    let events: Vec<StreamEvent> = events.into_iter().map(Result::unwrap).collect();
    let deltas: Vec<(usize, String)> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => Some((*index, partial_json.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec![(7, second), (3, first)]);
}

#[tokio::test]
async fn output_item_done_rejects_cross_type_reuse_and_untracked_function_call() {
    let cases = [
        (
            "Codex function_call item completion lacks a prior item",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_untracked","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.done","output_index":4,"item":{"id":"item_untracked","call_id":"call_untracked","type":"function_call","name":"Read","arguments":"{}"}}),
                json!({"type":"response.completed","response":{"id":"resp_untracked","model":"fixture-codex-model"}}),
            ]),
        ),
        (
            "Codex output index was reused across output items",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_text_function","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":2,"item":{"id":"message_two","type":"message"}}),
                json!({"type":"response.output_item.added","output_index":2,"item":{"id":"call_two","call_id":"call_two","type":"function_call","name":"Read"}}),
                json!({"type":"response.completed","response":{"id":"resp_text_function","model":"fixture-codex-model"}}),
            ]),
        ),
        (
            "Codex text content part reused a function_call output index",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_function_text","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":2,"item":{"id":"call_three","call_id":"call_three","type":"function_call","name":"Read"}}),
                json!({"type":"response.content_part.added","output_index":2,"part":{"type":"output_text"}}),
                json!({"type":"response.completed","response":{"id":"resp_function_text","model":"fixture-codex-model"}}),
            ]),
        ),
    ];
    for (expected, body) in cases {
        let (events, _) = collect(body, vec![]).await;
        assert_exact_stream_error(&events, expected);
    }
}

#[tokio::test]
async fn function_call_call_id_reuse_and_malformed_arguments_fail_at_terminal_completion() {
    let cases = [
        (
            "Codex function_call call_id was reused across output indices",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_call_reuse","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_one","call_id":"same_call","type":"function_call","name":"Read"}}),
                json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"item_one","arguments":"{}"}),
                json!({"type":"response.output_item.added","output_index":2,"item":{"id":"item_two","call_id":"same_call","type":"function_call","name":"Read"}}),
                json!({"type":"response.completed","response":{"id":"resp_call_reuse","model":"fixture-codex-model"}}),
            ]),
        ),
        (
            "Codex finalized function_call arguments were not a JSON object",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_bad_args","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_bad","call_id":"call_bad","type":"function_call","name":"Read"}}),
                json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"item_bad","arguments":"[]"}),
                json!({"type":"response.completed","response":{"id":"resp_bad_args","model":"fixture-codex-model"}}),
            ]),
        ),
    ];
    for (expected, body) in cases {
        let (events, _) = collect(body, vec![]).await;
        assert_exact_stream_error(&events, expected);
    }
}

#[tokio::test]
async fn reasoning_output_items_preserve_text_completion() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_reasoning","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"reasoning_first","type":"reasoning","summary":[]}}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"id":"reasoning_first","type":"reasoning","summary":[]}}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"id":"message_next","type":"message"}}),
        json!({"type":"response.output_text.delta","output_index":1,"delta":"complete"}),
        json!({"type":"response.output_item.done","output_index":1}),
        json!({"type":"response.completed","response":{"id":"resp_reasoning","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    assert!(events.iter().any(
        |event| matches!(event, Ok(StreamEvent::TextDelta { text, .. }) if text == "complete")
    ));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Ok(StreamEvent::MessageStop)))
            .count(),
        1
    );
}

#[tokio::test]
async fn empty_and_malformed_final_argument_strings_fail_closed() {
    for arguments in ["", "{", "[]", "null"] {
        let body = sse(&[
            json!({"type":"response.created","response":{"id":"resp_invalid_final","model":"fixture-codex-model"}}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"id":"item_invalid_final","call_id":"call_invalid_final","type":"function_call","name":"Read"}}),
            json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"item_invalid_final","arguments":arguments}),
            json!({"type":"response.completed","response":{"id":"resp_invalid_final","model":"fixture-codex-model"}}),
        ]);
        let (events, _) = collect(body, vec![]).await;
        assert_exact_stream_error(
            &events,
            "Codex finalized function_call arguments were not a JSON object",
        );
    }
}

#[tokio::test]
async fn output_item_state_is_bounded_for_non_tool_items() {
    let mut events = vec![
        json!({"type":"response.created","response":{"id":"resp_output_bound","model":"fixture-codex-model"}}),
    ];
    for index in 0..=MAX_CODEX_OUTPUT_ITEMS {
        events.push(json!({"type":"response.output_item.added","output_index":index,"item":{"id":format!("reasoning_{index}"),"type":"reasoning"}}));
    }
    events.push(json!({"type":"response.completed","response":{"id":"resp_output_bound","model":"fixture-codex-model"}}));
    let (events, _) = collect(sse(&events), vec![]).await;
    assert_exact_stream_error(&events, "Codex output item count exceeded the bound");
}

#[tokio::test]
async fn response_incomplete_rejects_tool_even_when_arguments_are_finalized() {
    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_incomplete_tool","model":"fixture-codex-model"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"item_incomplete_tool","call_id":"call_incomplete_tool","type":"function_call","name":"Read"}}),
        json!({"type":"response.function_call_arguments.done","output_index":0,"item_id":"item_incomplete_tool","arguments":"{}"}),
        json!({"type":"response.incomplete","response":{"id":"resp_incomplete_tool","model":"fixture-codex-model","incomplete_details":{"reason":"max_output_tokens"}}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert_exact_stream_error(
        &events,
        "Codex response became incomplete with a function_call",
    );
}

#[tokio::test]
async fn completed_text_and_other_items_cannot_reopen_or_complete_twice() {
    let cases = [
        (
            "Codex output item completion was duplicated",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_duplicate_text","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0,"item":{"id":"message_duplicate","type":"message"}}),
                json!({"type":"response.output_text.delta","output_index":0,"delta":"text"}),
                json!({"type":"response.output_item.done","output_index":0}),
                json!({"type":"response.output_item.done","output_index":0}),
            ]),
        ),
        (
            "Codex text delta arrived after output item completion",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_late_text","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0,"item":{"id":"message_late","type":"message"}}),
                json!({"type":"response.output_text.delta","output_index":0,"delta":"text"}),
                json!({"type":"response.output_item.done","output_index":0}),
                json!({"type":"response.output_text.delta","output_index":0,"delta":"late"}),
            ]),
        ),
        (
            "Codex content part arrived after output item completion",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_late_part","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0,"item":{"id":"message_late_part","type":"message"}}),
                json!({"type":"response.output_text.delta","output_index":0,"delta":"text"}),
                json!({"type":"response.output_item.done","output_index":0}),
                json!({"type":"response.content_part.added","output_index":0,"part":{"type":"output_text"}}),
            ]),
        ),
        (
            "Codex output item completion was duplicated",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_duplicate_reasoning","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0,"item":{"id":"reasoning_duplicate","type":"reasoning","summary":[]}}),
                json!({"type":"response.output_item.done","output_index":0,"item":{"id":"reasoning_duplicate","type":"reasoning","summary":[]}}),
                json!({"type":"response.output_item.done","output_index":0,"item":{"id":"reasoning_duplicate","type":"reasoning","summary":[]}}),
            ]),
        ),
    ];
    for (expected, body) in cases {
        let (events, _) = collect(body, vec![]).await;
        assert_exact_stream_error(&events, expected);
    }
}

#[tokio::test]
async fn text_paths_share_the_output_item_bound() {
    for use_content_part in [false, true] {
        let mut events = vec![
            json!({"type":"response.created","response":{"id":"resp_text_bound","model":"fixture-codex-model"}}),
        ];
        for index in 0..=MAX_CODEX_OUTPUT_ITEMS {
            if use_content_part {
                events.push(json!({"type":"response.content_part.added","output_index":index,"part":{"type":"output_text"}}));
            } else {
                events.push(
                    json!({"type":"response.output_text.delta","output_index":index,"delta":"x"}),
                );
            }
        }
        events.push(json!({"type":"response.completed","response":{"id":"resp_text_bound","model":"fixture-codex-model"}}));
        let (events, _) = collect(sse(&events), vec![]).await;
        assert_exact_stream_error(&events, "Codex output item count exceeded the bound");
    }
}

#[tokio::test]
async fn known_text_events_require_typed_fields_and_unknown_parts_remain_compatible() {
    let cases = [
        (
            "Codex output item event lacks an item object",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_missing_item","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0}),
            ]),
        ),
        (
            "Codex output item event lacks a string type",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_missing_item_type","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0,"item":{}}),
            ]),
        ),
        (
            "Codex content part event lacks a string type",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_missing_part_type","model":"fixture-codex-model"}}),
                json!({"type":"response.content_part.added","output_index":0,"part":{}}),
            ]),
        ),
        (
            "Codex content part event lacks a part object",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_missing_part","model":"fixture-codex-model"}}),
                json!({"type":"response.content_part.added","output_index":0}),
            ]),
        ),
        (
            "Codex text delta event lacks a string delta",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_bad_delta","model":"fixture-codex-model"}}),
                json!({"type":"response.output_text.delta","output_index":0,"delta":7}),
            ]),
        ),
        (
            "Codex function_call argument delta lacks a string delta",
            sse(&[
                json!({"type":"response.created","response":{"id":"resp_bad_argument_delta","model":"fixture-codex-model"}}),
                json!({"type":"response.output_item.added","output_index":0,"item":{"id":"item_bad_delta","call_id":"call_bad_delta","type":"function_call","name":"Read"}}),
                json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"item_bad_delta","delta":7}),
            ]),
        ),
    ];
    for (expected, body) in cases {
        let (events, _) = collect(body, vec![]).await;
        assert_exact_stream_error(&events, expected);
    }

    let body = sse(&[
        json!({"type":"response.created","response":{"id":"resp_unknown_part","model":"fixture-codex-model"}}),
        json!({"type":"response.content_part.added","output_index":0,"part":{"type":"future_output_part"}}),
        json!({"type":"response.completed","response":{"id":"resp_unknown_part","model":"fixture-codex-model"}}),
    ]);
    let (events, _) = collect(body, vec![]).await;
    assert!(events.iter().all(Result::is_ok), "events: {events:?}");
    assert!(events
        .iter()
        .any(|event| matches!(event, Ok(StreamEvent::MessageStop))));
}
