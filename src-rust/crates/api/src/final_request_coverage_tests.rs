use std::sync::Arc;
use std::time::Duration;

use claurst_core::oauth_config::CodexTokens;
use claurst_core::types::Message;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::client::{AnthropicClient, ClientConfig};
use crate::provider::LlmProvider;
use crate::provider_types::ProviderRequest;
use crate::provider_wire::{
    record_final_requests, MediaAccounting, RecordedFinalRequest, WireBudget,
};
use crate::providers::{
    catalog_entry, AnthropicProvider, AzureProvider, BedrockProvider, CodexProvider,
    CohereProvider, CopilotProvider, FreeEntry, FreeProvider, GoogleProvider, MinimaxProvider,
    OpenAiCompatProvider, OpenAiProvider,
};
use crate::streaming::NullStreamHandler;
use crate::types::CreateMessageRequest;

struct CapturedRequest {
    target: String,
    body: Vec<u8>,
}

fn request(model: &str) -> ProviderRequest {
    ProviderRequest {
        model: model.to_string(),
        messages: vec![Message::user("final request recorder fixture")],
        system_prompt: None,
        tools: Vec::new(),
        max_tokens: 32,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        stop_sequences: Vec::new(),
        thinking: None,
        provider_options: serde_json::json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            WireBudget::finite(100_000, 90_000, 8_000, 1_048_576, MediaAccounting::None)
                .expect("finite fixture budget"),
        ),
        dispatch: None,
    }
}

async fn capture_one(listener: TcpListener, response_status: &'static str) -> CapturedRequest {
    let (mut socket, _) = listener.accept().await.expect("fixture connection");
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut header_end = None;
    let mut content_length = None;

    loop {
        let read = socket
            .read(&mut buffer)
            .await
            .expect("fixture request read");
        assert!(read > 0, "fixture request closed before its body completed");
        raw.extend_from_slice(&buffer[..read]);

        if header_end.is_none() {
            header_end = raw
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4);
            if let Some(end) = header_end {
                let headers = std::str::from_utf8(&raw[..end]).expect("UTF-8 request headers");
                content_length = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length").then(|| {
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("numeric content length")
                    })
                });
            }
        }

        if let (Some(end), Some(length)) = (header_end, content_length) {
            if raw.len() >= end + length {
                break;
            }
        }
    }

    let end = header_end.expect("request header boundary");
    let headers = std::str::from_utf8(&raw[..end]).expect("UTF-8 request headers");
    let target = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("request target")
        .to_string();
    let length = content_length.expect("content length");
    let body = raw[end..end + length].to_vec();

    let response_body = r#"{"error":{"message":"fixture rejection"}}"#;
    let response = format!(
        "HTTP/1.1 {response_status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
        response_body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("fixture response");

    CapturedRequest { target, body }
}

fn endpoint_path(endpoint: &str) -> String {
    let parsed = url::Url::parse(endpoint).expect("recorded endpoint URL");
    parsed.path().to_string()
}

fn assert_record_matches(
    record: &RecordedFinalRequest,
    captured: &CapturedRequest,
    expected_provider: &str,
    expected_stream: bool,
) {
    assert_eq!(record.provider, expected_provider);
    assert_eq!(record.stream, expected_stream);
    assert_eq!(record.body, captured.body);
    assert_eq!(record.wire_bytes, captured.body.len() as u64);
    assert_eq!(
        record.wire_sha256,
        hex::encode(Sha256::digest(&captured.body))
    );
    assert_eq!(record.media, MediaAccounting::None);
    assert!(record.accounting.input_tokens() > 0);
    assert_eq!(
        endpoint_path(&record.endpoint),
        captured
            .target
            .split('?')
            .next()
            .expect("captured target path")
    );
}

async fn assert_adapter_entrypoints<F, S>(
    expected_provider: &str,
    model: &str,
    response_status: &'static str,
    expected_stream: S,
    factory: F,
) where
    F: Fn(&str) -> Box<dyn LlmProvider>,
    S: Fn(bool) -> bool,
{
    for streaming in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("fixture bind");
        let address = listener.local_addr().expect("fixture address");
        let provider = factory(&format!("http://{address}"));
        let capture = tokio::spawn(capture_one(listener, response_status));
        let request = request(model);

        let (result, records) = record_final_requests(async {
            if streaming {
                provider.create_message_stream(request).await.map(|_| ())
            } else {
                provider.create_message(request).await.map(|_| ())
            }
        })
        .await;
        assert!(
            result.is_err(),
            "fixture HTTP failure must reach the adapter"
        );
        let captured = tokio::time::timeout(Duration::from_secs(2), capture)
            .await
            .expect("fixture capture timeout")
            .expect("fixture capture task");
        let matching = records
            .iter()
            .rev()
            .find(|record| record.provider == expected_provider)
            .unwrap_or_else(|| panic!("missing {expected_provider} record: {records:?}"));
        assert_record_matches(
            matching,
            &captured,
            expected_provider,
            expected_stream(streaming),
        );
    }
}

#[tokio::test]
async fn every_provider_entrypoint_sends_the_exact_recorded_final_body() {
    assert_adapter_entrypoints(
        "openai",
        "gpt-4o",
        "400 Bad Request",
        |streaming| streaming,
        |base| Box::new(OpenAiProvider::new("fixture-key".into()).with_base_url(base.into())),
    )
    .await;
    assert_adapter_entrypoints(
        "google",
        "gemini-2.5-flash",
        "400 Bad Request",
        |streaming| streaming,
        |base| Box::new(GoogleProvider::new("fixture-key".into()).with_base_url(base)),
    )
    .await;
    assert_adapter_entrypoints(
        "minimax",
        "MiniMax-M3",
        "400 Bad Request",
        |_| true,
        |base| {
            Box::new(
                MinimaxProvider::new("fixture-key".into())
                    .with_base_url(format!("{base}/anthropic")),
            )
        },
    )
    .await;
    assert_adapter_entrypoints(
        "fixture-compat",
        "fixture-model",
        "400 Bad Request",
        |streaming| streaming,
        |base| {
            Box::new(OpenAiCompatProvider::new(
                "fixture-compat",
                "Fixture compat",
                format!("{base}/v1"),
            ))
        },
    )
    .await;
    assert_adapter_entrypoints(
        "cohere",
        "command-r-plus",
        "400 Bad Request",
        |streaming| streaming,
        |base| {
            Box::new(CohereProvider::new("fixture-key".into()).with_base_url(format!("{base}/v2")))
        },
    )
    .await;
    assert_adapter_entrypoints(
        "azure",
        "fixture-deployment",
        "400 Bad Request",
        |streaming| streaming,
        |base| {
            Box::new(
                AzureProvider::new("fixture-resource".into(), "fixture-key".into())
                    .with_base_url(base),
            )
        },
    )
    .await;
    assert_adapter_entrypoints(
        "amazon-bedrock",
        "anthropic.claude-fixture",
        "400 Bad Request",
        |streaming| streaming,
        |base| Box::new(BedrockProvider::test_fixture().with_base_url(base)),
    )
    .await;
    assert_adapter_entrypoints(
        "copilot",
        "gpt-4o",
        "400 Bad Request",
        |streaming| streaming,
        |base| Box::new(CopilotProvider::new("fixture-token".into()).with_base_url(base)),
    )
    .await;
    assert_adapter_entrypoints(
        "copilot",
        "gpt-5.4",
        "401 Unauthorized",
        |_| false,
        |base| Box::new(CopilotProvider::new("fixture-token".into()).with_base_url(base)),
    )
    .await;
    assert_adapter_entrypoints(
        "codex",
        "gpt-5-codex",
        "400 Bad Request",
        |streaming| streaming,
        |base| {
            Box::new(
                CodexProvider::new(CodexTokens {
                    access_token: "fixture-token".into(),
                    refresh_token: None,
                    account_id: Some("fixture-account".into()),
                    expires_at: None,
                })
                .with_base_url(format!("{base}/responses")),
            )
        },
    )
    .await;
    assert_adapter_entrypoints(
        "anthropic",
        "claude-sonnet-4-5",
        "400 Bad Request",
        |_| true,
        |base| {
            Box::new(AnthropicProvider::from_config(ClientConfig {
                api_key: "fixture-key".into(),
                api_base: base.to_string(),
                max_retries: 0,
                ..Default::default()
            }))
        },
    )
    .await;
}

#[tokio::test]
async fn legacy_anthropic_nonstream_sends_the_exact_recorded_final_body() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("fixture bind");
    let address = listener.local_addr().expect("fixture address");
    let client = AnthropicClient::new(ClientConfig {
        api_key: "fixture-key".into(),
        api_base: format!("http://{address}"),
        max_retries: 0,
        ..Default::default()
    })
    .expect("legacy client");
    let request = CreateMessageRequest::builder("claude-sonnet-4-5", 32)
        .wire_budget(
            WireBudget::finite(100_000, 90_000, 8_000, 1_048_576, MediaAccounting::None)
                .expect("finite fixture budget"),
        )
        .build();
    let capture = tokio::spawn(capture_one(listener, "400 Bad Request"));

    let (result, records) = record_final_requests(client.create_message(request)).await;
    assert!(
        result.is_err(),
        "fixture HTTP failure must reach the client"
    );
    let captured = tokio::time::timeout(Duration::from_secs(2), capture)
        .await
        .expect("fixture capture timeout")
        .expect("fixture capture task");
    let final_record = records.last().expect("legacy final request record");
    assert_record_matches(final_record, &captured, "anthropic", false);
}

#[tokio::test]
async fn legacy_anthropic_stream_sends_the_exact_recorded_final_body() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("fixture bind");
    let address = listener.local_addr().expect("fixture address");
    let client = AnthropicClient::new(ClientConfig {
        api_key: "fixture-key".into(),
        api_base: format!("http://{address}"),
        max_retries: 0,
        ..Default::default()
    })
    .expect("legacy client");
    let request = CreateMessageRequest::builder("claude-sonnet-4-5", 32)
        .wire_budget(
            WireBudget::finite(100_000, 90_000, 8_000, 1_048_576, MediaAccounting::None)
                .expect("finite fixture budget"),
        )
        .build();
    let capture = tokio::spawn(capture_one(listener, "400 Bad Request"));

    let (result, records) =
        record_final_requests(client.create_message_stream(request, Arc::new(NullStreamHandler)))
            .await;
    assert!(
        result.is_err(),
        "fixture HTTP failure must reach the client"
    );
    let captured = tokio::time::timeout(Duration::from_secs(2), capture)
        .await
        .expect("fixture capture timeout")
        .expect("fixture capture task");
    let final_record = records.last().expect("legacy final request record");
    assert_record_matches(final_record, &captured, "anthropic", true);
}

#[tokio::test]
async fn free_route_records_local_preflight_and_exact_upstream_body() {
    for streaming in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("fixture bind");
        let address = listener.local_addr().expect("fixture address");
        let upstream =
            OpenAiCompatProvider::new("groq", "Groq fixture", format!("http://{address}/v1"));
        let provider = FreeProvider::new(vec![FreeEntry {
            upstream: *catalog_entry("groq").expect("Groq catalog entry"),
            provider: Arc::new(upstream),
        }]);
        let capture = tokio::spawn(capture_one(listener, "400 Bad Request"));
        let request = request("free/auto");

        let (result, records) = record_final_requests(async {
            if streaming {
                provider.create_message_stream(request).await.map(|_| ())
            } else {
                provider.create_message(request).await.map(|_| ())
            }
        })
        .await;
        assert!(result.is_err(), "fixture HTTP 400 must reach the upstream");
        let captured = tokio::time::timeout(Duration::from_secs(2), capture)
            .await
            .expect("fixture capture timeout")
            .expect("fixture capture task");
        assert_eq!(records.len(), 2, "local and upstream records: {records:?}");
        assert_eq!(records[0].provider, "free");
        assert_eq!(records[0].endpoint, "free-route");
        assert_record_matches(&records[1], &captured, "groq", streaming);
    }
}
