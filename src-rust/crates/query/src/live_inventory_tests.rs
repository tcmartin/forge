//! Source-backed inventory integration regressions.
//!
//! These tests deliberately drive the public query loop with a local provider
//! stream.  They exercise the boundary where a completed tool batch is followed
//! by a refreshed host inventory; no network or global host state is involved.

use async_trait::async_trait;
use claurst_api::{
    LlmProvider, ProviderError, ProviderRegistry, ProviderRequest, ProviderResponse,
    ProviderStatus, StreamEvent,
};
use claurst_core::provider_id::ProviderId;
use claurst_core::types::{ContentBlock, Message, UsageInfo};
use claurst_core::{PermissionDecision, PermissionHandler, PermissionRequest};
use claurst_tools::{PermissionLevel, Tool, ToolContext, ToolResult, ToolSearchTool};
use futures::Stream;
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{
    run_query_loop, ChildInventoryCapability, HistoryIndex, InventoryError, InventoryGeneration,
    InventoryRequest, QueryConfig, QueryOutcome, ToolInventoryProfile, ToolInventorySnapshot,
    ToolInventorySource,
};

const PROVIDER: &str = "live-inventory-test";

#[tokio::test]
async fn legacy_anthropic_discovery_uses_source_handles_with_empty_caller_slice() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let ctx = context("legacy-source-inventory");
    let runs = Arc::new(AtomicUsize::new(0));
    let initial = inventory_tools(&ctx, 1, "DeferredA", Arc::clone(&runs));
    let source = Arc::new(SwitchingSource {
        refreshed: Arc::clone(&initial),
        initial,
        replaced: Arc::new(AtomicBool::new(false)),
    });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let server_capture = Arc::clone(&captured);
    let server = tokio::spawn(async move {
        for ordinal in 0..3 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let body_start = loop {
                let mut chunk = [0_u8; 8192];
                let read = socket.read(&mut chunk).await.unwrap();
                assert!(read > 0);
                bytes.extend_from_slice(&chunk[..read]);
                assert!(bytes.len() <= 1_000_000);
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .expect("request content length");
                    if bytes.len() >= end + 4 + length {
                        break end + 4;
                    }
                }
            };
            server_capture
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&bytes[body_start..]).unwrap());
            let event = |kind: &str, fields: Value| {
                let mut fields = fields.as_object().unwrap().clone();
                fields.insert("type".into(), json!(kind));
                format!("event: {kind}\ndata: {}\n\n", Value::Object(fields))
            };
            let mut body = event(
                "message_start",
                json!({"message":{
                    "id":format!("legacy-inventory-{ordinal}"), "type":"message", "role":"assistant",
                    "model":"claude-sonnet-4-20250514", "content":[],
                    "usage":{"input_tokens":1,"output_tokens":0}
                }}),
            );
            if ordinal < 2 {
                let (name, input) = if ordinal == 0 {
                    ("ToolSearch", json!({"query":"select:DeferredA"}))
                } else {
                    ("DeferredA", json!({}))
                };
                body.push_str(&event("content_block_start", json!({"index":0,"content_block":{
                    "type":"tool_use", "id":format!("legacy-tool-{ordinal}"), "name":name, "input":{}
                }})));
                body.push_str(&event(
                    "content_block_delta",
                    json!({"index":0,"delta":{
                        "type":"input_json_delta", "partial_json":input.to_string()
                    }}),
                ));
            } else {
                body.push_str(&event(
                    "content_block_start",
                    json!({"index":0,"content_block":{
                        "type":"text", "text":""
                    }}),
                ));
                body.push_str(&event(
                    "content_block_delta",
                    json!({"index":0,"delta":{
                        "type":"text_delta", "text":"legacy inventory complete"
                    }}),
                ));
            }
            body.push_str(&event("content_block_stop", json!({"index":0})));
            body.push_str(&event(
                "message_delta",
                json!({"delta":{
                "stop_reason":if ordinal < 2 {"tool_use"} else {"end_turn"}, "stop_sequence":null
            }, "usage":{"output_tokens":1}}),
            ));
            body.push_str(&event("message_stop", json!({})));
            let header = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            socket.write_all(header.as_bytes()).await.unwrap();
            socket.write_all(body.as_bytes()).await.unwrap();
        }
    });
    let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "legacy-inventory-fixture-key".into(),
        api_base: format!("http://{address}"),
        ..Default::default()
    })
    .unwrap();
    let mut config = QueryConfig {
        model: "claude-sonnet-4-20250514".into(),
        max_tokens: 1024,
        max_turns: 4,
        inventory_source: Some(source),
        ..QueryConfig::default()
    };
    config.context_budget.context_limit_tokens = Some(100_000);
    let mut messages = vec![Message::user("Exercise the source-owned legacy inventory.")];
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        run_query_loop(
            &client,
            &mut messages,
            &[],
            &ctx,
            &config,
            Arc::clone(&ctx.cost_tracker),
            None,
            tokio_util::sync::CancellationToken::new(),
            None,
        ),
    )
    .await;
    server.abort();
    let outcome = outcome.expect("legacy source query timeout");
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "source-owned tool must execute once"
    );
    let schemas: Vec<Vec<String>> = captured
        .lock()
        .unwrap()
        .iter()
        .map(|body| {
            body["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| tool["name"].as_str().unwrap().to_owned())
                .collect()
        })
        .collect();
    assert_eq!(
        schemas,
        vec![
            vec!["ToolSearch"],
            vec!["DeferredA", "ToolSearch"],
            vec!["DeferredA", "ToolSearch"]
        ]
    );
}

#[test]
fn child_profile_distinguishes_candidate_from_final_deny_all() {
    let candidate = ToolInventoryProfile::AgentChild {
        allowed_tools: None,
        capability: ChildInventoryCapability::ObservationOnly,
    };
    assert!(candidate.allows_tool("Read"));
    let deny_all = ToolInventoryProfile::AgentChild {
        allowed_tools: Some(Arc::from(Vec::<String>::new().into_boxed_slice())),
        capability: ChildInventoryCapability::ObservationOnly,
    };
    assert!(!deny_all.allows_tool("Read"));
    let selected = ToolInventoryProfile::AgentChild {
        allowed_tools: Some(Arc::from(vec!["Read".to_string()].into_boxed_slice())),
        capability: ChildInventoryCapability::ObservationOnly,
    };
    assert!(selected.allows_tool("Read"));
    assert!(!selected.allows_tool("Write"));
}

#[test]
fn history_transition_reopens_under_original_config_root() {
    let config_root = tempfile::tempdir().expect("history config root");
    let old =
        HistoryIndex::open(config_root.path(), "old-workspace", "session").expect("old index");
    assert_eq!(old.config_root(), config_root.path());

    let new = HistoryIndex::open(old.config_root(), "new-workspace", "session")
        .expect("fresh transition index");
    assert_eq!(new.config_root(), config_root.path());
    assert!(config_root.path().join("history").is_dir());
    assert!(!config_root.path().join("history").join("history").exists());
}

struct AllowAll;

impl PermissionHandler for AllowAll {
    fn check_permission(&self, _: &PermissionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }

    fn request_permission(&self, _: &PermissionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

fn context(session_id: &str) -> ToolContext {
    ToolContext {
        working_dir: std::env::current_dir().expect("inventory fixture directory"),
        permission_mode: claurst_core::config::PermissionMode::Default,
        permission_handler: Arc::new(AllowAll),
        cost_tracker: claurst_core::cost::CostTracker::new(),
        session_id: session_id.to_string(),
        file_history: Arc::new(parking_lot::Mutex::new(
            claurst_core::file_history::FileHistory::new(),
        )),
        current_turn: Arc::new(AtomicUsize::new(0)),
        non_interactive: true,
        mcp_manager: None,
        config: claurst_core::config::Config::default(),
        managed_agent_config: None,
        completion_notifier: None,
        pending_permissions: None,
        permission_manager: None,
        user_question_tx: None,
        cancel_token: tokio_util::sync::CancellationToken::new(),
        trusted_checker_catalog: Arc::new(claurst_tools::TrustedCheckerCatalog::empty()),
        catalog_view: None,
    }
}

struct FixtureTool {
    name: &'static str,
    runs: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for FixtureTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "inventory integration fixture"
    }
    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }
    fn advanced(&self) -> bool {
        true
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
        self.runs.fetch_add(1, Ordering::SeqCst);
        ToolResult::success(self.name)
    }
}

fn snapshot(
    ctx: &ToolContext,
    generation: InventoryGeneration,
    tools: Vec<Box<dyn Tool>>,
) -> Arc<ToolInventorySnapshot> {
    let scope = claurst_tools::ToolCatalogScope::from_context(ctx).expect("fixture scope");
    ToolInventorySnapshot::new(Arc::new(tools), generation, scope).expect("fixture snapshot")
}

struct SwitchingSource {
    initial: Arc<ToolInventorySnapshot>,
    refreshed: Arc<ToolInventorySnapshot>,
    replaced: Arc<AtomicBool>,
}

#[async_trait]
impl ToolInventorySource for SwitchingSource {
    async fn snapshot(
        &self,
        _: &InventoryRequest,
        _: &ToolContext,
    ) -> Result<Arc<ToolInventorySnapshot>, InventoryError> {
        if self.replaced.load(Ordering::SeqCst) {
            Ok(Arc::clone(&self.refreshed))
        } else {
            Ok(Arc::clone(&self.initial))
        }
    }

    fn current_generation(
        &self,
        _: &InventoryRequest,
        _: &ToolContext,
    ) -> Result<InventoryGeneration, InventoryError> {
        Ok(if self.replaced.load(Ordering::SeqCst) {
            self.refreshed.generation()
        } else {
            self.initial.generation()
        })
    }
}

/// The source keeps its generation stable but swaps the owning tool handles.
/// The loop must reject the replacement before making a second provider call.
struct SameGenerationReplacementSource {
    first: Arc<ToolInventorySnapshot>,
    replacement: Arc<ToolInventorySnapshot>,
    replaced: Arc<AtomicBool>,
}

#[async_trait]
impl ToolInventorySource for SameGenerationReplacementSource {
    async fn snapshot(
        &self,
        _: &InventoryRequest,
        _: &ToolContext,
    ) -> Result<Arc<ToolInventorySnapshot>, InventoryError> {
        if self.replaced.load(Ordering::SeqCst) {
            Ok(Arc::clone(&self.replacement))
        } else {
            Ok(Arc::clone(&self.first))
        }
    }

    fn current_generation(
        &self,
        _: &InventoryRequest,
        _: &ToolContext,
    ) -> Result<InventoryGeneration, InventoryError> {
        Ok(1)
    }
}

/// A generation change becomes visible while the ToolSearch result is being
/// finalized.  This models a host refresh racing receipt activation.
struct DirtyDuringSearchSource {
    initial: Arc<ToolInventorySnapshot>,
    refreshed: Arc<ToolInventorySnapshot>,
    dirty: Arc<AtomicBool>,
}

#[async_trait]
impl ToolInventorySource for DirtyDuringSearchSource {
    async fn snapshot(
        &self,
        _: &InventoryRequest,
        _: &ToolContext,
    ) -> Result<Arc<ToolInventorySnapshot>, InventoryError> {
        if self.dirty.load(Ordering::SeqCst) {
            Ok(Arc::clone(&self.refreshed))
        } else {
            Ok(Arc::clone(&self.initial))
        }
    }

    fn current_generation(
        &self,
        _: &InventoryRequest,
        _: &ToolContext,
    ) -> Result<InventoryGeneration, InventoryError> {
        Ok(if self.dirty.load(Ordering::SeqCst) {
            2
        } else {
            1
        })
    }
}

struct FlippingToolSearch {
    dirty: Arc<AtomicBool>,
    executions: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for FlippingToolSearch {
    fn name(&self) -> &str {
        "ToolSearch"
    }
    fn description(&self) -> &str {
        "fixture ToolSearch wrapper"
    }
    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }
    fn effect(&self) -> claurst_tools::ToolEffect {
        claurst_tools::ToolEffect::Observation
    }
    fn advanced(&self) -> bool {
        ToolSearchTool.advanced()
    }
    fn catalog_source(&self) -> claurst_tools::ToolCatalogSource {
        ToolSearchTool.catalog_source()
    }
    fn input_schema(&self) -> Value {
        ToolSearchTool.input_schema()
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        self.executions.fetch_add(1, Ordering::SeqCst);
        let result = ToolSearchTool.execute(input, ctx).await;
        // Keep the private receipt attached to the delegated result. The host
        // generation changes only after the real search has completed.
        self.dirty.store(true, Ordering::SeqCst);
        result
    }
}

struct RecordingProvider {
    id: ProviderId,
    requests: Arc<Mutex<Vec<Vec<String>>>>,
    flip_after_request: Option<(usize, Arc<AtomicBool>)>,
}

impl RecordingProvider {
    fn requests(&self) -> Vec<Vec<String>> {
        self.requests.lock().expect("requests lock").clone()
    }

    fn event_stream(
        index: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>> {
        let id = format!("inventory-message-{index}");
        let events = match index {
            0 => tool_events(id, "ToolSearch", json!({"query": "select:DeferredA"})),
            1 => tool_events(id, "DeferredA", json!({})),
            2 => tool_events(id, "ToolSearch", json!({"query": "select:DeferredB"})),
            3 => tool_events(id, "DeferredB", json!({})),
            _ => end_turn_events(id),
        };
        Box::pin(futures::stream::iter(events))
    }
}

#[async_trait]
impl LlmProvider for RecordingProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }
    fn name(&self) -> &str {
        "inventory-recording-provider"
    }

    async fn create_message(&self, _: ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        Err(ProviderError::Other {
            provider: self.id.clone(),
            message: "fixture only supports streams".to_string(),
            status: None,
            body: None,
        })
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let index = {
            let mut requests = self.requests.lock().expect("requests lock");
            requests.push(request.tools.iter().map(|tool| tool.name.clone()).collect());
            requests.len() - 1
        };
        if let Some((flip_index, flag)) = &self.flip_after_request {
            if index == *flip_index {
                flag.store(true, Ordering::SeqCst);
            }
        }
        Ok(Self::event_stream(index))
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> claurst_api::ProviderCapabilities {
        claurst_api::ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: claurst_api::SystemPromptStyle::TopLevel,
        }
    }
}

fn tool_events(
    message_id: String,
    name: &str,
    input: Value,
) -> Vec<Result<StreamEvent, ProviderError>> {
    let tool_id = format!("{message_id}-tool-{name}");
    vec![
        Ok(StreamEvent::MessageStart {
            id: message_id,
            model: "inventory-model".to_string(),
            usage: UsageInfo::default(),
        }),
        Ok(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: tool_id,
                name: name.to_string(),
                input: input.clone(),
                thought_signature: None,
            },
        }),
        Ok(StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: input.to_string(),
        }),
        Ok(StreamEvent::MessageDelta {
            stop_reason: Some(claurst_api::provider_types::StopReason::ToolUse),
            usage: Some(UsageInfo::default()),
        }),
        Ok(StreamEvent::MessageStop),
    ]
}

fn end_turn_events(message_id: String) -> Vec<Result<StreamEvent, ProviderError>> {
    vec![
        Ok(StreamEvent::MessageStart {
            id: message_id,
            model: "inventory-model".to_string(),
            usage: UsageInfo::default(),
        }),
        Ok(StreamEvent::TextDelta {
            index: 0,
            text: "done".to_string(),
        }),
        Ok(StreamEvent::MessageDelta {
            stop_reason: Some(claurst_api::provider_types::StopReason::EndTurn),
            usage: Some(UsageInfo::default()),
        }),
        Ok(StreamEvent::MessageStop),
    ]
}

async fn run_with_source(
    source: Arc<dyn ToolInventorySource>,
    ctx: &ToolContext,
    provider: Arc<RecordingProvider>,
) -> QueryOutcome {
    run_with_source_and_messages(source, ctx, provider).await.0
}

async fn run_with_source_and_messages(
    source: Arc<dyn ToolInventorySource>,
    ctx: &ToolContext,
    provider: Arc<RecordingProvider>,
) -> (QueryOutcome, Vec<Message>) {
    let mut registry = ProviderRegistry::new();
    registry.register(provider);
    let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "inventory-test-key".to_string(),
        ..Default::default()
    })
    .expect("local client");
    let mut loop_ctx = ctx.clone();
    loop_ctx.config.provider = Some(PROVIDER.to_string());
    let mut config = QueryConfig {
        model: "inventory-model".to_string(),
        max_tokens: 1024,
        max_turns: 8,
        provider_registry: Some(Arc::new(registry)),
        inventory_source: Some(source),
        ..QueryConfig::default()
    };
    config.context_budget.wire_byte_limit = 1_000_000;
    config.context_budget.context_limit_tokens = Some(100_000);
    let mut messages = vec![Message::user("inventory refresh")];
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        run_query_loop(
            &client,
            &mut messages,
            &[],
            &loop_ctx,
            &config,
            claurst_core::cost::CostTracker::new(),
            None,
            tokio_util::sync::CancellationToken::new(),
            None,
        ),
    )
    .await
    .expect("inventory fixture query must finish");
    (outcome, messages)
}

fn inventory_tools(
    ctx: &ToolContext,
    generation: InventoryGeneration,
    deferred_name: &'static str,
    runs: Arc<AtomicUsize>,
) -> Arc<ToolInventorySnapshot> {
    inventory_tools_with_search(
        ctx,
        generation,
        deferred_name,
        runs,
        Box::new(ToolSearchTool),
    )
}

fn inventory_tools_with_search(
    ctx: &ToolContext,
    generation: InventoryGeneration,
    deferred_name: &'static str,
    runs: Arc<AtomicUsize>,
    search: Box<dyn Tool>,
) -> Arc<ToolInventorySnapshot> {
    snapshot(
        ctx,
        generation,
        vec![
            search,
            Box::new(FixtureTool {
                name: deferred_name,
                runs,
            }),
        ],
    )
}

#[tokio::test]
async fn generation_replacement_revokes_old_discovery_and_executes_new_tool() {
    let ctx = context("inventory-generation");
    let a_runs = Arc::new(AtomicUsize::new(0));
    let b_runs = Arc::new(AtomicUsize::new(0));
    let initial = inventory_tools(&ctx, 1, "DeferredA", Arc::clone(&a_runs));
    let refreshed = inventory_tools(&ctx, 2, "DeferredB", Arc::clone(&b_runs));
    let source = Arc::new(SwitchingSource {
        initial,
        refreshed,
        replaced: Arc::new(AtomicBool::new(false)),
    });
    let replaced = Arc::clone(&source.replaced);
    let provider = Arc::new(RecordingProvider {
        id: ProviderId::new(PROVIDER),
        requests: Arc::new(Mutex::new(Vec::new())),
        flip_after_request: Some((1, replaced)),
    });

    let outcome = run_with_source(
        Arc::clone(&source) as Arc<dyn ToolInventorySource>,
        &ctx,
        Arc::clone(&provider),
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "unexpected outcome: {outcome:?}"
    );
    assert_eq!(
        a_runs.load(Ordering::SeqCst),
        0,
        "old-schema tool must be NotRun after refresh"
    );
    assert_eq!(
        b_runs.load(Ordering::SeqCst),
        1,
        "new-schema tool executes once"
    );
    let requests = provider.requests();
    assert!(
        requests.len() >= 4,
        "expected discovery, stale-call, discovery, and accepted call: {requests:?}"
    );
    assert_eq!(requests[0], vec!["ToolSearch"]);
    assert_eq!(requests[1], vec!["DeferredA", "ToolSearch"]);
    assert_eq!(requests[2], vec!["ToolSearch"]);
    assert_eq!(requests[3], vec!["DeferredB", "ToolSearch"]);
}

#[tokio::test]
async fn same_generation_handle_replacement_stops_before_second_provider_call() {
    let ctx = context("inventory-handle-replacement");
    let first = inventory_tools(&ctx, 1, "DeferredA", Arc::new(AtomicUsize::new(0)));
    let replacement = inventory_tools(&ctx, 1, "DeferredB", Arc::new(AtomicUsize::new(0)));
    let source = Arc::new(SameGenerationReplacementSource {
        first,
        replacement,
        replaced: Arc::new(AtomicBool::new(false)),
    });
    let replaced = Arc::clone(&source.replaced);
    let provider = Arc::new(RecordingProvider {
        id: ProviderId::new(PROVIDER),
        requests: Arc::new(Mutex::new(Vec::new())),
        flip_after_request: Some((0, replaced)),
    });

    let outcome = run_with_source(
        Arc::clone(&source) as Arc<dyn ToolInventorySource>,
        &ctx,
        Arc::clone(&provider),
    )
    .await;
    assert!(matches!(outcome, QueryOutcome::Error(_)));
    assert_eq!(
        provider.requests().len(),
        1,
        "replacement must be rejected before a second send"
    );
}

#[tokio::test]
async fn dirty_generation_during_search_keeps_receipt_inactive() {
    let ctx = context("inventory-dirty-search");
    let a_runs = Arc::new(AtomicUsize::new(0));
    let search_runs = Arc::new(AtomicUsize::new(0));
    let dirty = Arc::new(AtomicBool::new(false));
    let initial = inventory_tools_with_search(
        &ctx,
        1,
        "DeferredA",
        Arc::clone(&a_runs),
        Box::new(FlippingToolSearch {
            dirty: Arc::clone(&dirty),
            executions: Arc::clone(&search_runs),
        }),
    );
    let refreshed = inventory_tools(&ctx, 2, "DeferredB", Arc::new(AtomicUsize::new(0)));
    let source = Arc::new(DirtyDuringSearchSource {
        initial,
        refreshed,
        dirty,
    });
    let provider = Arc::new(RecordingProvider {
        id: ProviderId::new(PROVIDER),
        requests: Arc::new(Mutex::new(Vec::new())),
        flip_after_request: None,
    });

    let (outcome, messages) = run_with_source_and_messages(
        Arc::clone(&source) as Arc<dyn ToolInventorySource>,
        &ctx,
        Arc::clone(&provider),
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "dirty search should recover with fresh catalog: {outcome:?}"
    );
    assert_eq!(
        search_runs.load(Ordering::SeqCst),
        1,
        "ToolSearch must really execute before the flip"
    );
    assert_eq!(a_runs.load(Ordering::SeqCst), 0);
    let requests = provider.requests();
    assert!(
        requests.len() >= 4,
        "expected stale search, fresh discovery, and accepted request: {requests:?}"
    );
    assert_eq!(requests[0], vec!["ToolSearch"]);
    assert_eq!(requests[1], vec!["ToolSearch"]);
    assert_eq!(requests[2], vec!["ToolSearch"]);
    assert_eq!(requests[3], vec!["DeferredB", "ToolSearch"]);
    assert!(requests
        .iter()
        .skip(1)
        .all(|request| !request.iter().any(|name| name == "DeferredA")));
    let transcript = format!("{messages:?}");
    assert!(
        transcript.contains("Tool discovery selection was stale"),
        "stale receipt result missing: {transcript}"
    );
}
