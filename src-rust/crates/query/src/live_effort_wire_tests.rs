//! Real loopback coverage for effort selection through the Anthropic leaf.
//!
//! These tests intentionally use the production legacy client and the
//! AnthropicProvider wrapper.  The server is a local synthetic SSE endpoint;
//! no credentials or external network are involved.

#![cfg(unix)]

use crate::{run_query_loop, QueryConfig, QueryOutcome};
use claurst_api::client::{AnthropicClient, ClientConfig};
use claurst_api::dispatch_policy::{DispatchModelSpec, DispatchPolicy, EffortWireBinding};
use claurst_api::providers::AnthropicProvider;
use claurst_api::{LlmProvider, ProviderError, ProviderRegistry};
use claurst_core::config::PermissionMode;
use claurst_core::dispatch::DispatchRoute;
use claurst_core::effort::EffortLevel;
use claurst_core::file_history::FileHistory;
use claurst_core::permissions::{AutoPermissionHandler, PermissionHandler};
use claurst_core::provider_id::ProviderId;
use claurst_core::types::Message;
use claurst_tools::{
    EditCheckTool, FileReadTool, RegistrationSpec, Tool, ToolContext, TrustedCheckerCatalogBuilder,
    TrustedCheckerInvocation, TrustedCheckerOperation, TrustedCheckerRegistration,
    TrustedCheckerShell, TrustedCheckerSource,
};
use futures::Stream;
use serde_json::{json, Value};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

const MODEL: &str = "claude-live-effort-fixture";
const CHECK: &str = "test \"$(cat target.txt)\" = after; printf x >> checker-runs";

#[derive(Clone, Copy)]
enum RouteKind {
    Direct,
    Wrapper,
}

#[derive(Clone, Copy)]
enum Scenario {
    Verified,
    Fixed,
    ExplicitThinking,
    ExplicitEffort,
    FailedChecker,
    UnregisteredChecker,
    InlinePin,
}

#[derive(Clone)]
struct Capture {
    requests: Arc<Mutex<Vec<Value>>>,
}

struct ServerPlan {
    scenario: Scenario,
}

#[derive(Clone)]
enum OracleAction {
    None,
    LocalBudget {
        call: usize,
    },
    InvalidateTarget {
        call: usize,
        target: std::path::PathBuf,
    },
}

#[derive(Clone, Debug)]
struct RequestIdentity {
    logical_request_id: Option<String>,
    expected_route: Option<DispatchRoute>,
    model: String,
    max_tokens: u32,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    thinking: Option<(String, u32)>,
    provider_options: Value,
}

impl RequestIdentity {
    fn from_request(request: &claurst_api::ProviderRequest) -> Self {
        Self {
            logical_request_id: request
                .dispatch
                .as_ref()
                .map(|dispatch| dispatch.logical_request_id().as_str().to_string()),
            expected_route: request
                .dispatch
                .as_ref()
                .map(|dispatch| dispatch.expected_route().clone()),
            model: request.model.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            thinking: request
                .thinking
                .as_ref()
                .map(|thinking| (thinking.thinking_type.clone(), thinking.budget_tokens)),
            provider_options: request.provider_options.clone(),
        }
    }
}

struct OracleProvider {
    inner: Arc<AnthropicProvider>,
    id: ProviderId,
    action: OracleAction,
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<RequestIdentity>>>,
}

impl OracleProvider {
    fn new(
        inner: Arc<AnthropicProvider>,
        action: OracleAction,
        requests: Arc<Mutex<Vec<RequestIdentity>>>,
    ) -> Self {
        Self {
            inner,
            id: ProviderId::new(ProviderId::ANTHROPIC),
            action,
            calls: Arc::new(AtomicUsize::new(0)),
            requests,
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for OracleProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "oracle-anthropic-wrapper"
    }

    async fn create_message(
        &self,
        request: claurst_api::ProviderRequest,
    ) -> Result<claurst_api::ProviderResponse, ProviderError> {
        self.inner.create_message(request).await
    }

    async fn create_message_stream(
        &self,
        request: claurst_api::ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<claurst_api::StreamEvent, ProviderError>> + Send>>,
        ProviderError,
    > {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        self.requests
            .lock()
            .expect("oracle request lock")
            .push(RequestIdentity::from_request(&request));
        match &self.action {
            OracleAction::LocalBudget { call: rejected } if *rejected == call => {
                Err(ProviderError::budget(
                    self.id.clone(),
                    claurst_api::WireBudgetError::WireOverflow {
                        wire_bytes: 9_000,
                        max_wire_bytes: 8_000,
                        estimated_input_tokens: 2_250,
                        input_limit_tokens: 2_000,
                        context_limit_tokens: 32_000,
                        output_reserve_tokens: 4_096,
                        policy_revision: "oracle".to_string(),
                        budget_source: "oracle".to_string(),
                    },
                ))
            }
            OracleAction::InvalidateTarget {
                call: invalidated,
                target,
            } if *invalidated == call => {
                std::fs::write(target, b"foreign").expect("mutate target before leaf dispatch");
                self.inner.create_message_stream(request).await
            }
            _ => self.inner.create_message_stream(request).await,
        }
    }

    async fn health_check(&self) -> Result<claurst_api::ProviderStatus, ProviderError> {
        self.inner.health_check().await
    }

    fn capabilities(&self) -> claurst_api::ProviderCapabilities {
        self.inner.capabilities()
    }

    fn dispatch_route(
        &self,
        model: &str,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, claurst_core::dispatch::DispatchDenied> {
        self.inner.dispatch_route(model, effort)
    }
}

#[derive(Clone, Copy)]
enum OracleServerKind {
    LocalBudget,
    LateInvalidation,
    UncertainStream,
}

struct OracleServerControl {
    done: AtomicBool,
    notify: Notify,
    extra_requests: AtomicUsize,
}

impl OracleServerControl {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            done: AtomicBool::new(false),
            notify: Notify::new(),
            extra_requests: AtomicUsize::new(0),
        })
    }

    fn finish(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }
}

fn policy() -> Arc<DispatchPolicy> {
    Arc::new(
        DispatchPolicy::new(
            1,
            vec![DispatchModelSpec {
                model_id: MODEL.to_string(),
                efforts: vec![
                    EffortWireBinding::anthropic_thinking(EffortLevel::High),
                    EffortWireBinding::anthropic_thinking(EffortLevel::Medium),
                ],
            }],
        )
        .expect("fixture dispatch policy"),
    )
}

fn context(root: &Path, catalog: Arc<claurst_tools::TrustedCheckerCatalog>) -> ToolContext {
    ToolContext {
        working_dir: root.to_path_buf(),
        permission_mode: PermissionMode::BypassPermissions,
        permission_handler: Arc::new(AutoPermissionHandler {
            mode: PermissionMode::BypassPermissions,
        }) as Arc<dyn PermissionHandler>,
        cost_tracker: claurst_core::cost::CostTracker::new(),
        session_id: format!("live-effort-{}", uuid::Uuid::new_v4()),
        file_history: Arc::new(parking_lot::Mutex::new(FileHistory::new())),
        current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        non_interactive: true,
        mcp_manager: None,
        config: claurst_core::config::Config::default(),
        managed_agent_config: None,
        completion_notifier: None,
        pending_permissions: None,
        permission_manager: None,
        user_question_tx: None,
        cancel_token: tokio_util::sync::CancellationToken::new(),
        catalog_view: None,
        trusted_checker_catalog: catalog,
    }
}

fn checker_catalog(
    root: &Path,
    command: &str,
    session_id: &str,
) -> Arc<claurst_tools::TrustedCheckerCatalog> {
    let invocation = TrustedCheckerInvocation::foreground(
        TrustedCheckerShell::Bash,
        command,
        root,
        Duration::from_secs(5),
    )
    .expect("checker invocation");
    let registration = TrustedCheckerRegistration::new(RegistrationSpec {
        id: "live-effort-check".to_string(),
        revision: 1,
        source: TrustedCheckerSource::Application,
        workspace: root.to_path_buf(),
        session_id: session_id.to_string(),
        target: root.join("target.txt"),
        operation: TrustedCheckerOperation::Write,
        invocation,
    })
    .expect("checker registration");
    let mut builder = TrustedCheckerCatalogBuilder::new();
    builder
        .register(registration)
        .expect("catalog registration");
    Arc::new(builder.build())
}

async fn read_request_body(socket: &mut TcpStream) -> Vec<u8> {
    const MAX_REQUEST_BYTES: usize = 1_048_576;
    let mut bytes = Vec::new();
    let header_end;
    let content_length;
    loop {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.expect("request headers");
        assert!(count > 0, "client closed before request headers");
        assert!(
            bytes.len() + count <= MAX_REQUEST_BYTES,
            "request headers exceed fixture bound"
        );
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(position) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            header_end = position;
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
    let body_start = header_end + 4;
    while bytes.len() < body_start + content_length {
        let mut chunk = [0_u8; 4096];
        let count = socket.read(&mut chunk).await.expect("request body");
        assert!(count > 0, "client closed before request body");
        assert!(
            bytes.len() + count <= MAX_REQUEST_BYTES,
            "request body exceeds fixture bound"
        );
        bytes.extend_from_slice(&chunk[..count]);
    }
    bytes[body_start..body_start + content_length].to_vec()
}

fn sse(event: &str, data: Value, output: &mut Vec<u8>) {
    output.extend_from_slice(format!("event: {event}\ndata: {}\n\n", data).as_bytes());
}

fn response_for(scenario: Scenario, ordinal: usize) -> Vec<u8> {
    let mut output = Vec::new();
    sse(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": format!("live-{ordinal}"),
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0}
            }
        }),
        &mut output,
    );

    let tool = match (scenario, ordinal) {
        (Scenario::Verified, 1)
        | (Scenario::Fixed, 1)
        | (Scenario::ExplicitThinking, 1)
        | (Scenario::ExplicitEffort, 1)
        | (Scenario::InlinePin, 1)
        | (Scenario::FailedChecker, 1)
        | (Scenario::UnregisteredChecker, 1) => {
            let command = match scenario {
                Scenario::FailedChecker => "printf x >> checker-runs; exit 7",
                Scenario::UnregisteredChecker => "test \"$(cat target.txt)\" = after; printf x >> checker-runs; : unregistered-invocation",
                _ => CHECK,
            };
            Some((
                "EditCheck",
                json!({
                    "operation": "Write",
                    "edit": {"file_path": "target.txt", "content": "after"},
                    "check": {"tool": "Bash", "input": {"command": command, "timeout": 5000}}
                }),
            ))
        }
        (Scenario::Verified, 2)
        | (Scenario::Fixed, 2)
        | (Scenario::ExplicitThinking, 2)
        | (Scenario::ExplicitEffort, 2)
        | (Scenario::InlinePin, 2) => Some(("Read", json!({"file_path": "target.txt"}))),
        _ => None,
    };

    if let Some((name, input)) = tool {
        let tool_id = format!("tool-{ordinal}");
        sse(
            "content_block_start",
            json!({"type":"content_block_start", "index":0, "content_block":{"type":"tool_use", "id":tool_id, "name":name, "input":{}}}),
            &mut output,
        );
        sse(
            "content_block_delta",
            json!({"type":"content_block_delta", "index":0, "delta":{"type":"input_json_delta", "partial_json":input.to_string()}}),
            &mut output,
        );
        sse(
            "content_block_stop",
            json!({"type":"content_block_stop", "index":0}),
            &mut output,
        );
        sse(
            "message_delta",
            json!({"type":"message_delta", "delta":{"stop_reason":"tool_use", "stop_sequence":null}, "usage":{"output_tokens":2}}),
            &mut output,
        );
    } else {
        sse(
            "content_block_start",
            json!({"type":"content_block_start", "index":0, "content_block":{"type":"text", "text":""}}),
            &mut output,
        );
        sse(
            "content_block_delta",
            json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta", "text":"LIVE_EFFORT_OK"}}),
            &mut output,
        );
        sse(
            "content_block_stop",
            json!({"type":"content_block_stop", "index":0}),
            &mut output,
        );
        sse(
            "message_delta",
            json!({"type":"message_delta", "delta":{"stop_reason":"end_turn", "stop_sequence":null}, "usage":{"output_tokens":2}}),
            &mut output,
        );
    }
    sse("message_stop", json!({"type":"message_stop"}), &mut output);
    output
}

async fn serve(listener: TcpListener, capture: Capture, plan: ServerPlan) {
    let expected_requests = match plan.scenario {
        Scenario::Verified
        | Scenario::Fixed
        | Scenario::ExplicitThinking
        | Scenario::ExplicitEffort
        | Scenario::InlinePin => 3,
        Scenario::FailedChecker | Scenario::UnregisteredChecker => 2,
    };
    for ordinal in 1..=expected_requests {
        let (mut socket, _) = listener.accept().await.expect("loopback accept");
        let raw = read_request_body(&mut socket).await;
        let body: Value = serde_json::from_slice(&raw).expect("request JSON");
        capture.requests.lock().expect("capture lock").push(body);
        let response = response_for(plan.scenario, ordinal);
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.len()
        );
        socket.write_all(header.as_bytes()).await.expect("headers");
        socket.write_all(&response).await.expect("SSE response");
    }
}

fn uncertain_response() -> Vec<u8> {
    let mut output = Vec::new();
    sse(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": "uncertain-lower",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0}
            }
        }),
        &mut output,
    );
    sse(
        "content_block_start",
        json!({"type":"content_block_start", "index":0, "content_block":{"type":"text", "text":""}}),
        &mut output,
    );
    sse(
        "content_block_delta",
        json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta", "text":"partial"}}),
        &mut output,
    );
    sse(
        "content_block_stop",
        json!({"type":"content_block_stop", "index":0}),
        &mut output,
    );
    sse(
        "message_delta",
        json!({"type":"message_delta", "delta":{"stop_reason":"end_turn", "stop_sequence":null}, "usage":{"output_tokens":2}}),
        &mut output,
    );
    output
}

async fn write_oracle_response(mut socket: TcpStream, response: Vec<u8>) {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.len()
    );
    socket
        .write_all(header.as_bytes())
        .await
        .expect("oracle response headers");
    socket
        .write_all(&response)
        .await
        .expect("oracle response body");
}

async fn serve_oracle(
    listener: TcpListener,
    capture: Capture,
    kind: OracleServerKind,
    control: Arc<OracleServerControl>,
) {
    let responses = match kind {
        OracleServerKind::LocalBudget => vec![
            response_for(Scenario::Verified, 1),
            response_for(Scenario::Verified, 2),
            response_for(Scenario::Verified, 3),
        ],
        OracleServerKind::LateInvalidation => vec![
            response_for(Scenario::Verified, 1),
            response_for(Scenario::Verified, 3),
        ],
        OracleServerKind::UncertainStream => {
            vec![response_for(Scenario::Verified, 1), uncertain_response()]
        }
    };

    for response in responses {
        let (mut socket, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
            .await
            .expect("oracle accept timeout")
            .expect("oracle accept");
        let raw = read_request_body(&mut socket).await;
        let body: Value = serde_json::from_slice(&raw).expect("oracle request JSON");
        capture
            .requests
            .lock()
            .expect("oracle capture lock")
            .push(body);
        write_oracle_response(socket, response).await;
    }

    let wait_for_done = async {
        loop {
            if control.done.load(Ordering::Acquire) {
                break;
            }
            tokio::select! {
                _ = control.notify.notified() => {},
                accepted = listener.accept() => {
                    if let Ok((socket, _)) = accepted {
                        control.extra_requests.fetch_add(1, Ordering::SeqCst);
                        drop(socket);
                    }
                }
            }
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(3), wait_for_done).await;
}

fn make_query_config(scenario: Scenario, registry: Option<Arc<ProviderRegistry>>) -> QueryConfig {
    let mut config = QueryConfig {
        model: MODEL.to_string(),
        max_tokens: 32_768,
        max_turns: 5,
        context_budget: crate::ContextBudgetConfig {
            context_limit_tokens: Some(131_072),
            provider_reported_limit_tokens: Some(131_072),
            wire_byte_limit: 1_048_576,
            ..Default::default()
        },
        provider_registry: registry,
        ..Default::default()
    };
    if !matches!(scenario, Scenario::Fixed) {
        config.effort_policy =
            crate::effort_policy::EffortPolicy::verified(EffortLevel::High, EffortLevel::Medium)
                .expect("verified policy");
    }
    match scenario {
        Scenario::Fixed | Scenario::ExplicitEffort => {
            config.effort_level = Some(EffortLevel::High);
        }
        Scenario::ExplicitThinking => config.thinking_budget = Some(7_777),
        Scenario::Verified
        | Scenario::InlinePin
        | Scenario::FailedChecker
        | Scenario::UnregisteredChecker => {}
    }
    config
}

async fn run_case(
    route_kind: RouteKind,
    scenario: Scenario,
    prompt: &str,
) -> (QueryOutcome, Vec<Value>, Vec<u8>, Vec<u8>) {
    let root = tempfile::tempdir().expect("fixture root");
    std::fs::write(root.path().join("target.txt"), b"before").expect("target");
    let session_id = format!("live-effort-{}", uuid::Uuid::new_v4());
    let command = match scenario {
        Scenario::FailedChecker => "printf x >> checker-runs; exit 7",
        _ => CHECK,
    };
    let catalog = if matches!(
        scenario,
        Scenario::Verified
            | Scenario::Fixed
            | Scenario::ExplicitThinking
            | Scenario::ExplicitEffort
            | Scenario::InlinePin
            | Scenario::FailedChecker
            | Scenario::UnregisteredChecker
    ) {
        checker_catalog(root.path(), command, &session_id)
    } else {
        Arc::new(claurst_tools::TrustedCheckerCatalog::empty())
    };
    let mut ctx = context(root.path(), catalog);
    ctx.session_id = session_id;
    if matches!(route_kind, RouteKind::Wrapper) {
        ctx.config.provider = Some("anthropic".to_string());
    }

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let base_url = format!("http://{}", listener.local_addr().expect("address"));
    let capture = Capture {
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let server_capture = capture.clone();
    let server = tokio::spawn(serve(listener, server_capture, ServerPlan { scenario }));
    let policy = policy();
    let (client, registry) = match route_kind {
        RouteKind::Direct => {
            let client = AnthropicClient::new(ClientConfig {
                api_key: "synthetic-live-effort-key".to_string(),
                api_base: base_url,
                max_retries: 0,
                request_timeout: Duration::from_secs(3),
                ..Default::default()
            })
            .expect("direct client")
            .with_dispatch_policy(policy);
            (client, None)
        }
        RouteKind::Wrapper => {
            let wrapper_client = AnthropicClient::new(ClientConfig {
                api_key: "synthetic-live-effort-key".to_string(),
                api_base: base_url.clone(),
                max_retries: 0,
                request_timeout: Duration::from_secs(3),
                ..Default::default()
            })
            .expect("wrapper client")
            .with_dispatch_policy(policy);
            let mut registry = ProviderRegistry::new();
            registry.register(Arc::new(AnthropicProvider::new(Arc::new(wrapper_client))));
            let legacy = AnthropicClient::new(ClientConfig {
                api_key: String::new(),
                api_base: base_url,
                max_retries: 0,
                request_timeout: Duration::from_secs(3),
                ..Default::default()
            })
            .expect("legacy fallback client");
            (legacy, Some(Arc::new(registry)))
        }
    };
    let config = make_query_config(scenario, registry);
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EditCheckTool), Box::new(FileReadTool)];
    let cancel = tokio_util::sync::CancellationToken::new();
    let mut messages = vec![Message::user(prompt)];
    let outcome = tokio::time::timeout(
        Duration::from_secs(12),
        run_query_loop(
            &client,
            &mut messages,
            &tools,
            &ctx,
            &config,
            claurst_core::cost::CostTracker::new(),
            None,
            cancel,
            None,
        ),
    )
    .await
    .expect("query loop timeout");
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("server completion timeout")
        .expect("server task");
    let requests = capture.requests.lock().expect("capture lock").clone();
    let target = std::fs::read(root.path().join("target.txt")).expect("target");
    let marker = std::fs::read(root.path().join("checker-runs")).unwrap_or_default();
    (outcome, requests, target, marker)
}

struct OracleRun {
    outcome: QueryOutcome,
    requests: Vec<Value>,
    identities: Vec<RequestIdentity>,
    target: Vec<u8>,
    marker: Vec<u8>,
    provider_calls: usize,
    extra_requests: usize,
}

async fn run_oracle_case(action: OracleAction, server_kind: OracleServerKind) -> OracleRun {
    let root = tempfile::tempdir().expect("oracle fixture root");
    std::fs::write(root.path().join("target.txt"), b"before").expect("oracle target");
    let session_id = format!("oracle-live-effort-{}", uuid::Uuid::new_v4());
    let catalog = checker_catalog(root.path(), CHECK, &session_id);
    let mut ctx = context(root.path(), catalog);
    ctx.session_id = session_id;
    ctx.config.provider = Some("anthropic".to_string());

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("oracle listener");
    let base_url = format!("http://{}", listener.local_addr().expect("oracle address"));
    let capture = Capture {
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let control = OracleServerControl::new();
    let server = tokio::spawn(serve_oracle(
        listener,
        capture.clone(),
        server_kind,
        control.clone(),
    ));

    let actual_client = AnthropicClient::new(ClientConfig {
        api_key: "synthetic-oracle-key".to_string(),
        api_base: base_url.clone(),
        max_retries: 0,
        request_timeout: Duration::from_secs(3),
        ..Default::default()
    })
    .expect("oracle actual client")
    .with_dispatch_policy(policy());
    let inner = Arc::new(AnthropicProvider::new(Arc::new(actual_client)));
    let identities = Arc::new(Mutex::new(Vec::new()));
    let action = match action {
        OracleAction::InvalidateTarget { call, target }
            if target == std::path::Path::new("target.txt") =>
        {
            OracleAction::InvalidateTarget {
                call,
                target: root.path().join(target),
            }
        }
        action => action,
    };
    let oracle = Arc::new(OracleProvider::new(inner, action, identities.clone()));
    let provider_calls = oracle.calls.clone();
    let mut registry = ProviderRegistry::new();
    registry.register(oracle);

    let client = AnthropicClient::new(ClientConfig {
        api_key: String::new(),
        api_base: base_url,
        max_retries: 0,
        request_timeout: Duration::from_secs(3),
        ..Default::default()
    })
    .expect("oracle host client");
    let config = make_query_config(Scenario::Verified, Some(Arc::new(registry)));
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EditCheckTool), Box::new(FileReadTool)];
    let mut messages = vec![
        Message::user("an earlier completed request"),
        Message::assistant("an earlier completed response"),
        Message::user("perform the oracle verified edit"),
    ];
    let outcome = tokio::time::timeout(
        Duration::from_secs(12),
        run_query_loop(
            &client,
            &mut messages,
            &tools,
            &ctx,
            &config,
            claurst_core::cost::CostTracker::new(),
            None,
            tokio_util::sync::CancellationToken::new(),
            None,
        ),
    )
    .await
    .expect("oracle query timeout");
    control.finish();
    tokio::time::timeout(Duration::from_secs(4), server)
        .await
        .expect("oracle server completion timeout")
        .expect("oracle server task");

    let requests = capture
        .requests
        .lock()
        .expect("oracle capture lock")
        .clone();
    let identities = identities.lock().expect("oracle identity lock").clone();
    OracleRun {
        outcome,
        requests,
        identities,
        target: std::fs::read(root.path().join("target.txt")).expect("oracle final target"),
        marker: std::fs::read(root.path().join("checker-runs")).unwrap_or_default(),
        provider_calls: provider_calls.load(Ordering::SeqCst),
        extra_requests: control.extra_requests.load(Ordering::SeqCst),
    }
}

fn assert_same_dispatch_settings(first: &RequestIdentity, second: &RequestIdentity) {
    assert!(first.logical_request_id.is_some());
    assert!(second.logical_request_id.is_some());
    assert_eq!(first.logical_request_id, second.logical_request_id);
    assert_eq!(first.expected_route, second.expected_route);
    assert_eq!(first.model, second.model);
    assert_eq!(first.max_tokens, second.max_tokens);
    assert_eq!(first.temperature, second.temperature);
    assert_eq!(first.top_p, second.top_p);
    assert_eq!(first.top_k, second.top_k);
    assert_eq!(first.thinking, second.thinking);
    assert_eq!(first.provider_options, second.provider_options);
}

fn assert_no_authority(body: &Value) {
    assert_eq!(body["stream"], true);
    for key in ["dispatch", "logical_request_id", "permit"] {
        assert!(body.get(key).is_none(), "serialized authority field: {key}");
    }
}

fn budgets(requests: &[Value]) -> Vec<Option<u64>> {
    requests
        .iter()
        .map(|body| {
            body.pointer("/thinking/budget_tokens")
                .and_then(Value::as_u64)
        })
        .collect()
}

#[tokio::test]
async fn direct_anthropic_leaf_runs_verified_effort_round_trip() {
    let (outcome, requests, target, marker) = run_case(
        RouteKind::Direct,
        Scenario::Verified,
        "perform the verified edit",
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );
    assert_eq!(requests.len(), 3);
    assert_eq!(
        budgets(&requests),
        vec![Some(10_000), Some(5_000), Some(10_000)]
    );
    assert!(requests.iter().all(|body| body["model"] == MODEL));
    requests.iter().for_each(assert_no_authority);
    assert_eq!(target, b"after");
    assert_eq!(marker, b"x");
}

#[tokio::test]
async fn anthropic_provider_wrapper_preserves_verified_effort_wire_round_trip() {
    let (outcome, requests, target, marker) = run_case(
        RouteKind::Wrapper,
        Scenario::Verified,
        "perform the wrapped verified edit",
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );
    assert_eq!(requests.len(), 3);
    assert_eq!(
        budgets(&requests),
        vec![Some(10_000), Some(5_000), Some(10_000)]
    );
    assert!(requests.iter().all(|body| body["model"] == MODEL));
    requests.iter().for_each(assert_no_authority);
    assert_eq!(target, b"after");
    assert_eq!(marker, b"x");
}

#[tokio::test]
async fn fixed_and_explicit_thinking_controls_preserve_real_proof_flow() {
    let (fixed, fixed_requests, fixed_target, fixed_marker) =
        run_case(RouteKind::Direct, Scenario::Fixed, "fixed request").await;
    assert!(matches!(fixed, QueryOutcome::EndTurn { .. }), "{fixed:?}");
    assert_eq!(
        budgets(&fixed_requests),
        vec![Some(10_000), Some(10_000), Some(10_000)]
    );
    fixed_requests.iter().for_each(assert_no_authority);
    assert_eq!(fixed_target, b"after");
    assert_eq!(fixed_marker, b"x");

    let (explicit, explicit_requests, explicit_target, explicit_marker) = run_case(
        RouteKind::Direct,
        Scenario::ExplicitThinking,
        "explicit thinking request",
    )
    .await;
    assert!(
        matches!(explicit, QueryOutcome::EndTurn { .. }),
        "{explicit:?}"
    );
    assert_eq!(
        budgets(&explicit_requests),
        vec![Some(7_777), Some(7_777), Some(7_777)]
    );
    explicit_requests.iter().for_each(assert_no_authority);
    assert_eq!(explicit_target, b"after");
    assert_eq!(explicit_marker, b"x");
}

#[tokio::test]
async fn explicit_effort_and_inline_pins_preserve_authenticated_proof_flow() {
    let (explicit, explicit_requests, explicit_target, explicit_marker) = run_case(
        RouteKind::Direct,
        Scenario::ExplicitEffort,
        "explicit effort request",
    )
    .await;
    assert!(
        matches!(explicit, QueryOutcome::EndTurn { .. }),
        "{explicit:?}"
    );
    assert_eq!(
        budgets(&explicit_requests),
        vec![Some(10_000), Some(10_000), Some(10_000)]
    );
    explicit_requests.iter().for_each(assert_no_authority);
    assert_eq!(explicit_target, b"after");
    assert_eq!(explicit_marker, b"x");

    let (inline, inline_requests, inline_target, inline_marker) = run_case(
        RouteKind::Direct,
        Scenario::InlinePin,
        "ultracode inspect this fixture",
    )
    .await;
    assert!(matches!(inline, QueryOutcome::EndTurn { .. }), "{inline:?}");
    assert_eq!(
        budgets(&inline_requests),
        vec![Some(20_000), Some(20_000), Some(20_000)]
    );
    inline_requests.iter().for_each(assert_no_authority);
    assert_eq!(inline_target, b"after");
    assert_eq!(inline_marker, b"x");
}

#[tokio::test]
async fn failed_and_unregistered_checkers_do_not_authorize_lower_effort() {
    let (failed, failed_requests, failed_target, failed_marker) =
        run_case(RouteKind::Direct, Scenario::FailedChecker, "failed checker").await;
    assert!(matches!(failed, QueryOutcome::EndTurn { .. }), "{failed:?}");
    assert_eq!(budgets(&failed_requests), vec![Some(10_000), Some(10_000)]);
    assert_eq!(failed_target, b"after");
    assert_eq!(failed_marker, b"x");

    let (unregistered, unregistered_requests, unregistered_target, unregistered_marker) = run_case(
        RouteKind::Direct,
        Scenario::UnregisteredChecker,
        "unregistered checker",
    )
    .await;
    assert!(
        matches!(unregistered, QueryOutcome::EndTurn { .. }),
        "{unregistered:?}"
    );
    assert_eq!(
        budgets(&unregistered_requests),
        vec![Some(10_000), Some(10_000)]
    );
    assert_eq!(unregistered_target, b"after");
    assert_eq!(unregistered_marker, b"x");
}

#[tokio::test]
async fn inline_effort_pin_is_visible_on_real_wire() {
    let (outcome, requests, target, marker) = run_case(
        RouteKind::Direct,
        Scenario::InlinePin,
        "ultracode inspect this fixture",
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );
    assert_eq!(
        budgets(&requests),
        vec![Some(20_000), Some(20_000), Some(20_000)]
    );
    assert_eq!(target, b"after");
    assert_eq!(marker, b"x");
}

#[tokio::test]
async fn local_wire_overflow_preserves_dispatch_identity_and_unused_permit() {
    let run = run_oracle_case(
        OracleAction::LocalBudget { call: 2 },
        OracleServerKind::LocalBudget,
    )
    .await;
    assert!(
        matches!(run.outcome, QueryOutcome::EndTurn { .. }),
        "{:?}",
        run.outcome
    );
    assert_eq!(
        run.requests.len(),
        3,
        "local overflow must open only real accepted transports"
    );
    assert_eq!(
        run.provider_calls, 4,
        "one local rejection plus three real provider calls"
    );
    assert_eq!(run.extra_requests, 0);
    assert_eq!(run.identities.len(), 4);
    assert_same_dispatch_settings(&run.identities[1], &run.identities[2]);
    assert_eq!(
        budgets(&run.requests),
        vec![Some(10_000), Some(5_000), Some(10_000)]
    );
    assert_eq!(run.target, b"after");
    assert_eq!(run.marker, b"x");
    run.requests.iter().for_each(assert_no_authority);
}

#[tokio::test]
async fn late_leaf_proof_invalidation_allows_one_fresh_base_without_tool_replay() {
    let run = run_oracle_case(
        OracleAction::InvalidateTarget {
            call: 2,
            target: std::path::PathBuf::from("target.txt"),
        },
        OracleServerKind::LateInvalidation,
    )
    .await;
    assert!(
        matches!(run.outcome, QueryOutcome::EndTurn { .. }),
        "{:?}",
        run.outcome
    );
    assert_eq!(
        run.requests.len(),
        2,
        "one initial request plus one fresh base request"
    );
    assert_eq!(
        run.provider_calls, 3,
        "initial, rejected lower leaf, fresh base"
    );
    assert_eq!(run.extra_requests, 0);
    assert_eq!(run.identities.len(), 3);
    assert!(run.identities[2].logical_request_id.is_none());
    assert!(run.identities[2].expected_route.is_none());
    assert_eq!(budgets(&run.requests), vec![Some(10_000), Some(10_000)]);
    assert_eq!(
        run.marker, b"x",
        "the authenticated EditCheck must not replay"
    );
    assert_eq!(run.target, b"foreign");
    run.requests.iter().for_each(assert_no_authority);
}

#[tokio::test]
async fn uncertain_lower_stream_aborts_without_changed_resend_or_tool_replay() {
    let run = run_oracle_case(OracleAction::None, OracleServerKind::UncertainStream).await;
    assert!(
        matches!(run.outcome, QueryOutcome::Error(_)),
        "{:?}",
        run.outcome
    );
    assert_eq!(
        run.requests.len(),
        2,
        "uncertain stream must not trigger another inference"
    );
    assert_eq!(run.provider_calls, 2);
    assert_eq!(
        run.extra_requests, 0,
        "listener stayed armed through query completion"
    );
    assert_eq!(run.identities.len(), 2);
    assert!(run.identities[1].logical_request_id.is_some());
    assert_eq!(budgets(&run.requests), vec![Some(10_000), Some(5_000)]);
    assert_eq!(run.marker, b"x", "uncertain response must not replay tools");
    assert_eq!(run.target, b"after");
    run.requests.iter().for_each(assert_no_authority);
}
