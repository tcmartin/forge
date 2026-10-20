#![cfg(feature = "a016-openai-loopback")]

//! Explicit-feature-only proof that startup authority reaches the OpenAI
//! adapter's final bytes through the public query loop. The loopback URL is
//! transport-only test state; the dispatch identity remains api.openai.com.

use claurst_api::{AnthropicClient, OpenAiProvider, ProviderRegistry};
use claurst_core::config::PermissionMode;
use claurst_core::permissions::AutoPermissionHandler;
use claurst_core::types::Message;
use claurst_query::startup_policy::LoadedVerifiedPolicy;
use claurst_query::{run_query_loop, ContextBudgetConfig, QueryConfig, QueryOutcome};
use claurst_tools::{EditCheckTool, Tool, ToolContext, TrustedCheckerCatalog};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

static AUTH_HOME_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn read_request(socket: &mut TcpStream) -> (String, Vec<u8>) {
    let mut request = Vec::new();
    loop {
        let mut chunk = [0_u8; 4096];
        let size = socket.read(&mut chunk).await.expect("request read");
        assert!(size > 0, "request closed before its body completed");
        request.extend_from_slice(&chunk[..size]);
        assert!(
            request.len() <= 512 * 1024,
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
            return (
                String::from_utf8_lossy(&request[..header_end]).into_owned(),
                request[body_start..body_start + content_length].to_vec(),
            );
        }
    }
}

async fn read_request_body(socket: &mut TcpStream) -> Vec<u8> {
    read_request(socket).await.1
}
fn sse(events: &[Value]) -> Vec<u8> {
    let mut body = Vec::new();
    for event in events {
        body.extend_from_slice(b"data: ");
        body.extend_from_slice(event.to_string().as_bytes());
        body.extend_from_slice(b"\n\n");
    }
    body.extend_from_slice(b"data: [DONE]\n\n");
    body
}

async fn reply_sse(socket: &mut TcpStream, body: &[u8]) {
    socket
        .write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .expect("response headers");
    socket.write_all(body).await.expect("response body");
}

fn policy_document() -> Value {
    json!({
        "schema_version": 1,
        "revision": 7,
        "provider": "openai",
        "base": "high",
        "continuation": "medium",
        "models": [{"id": "gpt-5.5", "efforts": ["high", "medium"]}],
        "checkers": [{
            "id": "openai-query-check",
            "revision": 1,
            "target": "target",
            "operation": "write",
            "shell": "bash",
            "command": "test \"$(cat target)\" = after; printf x >> checker-runs",
            "cwd": ".",
            "timeout_ms": 5000
        }]
    })
}

fn tool_call_response() -> Vec<u8> {
    sse(&[
        json!({
            "id": "chatcmpl-first",
            "model": "gpt-5.5",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_edit",
                        "type": "function",
                        "function": {
                            "name": "EditCheck",
                            "arguments": "{\"operation\":\"Write\",\"edit\":{\"file_path\":\"target\",\"content\":\"after\"},\"check\":{\"tool\":\"Bash\",\"input\":{\"command\":\"test \\\"$(cat target)\\\" = after; printf x >> checker-runs\",\"timeout\":5000}}}"
                        }
                    }]
                },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-first",
            "model": "gpt-5.5",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }),
        json!({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2}}),
    ])
}

fn final_response() -> Vec<u8> {
    sse(&[
        json!({
            "id": "chatcmpl-final",
            "model": "gpt-5.5",
            "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]
        }),
        json!({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 1}}),
    ])
}

fn context(root: &Path) -> ToolContext {
    ToolContext {
        working_dir: root.to_path_buf(),
        permission_mode: PermissionMode::BypassPermissions,
        permission_handler: Arc::new(AutoPermissionHandler {
            mode: PermissionMode::BypassPermissions,
        }),
        cost_tracker: claurst_core::cost::CostTracker::new(),
        session_id: "a016-openai-query-proof".to_string(),
        file_history: Arc::new(parking_lot::Mutex::new(
            claurst_core::file_history::FileHistory::new(),
        )),
        current_turn: Arc::new(AtomicUsize::new(0)),
        non_interactive: true,
        mcp_manager: None,
        config: claurst_core::config::Config {
            provider: Some("openai".to_string()),
            project_dir: Some(root.to_path_buf()),
            workspace_paths: vec![root.to_path_buf()],
            permission_mode: PermissionMode::BypassPermissions,
            ..Default::default()
        },
        managed_agent_config: None,
        completion_notifier: None,
        pending_permissions: None,
        permission_manager: None,
        user_question_tx: None,
        cancel_token: CancellationToken::new(),
        trusted_checker_catalog: Arc::new(TrustedCheckerCatalog::empty()),
        catalog_view: None,
    }
}

#[tokio::test]
async fn startup_openai_policy_reaches_exact_fixed_and_adaptive_wire_bytes() {
    let root = tempfile::tempdir().expect("fixture root");
    std::fs::write(root.path().join("target"), b"before").expect("target");
    std::fs::write(
        root.path().join("policy.json"),
        policy_document().to_string(),
    )
    .expect("policy");

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("listener address");
    let server = tokio::spawn(async move {
        let mut bodies = Vec::new();
        for response in [tool_call_response(), final_response()] {
            let (mut socket, _) = listener.accept().await.expect("accept");
            bodies.push(read_request_body(&mut socket).await);
            reply_sse(&mut socket, &response).await;
        }
        bodies
    });

    let policy = LoadedVerifiedPolicy::load(&root.path().join("policy.json"), root.path())
        .expect("valid startup policy");
    let provider = OpenAiProvider::new("synthetic-openai-key".to_string())
        .with_test_loopback_base_url(format!("http://{address}"));
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(provider));

    let query_config = QueryConfig {
        model: "gpt-5.5".to_string(),
        max_tokens: 321,
        max_turns: 3,
        working_directory: Some(root.path().display().to_string()),
        provider_registry: Some(Arc::new(registry)),
        verified_startup_policy: Some(Arc::new(policy)),
        context_budget: ContextBudgetConfig {
            wire_byte_limit: 1_000_000,
            ..Default::default()
        },
        ..Default::default()
    };

    let client = AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "unused-legacy-key".to_string(),
        ..Default::default()
    })
    .expect("legacy client fixture");
    let mut messages = vec![Message::user("perform the verified edit")];
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(EditCheckTool)];
    let ctx = context(root.path());
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        run_query_loop(
            &client,
            &mut messages,
            &tools,
            &ctx,
            &query_config,
            claurst_core::cost::CostTracker::new(),
            None,
            CancellationToken::new(),
            None,
        ),
    )
    .await
    .expect("query loop timeout");
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );

    let bodies = tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("server timeout")
        .expect("server task");
    assert_eq!(bodies.len(), 2);
    let fixed: Value = serde_json::from_slice(&bodies[0]).expect("fixed JSON");
    let adaptive: Value = serde_json::from_slice(&bodies[1]).expect("adaptive JSON");

    assert_eq!(fixed["model"], "gpt-5.5");
    assert_eq!(fixed["stream"], true);
    assert_eq!(fixed["max_completion_tokens"], 321);
    // The fixed/base request retains the query runner's configured high
    // control. Only the verified continuation changes to the bound medium
    // snake-case field below.
    assert_eq!(fixed["reasoning_effort"], "high");
    assert!(fixed.get("max_tokens").is_none());

    assert_eq!(adaptive["model"], "gpt-5.5");
    assert_eq!(adaptive["stream"], true);
    assert_eq!(adaptive["reasoning_effort"], "medium");
    assert_eq!(adaptive["max_completion_tokens"], 321);
    assert!(adaptive.get("max_tokens").is_none());
    assert!(adaptive.get("reasoningEffort").is_none());

    assert_eq!(
        std::fs::read(root.path().join("target")).expect("target"),
        b"after"
    );
    assert_eq!(
        std::fs::read(root.path().join("checker-runs")).expect("checker marker"),
        b"x"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn disabled_openai_rejects_stale_registry_and_runtime_credential_without_traffic() {
    let _env_guard = AUTH_HOME_ENV_LOCK.lock().await;
    let root = tempfile::tempdir().expect("fixture root");
    std::fs::write(root.path().join("target"), b"before").expect("target fixture");
    std::fs::write(
        root.path().join("policy.json"),
        policy_document().to_string(),
    )
    .expect("policy fixture");
    let policy = LoadedVerifiedPolicy::load(&root.path().join("policy.json"), root.path())
        .expect("valid startup policy");
    let auth_home = root.path().join("auth-home");
    std::fs::create_dir_all(&auth_home).expect("auth home");
    std::fs::write(
        auth_home.join("auth.json"),
        r#"{"credentials":{"openai":{"type":"api","key":"fixture-runtime-key"}}}"#,
    )
    .expect("runtime credential fixture");
    let previous_auth_home = std::env::var_os("CLAURST_HOME");
    std::env::set_var("CLAURST_HOME", &auth_home);

    let stale_listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let stale_address = stale_listener.local_addr().expect("listener address");
    let stale_probe = tokio::spawn(async move {
        tokio::time::timeout(
            std::time::Duration::from_millis(250),
            stale_listener.accept(),
        )
        .await
        .is_ok()
    });
    let stale_provider = OpenAiProvider::new("stale-registry-key".to_string())
        .with_test_loopback_base_url(format!("http://{stale_address}"));
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(stale_provider));

    let mut ctx = context(root.path());
    ctx.config.provider_configs.insert(
        "openai".to_string(),
        claurst_core::config::ProviderConfig {
            enabled: false,
            ..Default::default()
        },
    );
    let query_config = QueryConfig {
        model: "gpt-5.5".to_string(),
        max_tokens: 321,
        max_turns: 1,
        working_directory: Some(root.path().display().to_string()),
        provider_registry: Some(Arc::new(registry)),
        verified_startup_policy: Some(Arc::new(policy)),
        ..Default::default()
    };
    let client = AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "unused-legacy-key".to_string(),
        ..Default::default()
    })
    .expect("legacy client fixture");
    let mut messages = vec![Message::user("must fail before provider selection")];
    let outcome = run_query_loop(
        &client,
        &mut messages,
        &[],
        &ctx,
        &query_config,
        claurst_core::cost::CostTracker::new(),
        None,
        CancellationToken::new(),
        None,
    )
    .await;
    let no_stale_traffic = !stale_probe.await.expect("stale probe task");

    match previous_auth_home {
        Some(value) => std::env::set_var("CLAURST_HOME", value),
        None => std::env::remove_var("CLAURST_HOME"),
    }
    match outcome {
        QueryOutcome::Error(error) => {
            assert!(
                error.to_string().contains("provider is disabled"),
                "{error}"
            );
        }
        other => panic!("disabled provider unexpectedly reached query loop: {other:?}"),
    }
    assert!(no_stale_traffic);
}

#[tokio::test]
async fn binding_without_loaded_policy_is_rejected_before_provider_or_tool_traffic() {
    let root = tempfile::tempdir().expect("fixture root");
    std::fs::write(root.path().join("target"), b"before").expect("target");
    std::fs::write(
        root.path().join("policy.json"),
        policy_document().to_string(),
    )
    .expect("policy");
    let policy = LoadedVerifiedPolicy::load(&root.path().join("policy.json"), root.path())
        .expect("valid startup policy");
    let bound = Arc::new(
        policy
            .bind_session(root.path(), "foreign-session")
            .expect("session binding"),
    );
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("foreign-binding listener");
    let address = listener.local_addr().expect("foreign-binding address");
    let probe = tokio::spawn(async move {
        tokio::time::timeout(std::time::Duration::from_millis(250), listener.accept())
            .await
            .is_ok()
    });
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(
        OpenAiProvider::new("foreign-binding-key".to_string())
            .with_test_loopback_base_url(format!("http://{address}")),
    ));
    let query_config = QueryConfig {
        model: "gpt-5.5".to_string(),
        max_tokens: 321,
        max_turns: 1,
        working_directory: Some(root.path().display().to_string()),
        provider_registry: Some(Arc::new(registry)),
        verified_startup_binding: Some(bound),
        ..Default::default()
    };
    let client = AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "unused-legacy-key".to_string(),
        ..Default::default()
    })
    .expect("legacy client fixture");
    let mut messages = vec![Message::user("must reject foreign binding")];
    let outcome = run_query_loop(
        &client,
        &mut messages,
        &[],
        &context(root.path()),
        &query_config,
        claurst_core::cost::CostTracker::new(),
        None,
        CancellationToken::new(),
        None,
    )
    .await;
    match outcome {
        QueryOutcome::Error(error) => {
            assert!(
                error
                    .to_string()
                    .contains("requires a loaded startup policy"),
                "{error}"
            );
        }
        other => panic!("foreign binding unexpectedly reached query loop: {other:?}"),
    }
    assert!(!probe.await.expect("foreign-binding probe task"));
}

#[tokio::test]
async fn explicit_custom_endpoint_keeps_current_account_fixed_and_unbound() {
    let root = tempfile::tempdir().expect("fixture root");
    std::fs::write(root.path().join("target"), b"before").expect("target");
    std::fs::write(
        root.path().join("policy.json"),
        policy_document().to_string(),
    )
    .expect("policy");

    let official_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("official listener");
    let official_address = official_listener.local_addr().expect("official address");
    let official_probe = tokio::spawn(async move {
        tokio::time::timeout(
            std::time::Duration::from_millis(250),
            official_listener.accept(),
        )
        .await
        .is_ok()
    });
    let custom_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("custom listener");
    let custom_address = custom_listener.local_addr().expect("custom address");
    let custom_server = tokio::spawn(async move {
        let (mut socket, _) = custom_listener.accept().await.expect("custom accept");
        let (headers, body) = read_request(&mut socket).await;
        assert!(headers.lines().any(|line| {
            line.to_ascii_lowercase()
                .starts_with("authorization: bearer current-custom-key")
        }));
        reply_sse(&mut socket, &final_response()).await;
        body
    });

    let policy = LoadedVerifiedPolicy::load(&root.path().join("policy.json"), root.path())
        .expect("valid startup policy");
    let provider = OpenAiProvider::new("stale-startup-key".to_string())
        .with_test_loopback_base_url(format!("http://{official_address}"));
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(provider));

    let mut ctx = context(root.path());
    ctx.config.provider_configs.insert(
        "openai".to_string(),
        claurst_core::config::ProviderConfig {
            api_key: Some("current-custom-key".to_string()),
            api_base: Some(format!("http://{custom_address}")),
            ..Default::default()
        },
    );
    let query_config = QueryConfig {
        model: "gpt-5.5".to_string(),
        max_tokens: 321,
        max_turns: 1,
        working_directory: Some(root.path().display().to_string()),
        provider_registry: Some(Arc::new(registry)),
        verified_startup_policy: Some(Arc::new(policy)),
        context_budget: ContextBudgetConfig {
            wire_byte_limit: 1_000_000,
            ..Default::default()
        },
        ..Default::default()
    };
    let client = AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "unused-legacy-key".to_string(),
        ..Default::default()
    })
    .expect("legacy client fixture");
    let mut messages = vec![Message::user("use the explicit custom endpoint")];
    let outcome = run_query_loop(
        &client,
        &mut messages,
        &[],
        &ctx,
        &query_config,
        claurst_core::cost::CostTracker::new(),
        None,
        CancellationToken::new(),
        None,
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );

    let body = custom_server.await.expect("custom server task");
    let body: Value = serde_json::from_slice(&body).expect("custom JSON body");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["max_completion_tokens"], 321);
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("reasoningEffort").is_none());
    assert!(!official_probe.await.expect("official probe task"));
}

#[tokio::test]
async fn no_base_key_rotation_binds_current_account_before_stale_registry() {
    let root = tempfile::tempdir().expect("fixture root");
    std::fs::write(root.path().join("target"), b"before").expect("target");
    std::fs::write(
        root.path().join("policy.json"),
        policy_document().to_string(),
    )
    .expect("policy");

    let stale_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("stale listener");
    let stale_address = stale_listener.local_addr().expect("stale address");
    let stale_probe = tokio::spawn(async move {
        tokio::time::timeout(
            std::time::Duration::from_millis(250),
            stale_listener.accept(),
        )
        .await
        .is_ok()
    });
    let current_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("current listener");
    let current_address = current_listener.local_addr().expect("current address");
    let current_server = tokio::spawn(async move {
        let (mut socket, _) = current_listener.accept().await.expect("current accept");
        let (headers, body) = read_request(&mut socket).await;
        assert!(headers.lines().any(|line| {
            line.to_ascii_lowercase()
                .starts_with("authorization: bearer current-key-b")
        }));
        reply_sse(&mut socket, &final_response()).await;
        body
    });

    let policy = LoadedVerifiedPolicy::load(&root.path().join("policy.json"), root.path())
        .expect("valid startup policy");
    let stale_provider = OpenAiProvider::new("stale-key-a".to_string())
        .with_test_loopback_base_url(format!("http://{stale_address}"));
    let mut registry = ProviderRegistry::new();
    registry.register(Arc::new(stale_provider));

    let mut options = std::collections::HashMap::new();
    options.insert(
        "a016_test_loopback_base_url".to_string(),
        json!(format!("http://{current_address}")),
    );
    let mut ctx = context(root.path());
    ctx.config.provider_configs.insert(
        "openai".to_string(),
        claurst_core::config::ProviderConfig {
            api_key: Some("current-key-b".to_string()),
            options,
            ..Default::default()
        },
    );
    let query_config = QueryConfig {
        model: "gpt-5.5".to_string(),
        max_tokens: 321,
        max_turns: 1,
        working_directory: Some(root.path().display().to_string()),
        provider_registry: Some(Arc::new(registry)),
        verified_startup_policy: Some(Arc::new(policy)),
        context_budget: ContextBudgetConfig {
            wire_byte_limit: 1_000_000,
            ..Default::default()
        },
        ..Default::default()
    };
    let client = AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "unused-legacy-key".to_string(),
        ..Default::default()
    })
    .expect("legacy client fixture");
    let mut messages = vec![Message::user("use the rotated account")];
    let outcome = run_query_loop(
        &client,
        &mut messages,
        &[],
        &ctx,
        &query_config,
        claurst_core::cost::CostTracker::new(),
        None,
        CancellationToken::new(),
        None,
    )
    .await;
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );

    let body = current_server.await.expect("current server task");
    let body: Value = serde_json::from_slice(&body).expect("current JSON body");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["max_completion_tokens"], 321);
    assert!(body.get("max_tokens").is_none());
    assert!(!stale_probe.await.expect("stale probe task"));
}
