use super::{deny_all_context, make_config, precedence_provider};
use crate::resolved_target::ResolvedTarget;
use crate::startup_policy::{BoundVerifiedPolicy, LoadedVerifiedPolicy};
use crate::{EffortSelectionMode, EffortSelectionReason, QueryEvent, QueryOutcome};
use claurst_api::{AnthropicClient, LlmProvider, ProviderRegistry};
use claurst_core::config::ProviderConfig;
use claurst_core::dispatch::DispatchDenied;
use claurst_core::effort::EffortLevel;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn policy(root: &Path, provider: &str) -> (Arc<LoadedVerifiedPolicy>, Arc<BoundVerifiedPolicy>) {
    let path = root.join(format!("{provider}-policy.json"));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1, "revision": 7, "provider": provider,
            "base": "high", "continuation": "medium",
            "models": [{"id": "test-model", "efforts": ["high", "medium"]}],
            "checkers": [{
                "id": "check-target", "revision": 1, "target": "target",
                "operation": "edit", "shell": "bash", "command": "test -s target",
                "cwd": ".", "timeout_ms": 1000
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let loaded = Arc::new(LoadedVerifiedPolicy::load(&path, root).unwrap());
    let bound = Arc::new(loaded.bind_session(root, "resolver-binding-test").unwrap());
    (loaded, bound)
}

fn assert_verified_denied(target: &ResolvedTarget<'_>) {
    assert_eq!(
        target.dispatch_route(EffortLevel::High),
        Err(DispatchDenied::UnsupportedRoute)
    );
    assert_eq!(
        target.verified_route_pair(EffortLevel::High, EffortLevel::Medium),
        Err(DispatchDenied::UnsupportedRoute)
    );
}

#[tokio::test]
async fn actual_provider_resolution_and_query_status_are_isolated() {
    let temporary = tempfile::Builder::new()
        .prefix("forge-resolver-binding-")
        .tempdir()
        .unwrap();
    let root = temporary.path();
    std::fs::create_dir(root.join("home")).unwrap();
    std::fs::create_dir(root.join("data")).unwrap();
    let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "tests::resolver_binding_regressions::real_resolver_process_fixture",
            "--ignored",
            "--nocapture",
        ])
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("HOME", root.join("home"))
        .env("TMPDIR", root)
        .env("CLAURST_HOME", root.join("data"))
        .env("FORGE_HOME", root.join("data"))
        .env("CLAURST_DISABLE_MODELS_FETCH", "1")
        .env("RUST_TEST_THREADS", "1")
        .env("FORGE_RESOLVER_BINDING_FIXTURE", "1")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(20), command.output())
        .await
        .expect("isolated resolver fixture deadline")
        .unwrap();
    assert!(
        output.status.success(),
        "resolver fixture failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
#[ignore = "run by actual_provider_resolution_and_query_status_are_isolated with synthetic isolated credentials"]
async fn real_resolver_process_fixture() {
    assert_eq!(
        std::env::var("FORGE_RESOLVER_BINDING_FIXTURE").as_deref(),
        Ok("1")
    );
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    std::fs::write(root.join("target"), "before").unwrap();
    let (loaded, bound) = policy(&root, "openai");
    let client = AnthropicClient::new(claurst_api::client::ClientConfig {
        api_key: "synthetic-legacy-key".into(),
        ..Default::default()
    })
    .unwrap();
    let mut ctx = deny_all_context();
    ctx.working_dir = root.clone();
    ctx.session_id = "resolver-binding-test".into();
    ctx.config.provider = Some("openai".into());
    let mut config = make_config(None, None);
    config.model = "test-model".into();
    config.model_registry = Some(Arc::new(claurst_api::ModelRegistry::new()));
    config.effort_policy = bound.effort_policy();
    config.verified_startup_policy = Some(loaded);
    config.verified_startup_binding = Some(bound);

    // Production registry selection reaches the raw-successful/bind-failed
    // condition. No selection injection or target flag mutation is involved.
    let (rejecting, _, _) = precedence_provider("openai");
    assert!(rejecting
        .dispatch_route("test-model", EffortLevel::High)
        .is_ok());
    let raw: Arc<dyn LlmProvider> = rejecting.clone();
    let mut registry = ProviderRegistry::new();
    registry.register(raw.clone());
    config.provider_registry = Some(Arc::new(registry));
    let rejected = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(Arc::ptr_eq(rejected.provider().unwrap(), &raw));
    assert_verified_denied(&rejected);
    assert_eq!(*rejecting.accepted_requests.lock().unwrap(), 0);

    // The actual query still sends an ordinary fixed request and emits the
    // categorical reason. The fake provider performs no network inference.
    config.max_turns = 1;
    config.max_tokens = 1024;
    config.context_budget.context_limit_tokens = Some(100_000);
    config.context_budget.wire_byte_limit = 1_000_000;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut messages = vec![claurst_core::types::Message::user("record fixed fallback")];
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        crate::run_query_loop(
            &client,
            &mut messages,
            &[],
            &ctx,
            &config,
            claurst_core::cost::CostTracker::new(),
            Some(tx),
            tokio_util::sync::CancellationToken::new(),
            None,
        ),
    )
    .await
    .unwrap();
    assert!(
        matches!(outcome, QueryOutcome::EndTurn { .. }),
        "{outcome:?}"
    );
    let selection = std::iter::from_fn(|| rx.try_recv().ok())
        .find_map(|event| match event {
            QueryEvent::EffortSelection(selection) => Some(selection),
            _ => None,
        })
        .expect("effort selection event");
    assert_eq!(selection.requested(), EffortSelectionMode::Verified);
    assert_eq!(selection.applied(), EffortSelectionMode::Fixed);
    assert_eq!(selection.reason(), EffortSelectionReason::UnsupportedRoute);
    assert_eq!(selection.selected(), Some(EffortLevel::High));
    assert_eq!(*rejecting.dispatch_per_request.lock().unwrap(), vec![false]);
    assert_eq!(
        *rejecting.thinking_budgets_per_request.lock().unwrap(),
        vec![Some(10_000)]
    );

    // A current configured key takes precedence over the rejecting registry.
    ctx.config.provider_configs.insert(
        "openai".into(),
        ProviderConfig {
            api_key: Some("synthetic-configured-key".into()),
            ..Default::default()
        },
    );
    let configured = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(!Arc::ptr_eq(configured.provider().unwrap(), &raw));
    assert_eq!(
        configured.verified_route_pair(EffortLevel::High, EffortLevel::Medium),
        Ok(())
    );
    let repeated = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(configured.same_transport_identity(&repeated));
    ctx.config
        .provider_configs
        .get_mut("openai")
        .unwrap()
        .api_key = Some("synthetic-rotated-key".into());
    let rotated = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(!configured.same_transport_identity(&rotated));

    ctx.config
        .provider_configs
        .get_mut("openai")
        .unwrap()
        .api_base = Some("https://custom.example.invalid/v1".into());
    let custom = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(
        custom.provider().is_some(),
        "fixed custom provider remains available"
    );
    assert_verified_denied(&custom);
    assert!(!rotated.same_transport_identity(&custom));
    ctx.config.provider_configs.clear();

    // The registry owner participates in cache identity even with identical
    // key/base fingerprints. Production resolve_provider retains that owner.
    let registry_for = || {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(claurst_api::OpenAiProvider::new(
            "synthetic-registry-key".into(),
        )));
        Arc::new(registry)
    };
    config.provider_registry = Some(registry_for());
    let first_owner = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    let same_owner = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(first_owner.same_transport_identity(&same_owner));
    config.provider_registry = Some(registry_for());
    let second_owner = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert!(!first_owner.same_transport_identity(&second_owner));
    assert_eq!(
        second_owner.verified_route_pair(EffortLevel::High, EffortLevel::Medium),
        Ok(())
    );

    // AuthStore keys are real configured-resolution inputs, not a synthetic
    // Runtime label. Writes stay inside this subprocess's isolated data root.
    let mut store = claurst_core::AuthStore::default();
    store.set(
        "openai",
        claurst_core::auth_store::StoredCredential::ApiKey {
            key: "synthetic-store-key".into(),
        },
    );
    let stored = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert_eq!(
        stored.verified_route_pair(EffortLevel::High, EffortLevel::Medium),
        Ok(())
    );
    assert!(!second_owner.same_transport_identity(&stored));
    std::fs::remove_file(claurst_core::AuthStore::path()).unwrap();

    ctx.config.provider = Some("ollama".into());
    config.provider_registry = None;
    let runtime = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert_eq!(runtime.provider_id(), "ollama");
    assert!(runtime.provider().is_some());
    assert_verified_denied(&runtime);

    // Reselection also denies a raw-successful different registry provider.
    let (other, _, _) = precedence_provider("google");
    assert!(other
        .dispatch_route("test-model", EffortLevel::High)
        .is_ok());
    let mut registry = ProviderRegistry::new();
    registry.register(other);
    config.provider_registry = Some(Arc::new(registry));
    ctx.config.provider = Some("google".into());
    let reselected = ResolvedTarget::resolve(&config, &ctx, "test-model", &client).unwrap();
    assert_verified_denied(&reselected);
    assert!(!second_owner.same_transport_identity(&reselected));

    // A legacy client can carry its own valid route; it still cannot inherit
    // authority from an OpenAI startup policy after an Anthropic fallback.
    let (_, anthropic_bound) = policy(&root, "anthropic");
    let legacy = client.with_dispatch_policy(Arc::clone(anthropic_bound.dispatch_policy()));
    assert!(legacy
        .dispatch_route("test-model", EffortLevel::High)
        .is_ok());
    ctx.config.provider = Some("anthropic".into());
    let fallback = ResolvedTarget::resolve(&config, &ctx, "anthropic/test-model", &legacy).unwrap();
    assert!(fallback.legacy_client().is_some());
    assert_verified_denied(&fallback);
}
