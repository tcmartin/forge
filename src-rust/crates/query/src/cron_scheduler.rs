// cron_scheduler: background task that fires cron-scheduled prompts.
//
// Runs as a long-lived tokio task. Every minute it checks the global CRON_STORE
// (in cc-tools) for tasks whose cron expression matches the current wall-clock
// minute. Matching tasks are fired by spawning a sub-query loop, exactly like
// the AgentTool does for sub-agents.
//
// One-shot tasks (recurring=false) are automatically removed from the store
// by `pop_due_tasks` after they are returned.

use crate::{run_query_loop, QueryConfig, QueryOutcome};
use chrono::Timelike;
use claurst_core::types::Message;
use claurst_tools::Tool;
use claurst_tools::ToolContext;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::time::{sleep, Duration};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

/// Builds the policy-filtered tool set for one durable cron workspace.
///
/// The factory lives at the application boundary because only the caller has
/// the MCP manager and policy configuration. Query owns the index lifecycle;
/// the factory must use `make_history_read_tool_for_workspace` when it admits
/// `HistoryRead`.
pub type CronToolFactory = Arc<
    dyn Fn(Arc<parking_lot::Mutex<crate::HistoryIndex>>, &Path) -> Arc<Vec<Box<dyn Tool>>>
        + Send
        + Sync,
>;

/// Complete runtime ownership for one cron scheduler generation.
///
/// Reconnects replace this snapshot as one watch value.  A cron invocation
/// clones the fields it needs before spawning, so already-running work keeps
/// its original client, context, and policy while idle scheduling observes
/// the replacement and releases the old generation.
#[derive(Clone)]
pub struct CronSchedulerRuntime {
    pub client: Arc<claurst_api::AnthropicClient>,
    pub tool_factory: CronToolFactory,
    pub tool_ctx: ToolContext,
    pub query_config: QueryConfig,
    pub history_root: PathBuf,
    pub workspace: PathBuf,
    tools: Option<Arc<Vec<Box<dyn Tool>>>>,
}

impl CronSchedulerRuntime {
    pub fn new(
        client: Arc<claurst_api::AnthropicClient>,
        tool_factory: CronToolFactory,
        tool_ctx: ToolContext,
        query_config: QueryConfig,
        history_root: PathBuf,
        workspace: PathBuf,
    ) -> Self {
        Self {
            client,
            tool_factory,
            tool_ctx,
            query_config,
            history_root,
            workspace,
            tools: None,
        }
    }

    fn with_static_tools(
        client: Arc<claurst_api::AnthropicClient>,
        tools: Arc<Vec<Box<dyn Tool>>>,
        tool_ctx: ToolContext,
        query_config: QueryConfig,
    ) -> Self {
        Self {
            client,
            tool_factory: Arc::new(|_, _| Arc::new(Vec::new())),
            tool_ctx,
            query_config,
            history_root: PathBuf::new(),
            workspace: PathBuf::new(),
            tools: Some(tools),
        }
    }
}

/// Handle used by the host to replace all cron runtime ownership on reconnect.
#[derive(Clone)]
pub struct CronSchedulerControl {
    runtime_tx: watch::Sender<Arc<CronSchedulerRuntime>>,
}

impl CronSchedulerControl {
    /// Atomically replace the runtime used by the next cron dispatch.
    ///
    /// A disconnected scheduler returns the supplied runtime so callers can
    /// retain or explicitly drop it; normal updates return `Ok(())`.
    pub fn update(&self, runtime: CronSchedulerRuntime) -> Result<(), Arc<CronSchedulerRuntime>> {
        match self.runtime_tx.send(Arc::new(runtime)) {
            Ok(()) => Ok(()),
            Err(error) => Err(error.0),
        }
    }
}

/// Start the background cron scheduler.
///
/// Returns immediately; the scheduler runs as a detached tokio task.
/// Call `cancel.cancel()` to stop it gracefully.
pub fn start_cron_scheduler(
    client: Arc<claurst_api::AnthropicClient>,
    tools: Arc<Vec<Box<dyn Tool>>>,
    tool_ctx: ToolContext,
    query_config: QueryConfig,
    cancel: CancellationToken,
) {
    let runtime = Arc::new(CronSchedulerRuntime::with_static_tools(
        client,
        tools,
        tool_ctx,
        query_config,
    ));
    let (runtime_tx, runtime_rx) = watch::channel(runtime);
    tokio::spawn(async move {
        let _runtime_tx = runtime_tx;
        run_scheduler_loop(runtime_rx, cancel, None).await;
    });
}

/// Start cron with an application-supplied, policy-filtered tool factory.
/// Each task receives a distinct durable session index bound to the canonical
/// workspace, including a workspace-bound `HistoryRead` instance when policy
/// allows it.
pub fn start_cron_scheduler_with_history(
    client: Arc<claurst_api::AnthropicClient>,
    tool_factory: CronToolFactory,
    tool_ctx: ToolContext,
    query_config: QueryConfig,
    history_root: PathBuf,
    workspace: PathBuf,
    cancel: CancellationToken,
) -> CronSchedulerControl {
    let runtime = Arc::new(CronSchedulerRuntime::new(
        client,
        tool_factory,
        tool_ctx,
        query_config,
        history_root,
        workspace,
    ));
    let (runtime_tx, runtime_rx) = watch::channel(runtime);
    let task_runtime_tx = runtime_tx.clone();
    tokio::spawn(async move {
        let _runtime_tx = task_runtime_tx;
        run_scheduler_loop(runtime_rx, cancel, None).await;
    });
    CronSchedulerControl { runtime_tx }
}

async fn run_scheduler_loop(
    mut runtime_rx: watch::Receiver<Arc<CronSchedulerRuntime>>,
    cancel: CancellationToken,
    mut tick_rx: Option<mpsc::Receiver<()>>,
) {
    info!("Cron scheduler started");

    loop {
        let Some(runtime) = next_runtime_tick(&mut runtime_rx, &cancel, &mut tick_rx).await else {
            info!("Cron scheduler stopped");
            return;
        };

        let tick_time = chrono::Local::now();
        debug!("Cron scheduler tick");

        // Find tasks due at this minute.
        let due = claurst_tools::cron::pop_due_tasks(&tick_time).await;

        for task in due {
            info!("Firing cron task");

            let Some(invocation) = prepare_cron_invocation(&runtime, &task) else {
                continue;
            };
            let CronInvocation {
                client,
                tools,
                tool_ctx,
                query_config,
                task_id,
            } = invocation;
            let cost_tracker = tool_ctx.cost_tracker.clone();
            let cancel_child = cancel.clone();

            tokio::spawn(async move {
                let mut messages = vec![Message::user(task.prompt.clone())];

                let outcome = run_query_loop(
                    client.as_ref(),
                    &mut messages,
                    &tools,
                    &tool_ctx,
                    &query_config,
                    cost_tracker,
                    None, // background — no UI event channel
                    cancel_child,
                    None, // no pending message queue for cron tasks
                )
                .await;

                match outcome {
                    QueryOutcome::EndTurn { .. } => {
                        info!("Cron task completed");
                    }
                    QueryOutcome::Error(_) => {
                        error!("Cron task failed");
                    }
                    QueryOutcome::MaxTokens { .. } => {
                        info!("Cron task hit max tokens");
                    }
                    QueryOutcome::Cancelled => {
                        debug!("Cron task cancelled");
                    }
                    QueryOutcome::BudgetExceeded {
                        cost_usd,
                        limit_usd,
                    } => {
                        eprintln!(
                            "[cron] task {} budget exceeded: spent ${:.4} of ${:.4}",
                            task_id, cost_usd, limit_usd
                        );
                    }
                }
            });
        }
    }
}

async fn next_runtime_tick(
    runtime_rx: &mut watch::Receiver<Arc<CronSchedulerRuntime>>,
    cancel: &CancellationToken,
    tick_rx: &mut Option<mpsc::Receiver<()>>,
) -> Option<Arc<CronSchedulerRuntime>> {
    loop {
        // Sleep until the next whole-minute boundary (±1s tolerance).
        let now = chrono::Local::now();
        let secs_into_minute = now.second() as u64;
        let nanos_ms = now.nanosecond() as u64 / 1_000_000;
        // How many ms until the next minute starts? Use saturating sub to avoid underflow.
        let ms_to_next_minute = (60u64.saturating_sub(secs_into_minute))
            .saturating_mul(1_000)
            .saturating_sub(nanos_ms)
            .max(1); // always sleep at least 1ms

        let tick = async {
            match tick_rx.as_mut() {
                Some(receiver) => receiver.recv().await,
                None => {
                    sleep(Duration::from_millis(ms_to_next_minute)).await;
                    Some(())
                }
            }
        };
        tokio::pin!(tick);
        tokio::select! {
            _ = cancel.cancelled() => return None,
            changed = runtime_rx.changed() => {
                if changed.is_err() {
                    return None;
                }
                let _ = runtime_rx.borrow_and_update();
                continue;
            }
            ticked = &mut tick => {
                ticked?;
                return Some(Arc::clone(&*runtime_rx.borrow_and_update()));
            }
        }
    }
}

struct CronInvocation {
    client: Arc<claurst_api::AnthropicClient>,
    tools: Arc<Vec<Box<dyn Tool>>>,
    tool_ctx: ToolContext,
    query_config: QueryConfig,
    task_id: String,
}

fn prepare_cron_invocation(
    runtime: &CronSchedulerRuntime,
    task: &claurst_tools::cron::CronTask,
) -> Option<CronInvocation> {
    let client = runtime.client.clone();
    let mut tool_ctx = runtime.tool_ctx.clone();
    let mut query_config = runtime.query_config.clone();
    // Cron runs are independent durable sessions.  They must not inherit
    // startup-bound checker authority or an Anthropic client factory tied to
    // the interactive workspace/session.
    query_config.verified_startup_policy = None;
    query_config.anthropic_client_factory = None;
    query_config.effort_policy = crate::effort_policy::EffortPolicy::Fixed;
    query_config.inventory_profile = crate::ToolInventoryProfile::Cron;
    // A cron invocation receives a fresh index below when the durable history
    // factory is available.  Never reuse the interactive session's index while
    // retaining the host inventory source.
    query_config.history_index = None;
    tool_ctx.trusted_checker_catalog = Arc::new(claurst_tools::TrustedCheckerCatalog::empty());
    // Cron runs have a distinct durable session/workspace.  Never inherit a
    // parent catalog bound to a different scope.
    tool_ctx.catalog_view = None;
    let cron_session_id = format!("cron-{}", uuid::Uuid::new_v4());
    tool_ctx.session_id = cron_session_id.clone();
    let tools = if let Some(tools) = runtime.tools.as_ref() {
        tools.clone()
    } else {
        let canonical_workspace = runtime
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| runtime.workspace.clone());
        let scope = crate::workspace_scope_id(&canonical_workspace);
        let index = match crate::HistoryIndex::open(&runtime.history_root, &scope, &cron_session_id)
        {
            Ok(index) => Arc::new(parking_lot::Mutex::new(index)),
            Err(_) => {
                error!("Cron history index failed to open");
                return None;
            }
        };
        let tools = (runtime.tool_factory)(index.clone(), &canonical_workspace);
        tool_ctx.working_dir = canonical_workspace;
        query_config.history_index = Some(index);
        tools
    };
    Some(CronInvocation {
        client,
        tools,
        tool_ctx,
        query_config,
        task_id: task.id.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Poll;

    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    fn test_context(working_dir: PathBuf, session_id: &str) -> ToolContext {
        ToolContext {
            working_dir,
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(claurst_core::permissions::AutoPermissionHandler {
                mode: claurst_core::config::PermissionMode::Default,
            }),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: session_id.to_owned(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: claurst_core::config::Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: CancellationToken::new(),
            trusted_checker_catalog: Arc::new(claurst_tools::TrustedCheckerCatalog::empty()),
            catalog_view: None,
        }
    }

    fn test_client() -> Arc<claurst_api::AnthropicClient> {
        Arc::new(
            claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
                api_key: "cron-test".to_owned(),
                ..Default::default()
            })
            .expect("test client should build"),
        )
    }

    #[tokio::test]
    async fn idle_runtime_update_releases_old_generation_and_dispatches_new_snapshot() {
        let old_dropped = Arc::new(AtomicBool::new(false));
        let old_probe = DropProbe(old_dropped.clone());
        let old_factory: CronToolFactory = Arc::new(move |_, _| {
            let _ = &old_probe;
            Arc::new(Vec::new())
        });
        let old_workspace = tempfile::tempdir().unwrap();
        let old_history = tempfile::tempdir().unwrap();
        let old_runtime = CronSchedulerRuntime::new(
            test_client(),
            old_factory,
            test_context(old_workspace.path().to_path_buf(), "old-session"),
            QueryConfig::default(),
            old_history.path().to_path_buf(),
            old_workspace.path().to_path_buf(),
        );
        let (runtime_tx, runtime_rx) = watch::channel(Arc::new(old_runtime));
        let control = CronSchedulerControl {
            runtime_tx: runtime_tx.clone(),
        };
        let cancel = CancellationToken::new();
        let (tick_tx, tick_rx) = mpsc::channel(1);
        let mut runtime_rx = runtime_rx;
        let mut tick_rx = Some(tick_rx);
        let mut next_tick = Box::pin(next_runtime_tick(&mut runtime_rx, &cancel, &mut tick_rx));
        let pending = futures::future::poll_fn(|cx| match next_tick.as_mut().poll(cx) {
            Poll::Pending => Poll::Ready(true),
            Poll::Ready(_) => Poll::Ready(false),
        })
        .await;
        assert!(
            pending,
            "scheduler wait should remain pending before its tick"
        );

        let new_workspace = tempfile::tempdir().unwrap();
        let new_history = tempfile::tempdir().unwrap();
        let (factory_tx, mut factory_rx) = mpsc::channel(1);
        let new_factory: CronToolFactory = Arc::new(move |_, workspace| {
            let _ = factory_tx.try_send(workspace.to_path_buf());
            Arc::new(Vec::new())
        });
        let new_runtime = CronSchedulerRuntime::new(
            test_client(),
            new_factory,
            test_context(new_workspace.path().to_path_buf(), "new-session"),
            QueryConfig::default(),
            new_history.path().to_path_buf(),
            new_workspace.path().to_path_buf(),
        );
        assert!(control.update(new_runtime.clone()).is_ok());

        tokio::time::timeout(Duration::from_secs(1), async {
            while !old_dropped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("idle scheduler should release its old runtime promptly");

        tick_tx.send(()).await.unwrap();
        let captured = tokio::time::timeout(Duration::from_secs(1), &mut next_tick)
            .await
            .unwrap()
            .expect("scheduler should capture the post-update runtime at its tick");
        let task = claurst_tools::cron::CronTask {
            id: "new-dispatch".to_owned(),
            cron: "* * * * *".to_owned(),
            prompt: "test".to_owned(),
            recurring: false,
            durable: false,
            created_at: 0,
        };
        let invocation = prepare_cron_invocation(&captured, &task)
            .expect("new runtime should prepare the next dispatch");
        let factory_workspace = tokio::time::timeout(Duration::from_secs(1), factory_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let expected_workspace = new_workspace.path().canonicalize().unwrap();
        assert_eq!(factory_workspace, expected_workspace);
        assert_eq!(invocation.tool_ctx.working_dir, expected_workspace);
        assert_eq!(invocation.task_id, "new-dispatch");
    }
}
