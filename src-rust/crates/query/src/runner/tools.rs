// Tool execution helpers: argument parsing, permission gating, and the
// single-tool / batch execution paths. Extracted from lib.rs (issue #232).
// Behavior-preserving move.

use crate::active_schema::ActiveSchemaSet;
use crate::*;
use claurst_tools::{ExecutionStatus, ToolEffect};
use futures::{FutureExt, StreamExt};

/// Maximum number of trusted observation calls polled at once.
pub(crate) const DEFAULT_TOOL_OBSERVATION_CAP: usize = 4;

/// Parse the accumulated JSON arguments of a streamed tool call.
///
/// Providers stream a tool call's arguments as a sequence of partial-JSON
/// deltas which we concatenate into a single buffer. A well-behaved
/// no-argument call yields an empty (or whitespace-only) buffer, which we
/// map to an empty object. Any *non-empty* buffer that fails to parse is
/// returned as an error rather than being silently replaced with `{}` — a
/// truncated stream must never cause a tool (e.g. Edit/Write) to run with
/// empty arguments (issue #215).
pub(crate) fn parse_tool_args(json_str: &str) -> Result<Value, serde_json::Error> {
    let trimmed = json_str.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(trimmed)
}

/// Whether a `PermissionLevel` must be gated by the central backstop.
///
/// Only `None` and `ReadOnly` are exempt; every other level (`Write`,
/// `Execute`, `Dangerous`, `Forbidden`) represents a side-effecting action that
/// the backstop must confirm before it runs.
#[cfg(test)]
pub(crate) fn permission_level_is_gated(level: PermissionLevel) -> bool {
    !matches!(level, PermissionLevel::None | PermissionLevel::ReadOnly)
}

/// Execute a single tool invocation.
pub(crate) async fn execute_tool(
    name: &str,
    input: &Value,
    tools: &[Box<dyn Tool>],
    ctx: &ToolContext,
) -> ToolResult {
    let tool = tools.iter().find(|t| t.name() == name);

    match tool {
        Some(tool) => {
            debug!(tool = name, "Executing tool");
            claurst_tools::execute_authorized(tool.as_ref(), input.clone(), ctx).await
        }
        None => {
            warn!(tool = name, "Unknown tool requested");
            ToolResult::error(format!("Unknown tool: {}", name))
        }
    }
}

/// Return a typed denial before any configured or plugin hook can run.
/// Interactive `Ask` decisions remain owned by `execute_authorized` at the
/// final execution boundary.
pub(crate) fn pre_hook_authorization_denial(
    name: &str,
    input: &Value,
    tools: &[Box<dyn Tool>],
    ctx: &ToolContext,
) -> Option<ToolResult> {
    let tool = tools.iter().find(|tool| tool.name() == name)?;
    claurst_tools::authorization_preflight(tool.as_ref(), input, ctx)
        .err()
        .map(|result| *result)
}

/// Execute a model-requested tool only when its schema was disclosed from the
/// current host catalog.  This check intentionally runs before the central
/// permission/mutation broker: guessed or stale names must not prompt, mutate,
/// or reach a tool implementation.
#[cfg(test)]
pub(crate) async fn execute_tool_with_schema(
    name: &str,
    input: &Value,
    tools: &[Box<dyn Tool>],
    ctx: &ToolContext,
    active_schemas: &ActiveSchemaSet,
    catalog_view: &claurst_tools::ToolCatalogView,
) -> ToolResult {
    execute_tool_with_inventory(name, input, tools, ctx, active_schemas, catalog_view, None).await
}

/// Execute a disclosed tool with an optional host inventory generation guard.
/// The guard runs immediately before permission/broker admission, after the
/// query has joined its pre-hook phase.
pub(crate) async fn execute_tool_with_inventory(
    name: &str,
    input: &Value,
    tools: &[Box<dyn Tool>],
    ctx: &ToolContext,
    active_schemas: &ActiveSchemaSet,
    catalog_view: &claurst_tools::ToolCatalogView,
    inventory_guard: Option<crate::tool_inventory::InventoryGuard>,
) -> ToolResult {
    if active_schemas.catalog_id() != catalog_view.catalog_id()
        || !catalog_view.matches_context(ctx)
        || !active_schemas.is_disclosed(name)
    {
        return ToolResult::error(
            "Tool call was not run because the tool is unavailable in the current disclosed catalog.",
        )
        .with_execution_status(ExecutionStatus::NotRun);
    }
    if inventory_guard
        .as_ref()
        .is_some_and(|guard| !guard.is_current(ctx))
    {
        return ToolResult::error(
            "Tool call was not run because the authorized tool inventory is stale.",
        )
        .with_execution_status(ExecutionStatus::NotRun);
    }
    let source = inventory_guard.as_ref().map(|guard| guard.source());
    crate::tool_inventory::with_inventory_source(source, execute_tool(name, input, tools, ctx))
        .await
}

/// Exercise the conservative default in cancellation regressions. Production
/// dispatch passes trusted effects to `run_tool_batch_with_effects` directly.
#[cfg(test)]
pub(crate) async fn run_tool_batch<F>(
    exec_futures: Vec<F>,
    cancel_token: &tokio_util::sync::CancellationToken,
) -> (Vec<ToolResult>, bool)
where
    F: std::future::Future<Output = ToolResult>,
{
    let effects = vec![ToolEffect::Unknown; exec_futures.len()];
    run_tool_batch_with_effects(exec_futures, effects, cancel_token).await
}

/// Execute calls in source order while overlapping only contiguous trusted
/// observations. Mutation and unknown calls are barriers. On cancellation,
/// completed calls retain their real results; unfinished observations are
/// canceled, started barriers are marked uncertain to prevent replay, and
/// queued calls that never started are marked as not run.
pub(crate) async fn run_tool_batch_with_effects<F>(
    exec_futures: Vec<F>,
    effects: Vec<ToolEffect>,
    cancel_token: &tokio_util::sync::CancellationToken,
) -> (Vec<ToolResult>, bool)
where
    F: std::future::Future<Output = ToolResult>,
{
    let count = exec_futures.len();
    if count == 0 {
        return (Vec::new(), false);
    }

    let mut pending: std::collections::VecDeque<(usize, F)> =
        exec_futures.into_iter().enumerate().collect();
    let mut results: Vec<Option<ToolResult>> = (0..count).map(|_| None).collect();
    let mut started = vec![false; count];
    let cap = DEFAULT_TOOL_OBSERVATION_CAP.max(1);
    let mut cancelled = false;

    while let Some((index, future)) = pending.pop_front() {
        if cancel_token.is_cancelled() {
            cancelled = true;
            break;
        }

        if effects.get(index).copied().unwrap_or(ToolEffect::Unknown) == ToolEffect::Observation {
            let mut group = vec![(index, future)];
            while group.len() < cap {
                let Some((next_index, _)) = pending.front() else {
                    break;
                };
                if effects
                    .get(*next_index)
                    .copied()
                    .unwrap_or(ToolEffect::Unknown)
                    != ToolEffect::Observation
                {
                    break;
                }
                if let Some(item) = pending.pop_front() {
                    group.push(item);
                } else {
                    break;
                }
            }

            for (group_index, _) in &group {
                started[*group_index] = true;
            }
            if run_tool_group(group, &mut results, cancel_token).await {
                cancelled = true;
                break;
            }
        } else {
            let group = vec![(index, future)];
            started[index] = true;
            if run_tool_group(group, &mut results, cancel_token).await {
                cancelled = true;
                break;
            }
        }
    }

    if cancelled {
        for index in 0..count {
            if results[index].is_none() {
                let effect = effects.get(index).copied().unwrap_or(ToolEffect::Unknown);
                results[index] = Some(if !started[index] {
                    ToolResult::error(
                        "Tool execution was cancelled before it started; this call was not run.",
                    )
                    .with_execution_status(ExecutionStatus::NotRun)
                } else {
                    match effect {
                        ToolEffect::Observation => ToolResult::error(TOOL_CANCELLED_MSG)
                            .with_execution_status(ExecutionStatus::Cancelled),
                        ToolEffect::Mutation | ToolEffect::Unknown => ToolResult::error(
                            "Tool execution was cancelled before completion; its effect is uncertain and it must not be replayed automatically.",
                        ).with_execution_status(ExecutionStatus::Unknown),
                    }
                });
            }
        }
    }

    (
        results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| ToolResult::error("Tool scheduler lost a result."))
            })
            .collect(),
        cancelled,
    )
}

async fn run_tool_group<F>(
    group: Vec<(usize, F)>,
    results: &mut [Option<ToolResult>],
    cancel_token: &tokio_util::sync::CancellationToken,
) -> bool
where
    F: std::future::Future<Output = ToolResult>,
{
    let mut pending = futures::stream::FuturesUnordered::new();
    for (index, future) in group {
        pending.push(async move { (index, future.await) });
    }

    while !pending.is_empty() {
        tokio::select! {
            biased;
            Some((index, result)) = pending.next() => {
                results[index] = Some(result);
            }
            _ = cancel_token.cancelled() => {
                // Poll ready outputs once after cancellation so results that
                // completed at the same boundary are retained and never replayed.
                while let Some(Some((index, result))) = pending.next().now_or_never() {
                    results[index] = Some(result);
                }
                return true;
            }
        }
    }
    false
}

/// Load persisted todos for `session_id` and return a nudge string if any are
/// incomplete (status != "completed"). Returns empty string otherwise.
pub(crate) fn build_todo_nudge(session_id: &str) -> String {
    let todos = claurst_tools::todo_write::load_todos(session_id);
    let incomplete_count = todos
        .iter()
        .filter(|t| t["status"].as_str() != Some("completed"))
        .count();
    if incomplete_count == 0 {
        String::new()
    } else {
        format!(
            "You have {} incomplete task{} in your TodoWrite list. \
             Make sure to complete all tasks before ending your response.",
            incomplete_count,
            if incomplete_count == 1 { "" } else { "s" }
        )
    }
}

#[cfg(test)]
mod discovery_gate_tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct DeferredTool {
        calls: Arc<AtomicUsize>,
    }

    struct AllowAll;

    struct DenyAll;

    impl claurst_core::permissions::PermissionHandler for AllowAll {
        fn check_permission(
            &self,
            _: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }

        fn request_permission(
            &self,
            _: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }
    }

    impl claurst_core::permissions::PermissionHandler for DenyAll {
        fn check_permission(
            &self,
            _: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Deny
        }

        fn request_permission(
            &self,
            _: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            panic!("deny-only preflight must not request interactive permission")
        }
    }

    fn allow_all_context() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(AllowAll),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "discovery-gate-test".to_string(),
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
            cancel_token: tokio_util::sync::CancellationToken::new(),
            trusted_checker_catalog: Arc::new(claurst_tools::TrustedCheckerCatalog::empty()),
            catalog_view: None,
        }
    }

    #[async_trait]
    impl Tool for DeferredTool {
        fn name(&self) -> &str {
            "DeferredTool"
        }

        fn description(&self) -> &str {
            "A deferred test tool"
        }

        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }

        fn advanced(&self) -> bool {
            true
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            ToolResult::success("executed")
        }
    }

    #[tokio::test]
    async fn undisclosed_tool_is_not_run_before_authorized_execution() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DeferredTool {
            calls: calls.clone(),
        })];
        let mut ctx = allow_all_context();
        let view = claurst_tools::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        ctx.catalog_view = Some(view.clone());
        let active = ActiveSchemaSet::initial(&view, &tools, true).unwrap();

        let result =
            execute_tool_with_schema("DeferredTool", &json!({}), &tools, &ctx, &active, &view)
                .await;

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.execution_status(), Some(ExecutionStatus::NotRun));
    }

    #[tokio::test]
    async fn schema_from_another_catalog_is_not_authority() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DeferredTool {
            calls: calls.clone(),
        })];
        let ctx = allow_all_context();
        let first = claurst_tools::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let second = claurst_tools::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&first, &tools, false).unwrap();

        let result =
            execute_tool_with_schema("DeferredTool", &json!({}), &tools, &ctx, &active, &second)
                .await;

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.execution_status(), Some(ExecutionStatus::NotRun));
    }

    #[test]
    fn capability_denial_is_terminal_before_hooks() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(DeferredTool {
            calls: calls.clone(),
        })];
        let mut ctx = allow_all_context();
        ctx.permission_handler = Arc::new(DenyAll);

        let denial = pre_hook_authorization_denial("DeferredTool", &json!({}), &tools, &ctx)
            .expect("explicit capability deny");

        assert_eq!(denial.execution_status(), Some(ExecutionStatus::NotRun));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
