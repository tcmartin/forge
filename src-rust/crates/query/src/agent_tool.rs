// AgentTool: spawn a sub-agent to handle a complex sub-task.
//
// Lives in cc-query (not cc-tools) to avoid a circular dependency:
//   cc-tools would need cc-query, but cc-query already needs cc-tools.
//
// The AgentTool creates a nested query loop with its own context, enabling
// the model to delegate complex work to specialized sub-agents. Each sub-agent:
//   - Runs its own agentic loop
//   - Has access to all tools (except AgentTool itself, preventing infinite recursion)
//   - Returns its final output as the tool result
//
// New capabilities (TS parity):
//   - `isolation: "worktree"` — run the agent in a dedicated git worktree so
//     file edits don't conflict with the parent checkout or sibling agents.
//   - `run_in_background: true` — fire-and-forget; returns agent_id immediately.
//     Use the `monitor` tool to check completion status/output.

use async_trait::async_trait;
use claurst_api::client::ClientConfig;
use claurst_api::{AnthropicClient, ModelRegistry, ProviderRegistry};
use claurst_core::types::Message;
use claurst_tools::{
    agent_tasks::{self, AgentTaskStatus},
    ExecutionStatus, PermissionLevel, Tool, ToolContext, ToolResult,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

use crate::child_policy::{
    acquire_child_admission, contain_writers, plan_tools, ChildCapabilityProfile, ChildToolPlan,
};
use crate::effort_policy::{PinProvenance, ValueProvenance, ValueSource};
use crate::{run_query_loop, QueryConfig, QueryOutcome};

// ---------------------------------------------------------------------------
// Worktree isolation helpers
// ---------------------------------------------------------------------------

const GIT_COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
const CHILD_QUIESCENCE_TIMEOUT: Duration = Duration::from_secs(3);

async fn bounded_git_output(mut command: tokio::process::Command) -> Option<std::process::Output> {
    command.kill_on_drop(true);
    tokio::time::timeout(GIT_COMMAND_TIMEOUT, command.output())
        .await
        .ok()?
        .ok()
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

async fn create_worktree(git_root: &Path, agent_id: &str) -> Option<PathBuf> {
    let worktree_root = claurst_core::config::Settings::config_dir().join("worktrees");
    tokio::fs::create_dir_all(&worktree_root).await.ok()?;
    let worktree_dir = worktree_root.join(format!("child-agent-{agent_id}"));
    let mut command = tokio::process::Command::new("git");
    command
        .args([
            "worktree",
            "add",
            "--detach",
            worktree_dir.to_str().unwrap_or_default(),
            "HEAD",
        ])
        .current_dir(git_root);
    let output = bounded_git_output(command).await?;
    if output.status.success() {
        Some(worktree_dir)
    } else {
        warn!("git worktree add failed");
        None
    }
}

/// Remove a child worktree only when it is clean. A dirty worktree is
/// deliberately preserved for the caller to inspect and recover edits from.
async fn cleanup_worktree(git_root: &Path, worktree_dir: &Path) -> bool {
    let mut status_command = tokio::process::Command::new("git");
    status_command
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(worktree_dir);
    let status = bounded_git_output(status_command).await;
    let is_clean = status
        .as_ref()
        .is_some_and(|output| output.status.success() && output.stdout.is_empty());
    if !is_clean {
        return true;
    }

    let mut remove_command = tokio::process::Command::new("git");
    remove_command
        .args([
            "worktree",
            "remove",
            worktree_dir.to_str().unwrap_or_default(),
        ])
        .current_dir(git_root);
    let output = bounded_git_output(remove_command).await;
    !output
        .as_ref()
        .map(|result| result.status.success())
        .unwrap_or(false)
}

fn child_context(
    parent: &ToolContext,
    working_dir: PathBuf,
    agent_id: &str,
    cancel_token: tokio_util::sync::CancellationToken,
    isolated: bool,
) -> ToolContext {
    let mut child = parent.clone();
    child.working_dir = working_dir.clone();
    child.session_id = format!("child-{agent_id}");
    child.cancel_token = cancel_token;
    // A catalog is bound to the parent workspace/session.  Child execution
    // gets a fresh query-owned catalog from its actual tool slice.
    child.catalog_view = None;
    // Hooks and shadow commits are parent-session behavior. Running them from
    // a child would create implicit writes outside the declared inventory.
    child.config.hooks.clear();
    child.config.formatter.clear();
    child.config.auto_commits = Some(false);
    if isolated {
        child.config.project_dir = Some(working_dir.clone());
        child.config.workspace_paths = vec![working_dir];
        child.config.additional_dirs.clear();
    }
    child
}

fn provenance(
    agent_id: &str,
    session_id: &str,
    working_dir: &Path,
    worktree_path: Option<&Path>,
    worktree_preserved: bool,
    outcome: &str,
) -> Value {
    json!({
        "child_agent_id": agent_id,
        "child_session_id": session_id,
        "working_dir": working_dir.display().to_string(),
        "worktree_path": worktree_path.map(|path| path.display().to_string()),
        "worktree_preserved": worktree_preserved,
        "child_outcome": outcome,
    })
}

fn annotate_child_lifecycle(
    mut result: ToolResult,
    shutdown: agent_tasks::ChildShutdown,
    worktree_preserved: bool,
    pending_parent_acceptance: bool,
) -> ToolResult {
    let cleanup_confirmed = matches!(shutdown.cleanup, agent_tasks::ChildCleanup::ConfirmedClean);
    let mut metadata = match result.metadata.take() {
        Some(Value::Object(metadata)) => metadata,
        Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
        None => serde_json::Map::new(),
    };
    metadata.insert(
        "child_quiescence".into(),
        json!({
            "cleanup_status": if cleanup_confirmed { "confirmed_clean" } else { "unknown" },
            "repl_sessions": shutdown.repl_sessions,
            "background_processes": shutdown.background_processes,
            "worktree_preserved": worktree_preserved,
        }),
    );
    if pending_parent_acceptance {
        metadata.insert("pending_parent_acceptance".into(), Value::Bool(true));
    }
    if !cleanup_confirmed {
        metadata.insert("cleanup_status".into(), Value::String("unknown".into()));
        metadata.insert(
            "cleanup_error".into(),
            Value::String("Child-owned services did not reach confirmed quiescence".into()),
        );
        result.is_error = true;
        result.content.push_str("\n\nChild-owned service cleanup is unknown; worktree and pending barriers were retained.");
        result = result.with_execution_status(ExecutionStatus::Unknown);
    }
    result.metadata = Some(Value::Object(metadata));
    result
}

// ---------------------------------------------------------------------------
// AgentTool
// ---------------------------------------------------------------------------

pub struct AgentTool;

fn build_model_registry() -> ModelRegistry {
    let mut registry = ModelRegistry::new();
    let cache_path = claurst_core::config::Settings::config_dir().join("models_dev.json");
    registry.load_cache(&cache_path);
    registry
}

fn resolve_subagent_model(params: &AgentInput, ctx: &ToolContext) -> String {
    let base_model = params
        .model
        .clone()
        .filter(|m| !m.is_empty())
        .or_else(|| {
            ctx.managed_agent_config
                .as_ref()
                .map(|c| c.executor_model.clone())
                .filter(|m| !m.is_empty())
        })
        .unwrap_or_else(|| ctx.config.effective_model().to_string());

    if base_model.contains('/') {
        base_model
    } else {
        let provider_id = ctx.config.selected_provider_id();
        if provider_id != "anthropic" {
            format!("{}/{}", provider_id, base_model)
        } else {
            base_model
        }
    }
}

#[derive(Debug, Deserialize)]
struct AgentInput {
    /// Short description of the agent's task.
    #[allow(dead_code)]
    description: String,
    /// The complete task prompt to send as the first user message.
    prompt: String,
    /// Optional: which tools to make available, intersected with parent policy
    /// and the selected capability profile.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Optional: system prompt override for the sub-agent.
    #[serde(default)]
    system_prompt: Option<String>,
    /// Optional: max turns for the sub-agent (default 10).
    #[serde(default)]
    max_turns: Option<u32>,
    /// Optional: model override for this sub-agent.
    #[serde(default)]
    model: Option<String>,
    /// Set to "worktree" to run the agent in an isolated git worktree.
    /// Omit (or set to null) for shared working directory.
    #[serde(default)]
    isolation: Option<String>,
    /// Child capability profile. Read-only defaults to trusted observations;
    /// mutating requires a successfully created worktree.
    #[serde(default, alias = "profile")]
    capability_profile: Option<String>,
    /// If true, start the agent in the background and return agent_id immediately.
    /// Default: false (wait for completion).
    #[serde(default)]
    run_in_background: bool,
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        claurst_core::constants::TOOL_NAME_AGENT
    }

    fn description(&self) -> &str {
        "Launch a new agent to handle complex, multi-step tasks autonomously. \
         The agent runs its own agentic loop with access to tools and returns \
         its final result. Use this to delegate sub-tasks, run parallel \
         workstreams, or handle tasks that require many tool calls."
    }

    fn permission_level(&self) -> PermissionLevel {
        // Launching a child agent is an executable side effect.
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of the agent's task (3-5 words)"
                },
                "prompt": {
                    "type": "string",
                    "description": "The complete task for the agent to perform"
                },
                "tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of tool names to make available after parent policy and capability filtering."
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt override for the sub-agent"
                },
                "max_turns": {
                    "type": "number",
                    "description": "Maximum number of turns for the sub-agent (default 10)"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model to use for this agent"
                },
                "isolation": {
                    "type": "string",
                    "enum": ["worktree"],
                    "description": "Set to \"worktree\" to run the agent in an isolated git worktree. \
                                    Prevents file-edit conflicts when multiple agents run in parallel."
                },
                "capability_profile": {
                    "type": "string",
                    "enum": ["read_only", "mutating"],
                    "description": "Read-only exposes trusted observation tools. Mutating requires worktree isolation and background mode so the parent can verify and accept the task artifact."
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "If true, the agent starts immediately and this call returns an \
                                    agent_id without waiting for completion. Use the monitor tool \
                                    with action=status/output and task_id=agent_id. Default: false."
                }
            },
            "required": ["description", "prompt"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: AgentInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let profile = match ChildCapabilityProfile::parse(params.capability_profile.as_deref()) {
            Ok(profile) => profile,
            Err(error) => return ToolResult::error(error),
        };
        if profile == ChildCapabilityProfile::Mutating && !params.run_in_background {
            return ToolResult::error(
                "Mutating children require run_in_background=true so TaskAccept can verify their isolated artifacts before completion",
            );
        }

        // Resolve model: explicit override > managed config executor model > provider default.
        let model = resolve_subagent_model(&params, ctx);

        let system_prompt = params.system_prompt.unwrap_or_else(|| {
            let mut prompt = "You are a specialized AI agent helping with a specific sub-task. \
             Complete the task thoroughly and return your findings."
                .to_string();

            // Append plugin-contributed agent definitions so the sub-agent
            // is aware of any specialised agents declared by plugins.
            if let Some(registry) = claurst_plugins::global_plugin_registry() {
                let mut agent_defs = String::new();
                for agent_dir in registry.all_agent_paths() {
                    if let Ok(entries) = std::fs::read_dir(&agent_dir) {
                        for entry in entries.flatten() {
                            let p = entry.path();
                            if p.extension().is_some_and(|e| e == "md") {
                                if let Ok(content) = std::fs::read_to_string(&p) {
                                    let name =
                                        p.file_stem().and_then(|s| s.to_str()).unwrap_or("agent");
                                    agent_defs.push_str(&format!(
                                        "\n\n## Agent: {}\n{}",
                                        name,
                                        content.trim()
                                    ));
                                }
                            }
                        }
                    }
                }
                if !agent_defs.is_empty() {
                    prompt.push_str("\n\nThe following specialized agents are available:");
                    prompt.push_str(&agent_defs);
                }
            }

            prompt
        });

        // Resolve max_turns: explicit > managed config executor_max_turns > default.
        let resolved_max_turns = params.max_turns.unwrap_or_else(|| {
            ctx.managed_agent_config
                .as_ref()
                .map(|c| c.executor_max_turns)
                .unwrap_or(10)
        });

        // Resolve isolation: explicit param > managed config executor_isolation.
        let resolved_isolation = params.isolation.clone().or_else(|| {
            if ctx
                .managed_agent_config
                .as_ref()
                .map(|c| c.executor_isolation)
                .unwrap_or(false)
            {
                Some("worktree".to_string())
            } else {
                None
            }
        });

        if let Some(ref isolation) = resolved_isolation {
            if isolation != "worktree" {
                return ToolResult::error(format!(
                    "unsupported child isolation '{isolation}'; use 'worktree'"
                ));
            }
        }

        // -----------------------------------------------------------------------
        // Determine working directory - optionally isolate in a git worktree.
        // -----------------------------------------------------------------------
        let use_isolation = resolved_isolation.as_deref() == Some("worktree");
        let agent_id = uuid::Uuid::new_v4().to_string();
        // Preserve the parent scope before any child worktree is created. The
        // cleanup path can remove the live child directory before foreground
        // evidence retention runs.
        let parent_workspace = claurst_tools::worktree::workspace_root(ctx);

        let (working_dir_str, worktree_path, git_root): (String, Option<PathBuf>, Option<PathBuf>) =
            if use_isolation {
                let git_root = find_git_root(&ctx.working_dir);
                if let Some(ref root) = git_root {
                    if let Some(wt) = create_worktree(root, &agent_id).await {
                        let wd = wt.display().to_string();
                        (wd, Some(wt), git_root)
                    } else {
                        return ToolResult::error(
                            "Requested worktree isolation could not be created; child was not started",
                        );
                    }
                } else {
                    return ToolResult::error(
                        "Requested worktree isolation requires a git repository; child was not started",
                    );
                }
            } else {
                (ctx.working_dir.display().to_string(), None, None)
            };

        let worktree_ready = worktree_path.is_some();
        let working_dir = PathBuf::from(&working_dir_str);
        let cancel = ctx.cancel_token.child_token();
        let child_session_id = format!("child-{agent_id}");
        let child_ctx = child_context(
            ctx,
            working_dir.clone(),
            &agent_id,
            cancel.clone(),
            worktree_ready,
        );
        let inventory_source = crate::tool_inventory::current_inventory_source();
        let child_history_index = match crate::HistoryIndex::open(
            claurst_core::config::Settings::config_dir(),
            &crate::workspace_scope_id(&working_dir),
            &child_session_id,
        ) {
            Ok(index) => Some(Arc::new(parking_lot::Mutex::new(index))),
            Err(error) if inventory_source.is_some() => {
                if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                    cleanup_worktree(root, wt).await;
                }
                return ToolResult::error(format!(
                    "Child history index could not be opened: {error}"
                ));
            }
            Err(_) => None,
        };
        let inventory_profile = crate::ToolInventoryProfile::AgentChild {
            // Source-backed planning starts with the host ceiling and applies
            // the child capability/name intersection below.  `Some` is
            // reserved for the exact accepted plan used by the query loop.
            allowed_tools: None,
            capability: if profile == ChildCapabilityProfile::Mutating && worktree_ready {
                crate::ChildInventoryCapability::ScopedFilesystemWriters
            } else {
                crate::ChildInventoryCapability::ObservationOnly
            },
        };
        let child_plan = if let Some(source) = inventory_source.as_ref() {
            let request = match crate::tool_inventory::InventoryRequest::for_context(
                &child_ctx,
                inventory_profile.clone(),
                child_history_index.as_ref(),
            ) {
                Ok(request) => request,
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(error.to_string());
                }
            };
            let snapshot = match source.snapshot(&request, &child_ctx).await {
                Ok(snapshot)
                    if snapshot.scope().matches_context(&child_ctx)
                        && snapshot
                            .tools()
                            .iter()
                            .all(|tool| request.profile().allows_handle(tool.as_ref())) =>
                {
                    snapshot
                }
                Ok(_) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(
                        "Child inventory scope or profile does not match the child context.",
                    );
                }
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(error.to_string());
                }
            };
            let generation = match source.current_generation(&request, &child_ctx) {
                Ok(generation) => generation,
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(error.to_string());
                }
            };
            if snapshot.generation() != generation {
                if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                    cleanup_worktree(root, wt).await;
                }
                return ToolResult::error("Child inventory changed during snapshot.");
            }
            match crate::child_policy::plan_tools_from_inventory(
                &ctx.config,
                snapshot,
                params.tools.as_deref(),
                profile,
                worktree_ready,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(error);
                }
            }
        } else {
            match plan_tools(
                &ctx.config,
                params.tools.as_deref(),
                profile,
                worktree_ready,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(error);
                }
            }
        };
        let child_plan = if worktree_ready {
            match contain_writers(child_plan, Path::new(&working_dir_str)) {
                Ok(plan) => plan,
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    return ToolResult::error(error);
                }
            }
        } else {
            child_plan
        };
        // Admission happens after isolation and inventory validation but before
        // constructing or invoking the provider request.
        let admission = match acquire_child_admission(ctx.cancel_token.clone()).await {
            Ok(permit) => permit,
            Err(error) => {
                if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                    cleanup_worktree(root, wt).await;
                }
                return ToolResult::error(error);
            }
        };

        let anthropic_key = ctx.config.resolve_anthropic_api_key().unwrap_or_default();
        let anthropic_base = ctx.config.resolve_anthropic_api_base();
        let client = match AnthropicClient::new(ClientConfig {
            api_key: anthropic_key.clone(),
            api_base: anthropic_base,
            ..Default::default()
        }) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                    cleanup_worktree(root, wt).await;
                }
                return ToolResult::error(format!("Failed to create client: {e}"));
            }
        };

        let provider_registry = ProviderRegistry::from_config(
            &ctx.config,
            ClientConfig {
                api_key: anthropic_key,
                api_base: ctx.config.resolve_anthropic_api_base(),
                ..Default::default()
            },
        );
        let model_registry = Arc::new(build_model_registry());
        let ChildToolPlan {
            tools: agent_tools,
            names: agent_tool_names,
        } = child_plan;
        let query_inventory_profile = if params.tools.as_ref().is_some_and(Vec::is_empty) {
            crate::ToolInventoryProfile::AgentChild {
                allowed_tools: Some(Arc::from(vec!["\0".to_string()].into_boxed_slice())),
                capability: if profile == ChildCapabilityProfile::Mutating && worktree_ready {
                    crate::ChildInventoryCapability::ScopedFilesystemWriters
                } else {
                    crate::ChildInventoryCapability::ObservationOnly
                },
            }
        } else {
            crate::ToolInventoryProfile::AgentChild {
                allowed_tools: Some(Arc::from(agent_tool_names.clone().into_boxed_slice())),
                capability: if profile == ChildCapabilityProfile::Mutating && worktree_ready {
                    crate::ChildInventoryCapability::ScopedFilesystemWriters
                } else {
                    crate::ChildInventoryCapability::ObservationOnly
                },
            }
        };

        let query_config = QueryConfig {
            model,
            capture_mcp_diagnostics: false,
            effort_policy: crate::effort_policy::EffortPolicy::Fixed,
            // Child model selection is an agent-owned override. Mark it before
            // normalization so future adaptive policy cannot mistake the
            // inherited/default wire model for an unpinned user choice.
            pin_provenance: PinProvenance::default()
                .with_model_source(ValueProvenance::new(ValueSource::Agent, true)),
            verified_startup_policy: None,
            verified_startup_binding: None,
            anthropic_client_factory: None,
            max_tokens: claurst_core::constants::DEFAULT_MAX_TOKENS,
            max_turns: resolved_max_turns,
            system_prompt: Some(system_prompt),
            append_system_prompt: None,
            output_style: ctx.config.effective_output_style(),
            output_style_prompt: ctx.config.resolve_output_style_prompt(),
            working_directory: Some(working_dir_str),
            thinking_budget: None,
            temperature: None,
            tool_result_budget: 50_000,
            context_budget: crate::ContextBudgetConfig::default(),
            history_index: child_history_index,
            effort_level: None,
            command_queue: None,
            skill_index: None,
            max_budget_usd: None,
            fallback_model: None,
            provider_registry: Some(Arc::new(provider_registry)),
            agent_name: None,
            agent_definition: None,
            model_registry: Some(model_registry),
            managed_agents: None,
            // Progressive tool disclosure (issue #233): the sub-agent's system
            // prompt only needs guideline blocks for the tools it actually has.
            enabled_tools: Some(agent_tool_names),
            // Sub-agents run to their own completion and never drive goal
            // continuation — stop after one turn like every non-goal run.
            continuation: crate::continuation::ContinuationMode::Default,
            child_execution: true,
            session_memory_enabled: false,
            session_memory_state: Default::default(),
            inventory_source: inventory_source.clone(),
            inventory_profile: query_inventory_profile,
        };
        // -----------------------------------------------------------------------
        // Background mode: spawn and return agent_id immediately.
        // -----------------------------------------------------------------------
        if params.run_in_background {
            let profile_bg = profile;
            let completion_owner = match agent_tasks::register(
                agent_id.clone(),
                "subagent".to_string(),
                ctx,
                child_session_id.clone(),
                working_dir.clone(),
                worktree_path.clone(),
                cancel.clone(),
            ) {
                Ok(owner) => owner,
                Err(error) => {
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
                        cleanup_worktree(root, wt).await;
                    }
                    drop(admission);
                    return ToolResult::error(error);
                }
            };

            let client_bg = client.clone();
            let ctx_bg = child_ctx.clone();
            let parent_ctx_bg = ctx.clone();
            let completion_owner_bg = completion_owner.clone();
            let config_bg = query_config.clone();
            let cost_tracker_bg = ctx.cost_tracker.clone();
            let prompt_bg = params.prompt.clone();
            let agent_id_bg = agent_id.clone();
            let child_session_id_bg = child_session_id.clone();
            let working_dir_bg = working_dir.clone();
            let worktree_path_bg = worktree_path.clone();
            let agent_tools_bg = agent_tools;

            tokio::spawn(async move {
                let mut messages = vec![Message::user(prompt_bg)];
                let outcome = run_query_loop(
                    client_bg.as_ref(),
                    &mut messages,
                    &agent_tools_bg,
                    &ctx_bg,
                    &config_bg,
                    cost_tracker_bg,
                    None,
                    cancel,
                    None,
                )
                .await;

                let shutdown =
                    agent_tasks::shutdown_child_services(&ctx_bg, CHILD_QUIESCENCE_TIMEOUT).await;
                let cleanup_confirmed =
                    matches!(shutdown.cleanup, agent_tasks::ChildCleanup::ConfirmedClean);
                let pending_parent_acceptance =
                    profile_bg == ChildCapabilityProfile::Mutating && worktree_path_bg.is_some();
                // Read-only clean worktrees can be removed after confirmed
                // quiescence. Mutating worktrees remain for explicit parent
                // acceptance, and unknown cleanup always preserves artifacts.
                let worktree_preserved =
                    if let (Some(root), Some(wt)) = (&git_root, &worktree_path_bg) {
                        if cleanup_confirmed && !pending_parent_acceptance {
                            cleanup_worktree(root, wt).await
                        } else {
                            true
                        }
                    } else {
                        false
                    };

                let outcome_name = outcome_name(&outcome);
                let status = if cleanup_confirmed {
                    agent_task_status(&outcome)
                } else {
                    AgentTaskStatus::Failed
                };
                let result = annotate_child_lifecycle(
                    tool_result_from_outcome_with_provenance(
                        outcome,
                        provenance(
                            &agent_id_bg,
                            &child_session_id_bg,
                            &working_dir_bg,
                            worktree_path_bg.as_deref(),
                            worktree_preserved,
                            outcome_name,
                        ),
                    ),
                    shutdown,
                    worktree_preserved,
                    pending_parent_acceptance,
                );
                if agent_tasks::finish(
                    &agent_id_bg,
                    &completion_owner_bg,
                    &parent_ctx_bg,
                    status,
                    shutdown.cleanup,
                    result,
                )
                .await
                .is_err()
                {
                    warn!("Failed to publish background agent result");
                }

                debug!("Background agent completed");
                drop(admission);
            });

            return ToolResult::success(
                serde_json::json!({
                    "agent_id": agent_id,
                    "status": "running",
                    "message": format!(
                        "Agent started in background. Use monitor with action=status/output and task_id='{}'.",
                        agent_id
                    ),
                    "child_session_id": child_session_id,
                    "working_dir": working_dir.display().to_string(),
                    "worktree_path": worktree_path.as_ref().map(|path| path.display().to_string()),
                })
                .to_string(),
            );
        }

        // -----------------------------------------------------------------------
        // Synchronous mode: run the sub-agent loop and wait for completion.
        // -----------------------------------------------------------------------
        let mut messages = vec![Message::user(params.prompt)];

        let outcome = run_query_loop(
            client.as_ref(),
            &mut messages,
            &agent_tools,
            &child_ctx,
            &query_config,
            ctx.cost_tracker.clone(),
            None, // no event forwarding for sub-agents
            cancel,
            None, // no pending message queue for sub-agents
        )
        .await;

        let shutdown =
            agent_tasks::shutdown_child_services(&child_ctx, CHILD_QUIESCENCE_TIMEOUT).await;
        let cleanup_confirmed =
            matches!(shutdown.cleanup, agent_tasks::ChildCleanup::ConfirmedClean);
        let pending_parent_acceptance =
            profile == ChildCapabilityProfile::Mutating && worktree_path.is_some();
        let worktree_preserved = if let (Some(root), Some(wt)) = (&git_root, &worktree_path) {
            if cleanup_confirmed && !pending_parent_acceptance {
                cleanup_worktree(root, wt).await
            } else {
                true
            }
        } else {
            false
        };
        let outcome_name = outcome_name(&outcome);
        let outcome_status = if !cleanup_confirmed {
            AgentTaskStatus::Failed
        } else if matches!(&outcome, QueryOutcome::EndTurn { .. })
            && ctx.cancel_token.is_cancelled()
        {
            AgentTaskStatus::CompletedAfterCancellationRequest
        } else {
            agent_task_status(&outcome)
        };
        let result = annotate_child_lifecycle(
            tool_result_from_outcome_with_provenance(
                outcome,
                provenance(
                    &agent_id,
                    &child_session_id,
                    &working_dir,
                    worktree_path.as_deref(),
                    worktree_preserved,
                    outcome_name,
                ),
            ),
            shutdown,
            worktree_preserved,
            pending_parent_acceptance,
        );
        let mut evidence_ctx = ctx.clone();
        evidence_ctx.working_dir = parent_workspace;
        claurst_tools::agent_tasks::retain_foreground_result(&evidence_ctx, outcome_status, result)
    }
}

/// Convert a nested query outcome into the parent tool result while retaining
/// partial output. A cancelled or max-token child may have already performed a
/// mutation, so its execution status is conservatively `Unknown`.
fn tool_result_from_outcome(outcome: QueryOutcome) -> ToolResult {
    match outcome {
        QueryOutcome::EndTurn { message, .. } => ToolResult::success(message.get_all_text()),
        QueryOutcome::MaxTokens {
            partial_message, ..
        } => ToolResult::error(format!(
            "{}\n\n[Note: Agent hit max_tokens limit]",
            partial_message.get_all_text()
        ))
        .with_metadata(json!({"agent_outcome": "max_tokens"}))
        .with_execution_status(ExecutionStatus::Unknown),
        QueryOutcome::Cancelled => ToolResult::error("Sub-agent was cancelled")
            .with_metadata(json!({"agent_outcome": "cancelled"}))
            .with_execution_status(ExecutionStatus::Unknown),
        QueryOutcome::Error(e) => ToolResult::error(format!("Sub-agent error: {}", e))
            .with_metadata(json!({"agent_outcome": "error"}))
            .with_execution_status(ExecutionStatus::Unknown),
        QueryOutcome::BudgetExceeded {
            cost_usd,
            limit_usd,
        } => ToolResult::error(format!(
            "Sub-agent stopped: budget ${:.4} exceeded (limit ${:.4})",
            cost_usd, limit_usd
        ))
        .with_metadata(json!({"agent_outcome": "budget_exceeded"}))
        .with_execution_status(ExecutionStatus::Unknown),
    }
}

fn outcome_name(outcome: &QueryOutcome) -> &'static str {
    match outcome {
        QueryOutcome::EndTurn { .. } => "completed",
        QueryOutcome::MaxTokens { .. } => "max_tokens",
        QueryOutcome::Cancelled => "cancelled",
        QueryOutcome::Error(_) => "error",
        QueryOutcome::BudgetExceeded { .. } => "budget_exceeded",
    }
}

fn agent_task_status(outcome: &QueryOutcome) -> AgentTaskStatus {
    match outcome {
        QueryOutcome::EndTurn { .. } => AgentTaskStatus::Completed,
        QueryOutcome::MaxTokens { .. } => AgentTaskStatus::MaxTokens,
        QueryOutcome::Cancelled => AgentTaskStatus::Cancelled,
        QueryOutcome::Error(_) => AgentTaskStatus::Failed,
        QueryOutcome::BudgetExceeded { .. } => AgentTaskStatus::BudgetExceeded,
    }
}

fn tool_result_from_outcome_with_provenance(
    outcome: QueryOutcome,
    provenance: Value,
) -> ToolResult {
    let mut result = tool_result_from_outcome(outcome);
    let mut metadata = result.metadata.take().unwrap_or_else(|| json!({}));
    if let Some(object) = metadata.as_object_mut() {
        object.insert("child_provenance".to_string(), provenance);
    }
    result.metadata = Some(metadata);
    result
}

// ---------------------------------------------------------------------------
// Helper: convert a QueryOutcome into a result string for background agents
// ---------------------------------------------------------------------------

fn format_outcome(outcome: QueryOutcome) -> String {
    match outcome {
        QueryOutcome::EndTurn { message, .. } => message.get_all_text(),
        QueryOutcome::MaxTokens {
            partial_message, ..
        } => format!(
            "{}\n\n[Note: Agent hit max_tokens limit]",
            partial_message.get_all_text()
        ),
        QueryOutcome::Cancelled => "[Agent was cancelled]".to_string(),
        QueryOutcome::Error(e) => format!("[Agent error: {}]", e),
        QueryOutcome::BudgetExceeded {
            cost_usd,
            limit_usd,
        } => format!(
            "[Agent stopped: budget ${:.4} exceeded (limit ${:.4})]",
            cost_usd, limit_usd
        ),
    }
}

// ---------------------------------------------------------------------------
// Team swarm runner injection
// ---------------------------------------------------------------------------
//
// Called once at process startup (e.g. from main.rs) to inject a real agent
// runner into cc-tools so that TeamCreateTool can spawn sub-agents via
// run_query_loop without creating a circular crate dependency.

/// Register the cc-query-backed agent runner with cc-tools.
///
/// After this call, `TeamCreateTool` will actually invoke `run_query_loop` for
/// each agent instead of returning stub output.
///
/// # Panics
/// Panics if the runner was already registered.
pub fn init_team_swarm_runner() {
    let runner: claurst_tools::AgentRunFn = Arc::new(
        |description: String,
         prompt: String,
         tools: Option<Vec<String>>,
         system: Option<String>,
         max_turns: Option<u32>,
         ctx: Arc<claurst_tools::ToolContext>| {
            // We must return a Pin<Box<dyn Future<...> + Send>>.
            Box::pin(async move {
                let cancel = ctx.cancel_token.child_token();
                let child_id = uuid::Uuid::new_v4().to_string();
                let child_ctx = child_context(
                    &ctx,
                    ctx.working_dir.clone(),
                    &child_id,
                    cancel.clone(),
                    false,
                );
                let inventory_source = crate::tool_inventory::current_inventory_source();
                let child_history_index = match crate::HistoryIndex::open(
                    claurst_core::config::Settings::config_dir(),
                    &crate::workspace_scope_id(&child_ctx.working_dir),
                    &child_ctx.session_id,
                ) {
                    Ok(index) => Some(Arc::new(parking_lot::Mutex::new(index))),
                    Err(error) if inventory_source.is_some() => {
                        return format!("[Agent child history unavailable: {error}]");
                    }
                    Err(_) => None,
                };
                let inventory_profile = crate::ToolInventoryProfile::AgentChild {
                    // Source-backed planning starts with the host ceiling and
                    // applies the child capability/name intersection below.
                    allowed_tools: None,
                    // Team runners do not create an isolated worktree here;
                    // keep the candidate and final profile observation-only.
                    capability: crate::ChildInventoryCapability::ObservationOnly,
                };
                let child_plan = if let Some(source) = inventory_source.as_ref() {
                    let request = match crate::tool_inventory::InventoryRequest::for_context(
                        &child_ctx,
                        inventory_profile.clone(),
                        child_history_index.as_ref(),
                    ) {
                        Ok(request) => request,
                        Err(error) => return format!("[Agent child inventory rejected: {error}]"),
                    };
                    let snapshot = match source.snapshot(&request, &child_ctx).await {
                        Ok(snapshot)
                            if snapshot.scope().matches_context(&child_ctx)
                                && snapshot
                                    .tools()
                                    .iter()
                                    .all(|tool| request.profile().allows_handle(tool.as_ref())) =>
                        {
                            snapshot
                        }
                        Ok(_) => {
                            return "[Agent child inventory scope or profile mismatch]".to_string()
                        }
                        Err(error) => {
                            return format!("[Agent child inventory unavailable: {error}]")
                        }
                    };
                    let generation = match source.current_generation(&request, &child_ctx) {
                        Ok(generation) => generation,
                        Err(error) => {
                            return format!("[Agent child inventory unavailable: {error}]");
                        }
                    };
                    if snapshot.generation() != generation {
                        return "[Agent child inventory changed during snapshot]".to_string();
                    }
                    match crate::child_policy::plan_tools_from_inventory(
                        &ctx.config,
                        snapshot,
                        tools.as_deref(),
                        ChildCapabilityProfile::ReadOnly,
                        false,
                    ) {
                        Ok(plan) => plan,
                        Err(error) => return format!("[Agent child policy rejected: {error}]"),
                    }
                } else {
                    match plan_tools(
                        &ctx.config,
                        tools.as_deref(),
                        ChildCapabilityProfile::ReadOnly,
                        false,
                    ) {
                        Ok(plan) => plan,
                        Err(error) => return format!("[Agent child policy rejected: {error}]"),
                    }
                };
                let admission = match acquire_child_admission(ctx.cancel_token.clone()).await {
                    Ok(permit) => permit,
                    Err(error) => return format!("[Agent child admission rejected: {error}]"),
                };
                let anthropic_key = ctx.config.resolve_anthropic_api_key().unwrap_or_default();
                let anthropic_base = ctx.config.resolve_anthropic_api_base();
                let client =
                    match claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
                        api_key: anthropic_key.clone(),
                        api_base: anthropic_base,
                        ..Default::default()
                    }) {
                        Ok(c) => Arc::new(c),
                        Err(e) => {
                            return format!(
                                "[Agent '{}' failed to create client: {}]",
                                description, e
                            )
                        }
                    };

                let provider_registry = ProviderRegistry::from_config(
                    &ctx.config,
                    claurst_api::client::ClientConfig {
                        api_key: anthropic_key,
                        api_base: ctx.config.resolve_anthropic_api_base(),
                        ..Default::default()
                    },
                );
                let model_registry = Arc::new(build_model_registry());

                let crate::child_policy::ChildToolPlan {
                    tools: agent_tools,
                    names,
                } = child_plan;
                let query_inventory_profile = if tools.as_ref().is_some_and(Vec::is_empty) {
                    crate::ToolInventoryProfile::AgentChild {
                        allowed_tools: Some(Arc::from(vec!["\0".to_string()].into_boxed_slice())),
                        capability: crate::ChildInventoryCapability::ObservationOnly,
                    }
                } else {
                    crate::ToolInventoryProfile::AgentChild {
                        allowed_tools: Some(Arc::from(names.clone().into_boxed_slice())),
                        capability: crate::ChildInventoryCapability::ObservationOnly,
                    }
                };
                let model = resolve_subagent_model(
                    &AgentInput {
                        description: description.clone(),
                        prompt: prompt.clone(),
                        tools: tools.clone(),
                        system_prompt: system.clone(),
                        max_turns,
                        model: None,
                        isolation: None,
                        capability_profile: None,
                        run_in_background: false,
                    },
                    &ctx,
                );

                let system_prompt = system.unwrap_or_else(|| {
                    "You are a specialized AI agent helping with a specific sub-task. \
                     Complete the task thoroughly and return your findings."
                        .to_string()
                });

                let query_config = crate::QueryConfig {
                    model,
                    effort_policy: crate::effort_policy::EffortPolicy::Fixed,
                    pin_provenance: PinProvenance::default()
                        .with_model_source(ValueProvenance::new(ValueSource::Agent, true)),
                    verified_startup_policy: None,
                    anthropic_client_factory: None,
                    max_tokens: claurst_core::constants::DEFAULT_MAX_TOKENS,
                    max_turns: max_turns.unwrap_or(10),
                    system_prompt: Some(system_prompt),
                    working_directory: Some(ctx.working_dir.display().to_string()),
                    output_style: ctx.config.effective_output_style(),
                    output_style_prompt: ctx.config.resolve_output_style_prompt(),
                    provider_registry: Some(Arc::new(provider_registry)),
                    model_registry: Some(model_registry),
                    history_index: child_history_index,
                    // Progressive tool disclosure (issue #233): only emit
                    // per-tool guidance for tools this team sub-agent has.
                    enabled_tools: Some(names),
                    // Nested team runs must not trigger implicit memory or
                    // AutoDream side effects.
                    child_execution: true,
                    inventory_source,
                    inventory_profile: query_inventory_profile,
                    ..Default::default()
                };

                let mut messages = vec![claurst_core::types::Message::user(prompt)];
                let outcome = crate::run_query_loop(
                    client.as_ref(),
                    &mut messages,
                    &agent_tools,
                    &child_ctx,
                    &query_config,
                    ctx.cost_tracker.clone(),
                    None,
                    cancel,
                    None,
                )
                .await;

                let result = format_outcome(outcome);
                drop(admission);
                result
            }) as Pin<Box<dyn std::future::Future<Output = String> + Send>>
        },
    );

    claurst_tools::register_agent_runner(runner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::types::Message;

    struct TestPermissionHandler;

    impl claurst_core::permissions::PermissionHandler for TestPermissionHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }

        fn request_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }
    }

    fn test_context() -> ToolContext {
        ToolContext {
            working_dir: PathBuf::from("/parent"),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(TestPermissionHandler),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "parent-session".to_string(),
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

    #[test]
    fn child_context_uses_actual_worktree_and_independent_cancel_identity() {
        let mut parent = test_context();
        parent.config.formatter.insert(
            "sentinel".to_string(),
            claurst_core::config::FormatterConfig {
                command: vec!["sentinel-formatter".to_string()],
                ..Default::default()
            },
        );
        let child_cancel = parent.cancel_token.child_token();
        let child = child_context(
            &parent,
            PathBuf::from("/worktrees/child"),
            "agent-123",
            child_cancel.clone(),
            true,
        );
        assert_eq!(child.working_dir, PathBuf::from("/worktrees/child"));
        assert_eq!(child.session_id, "child-agent-123");
        assert!(!child.cancel_token.is_cancelled());
        parent.cancel_token.cancel();
        assert!(child.cancel_token.is_cancelled());
        assert_eq!(
            child.config.workspace_paths,
            vec![PathBuf::from("/worktrees/child")]
        );
        assert!(child.config.additional_dirs.is_empty());
        assert!(child.config.formatter.is_empty());
        assert!(child.config.hooks.is_empty());
        assert_eq!(child.config.auto_commits, Some(false));
    }

    #[test]
    fn cancelled_agent_is_an_unknown_error() {
        let result = tool_result_from_outcome(QueryOutcome::Cancelled);

        assert!(result.is_error);
        assert_eq!(result.content, "Sub-agent was cancelled");
        let metadata = result.metadata.expect("cancellation metadata");
        assert_eq!(metadata["agent_outcome"], "cancelled");
        assert_eq!(metadata["execution_status"], "unknown");
    }

    #[test]
    fn max_tokens_agent_is_an_unknown_error_with_partial_output() {
        let result = tool_result_from_outcome(QueryOutcome::MaxTokens {
            partial_message: Message::user("partial output"),
            usage: Default::default(),
        });

        assert!(result.is_error);
        assert!(result.content.contains("partial output"));
        assert!(result.content.contains("max_tokens limit"));
        let metadata = result.metadata.expect("max-tokens metadata");
        assert_eq!(metadata["agent_outcome"], "max_tokens");
        assert_eq!(metadata["execution_status"], "unknown");
    }

    #[test]
    fn failed_and_budget_limited_agents_are_not_completed() {
        let error = tool_result_from_outcome(QueryOutcome::Error(
            claurst_core::error::ClaudeError::Api("provider failed".to_string()),
        ));
        assert!(error.is_error);
        assert_eq!(error.metadata.as_ref().unwrap()["agent_outcome"], "error");
        assert_eq!(
            error.metadata.as_ref().unwrap()["execution_status"],
            "unknown"
        );

        let budget = tool_result_from_outcome(QueryOutcome::BudgetExceeded {
            cost_usd: 1.25,
            limit_usd: 1.0,
        });
        assert!(budget.is_error);
        assert_eq!(
            budget.metadata.as_ref().unwrap()["agent_outcome"],
            "budget_exceeded"
        );
        assert_eq!(
            budget.metadata.as_ref().unwrap()["execution_status"],
            "unknown"
        );
    }

    #[test]
    fn completed_agent_remains_a_success_without_extra_metadata() {
        let result = tool_result_from_outcome(QueryOutcome::EndTurn {
            message: Message::user("completed output"),
            usage: Default::default(),
        });

        assert!(!result.is_error);
        assert_eq!(result.content, "completed output");
        assert!(result.metadata.is_none());
    }

    #[tokio::test]
    async fn mutating_child_requires_background_task_acceptance_identity() {
        let result = AgentTool
            .execute(
                json!({
                    "description": "mutating child",
                    "prompt": "change an artifact",
                    "capability_profile": "mutating",
                    "isolation": "worktree",
                    "run_in_background": false,
                }),
                &test_context(),
            )
            .await;

        assert!(result.is_error);
        assert!(result.content.contains("TaskAccept"));
    }
}
