// cc-query: The core agentic query loop.
//
// This crate implements the main conversation loop that:
// 1. Sends messages to the Anthropic API
// 2. Processes streaming responses
// 3. Detects tool-use requests and dispatches them
// 4. Feeds tool results back to the model
// 5. Handles auto-compact when the context window fills up
// 6. Manages stop conditions (end_turn, max_turns, cancellation)

// too_many_arguments: `run_query_loop` and related orchestration entrypoints
// thread many parameters by design; splitting would obscure the control flow.
#![allow(clippy::too_many_arguments)]

pub(crate) mod active_schema;
pub mod agent_tool;
pub mod auto_dream;
pub mod away_summary;
pub(crate) mod child_policy;
pub mod command_queue;
pub mod compact;
pub mod context_analyzer;
pub mod context_history;
pub mod context_plan;
pub mod continuation;
pub(crate) mod continuation_state;
pub mod coordinator;
pub mod cron_scheduler;
pub mod effort_policy;
pub mod event_channel;
pub mod goal_loop;
pub mod history_read;
pub mod host_inventory;
pub mod managed_orchestrator;
pub mod native_mcp_capture;
pub(crate) mod resolved_target;
pub mod sanitize;
pub mod session_memory;
pub mod skill_prefetch;
pub mod startup_policy;
pub mod tool_inventory;
pub mod verified_batch;

#[cfg(test)]
mod live_effort_tests;
#[cfg(test)]
mod live_effort_wire_tests;
#[cfg(test)]
mod live_inventory_tests;

mod runner;
pub use agent_tool::{init_team_swarm_runner, AgentTool};
pub use command_queue::{drain_command_queue, CommandPriority, CommandQueue, QueuedCommand};
pub use compact::{
    auto_compact_if_needed, calculate_messages_to_keep_index, calculate_token_warning_state,
    calculate_token_warning_state_for_window, compact_conversation, context_collapse,
    context_window_for_model, estimate_context_tokens, format_compact_summary, get_compact_prompt,
    group_messages_for_compact, micro_compact_if_needed, reactive_compact, resolve_context_window,
    should_auto_compact, should_auto_compact_for_window, should_compact, should_context_collapse,
    snip_compact, AutoCompactState, CompactResult, CompactTrigger, MessageGroup,
    MicroCompactConfig, TokenWarningState,
};
pub use context_history::{workspace_scope_id, HistoryIndex, HistoryRef};
pub use context_plan::{BudgetSource, ContextAssembler, ContextPlanError, RequestBudget};
pub use continuation::{
    ContinuationDecision, ContinuationMode, ContinuationPolicy, StopPolicy, TurnEndContext,
};
pub use cron_scheduler::{
    start_cron_scheduler, start_cron_scheduler_with_history, CronSchedulerControl,
    CronSchedulerRuntime, CronToolFactory,
};
pub use goal_loop::{
    check_and_continue_goal, decide_goal_continuation, mark_goal_complete, GoalContinuation,
    StopReason,
};
pub use history_read::{
    make_history_read_tool, make_history_read_tool_for_workspace, HistoryReadTool,
};
pub use runner::*;
pub use sanitize::sanitize_history;
pub use session_memory::{
    ExtractedMemory, MemoryCategory, SessionMemoryExtractor, SessionMemoryState,
};
pub use skill_prefetch::{
    format_skill_listing, prefetch_skills, SharedSkillIndex, SkillDefinition, SkillIndex,
};
pub use tool_inventory::{
    ChildInventoryCapability, HistoryBindingRequest, InventoryError, InventoryGeneration,
    InventoryRequest, ToolInventoryProfile, ToolInventorySnapshot, ToolInventorySource,
};

use claurst_api::{
    AnthropicStreamEvent, ApiMessage, ApiToolDefinition, CreateMessageRequest, DispatchRequest,
    StreamAccumulator, StreamHandler, SystemPrompt, ThinkingConfig,
};
use claurst_core::config::Config;
use claurst_core::cost::CostTracker;
use claurst_core::error::ClaudeError;
use claurst_core::types::{ContentBlock, Message, Role, ToolResultContent, UsageInfo};
#[cfg(test)]
use claurst_tools::PermissionLevel;
use claurst_tools::{
    with_mcp_query_operation_observer, McpOperationBackend, McpOperationObserver,
    McpQueryOperationIdentity,
};
use claurst_tools::{
    DiscoveryReceipt, NativeBashExecutionWitness, Tool, ToolCatalogSource, ToolCatalogView,
    ToolContext, ToolEffect, ToolResult, TrustedVerificationProof,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

const MAX_UNSENT_INVENTORY_RESTARTS: u8 = 16;

/// Resolve one native MCP selection witness for one dispatch boundary.  The
/// returned value is cloned for both the event and task-local scope; callers
/// must not re-query the selector and bind those observations independently.
fn selected_native_backend(
    identity: &native_mcp_capture::NativeMcpSelectedInvocation,
) -> McpOperationBackend {
    McpOperationBackend {
        server_name: identity.server_name.clone(),
        raw_name: identity.raw_name.clone(),
        callable_name: identity.callable_name.clone(),
        source_identity: identity.source_identity.clone(),
        connection_generation: identity.connection_generation,
        catalog_generation: identity.catalog_generation,
        definition_digest: identity.definition_digest.clone(),
    }
}

fn selected_native_mcp_identity(
    manager: Option<&claurst_tools::McpManager>,
    name: &str,
    activation: Option<&native_mcp_capture::NativeMcpActivationBinding>,
    active_schemas: &active_schema::ActiveSchemaSet,
    tools: &[Box<dyn Tool>],
) -> Option<native_mcp_capture::NativeMcpSelectedInvocation> {
    let manager = manager?;
    let activation = activation?;
    let tool = tools.iter().find(|tool| tool.name() == name)?;
    // Only the query-owned MCP wrapper (or its filtered SharedTool view) can
    // provide an executable descriptor. Catalog metadata on an arbitrary
    // inventory tool is insufficient for provider identity attribution.
    let executable = host_inventory::trusted_mcp_descriptor(tool.as_ref(), manager)?;
    let identity =
        native_mcp_capture::invocation_identity(Some(manager), name, activation, active_schemas)
            .ok()
            .flatten()?;
    (McpOperationBackend::from_descriptor(&executable) == selected_native_backend(&identity))
        .then_some(identity)
}

fn execute_with_selected_mcp_identity<'a, F>(
    identity: Option<native_mcp_capture::NativeMcpSelectedInvocation>,
    tool_id: String,
    event_tx: Option<&'a event_channel::QueryEventSender>,
    execution: F,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ToolResult> + Send + 'a>>
where
    F: std::future::Future<Output = ToolResult> + Send + 'a,
{
    Box::pin(async move {
        let Some(identity) = identity else {
            return execution.await;
        };
        let expected_backend = selected_native_backend(&identity);
        let tx = event_tx.cloned();
        let provider_tool_id = tool_id.clone();
        let observer: McpOperationObserver = Arc::new(move |context| {
            if let Some(tx) = tx.as_ref() {
                let _ = tx.send(QueryEvent::McpInvocation {
                    tool_id: tool_id.clone(),
                    identity: identity.clone(),
                    operation_context: Box::new(context.clone()),
                });
            }
        });
        with_mcp_query_operation_observer(
            McpQueryOperationIdentity {
                provider_tool_id: Some(provider_tool_id),
                observed_assistant_response_id: None,
                query_generated_uuid: None,
                expected_backend: Some(expected_backend),
            },
            observer,
            execution,
        )
        .await
    })
}

fn take_unsent_inventory_restart(restarts: &mut u8) -> bool {
    if *restarts >= MAX_UNSENT_INVENTORY_RESTARTS {
        return false;
    }
    *restarts = (*restarts).saturating_add(1);
    true
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Outcome of a single query-loop run.
#[derive(Debug)]
pub enum QueryOutcome {
    /// The model finished its turn (end_turn stop reason).
    EndTurn { message: Message, usage: UsageInfo },
    /// The model hit max_tokens.
    MaxTokens {
        partial_message: Message,
        usage: UsageInfo,
    },
    /// The conversation was cancelled by the user.
    Cancelled,
    /// An unrecoverable error occurred.
    Error(ClaudeError),
    /// The configured USD budget was exceeded.
    BudgetExceeded { cost_usd: f64, limit_usd: f64 },
}

/// Configuration for a single query-loop invocation.
#[derive(Clone)]
pub struct QueryConfig {
    pub model: String,
    /// Host opt-in for bounded MCP diagnostic projection. Ordinary queries do
    /// not construct capture bindings, serialize catalogs, or emit these events.
    pub capture_mcp_diagnostics: bool,
    /// Host-owned effort selection policy.  It has no serde or model-visible
    /// representation and defaults to the historical fixed behavior.
    pub effort_policy: effort_policy::EffortPolicy,
    /// Host-owned provenance captured before model and wire settings are
    /// normalized. Model selection is retained here for later adaptive
    /// policy decisions; it is never read from model-visible input.
    pub pin_provenance: effort_policy::PinProvenance,
    /// Immutable host policy loaded during startup.  The query loop binds it
    /// to its current session/workspace before selecting a verified route.
    pub verified_startup_policy: Option<Arc<startup_policy::LoadedVerifiedPolicy>>,
    /// Session-bound policy retained until provider resolution selects the
    /// current account and endpoint. Binding happens to that exact provider,
    /// before the resolved target snapshots its route identity.
    pub verified_startup_binding: Option<Arc<startup_policy::BoundVerifiedPolicy>>,
    /// Factory carrying the startup dispatch policy for fresh Anthropic
    /// clients.  It remains host-private and is absent for ordinary runs.
    pub anthropic_client_factory: Option<claurst_api::AnthropicClientFactory>,
    pub max_tokens: u32,
    pub max_turns: u32,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<String>,
    pub output_style: claurst_core::system_prompt::OutputStyle,
    pub output_style_prompt: Option<String>,
    pub working_directory: Option<String>,
    pub thinking_budget: Option<u32>,
    pub temperature: Option<f32>,
    /// Maximum cumulative character count of all tool results in the message
    /// Legacy compatibility field retained for callers that persist query
    /// settings. Live request selection uses the finite context planner and
    /// never mutates canonical tool results. Default: 50_000.
    pub tool_result_budget: usize,
    /// Finite request-context policy. The explicit limit takes precedence over
    /// the model registry and reported limit; unknown models use the finite
    /// fallback instead of an unlimited or heuristic budget.
    pub context_budget: ContextBudgetConfig,
    /// Optional durable session index used for requirement ingestion and
    /// omitted-history retrieval. The loop only holds its mutex while doing
    /// synchronous journal work, never across provider awaits.
    pub history_index: Option<Arc<parking_lot::Mutex<context_history::HistoryIndex>>>,
    /// Optional effort level.  When set and `thinking_budget` is `None`,
    /// the effort level's `thinking_budget_tokens()` is used as the
    /// thinking budget.  Also provides a temperature override when the
    /// level specifies one.
    pub effort_level: Option<claurst_core::effort::EffortLevel>,
    /// T1-4: Optional shared command queue.
    ///
    /// When set, the query loop drains this queue before each API call and
    /// injects any resulting messages into the conversation.  The queue is
    /// shared (Arc-backed) so the TUI input thread can push commands while the
    /// loop is waiting for a model response.
    pub command_queue: Option<CommandQueue>,
    /// T1-5: Optional shared skill index.
    ///
    /// When set, `prefetch_skills` is spawned once before the loop begins and
    /// the resulting index is used to inject a skill listing attachment into
    /// the conversation context.
    pub skill_index: Option<SharedSkillIndex>,
    /// Optional USD spend cap. The query loop checks accumulated cost after
    /// each turn and aborts with `QueryOutcome::BudgetExceeded` when exceeded.
    pub max_budget_usd: Option<f64>,
    /// Fallback model name. Used when the primary model returns overloaded /
    /// rate-limit errors (mirrors TS `--fallback-model`).
    pub fallback_model: Option<String>,
    /// Optional ProviderRegistry for dispatching to non-Anthropic providers.
    /// When `config.provider` is set to something other than "anthropic" and
    /// this registry contains that provider, the registry's provider is used
    /// instead of `AnthropicClient`.
    pub provider_registry: Option<std::sync::Arc<claurst_api::ProviderRegistry>>,
    /// Active agent name (e.g., "build", "plan", "explore", or None for default).
    pub agent_name: Option<String>,
    /// Resolved agent definition for the current session.
    pub agent_definition: Option<claurst_core::AgentDefinition>,
    /// Optional shared model registry for dynamic provider and model resolution.
    /// When set, the query loop uses this instead of constructing a fresh registry.
    pub model_registry: Option<std::sync::Arc<claurst_api::ModelRegistry>>,
    /// Managed agent (manager-executor) configuration.
    pub managed_agents: Option<claurst_core::ManagedAgentConfig>,
    /// Names of the tools enabled for this session (issue #233).
    ///
    /// When populated, `build_system_prompt` forwards these to
    /// `SystemPromptOptions::enabled_tools` so the "Tool use guidelines"
    /// section only emits per-tool guidance for tools that are actually
    /// loaded. `None` means "unknown" and emits no tool-specific claims;
    /// an empty vector means the caller explicitly exposes no tools.
    ///
    // Populated in-loop (issue #233 completion): when left `None`,
    // `run_query_loop` fills this from its live `tools: &[Box<dyn Tool>]`
    // argument before assembling the system prompt, so the top-level
    // interactive session gets progressive tool disclosure. Callers that build
    // both the tool vec and the config (e.g. sub-agents) may still set it
    // explicitly; the loop only fills an unset field.
    pub enabled_tools: Option<Vec<String>>,
    /// End-of-turn continuation policy (issue #230 / MI-3).
    ///
    /// `Default` stops after one turn (normal, non-goal behaviour). Goal-driven
    /// autonomy selects `Goal`, which keeps the loop running while an active
    /// goal's guards allow, injecting the goal continuation message as the next
    /// user turn — instead of the CLI REPL re-dispatching a fresh turn.
    pub continuation: crate::continuation::ContinuationMode,
    /// Child runs must not trigger implicit memory extraction or AutoDream.
    /// Interactive/top-level queries retain the default `false` behavior.
    pub child_execution: bool,
    /// Explicit host opt-in for session-memory extraction. This is disabled
    /// by default and is never inferred from project files or credentials.
    pub session_memory_enabled: bool,
    /// Cursor/state shared by turns in this host-owned session. Extraction
    /// updates it only after a successful, provider-bound request.
    pub session_memory_state: Arc<parking_lot::Mutex<session_memory::SessionMemoryState>>,
    /// Optional host-owned source for fresh policy-filtered tool snapshots.
    /// Manual callers leave this unset and retain the borrowed tool-slice API.
    pub inventory_source: Option<Arc<dyn ToolInventorySource>>,
    /// Effective host profile used when requesting source-backed inventory.
    /// Child and cron callers replace this with a fresh restricted profile
    /// while preserving the host source itself.
    pub inventory_profile: ToolInventoryProfile,
}

#[derive(Clone, Debug)]
pub struct ContextBudgetConfig {
    pub context_limit_tokens: Option<u64>,
    pub provider_reported_limit_tokens: Option<u64>,
    pub unknown_fallback_tokens: u64,
    pub wire_byte_limit: u64,
}

impl Default for ContextBudgetConfig {
    fn default() -> Self {
        Self {
            context_limit_tokens: None,
            provider_reported_limit_tokens: None,
            unknown_fallback_tokens: context_plan::UNKNOWN_CONTEXT_LIMIT,
            wire_byte_limit: 8 * 1024 * 1024,
        }
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            model: claurst_core::constants::DEFAULT_MODEL.to_string(),
            capture_mcp_diagnostics: false,
            effort_policy: effort_policy::EffortPolicy::Fixed,
            pin_provenance: effort_policy::PinProvenance::default(),
            verified_startup_policy: None,
            verified_startup_binding: None,
            anthropic_client_factory: None,
            max_tokens: claurst_core::constants::DEFAULT_MAX_TOKENS,
            max_turns: claurst_core::constants::MAX_TURNS_DEFAULT,
            system_prompt: None,
            append_system_prompt: None,
            output_style: claurst_core::system_prompt::OutputStyle::Default,
            output_style_prompt: None,
            working_directory: None,
            thinking_budget: None,
            temperature: None,
            tool_result_budget: 50_000,
            context_budget: ContextBudgetConfig::default(),
            history_index: None,
            effort_level: None,
            command_queue: None,
            skill_index: None,
            max_budget_usd: None,
            fallback_model: None,
            provider_registry: None,
            agent_name: None,
            agent_definition: None,
            model_registry: None,
            managed_agents: None,
            enabled_tools: None,
            continuation: crate::continuation::ContinuationMode::Default,
            child_execution: false,
            session_memory_enabled: false,
            session_memory_state: Arc::new(parking_lot::Mutex::new(
                session_memory::SessionMemoryState::new(),
            )),
            inventory_source: None,
            inventory_profile: ToolInventoryProfile::default(),
        }
    }
}

impl QueryConfig {
    /// Process hooks are parent-session behavior. Child runs must never spawn
    /// configured or process-global hook shells outside their scoped tool set.
    fn process_hooks_enabled(&self) -> bool {
        !self.child_execution
    }

    pub fn from_config(cfg: &Config) -> Self {
        let model_source = if cfg.model.is_some() {
            effort_policy::ValueProvenance::new(effort_policy::ValueSource::Settings, true)
        } else {
            effort_policy::ValueProvenance::default()
        };
        Self {
            model: cfg.effective_model().to_string(),
            pin_provenance: effort_policy::PinProvenance::default().with_model_source(model_source),
            max_tokens: cfg.effective_max_tokens(),
            output_style: cfg.effective_output_style(),
            output_style_prompt: cfg.resolve_output_style_prompt(),
            working_directory: cfg.project_dir.as_ref().map(|p| p.display().to_string()),
            managed_agents: cfg.managed_agents.clone(),
            ..Default::default()
        }
    }

    /// Build a QueryConfig using dynamic model resolution from the model registry.
    ///
    /// Prefers the best model for the configured provider (from models.dev data)
    /// over the hardcoded defaults.
    pub fn from_config_with_registry(cfg: &Config, registry: &claurst_api::ModelRegistry) -> Self {
        // We can't move the Arc here, but we need a clone for the query loop.
        // Callers typically wrap the registry in an Arc already.
        let model_source = if cfg.model.is_some() {
            effort_policy::ValueProvenance::new(effort_policy::ValueSource::Settings, true)
        } else {
            effort_policy::ValueProvenance::default()
        };
        Self {
            model: claurst_api::effective_model_for_config(cfg, registry),
            pin_provenance: effort_policy::PinProvenance::default().with_model_source(model_source),
            max_tokens: cfg.effective_max_tokens(),
            output_style: cfg.effective_output_style(),
            output_style_prompt: cfg.resolve_output_style_prompt(),
            working_directory: cfg.project_dir.as_ref().map(|p| p.display().to_string()),
            managed_agents: cfg.managed_agents.clone(),
            ..Default::default()
        }
    }
}

/// Query-local provider-managed response state. It never crosses a query,
/// workspace/session scope, or exact resolved transport identity.
struct ProviderResponseContinuity<'a> {
    target: Arc<resolved_target::ResolvedTarget<'a>>,
    scope: claurst_tools::ToolCatalogScope,
    anchor: claurst_api::ResponseContinuation,
}

/// Events emitted by the query loop for the TUI to render.
#[derive(Debug, Clone)]
pub enum QueryEvent {
    /// Bounded Codex wire/native binding metadata. It is not user-facing
    /// Anthropic content and is only rendered by stream-json.
    ProviderWireWitness(claurst_api::ProviderWireWitness),
    /// Checked MCP catalog/schema metadata captured after receipt activation.
    /// This is diagnostic-only and never execution authority.
    McpDiscovery(native_mcp_capture::NativeMcpDiscoveryWitness),
    /// Completed MCP wire operation emitted by the actual backend after exact
    /// request/response binding and executable source validation.
    McpInvocation {
        tool_id: String,
        identity: native_mcp_capture::NativeMcpSelectedInvocation,
        operation_context: Box<claurst_tools::McpOperationContext>,
    },
    /// A stream event from the API.
    Stream(AnthropicStreamEvent),
    /// A tool is about to be executed.
    ToolStart {
        tool_name: String,
        tool_id: String,
        input_json: String,
    },
    /// A tool has finished executing.
    ToolEnd {
        tool_name: String,
        tool_id: String,
        result: String,
        is_error: bool,
        /// Present only on stream-json for a registered built-in PTY Bash
        /// handle. Other surfaces intentionally discard this additive field.
        execution_witness: Option<NativeBashExecutionWitness>,
    },
    /// The model finished a turn.
    TurnComplete {
        turn: u32,
        stop_reason: String,
        usage: Option<UsageInfo>,
    },
    /// An informational status message.
    Status(String),
    /// An error.
    Error(String),
    /// Token usage has crossed a warning threshold.
    /// `state` is Warning (≥ 80 %) or Critical (≥ 95 %).
    /// `pct_used` is the fraction of the context window consumed (0.0–1.0).
    TokenWarning {
        state: TokenWarningState,
        pct_used: f64,
    },
    /// Host-owned effort selection status.  The payload is categorical and
    /// contains no account, route, body, or checker descriptor values.
    EffortSelection(EffortSelection),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffortSelectionMode {
    Fixed,
    Verified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffortSelectionReason {
    Initial,
    Changed,
    Explicit,
    FixedPolicy,
    NoProof,
    Pinned,
    UnsupportedRoute,
    HooksConfigured,
    LocalBudgetReselection,
    ProofInvalidated,
    StartupScopeUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffortSelection {
    requested: EffortSelectionMode,
    applied: EffortSelectionMode,
    selected: Option<claurst_core::effort::EffortLevel>,
    reason: EffortSelectionReason,
}

impl EffortSelection {
    pub const fn new(
        requested: EffortSelectionMode,
        applied: EffortSelectionMode,
        selected: Option<claurst_core::effort::EffortLevel>,
        reason: EffortSelectionReason,
    ) -> Self {
        Self {
            requested,
            applied,
            selected,
            reason,
        }
    }

    pub const fn requested(self) -> EffortSelectionMode {
        self.requested
    }

    pub const fn applied(self) -> EffortSelectionMode {
        self.applied
    }

    pub const fn selected(self) -> Option<claurst_core::effort::EffortLevel> {
        self.selected
    }

    pub const fn reason(self) -> EffortSelectionReason {
        self.reason
    }

    /// Stable, user-facing status text.  Keep transport, account, and
    /// checker details out of this summary; those remain host-private.
    pub fn status_line(self) -> String {
        let selected = self
            .selected
            .map(|effort| effort.as_str())
            .unwrap_or("default");
        let mode = match self.applied {
            EffortSelectionMode::Fixed => "fixed",
            EffortSelectionMode::Verified => "verified",
        };
        let reason = match self.reason {
            EffortSelectionReason::Initial => "initial",
            EffortSelectionReason::Changed => "changed",
            EffortSelectionReason::Explicit => "explicit",
            EffortSelectionReason::FixedPolicy => "default",
            EffortSelectionReason::NoProof => "no proof",
            EffortSelectionReason::Pinned => "pinned",
            EffortSelectionReason::UnsupportedRoute => "unsupported route",
            EffortSelectionReason::HooksConfigured => "configured hooks",
            EffortSelectionReason::LocalBudgetReselection => "local budget",
            EffortSelectionReason::ProofInvalidated => "proof invalidated",
            EffortSelectionReason::StartupScopeUnavailable => "startup scope unavailable",
        };
        format!("Effort: {selected} ({mode}; {reason})")
    }

    /// Default fixed effort is intentionally silent in machine-facing stderr
    /// diagnostics.  Explicit effort and adaptive outcomes remain visible.
    pub const fn is_default_fixed(self) -> bool {
        matches!(self.reason, EffortSelectionReason::FixedPolicy)
    }
}

// ---------------------------------------------------------------------------
// Tool-result budgeting
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Query loop
// ---------------------------------------------------------------------------

/// Maximum number of max_tokens continuation attempts before surfacing the
/// partial response.  Mirrors `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT` in query.ts.
const MAX_TOKENS_RECOVERY_LIMIT: u32 = 3;

/// Message injected when the model hits its output-token limit.
/// Mirrors the TS recovery message in query.ts lines 1224-1228.
const MAX_TOKENS_RECOVERY_MSG: &str =
    "Output token limit hit. Resume directly — no apology, no recap of what \
     you were doing. Pick up mid-thought if that is where the cut happened. \
     Break remaining work into smaller pieces.";

/// Injected as the final user turn when `effective_max_turns` is reached. That
/// turn runs with tools DISABLED (graceful degradation, mirroring opencode's
/// max-steps `toolChoice:"none"` behaviour), so the model produces a plain-text
/// wrap-up instead of the loop returning cold.
const MAX_STEPS_DEGRADATION_MSG: &str =
    "You have reached the maximum number of steps for this run, so tools are now \
     disabled — do not attempt to call any tools. In plain text, briefly \
     summarize what you accomplished, what remains unfinished, and exactly where \
     you stopped, so the work can be resumed later.";

/// Content stored in the synthetic `tool_result` for a tool that was abandoned
/// mid-flight because the query loop was cancelled (issue #218). Every
/// outstanding `tool_use` still receives a matching `tool_result` carrying this
/// text so the message history stays well-formed.
const TOOL_CANCELLED_MSG: &str = "Tool execution was cancelled by the user before it completed.";
const HISTORY_MANIFEST_RESERVE_BYTES: usize = 4 * 1024;
const MAX_HISTORY_MANIFEST_REFS: usize = 64;
const MAX_HISTORY_MANIFEST_BYTES: usize = 4 * 1024;

fn merge_provider_stream_usage(current: &mut UsageInfo, update: &UsageInfo) {
    if update.total_input() > 0 {
        current.input_tokens = update.input_tokens;
        current.cache_read_input_tokens = update.cache_read_input_tokens;
        current.cache_creation_input_tokens = update.cache_creation_input_tokens;
    }
    if update.output_tokens > 0 {
        current.output_tokens = update.output_tokens;
    }
}

// Spinner verbs are imported from claurst_core::spinner

/// Resolve the effective effort level for a turn.
///
/// Ultracode is a keyword-activated effort: if the most recent user message
/// contains the `ultracode` keyword (whole-word, case-insensitive), the effort
/// for this turn is raised to [`EffortLevel::Ultracode`] — the model's top
/// reasoning plus the ultracode operating procedure (injected as a system
/// addendum by the loop). Otherwise the configured `config_effort` is used
/// unchanged. Checking only the *last* user message keeps the mode scoped to the
/// turn that asked for it (a later plain turn deactivates it automatically).
///
/// [`EffortLevel::Ultracode`]: claurst_core::effort::EffortLevel::Ultracode
#[cfg(test)]
fn effective_effort_for_turn(
    config_effort: Option<claurst_core::effort::EffortLevel>,
    messages: &[Message],
) -> Option<claurst_core::effort::EffortLevel> {
    if let Some(last_user) = messages.iter().rev().find(|m| m.role == Role::User) {
        if claurst_core::effort::text_triggers_ultracode(&last_user.get_all_text()) {
            return Some(claurst_core::effort::EffortLevel::Ultracode);
        }
    }
    config_effort
}

/// Return the inline ultracode state from a genuine plain-text user turn.
/// Tool results and host-generated user messages are represented as user-role
/// messages for provider compatibility, but they must not create or clear the
/// user's work-level effort pin.
fn initial_inline_ultracode_state(messages: &[Message]) -> bool {
    messages
        .iter()
        .rev()
        .find_map(|message| {
            if message.role != Role::User {
                return None;
            }
            match &message.content {
                claurst_core::types::MessageContent::Text(text) => {
                    Some(claurst_core::effort::text_triggers_ultracode(text))
                }
                claurst_core::types::MessageContent::Blocks(_) => None,
            }
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn effective_pin_provenance(
    config: &QueryConfig,
    effective_effort: Option<claurst_core::effort::EffortLevel>,
    effective_thinking_budget: Option<u32>,
    effective_temperature: Option<f32>,
    messages: &[Message],
) -> effort_policy::PinProvenance {
    effective_pin_provenance_for_inline(
        config,
        effective_effort,
        effective_thinking_budget,
        effective_temperature,
        initial_inline_ultracode_state(messages),
    )
}

fn effective_pin_provenance_for_inline(
    config: &QueryConfig,
    effective_effort: Option<claurst_core::effort::EffortLevel>,
    effective_thinking_budget: Option<u32>,
    effective_temperature: Option<f32>,
    inline_ultracode: bool,
) -> effort_policy::PinProvenance {
    let mut provenance = config.pin_provenance.conservative_for_present_wire(
        config.effort_level.is_some(),
        config.thinking_budget.is_some(),
        config.temperature.is_some(),
    );

    if config
        .agent_definition
        .as_ref()
        .and_then(|agent| agent.model.as_ref())
        .is_some()
    {
        provenance = provenance.with_model_source(effort_policy::ValueProvenance::new(
            effort_policy::ValueSource::Agent,
            true,
        ));
    }

    if config
        .managed_agents
        .as_ref()
        .is_some_and(|managed| managed.enabled && !managed.manager_model.is_empty())
    {
        provenance = provenance.with_model_source(effort_policy::ValueProvenance::new(
            effort_policy::ValueSource::Agent,
            true,
        ));
    }

    if inline_ultracode {
        provenance = provenance.with_effort(effort_policy::ValueProvenance::new(
            effort_policy::ValueSource::InlineKeyword,
            false,
        ));
    }

    // Values derived from the selected effort inherit its trusted source. A
    // default or verified effort therefore stays adaptable, while an unknown
    // raw override remains pinned conservatively.
    let derived_effort = provenance.effort();
    if effective_effort.is_some()
        && config.thinking_budget.is_none()
        && effective_thinking_budget.is_some()
    {
        provenance = provenance.with_thinking_budget(derived_effort);
    }

    if config.temperature.is_none()
        && config
            .agent_definition
            .as_ref()
            .and_then(|agent| agent.temperature)
            .is_some()
    {
        provenance = provenance.with_temperature(effort_policy::ValueProvenance::new(
            effort_policy::ValueSource::Agent,
            true,
        ));
    } else if effective_effort.is_some()
        && config.temperature.is_none()
        && effective_temperature.is_some()
    {
        provenance = provenance.with_temperature(derived_effort);
    }

    provenance
}

/// Resolve the effective output-style persona for a turn.
///
/// Personas (`rocky` / `caveman` / `normal`) mirror the ultracode keyword: an
/// **inline** persona word in the most recent user message applies to *that one
/// turn* (transient) and then reverts, while the persona chosen via `/rocky`,
/// `/caveman`, or `/output-style` lives in `config` and **persists** until
/// changed. Inline `normal` resets to the default (no persona) for the turn.
///
/// Returns the `(output_style, output_style_prompt)` pair to assemble the
/// system prompt with for this turn. When no inline persona keyword is present,
/// the configured (persistent) pair is returned unchanged. Checking only the
/// *last* user message keeps the mode scoped to the turn that asked for it.
fn effective_output_style_for_turn(
    config: &QueryConfig,
    messages: &[Message],
) -> (claurst_core::system_prompt::OutputStyle, Option<String>) {
    if let Some(last_user) = messages.iter().rev().find(|m| m.role == Role::User) {
        if let Some(style_name) =
            claurst_core::keywords::inline_persona_style(&last_user.get_all_text())
        {
            // Inline `normal` (→ "default") resets the persona for this turn.
            if style_name == "default" {
                return (claurst_core::system_prompt::OutputStyle::Default, None);
            }
            // Otherwise apply the named persona's prompt for this turn only.
            let prompt = claurst_core::output_styles::find_style(
                &claurst_core::output_styles::builtin_styles(),
                style_name,
            )
            .map(|style| style.prompt.clone())
            .filter(|prompt| !prompt.trim().is_empty());
            return (claurst_core::system_prompt::OutputStyle::Default, prompt);
        }
    }
    // No inline persona keyword — keep the persistent selection.
    (config.output_style, config.output_style_prompt.clone())
}

fn ensure_message_ids(messages: &mut [Message]) {
    for message in messages {
        ensure_message_id(message);
    }
}

fn ensure_message_id(message: &mut Message) {
    if message.uuid.as_deref().is_none_or(str::is_empty) {
        message.uuid = Some(uuid::Uuid::new_v4().to_string());
    }
}

fn ensure_unique_message_id(messages: &[Message], message: &mut Message) {
    ensure_message_id(message);
    if messages
        .iter()
        .any(|existing| existing.uuid == message.uuid)
    {
        message.uuid = Some(uuid::Uuid::new_v4().to_string());
    }
}

fn push_message_with_id(messages: &mut Vec<Message>, mut message: Message) {
    ensure_unique_message_id(messages, &mut message);
    messages.push(message);
}

fn user_request_id(
    messages: &[Message],
    runtime_ids: &std::collections::BTreeSet<String>,
) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| {
            message.role == Role::User
                && !message
                    .uuid
                    .as_ref()
                    .is_some_and(|id| runtime_ids.contains(id))
                && !message.get_all_text().trim().is_empty()
        })
        .and_then(|message| message.uuid.clone())
}

fn latest_genuine_user_text<'a>(
    messages: &'a [Message],
    runtime_ids: &std::collections::BTreeSet<String>,
) -> Option<&'a str> {
    messages.iter().rev().find_map(|message| {
        if message.role != Role::User
            || message
                .uuid
                .as_ref()
                .is_some_and(|id| runtime_ids.contains(id))
        {
            return None;
        }
        let claurst_core::types::MessageContent::Text(text) = &message.content else {
            return None;
        };
        let text = text.trim();
        if text.is_empty() || text.starts_with("[System]:") {
            None
        } else {
            Some(text)
        }
    })
}

fn ingest_history_index(
    index: &Arc<parking_lot::Mutex<context_history::HistoryIndex>>,
    messages: &[Message],
    runtime_ids: &mut std::collections::BTreeSet<String>,
) -> Result<(), String> {
    let mut index = index.lock();
    runtime_ids.extend(index.runtime_message_ids());
    for message in messages {
        if message.role != Role::User {
            continue;
        }
        let text = message.get_all_text();
        if text.trim().is_empty() {
            continue;
        }
        index
            .ingest(message.uuid.as_deref(), &text)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn append_runtime_user_message(
    messages: &mut Vec<Message>,
    runtime_ids: &mut std::collections::BTreeSet<String>,
    index: Option<&Arc<parking_lot::Mutex<context_history::HistoryIndex>>>,
    text: impl Into<String>,
    origin: context_history::RuntimeMessageOrigin,
) -> Result<(), String> {
    let text = text.into();
    let mut message = Message::user(&text);
    let message_id = uuid::Uuid::new_v4().to_string();
    message.uuid = Some(message_id.clone());
    runtime_ids.insert(message_id.clone());
    if let Some(index) = index {
        index
            .lock()
            .record_runtime_message(&message_id, &text, origin)
            .map_err(|error| error.to_string())?;
    }
    messages.push(message);
    Ok(())
}

fn append_active_requirements(
    system: SystemPrompt,
    index: Option<&Arc<parking_lot::Mutex<context_history::HistoryIndex>>>,
) -> Result<SystemPrompt, String> {
    let Some(index) = index else {
        return Ok(system);
    };
    let view = index
        .lock()
        .ledger()
        .active_view()
        .map_err(|error| error.to_string())?;
    if view.full.is_empty() && view.references.is_empty() {
        return Ok(system);
    }
    let mut block = String::from(
        "\n\n<active-requirements>\nThese durable requirements are authoritative for this request.\n",
    );
    for record in view.full {
        block.push_str(&format!("- [{}] {}\n", record.id, record.text));
    }
    for reference in view.references {
        block.push_str(&format!(
            "- [{}] status={:?}; retrieve historical evidence explicitly\n",
            reference.id, reference.status
        ));
    }
    block.push_str("</active-requirements>");
    let system = match system {
        SystemPrompt::Text(text) => SystemPrompt::Text(format!("{text}{block}")),
        blocks => blocks,
    };
    Ok(system)
}

fn append_history_manifest(
    system: SystemPrompt,
    refs: &[HistoryRef],
) -> Result<SystemPrompt, String> {
    if refs.is_empty() {
        return Ok(system);
    }
    // Keep the visible page finite. Every reference remains durable in the
    // scoped index and can be paged through HistoryRead; dropping references
    // from this page must never trigger further context eviction.
    let mut visible_refs = Vec::new();
    let mut sorted_refs = refs.to_vec();
    sorted_refs.sort_by(|left, right| {
        left.sequence
            .cmp(&right.sequence)
            .then_with(|| left.id.cmp(&right.id))
    });
    for reference in sorted_refs {
        if visible_refs.len() >= MAX_HISTORY_MANIFEST_REFS {
            break;
        }
        let mut candidate = visible_refs.clone();
        candidate.push(reference);
        let candidate_manifest = json!({
            "retrieval_status": "omitted",
            "refs": candidate,
            "authority": "historical",
            "reactivates_requirements": false,
            "total_count": refs.len(),
            "remaining_count": refs.len().saturating_sub(visible_refs.len() + 1),
            "has_more": refs.len() > visible_refs.len() + 1,
            "index_search": {"tool": "HistoryRead", "query": "", "offset": visible_refs.len() + 1, "limit": MAX_HISTORY_MANIFEST_REFS},
        });
        if serde_json::to_string(&candidate_manifest)
            .map_err(|error| error.to_string())?
            .len()
            > MAX_HISTORY_MANIFEST_BYTES
        {
            break;
        }
        visible_refs = candidate;
    }
    let manifest = serde_json::to_string(&json!({
        "retrieval_status": "omitted",
        "refs": visible_refs,
        "authority": "historical",
        "reactivates_requirements": false,
        "total_count": refs.len(),
        "remaining_count": refs.len().saturating_sub(visible_refs.len()),
        "has_more": refs.len() > visible_refs.len(),
        "index_search": {"tool": "HistoryRead", "query": "", "offset": visible_refs.len(), "limit": MAX_HISTORY_MANIFEST_REFS},
    }))
    .map_err(|error| error.to_string())?;
    if manifest.len() > MAX_HISTORY_MANIFEST_BYTES {
        return Err("history manifest metadata exceeds its fixed byte limit".into());
    }
    let block = format!(
        "\n\n<history-manifest>Retrieve omitted groups only through HistoryRead; this manifest is authoritative. {manifest}</history-manifest>"
    );
    Ok(match system {
        SystemPrompt::Text(text) => SystemPrompt::Text(format!("{text}{block}")),
        SystemPrompt::Blocks(mut blocks) => {
            if let Some(last) = blocks.last_mut() {
                last.text.push_str(&block);
            }
            SystemPrompt::Blocks(blocks)
        }
    })
}

fn reserve_history_manifest(system: &SystemPrompt) -> SystemPrompt {
    let reserve = format!(
        "\n<history-manifest-reserve>{}</history-manifest-reserve>",
        "x".repeat(HISTORY_MANIFEST_RESERVE_BYTES)
    );
    match system {
        SystemPrompt::Text(text) => SystemPrompt::Text(format!("{text}{reserve}")),
        SystemPrompt::Blocks(blocks) => {
            let mut blocks = blocks.clone();
            if let Some(last) = blocks.last_mut() {
                last.text.push_str(&reserve);
            }
            SystemPrompt::Blocks(blocks)
        }
    }
}

fn request_budget(
    config: &QueryConfig,
    target: &resolved_target::ResolvedTarget<'_>,
    output_reserve: u64,
) -> Result<(RequestBudget, claurst_api::WireBudget), ContextPlanError> {
    let budget = RequestBudget::resolve(
        config.context_budget.context_limit_tokens,
        target.catalog_context_limit_tokens(),
        config.context_budget.provider_reported_limit_tokens,
        Some(config.context_budget.unknown_fallback_tokens),
        output_reserve,
        config.context_budget.wire_byte_limit,
    )?;
    let wire = claurst_api::WireBudget::finite(
        budget.context_limit_tokens,
        budget.input_limit_tokens,
        budget.output_reserve_tokens,
        budget.wire_byte_limit,
        claurst_api::MediaAccounting::None,
    )
    .and_then(|wire| wire.with_source(format!("{:?}", budget.source)))
    .map_err(|error| ContextPlanError::Measurement(error.to_string()))?;
    Ok((budget, wire))
}

/// Non-agentic query helpers still need a finite transport budget. They do
/// not have a ToolContext or provider registry, so they use the same explicit
/// fallback policy as a normal query turn.
pub(crate) fn default_wire_budget(
    output_reserve: u64,
) -> Result<claurst_api::WireBudget, ContextPlanError> {
    let budget = RequestBudget::resolve(
        None,
        None,
        None,
        Some(ContextBudgetConfig::default().unknown_fallback_tokens),
        output_reserve,
        ContextBudgetConfig::default().wire_byte_limit,
    )?;
    claurst_api::WireBudget::finite(
        budget.context_limit_tokens,
        budget.input_limit_tokens,
        budget.output_reserve_tokens,
        budget.wire_byte_limit,
        claurst_api::MediaAccounting::None,
    )
    .and_then(|wire| wire.with_source(format!("{:?}", budget.source)))
    .map_err(|error| ContextPlanError::Measurement(error.to_string()))
}

fn context_plan_error(error: ContextPlanError) -> ClaudeError {
    match error {
        ContextPlanError::CurrentRequestTooLarge {
            input_tokens,
            wire_bytes,
            budget,
        } => ClaudeError::RequestBudgetExceeded {
            input_tokens,
            input_limit_tokens: budget.input_limit_tokens,
            wire_bytes,
            wire_byte_limit: budget.wire_byte_limit,
            context_limit_tokens: budget.context_limit_tokens,
            output_reserve_tokens: budget.output_reserve_tokens,
            budget_source: format!("{:?}", budget.source),
        },
        other => ClaudeError::Config(other.to_string()),
    }
}

fn measure_context_candidate(
    provider: &str,
    model: &str,
    messages: &[Message],
    omitted: &[context_plan::OmittedComponent],
    system: &SystemPrompt,
    tools: &[ApiToolDefinition],
) -> Result<context_plan::CandidateCost, ContextPlanError> {
    // Preview accounting must be freshly derived from the selected canonical
    // messages.  Persisted or provider-prepared metadata cannot authorize a
    // different candidate, and policy failures must remain typed.
    let media = claurst_api::MediaAccounting::from_messages(provider, model, messages)
        .map_err(ContextPlanError::MediaPolicy)?;
    let api_messages: Vec<ApiMessage> = messages.iter().map(ApiMessage::from).collect();
    let bytes = serde_json::to_vec(&(system, &api_messages, tools, omitted))
        .map_err(|error| ContextPlanError::Measurement(error.to_string()))?;
    let wire_bytes = u64::try_from(bytes.len())
        .map_err(|_| ContextPlanError::Measurement("preview wire byte count overflowed".into()))?;
    let input_tokens = match media {
        claurst_api::MediaAccounting::None => wire_bytes
            .checked_add(3)
            .and_then(|value| value.checked_div(4))
            .ok_or_else(|| {
                ContextPlanError::Measurement("preview token count overflowed".into())
            })?,
        claurst_api::MediaAccounting::Images(images) => {
            let encoded_bytes =
                u64::try_from(images.manifest().total_encoded_bytes()).map_err(|_| {
                    ContextPlanError::Measurement("validated media byte count overflowed".into())
                })?;
            let non_media_wire_bytes = wire_bytes.checked_sub(encoded_bytes).ok_or_else(|| {
                ContextPlanError::Measurement(
                    "validated media exceeds canonical preview wire bytes".into(),
                )
            })?;
            let non_media_tokens = non_media_wire_bytes
                .checked_add(3)
                .and_then(|value| value.checked_div(4))
                .ok_or_else(|| {
                    ContextPlanError::Measurement("preview token count overflowed".into())
                })?;
            non_media_tokens
                .checked_add(images.tokens())
                .ok_or_else(|| {
                    ContextPlanError::Measurement("media token count overflowed".into())
                })?
        }
    };
    Ok(context_plan::CandidateCost {
        input_tokens,
        wire_bytes,
    })
}

fn persist_omitted_groups(
    index: Option<&Arc<parking_lot::Mutex<context_history::HistoryIndex>>>,
    plan: &context_plan::ContextPlan,
    transcript: &[Message],
) -> Result<Vec<HistoryRef>, String> {
    let Some(index) = index else {
        return Ok(Vec::new());
    };
    let mut index = index.lock();
    let mut refs = Vec::new();
    for (sequence, omitted) in plan.omitted.iter().enumerate() {
        let ids = omitted
            .message_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        let members = transcript
            .iter()
            .filter(|message| message.uuid.as_ref().is_some_and(|id| ids.contains(id)))
            .cloned()
            .collect::<Vec<_>>();
        if members.is_empty() {
            continue;
        }
        let reason = match omitted.reason {
            context_plan::OmissionReason::Budget => context_history::OmitReason::Budget,
            context_plan::OmissionReason::MalformedToolPair => context_history::OmitReason::Summary,
            context_plan::OmissionReason::UnsupportedMedia => {
                context_history::OmitReason::ToolResultBudget
            }
        };
        refs.push(
            index
                .persist_omitted_group(&omitted.group_id, sequence as u64, members, reason)
                .map_err(|error| error.to_string())?,
        );
    }
    Ok(refs)
}

fn next_reselection_group(
    plan: &context_plan::ContextPlan,
    excluded: &std::collections::BTreeSet<String>,
) -> Option<String> {
    plan.optional_selected_groups
        .iter()
        .rev()
        .find(|group_id| !excluded.contains(*group_id))
        .cloned()
}

/// Extract a witness from the exact registered native handle that executed the
/// call. Same-name MCP/plugin tools have no native capability and therefore
/// cannot turn their metadata into execution authority.
fn native_bash_execution_witness(
    tools: &[Box<dyn Tool>],
    tool_name: &str,
    tool_id: &str,
    result: &ToolResult,
    emitted_result: &str,
) -> Option<NativeBashExecutionWitness> {
    let tool = tools.iter().find(|tool| tool.name() == tool_name)?;
    if tool_name != claurst_core::constants::TOOL_NAME_BASH
        || tool.catalog_source() != ToolCatalogSource::Native
    {
        return None;
    }
    tool.native_bash_execution_witness(tool_id, result, emitted_result)
}

/// Apply host discovery receipts after a complete captured tool batch.
///
/// Receipt activation is deliberately transactional: every receipt is checked
/// against the immutable catalog before the active set is replaced.  A bad
/// receipt leaves the previous set intact and changes only the corresponding
/// model-visible search result into a categorical failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscoveryApplyOutcome {
    None,
    Boundary,
    SchemaChanged,
}

fn apply_discovery_receipts(
    results: &mut [ToolResult],
    receipts: Vec<(usize, DiscoveryReceipt)>,
    active_schemas: &mut active_schema::ActiveSchemaSet,
    catalog_view: &ToolCatalogView,
    tool_ctx: &ToolContext,
    tools: &[Box<dyn Tool>],
    pending_proof: &mut Option<TrustedVerificationProof>,
    inventory_guard: Option<&tool_inventory::InventoryGuard>,
    event_tx: Option<&event_channel::QueryEventSender>,
    activation_binding: &mut Option<native_mcp_capture::NativeMcpActivationBinding>,
    capture_mcp_diagnostics: bool,
) -> DiscoveryApplyOutcome {
    if receipts.is_empty() {
        return DiscoveryApplyOutcome::None;
    }

    // ToolSearch may finish after the host replaced its catalog.  Never
    // activate a receipt against that new scope or generation; replace the
    // result atomically and retain the prior active schema set/cap marker.
    if inventory_guard.is_some_and(|guard| !guard.is_current(tool_ctx)) {
        for (index, _) in receipts {
            if let Some(result) = results.get_mut(index) {
                *result = claurst_tools::evidence::retain_result(
                    tool_ctx,
                    ToolResult::error(
                        "Tool discovery selection was stale; no schemas were activated.",
                    )
                    .with_execution_status(claurst_tools::ExecutionStatus::NotRun),
                );
            }
        }
        *pending_proof = None;
        return DiscoveryApplyOutcome::Boundary;
    }

    let receipt_values: Vec<DiscoveryReceipt> = receipts
        .iter()
        .map(|(_, receipt)| receipt.clone())
        .collect();
    // Capture the checked snapshot identity before activation and require the
    // same projection after activation. A refresh in either interval is an
    // integrity failure for this diagnostic, never a source of new authority.
    let snapshot_before = tool_ctx
        .mcp_manager
        .as_ref()
        .filter(|_| capture_mcp_diagnostics)
        .and_then(|manager| manager.checked_catalog_snapshot().ok());
    let snapshot_digest = snapshot_before
        .as_ref()
        .and_then(|snapshot| native_mcp_capture::checked_snapshot_digest(snapshot).ok());
    let (outcome, activation_succeeded) =
        match active_schemas.activate_receipts(catalog_view, tool_ctx, tools, &receipt_values) {
            Ok(next) => {
                if next.active_revision() != active_schemas.active_revision()
                    || next.digest() != active_schemas.digest()
                {
                    *active_schemas = next;
                    *pending_proof = None;
                    (DiscoveryApplyOutcome::SchemaChanged, true)
                } else {
                    (DiscoveryApplyOutcome::Boundary, true)
                }
            }
            Err(_) => {
                for (index, _) in &receipts {
                    if let Some(result) = results.get_mut(*index) {
                        *result = claurst_tools::evidence::retain_result(
                            tool_ctx,
                            ToolResult::error(
                                "Tool discovery selection was rejected; no schemas were activated.",
                            ),
                        );
                    }
                }
                *pending_proof = None;
                (DiscoveryApplyOutcome::Boundary, false)
            }
        };
    if !capture_mcp_diagnostics {
        *pending_proof = None;
        return outcome;
    }
    // Diagnostics observe activation; their independent size/integrity bounds
    // cannot revoke a successful schema receipt. Existing inventory and tool
    // dispatch guards still enforce authorization and catalog freshness.
    if activation_succeeded && tool_ctx.mcp_manager.is_some() {
        let Some(_snapshot_before) = snapshot_before.as_ref() else {
            *activation_binding = None;
            *pending_proof = None;
            return outcome;
        };
        let Some(pre_digest) = snapshot_digest else {
            *activation_binding = None;
            *pending_proof = None;
            return outcome;
        };
        let Some(manager) = tool_ctx.mcp_manager.as_deref() else {
            *activation_binding = None;
            *pending_proof = None;
            return outcome;
        };
        let post_snapshot = match manager.checked_catalog_snapshot() {
            Ok(snapshot) => snapshot,
            Err(_) => {
                *activation_binding = None;
                *pending_proof = None;
                return outcome;
            }
        };
        let post_digest = match native_mcp_capture::checked_snapshot_digest(&post_snapshot) {
            Ok(digest) => digest,
            Err(_) => {
                *activation_binding = None;
                *pending_proof = None;
                return outcome;
            }
        };
        if pre_digest != post_digest {
            *activation_binding = None;
            *pending_proof = None;
            return outcome;
        }
        *activation_binding =
            native_mcp_capture::activation_binding_from_snapshot(&post_snapshot, active_schemas)
                .ok();
    }
    if let (Some(tx), Some(binding)) = (event_tx, activation_binding.as_ref()) {
        if let Ok(Some(witness)) = native_mcp_capture::build_discovery_witness(
            tool_ctx.mcp_manager.as_deref(),
            &receipt_values,
            active_schemas,
            binding,
        ) {
            let _ = tx.send(QueryEvent::McpDiscovery(witness));
        }
    }
    *pending_proof = None;
    outcome
}

/// Run the agentic query loop.
///
/// This sends the conversation to the API, handles tool calls in a loop, and
/// returns when the model issues an end_turn or an error/limit is hit.
///
/// `pending_messages` is an optional queue of user messages that were enqueued
/// during tool execution (e.g. by the UI or a command queue).  Each string is
/// appended as a plain user message between turns.  Callers that do not need
/// command queuing may pass `None` or an empty `Vec`.
pub async fn run_query_loop(
    client: &claurst_api::AnthropicClient,
    messages: &mut Vec<Message>,
    tools: &[Box<dyn Tool>],
    tool_ctx: &ToolContext,
    config: &QueryConfig,
    cost_tracker: Arc<CostTracker>,
    event_tx: Option<mpsc::UnboundedSender<QueryEvent>>,
    cancel_token: tokio_util::sync::CancellationToken,
    pending_messages: Option<&mut Vec<String>>,
) -> QueryOutcome {
    run_query_loop_with_events(
        client,
        messages,
        tools,
        tool_ctx,
        config,
        cost_tracker,
        event_tx.map(Into::into),
        cancel_token,
        pending_messages,
    )
    .await
}

/// Explicit per-call event delivery. Bounded delivery is never inherited through
/// QueryConfig. Hosts must also join their consumer and inspect its final state.
pub async fn run_query_loop_with_events(
    client: &claurst_api::AnthropicClient,
    messages: &mut Vec<Message>,
    tools: &[Box<dyn Tool>],
    tool_ctx: &ToolContext,
    config: &QueryConfig,
    cost_tracker: Arc<CostTracker>,
    event_tx: Option<event_channel::QueryEventSender>,
    cancel_token: tokio_util::sync::CancellationToken,
    pending_messages: Option<&mut Vec<String>>,
) -> QueryOutcome {
    let delivery = event_tx
        .as_ref()
        .and_then(event_channel::QueryEventSender::delivery_state);
    let mut stop_hooks = StopHookScope::new(cancel_token.clone());
    let outcome = run_query_loop_inner(
        client,
        messages,
        tools,
        tool_ctx,
        config,
        cost_tracker,
        event_tx,
        cancel_token,
        pending_messages,
        &mut stop_hooks,
    )
    .await;
    stop_hooks.close().await;
    if delivery.is_some_and(|state| state.failure().is_some()) {
        return QueryOutcome::Error(ClaudeError::Config(
            "Query event delivery failed".to_string(),
        ));
    }
    outcome
}

async fn run_query_loop_inner(
    client: &claurst_api::AnthropicClient,
    messages: &mut Vec<Message>,
    tools: &[Box<dyn Tool>],
    tool_ctx: &ToolContext,
    config: &QueryConfig,
    cost_tracker: Arc<CostTracker>,
    event_tx: Option<event_channel::QueryEventSender>,
    cancel_token: tokio_util::sync::CancellationToken,
    mut pending_messages: Option<&mut Vec<String>>,
    stop_hooks: &mut StopHookScope,
) -> QueryOutcome {
    // Rebind the tool context to carry the loop's actual cancel token so the
    // parallel tool executor — and any tools or sub-agents that read
    // `ctx.cancel_token` — observe the same cancellation signal that drives this
    // loop (issue #218). Callers construct the context with a placeholder token;
    // making the loop authoritative here means a parent cancel reaches tools.
    let mut loop_ctx = tool_ctx.clone();
    loop_ctx.cancel_token = cancel_token.clone();
    let mut loop_config = config.clone();
    let mut startup_scope_unavailable = false;
    if !loop_config.child_execution
        && loop_config.verified_startup_policy.is_none()
        && loop_config.verified_startup_binding.is_some()
    {
        return QueryOutcome::Error(ClaudeError::Config(
            "verified startup binding requires a loaded startup policy".to_string(),
        ));
    }
    if loop_config.child_execution {
        // Child and cron executions are never allowed to inherit a parent's
        // startup authority or checker catalog.  They use the historical
        // fixed policy until a separately bound host session is provided.
        loop_config.effort_policy = effort_policy::EffortPolicy::Fixed;
        loop_config.verified_startup_policy = None;
        loop_config.verified_startup_binding = None;
        loop_config.anthropic_client_factory = None;
        loop_ctx.trusted_checker_catalog = Arc::new(claurst_tools::TrustedCheckerCatalog::empty());
        if loop_config.inventory_source.is_some()
            && !matches!(
                loop_config.inventory_profile,
                ToolInventoryProfile::AgentChild {
                    allowed_tools: Some(_),
                    ..
                }
            )
        {
            return QueryOutcome::Error(ClaudeError::Config(
                "child inventory profile must be finalized before query startup".to_string(),
            ));
        }
    } else if let Some(policy) = loop_config.verified_startup_policy.clone() {
        let workspace = claurst_tools::worktree::workspace_root(&loop_ctx);
        match policy
            .validate_workspace(&workspace)
            .and_then(|()| policy.bind_session(&workspace, &loop_ctx.session_id))
        {
            Ok(bound) => {
                let bound = Arc::new(bound);
                loop_config.effort_policy = bound.effort_policy();
                loop_config.verified_startup_binding = Some(Arc::clone(&bound));
                if bound.provider_id() == "anthropic" {
                    loop_config.anthropic_client_factory =
                        Some(claurst_api::AnthropicClientFactory::with_dispatch_policy(
                            Arc::clone(bound.dispatch_policy()),
                        ));
                }
                // Non-Anthropic binding is deferred until the resolver
                // selects the current configured/runtime provider.
                loop_ctx.trusted_checker_catalog = Arc::clone(bound.checker_catalog());
            }
            Err(_) => {
                // Keep the requested policy visible to the host while making
                // adaptive selection ineligible.  The categorical reason is
                // emitted by the first selection event below.
                loop_config.verified_startup_policy = None;
                loop_config.verified_startup_binding = None;
                loop_config.anthropic_client_factory = None;
                loop_ctx.trusted_checker_catalog =
                    Arc::new(claurst_tools::TrustedCheckerCatalog::empty());
                startup_scope_unavailable = true;
            }
        }
    }
    loop_ctx.catalog_view = None;
    let mut inventory_history_binding = match loop_config.history_index.as_ref() {
        Some(index) => {
            let scope = match claurst_tools::ToolCatalogScope::from_context(&loop_ctx) {
                Ok(scope) => scope,
                Err(error) if loop_config.inventory_source.is_some() => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
                Err(_) => {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "history binding scope unavailable".to_string(),
                    ));
                }
            };
            match tool_inventory::HistoryBindingRequest::from_index(Arc::clone(index), scope) {
                Ok(binding) => Some(binding),
                Err(error) if loop_config.inventory_source.is_some() => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
                Err(_) => None,
            }
        }
        None => None,
    };
    let mut inventory_request = match tool_inventory::InventoryRequest::for_context_with_binding(
        &loop_ctx,
        loop_config.inventory_profile.clone(),
        inventory_history_binding.as_ref(),
    ) {
        Ok(request) => Some(request),
        Err(error) if loop_config.inventory_source.is_some() => {
            return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
        }
        Err(_) => None,
    };
    let mut inventory = match (
        loop_config.inventory_source.as_ref(),
        inventory_request.as_ref(),
    ) {
        (Some(source), Some(request)) => {
            let snapshot = match source.snapshot(request, &loop_ctx).await {
                Ok(snapshot)
                    if snapshot.scope().matches_context(&loop_ctx)
                        && snapshot
                            .tools()
                            .iter()
                            .all(|tool| request.profile().allows_handle(tool.as_ref())) =>
                {
                    snapshot
                }
                Ok(_) => {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "tool inventory scope or profile does not match query context".to_string(),
                    ));
                }
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            let current_generation = match source.current_generation(request, &loop_ctx) {
                Ok(generation) => generation,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            if snapshot.generation() != current_generation {
                return QueryOutcome::Error(ClaudeError::Config(
                    "tool inventory changed during snapshot".to_string(),
                ));
            }
            tool_inventory::CapturedInventory::owned(snapshot)
        }
        (Some(_), None) => {
            return QueryOutcome::Error(ClaudeError::Config(
                "inventory request unavailable".to_string(),
            ))
        }
        (None, _) => tool_inventory::CapturedInventory::borrowed(tools),
    };
    // Always bind the catalog to the exact final authorized tool slice after
    // child/cron/startup scope resolution. An incoming view can be stale when
    // a host refreshes wrappers or replaces a workspace between turns.
    let mut catalog_view = match ToolCatalogView::from_tools(&loop_ctx, inventory.tools()) {
        Ok(view) => view,
        Err(error) => return QueryOutcome::Error(ClaudeError::Config(error.to_string())),
    };
    loop_ctx.catalog_view = Some(catalog_view.clone());
    let mut active_schemas = match active_schema::ActiveSchemaSet::initial(
        &catalog_view,
        inventory.tools(),
        catalog_view.contains_name("ToolSearch"),
    ) {
        Ok(active) => active,
        Err(error) => return QueryOutcome::Error(ClaudeError::Config(error.to_string())),
    };
    let mut mcp_activation_binding = if config.capture_mcp_diagnostics {
        native_mcp_capture::activation_binding(tool_ctx.mcp_manager.as_deref(), &active_schemas)
            .ok()
            .flatten()
    } else {
        None
    };
    let mut turn = 0u32;
    // Tracks how many consecutive max_tokens recoveries we've attempted so
    // we don't loop forever on a model that can't finish within any budget.
    let mut max_tokens_recovery_count: u32 = 0;
    let mut runtime_message_ids = std::collections::BTreeSet::new();
    // Active model — may switch to fallback on overloaded errors.
    // Agent model override takes priority over the session model when set.
    let mut effective_model = if let Some(ref agent) = loop_config.agent_definition {
        agent
            .model
            .clone()
            .unwrap_or_else(|| loop_config.model.clone())
    } else {
        loop_config.model.clone()
    };

    // If managed-agent mode is active, override the model to the manager model.
    if let Some(ref ma_config) = loop_config.managed_agents {
        if ma_config.enabled && !ma_config.manager_model.is_empty() {
            effective_model = ma_config.manager_model.clone();
        }
    }

    let mut used_fallback = false;
    // How many automatic retries remain when a stream stalls (no data for 45s).
    let mut retries_left: u32 = 2;
    // Max-steps graceful degradation (issue #230 / MI-3): set once the final
    // tool-less summary turn has been dispatched so it can never re-trigger
    // (anti-recursion guard).
    let mut degradation_done = false;

    // If an agent defines a max_turns override, respect it (agent wins over config).
    let effective_max_turns = loop_config
        .agent_definition
        .as_ref()
        .and_then(|a| a.max_turns)
        .unwrap_or(loop_config.max_turns);

    // In-loop continuation policy (issue #230 / MI-3). Consulted at the end of
    // every turn that finishes with `end_turn`. The default policy stops after
    // one turn; the goal policy keeps the loop running while an active goal's
    // guards allow. Built once per run.
    let continuation_policy = loop_config.continuation.policy();
    // Wall-clock start of the current "continuation turn" (a span from a user /
    // continuation message to the next `end_turn`). Reset on each accepted
    // continuation so goal time/turn accounting matches the old per-dispatch
    // measurement.
    let mut goal_turn_start = std::time::Instant::now();

    // Shadow-git snapshot: capture the worktree state before any tools run so we
    // can produce a per-turn file-change patch when the turn ends.
    let shadow_snap: Option<std::sync::Arc<claurst_core::snapshot::ShadowSnapshot>> =
        if loop_ctx.config.auto_commits == Some(true) {
            claurst_core::snapshot::get_or_create(&loop_ctx.working_dir)
        } else {
            None
        };
    // Pre-capture tree hash; refreshed at the start of each turn's tool phase.
    let initial_snapshot: Option<String> = if let Some(ref s) = shadow_snap {
        s.track().await
    } else {
        None
    };

    let mut wire_excluded_groups = std::collections::BTreeSet::new();
    let mut unsent_inventory_restarts = 0u8;
    // Keep the resolved runtime handle only through local pre-send budget
    // reselection. Once a stream opens, the next logical continuation must
    // resolve credentials and endpoint configuration afresh.
    let mut resolved_target: Option<Arc<resolved_target::ResolvedTarget<'_>>> = None;
    let mut last_completed_target: Option<Arc<resolved_target::ResolvedTarget<'_>>> = None;
    let mut provider_response_continuity: Option<ProviderResponseContinuity<'_>> = None;
    let mut last_effort_selection: Option<(
        EffortSelectionMode,
        EffortSelectionMode,
        Option<claurst_core::effort::EffortLevel>,
        EffortSelectionReason,
    )> = None;
    let mut next_selection_reason: Option<EffortSelectionReason> =
        startup_scope_unavailable.then_some(EffortSelectionReason::StartupScopeUnavailable);
    let mut continuation_state: Option<continuation_state::ContinuationState<'_>> = None;
    // A proof is retained only after a complete mutation/check batch. It is
    // private host authority and is consumed by at most one lower dispatch.
    let mut pending_proof: Option<TrustedVerificationProof> = None;
    let mut verified_lower_used = false;
    let mut proof_base_fallback_used = false;
    // Inline ultracode is a work-level pin.  It survives host-generated
    // continuation/tool-result messages and changes only when a new genuine
    // user turn is drained into this query.
    let mut inline_ultracode_active = initial_inline_ultracode_state(messages);

    'query_turn: loop {
        // Worktree moves and resumed sessions change the catalog scope.  Bind
        // a fresh metadata view before the next request and revoke any pending
        // proof/continuation authority; completed tool work is never replayed.
        let rebind_config_root = if inventory_history_binding.is_some() {
            let current_scope = match claurst_tools::ToolCatalogScope::from_context(&loop_ctx) {
                Ok(scope) => scope,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            inventory_history_binding
                .as_ref()
                .and_then(|binding| {
                    let same_workspace = binding.scope().workspace() == current_scope.workspace();
                    let same_session = binding.scope().session_id() == current_scope.session_id();
                    if binding.scope() != &current_scope && (!same_workspace || !same_session) {
                        Some(binding.index().lock().config_root().to_path_buf())
                    } else {
                        None
                    }
                })
                .map(|config_root| (config_root, current_scope))
        } else {
            None
        };
        if let Some((config_root, current_scope)) = rebind_config_root {
            // A planned worktree transition changes the canonical workspace
            // path. Open a new scoped index from the original config root;
            // never relabel the parent's index.
            let fresh_index = match context_history::HistoryIndex::open(
                config_root,
                &workspace_scope_id(current_scope.workspace()),
                current_scope.session_id(),
            ) {
                Ok(index) => Arc::new(parking_lot::Mutex::new(index)),
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(format!(
                        "history scope transition unavailable: {error}"
                    )));
                }
            };
            let fresh_binding = match tool_inventory::HistoryBindingRequest::from_index(
                Arc::clone(&fresh_index),
                current_scope,
            ) {
                Ok(binding) => binding,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            loop_config.history_index = Some(fresh_index);
            inventory_history_binding = Some(fresh_binding);
            mcp_activation_binding = if config.capture_mcp_diagnostics {
                native_mcp_capture::activation_binding(
                    tool_ctx.mcp_manager.as_deref(),
                    &active_schemas,
                )
                .ok()
                .flatten()
            } else {
                None
            };
            pending_proof = None;
            continuation_state = None;
            resolved_target = None;
            last_completed_target = None;
            provider_response_continuity = None;
            wire_excluded_groups.clear();
            next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
        }
        if let Some(source) = loop_config.inventory_source.as_ref() {
            let refreshed_request = match tool_inventory::InventoryRequest::for_context_with_binding(
                &loop_ctx,
                loop_config.inventory_profile.clone(),
                inventory_history_binding.as_ref(),
            ) {
                Ok(request) => request,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            inventory_request = Some(refreshed_request);
            let Some(request) = inventory_request.as_ref() else {
                return QueryOutcome::Error(ClaudeError::Config(
                    "inventory request unavailable".to_string(),
                ));
            };
            let Some(previous_generation) = inventory.generation() else {
                return QueryOutcome::Error(ClaudeError::Config(
                    "manual inventory cannot use a source".to_string(),
                ));
            };
            let current_generation = match source.current_generation(request, &loop_ctx) {
                Ok(generation) => generation,
                Err(error) => return QueryOutcome::Error(ClaudeError::Config(error.to_string())),
            };
            // The source owns snapshot reuse.  It may return the exact cached
            // Arc for an unchanged request; keeping this call here lets the
            // loop detect a same-generation handle replacement atomically.
            let replacement = match source.snapshot(request, &loop_ctx).await {
                Ok(snapshot)
                    if snapshot.scope().matches_context(&loop_ctx)
                        && snapshot
                            .tools()
                            .iter()
                            .all(|tool| request.profile().allows_handle(tool.as_ref())) =>
                {
                    snapshot
                }
                Ok(_) => {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "tool inventory scope or profile does not match query context".to_string(),
                    ));
                }
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            let scope_changed = inventory
                .scope()
                .map(|scope| !scope.matches_context(&loop_ctx))
                .unwrap_or(true);
            if scope_changed
                || current_generation != previous_generation
                || replacement.generation() != previous_generation
            {
                let after_generation = match source.current_generation(request, &loop_ctx) {
                    Ok(generation) => generation,
                    Err(error) => {
                        return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                    }
                };
                if replacement.generation() != after_generation {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "tool inventory changed during snapshot".to_string(),
                    ));
                }
                if replacement.generation() == previous_generation
                    && !inventory.same_handles(&replacement)
                {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "tool inventory handles changed without a new generation".to_string(),
                    ));
                }
                inventory = tool_inventory::CapturedInventory::owned(replacement);
                pending_proof = None;
                continuation_state = None;
                resolved_target = None;
                last_completed_target = None;
                provider_response_continuity = None;
                wire_excluded_groups.clear();
                next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
                loop_ctx.catalog_view = None;
            } else if !inventory.same_handles(&replacement) {
                return QueryOutcome::Error(ClaudeError::Config(
                    "tool inventory handles changed without a new generation".to_string(),
                ));
            }
        }
        if !catalog_view.matches_context(&loop_ctx)
            || (inventory.generation().is_some() && loop_ctx.catalog_view.is_none())
        {
            catalog_view = match ToolCatalogView::from_tools(&loop_ctx, inventory.tools()) {
                Ok(view) => view,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            loop_ctx.catalog_view = Some(catalog_view.clone());
            let search_authorized = catalog_view.contains_name("ToolSearch");
            active_schemas = match active_schema::ActiveSchemaSet::initial(
                &catalog_view,
                inventory.tools(),
                search_authorized,
            ) {
                Ok(active) => active,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::Config(error.to_string()));
                }
            };
            pending_proof = None;
            continuation_state = None;
            resolved_target = None;
            last_completed_target = None;
            provider_response_continuity = None;
            wire_excluded_groups.clear();
            next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
        }
        // A completed response clears the retained target before the next
        // logical selection. Revalidate only at that boundary; local wire
        // reselection retains the target and never rebuilds checker
        // descriptors. If the workspace was replaced or moved, revoke every
        // startup-bound authority before the new target is resolved.
        if resolved_target.is_none() && !startup_scope_unavailable {
            if let Some(policy) = loop_config.verified_startup_policy.clone() {
                let workspace = claurst_tools::worktree::workspace_root(&loop_ctx);
                if policy.validate_workspace(&workspace).is_err() {
                    startup_scope_unavailable = true;
                    loop_config.verified_startup_policy = None;
                    loop_config.anthropic_client_factory = None;
                    loop_ctx.trusted_checker_catalog =
                        Arc::new(claurst_tools::TrustedCheckerCatalog::empty());
                    pending_proof = None;
                    continuation_state = None;
                    verified_lower_used = false;
                    proof_base_fallback_used = false;
                    last_completed_target = None;
                    provider_response_continuity = None;
                    wire_excluded_groups.clear();
                    next_selection_reason = Some(EffortSelectionReason::StartupScopeUnavailable);
                }
            }
        }
        let config = &loop_config;
        let tool_ctx = &loop_ctx;
        let inventory_guard = match (
            loop_config.inventory_source.as_ref(),
            inventory_request.as_ref(),
            inventory.generation(),
        ) {
            (Some(source), Some(request), Some(generation)) => {
                Some(tool_inventory::InventoryGuard::new(
                    Arc::clone(source),
                    request.clone(),
                    generation,
                ))
            }
            _ => None,
        };
        turn += 1;
        tool_ctx
            .current_turn
            .store(turn as usize, std::sync::atomic::Ordering::Relaxed);
        if resolved_target
            .as_ref()
            .is_none_or(|target| target.requested_model() != effective_model)
        {
            resolved_target = Some(Arc::new(
                match resolved_target::ResolvedTarget::resolve(
                    config,
                    tool_ctx,
                    &effective_model,
                    client,
                ) {
                    Ok(target) => target,
                    Err(error) => return QueryOutcome::Error(ClaudeError::Config(error)),
                },
            ));
        }
        macro_rules! reset_for_genuine_user_work {
            () => {
                pending_proof = None;
                continuation_state = None;
                verified_lower_used = false;
                proof_base_fallback_used = false;
                wire_excluded_groups.clear();
                last_effort_selection = None;
                next_selection_reason = None;
                last_completed_target = None;
                provider_response_continuity = None;
                resolved_target = None;
            };
        }
        // Max-steps graceful degradation (issue #230 / MI-3). Rather than
        // returning cold when the turn cap is hit, run ONE final turn with tools
        // disabled that asks the model to summarize progress and its stopping
        // point (mirrors opencode's max-steps `toolChoice:"none"` fallback).
        // `degradation_done` is the anti-recursion guard: the summary turn is
        // dispatched exactly once, and re-exceeding the cap afterwards returns
        // cold. Applies to both goal and non-goal runs.
        let degradation_turn = if turn > effective_max_turns {
            if degradation_done {
                info!(
                    turns = turn,
                    "Max turns reached after degradation summary — returning"
                );
                let last_msg = messages
                    .last()
                    .cloned()
                    .unwrap_or_else(|| Message::assistant("Max turns reached."));
                return QueryOutcome::EndTurn {
                    message: last_msg,
                    usage: UsageInfo::default(),
                };
            }
            degradation_done = true;
            info!(
                turns = turn,
                max = effective_max_turns,
                "Max turns reached — running final tool-less summary turn"
            );
            if let Some(ref tx) = event_tx {
                let _ = tx.send(QueryEvent::Status(format!(
                    "Reached maximum turn limit ({}) — summarizing progress before stopping.",
                    effective_max_turns
                )));
            }
            // Inject the summary request as the next user turn. Tools are
            // disabled for this turn where `api_tools` / `provider_tools` are
            // built below.
            if let Err(error) = append_runtime_user_message(
                messages,
                &mut runtime_message_ids,
                loop_config.history_index.as_ref(),
                MAX_STEPS_DEGRADATION_MSG,
                context_history::RuntimeMessageOrigin::MaxStepsDegradation,
            ) {
                return QueryOutcome::Error(ClaudeError::Config(error));
            }
            true
        } else {
            false
        };

        // Continuation decision at `end_turn` (issue #230 / MI-3). Consults the
        // active continuation policy: `Continue` injects the follow-up message
        // as the next user turn and keeps looping (resetting the per-turn budget
        // so `effective_max_turns` bounds tool-rounds *within* a continuation
        // turn — the cross-turn cap is the policy's own guard, e.g. the goal
        // runaway limit); `Stop` surfaces any note and returns `EndTurn`.
        // Defined as a macro because it must `continue`/`return` the loop.
        macro_rules! continue_or_end {
            ($assistant_msg:expr, $usage:expr) => {{
                // The tool-less max-steps summary turn must never re-trigger
                // continuation (anti-recursion): return its wrap-up directly.
                if degradation_turn {
                    return QueryOutcome::EndTurn {
                        message: $assistant_msg,
                        usage: $usage,
                    };
                }
                let decision = continuation_policy.decide(&crate::continuation::TurnEndContext {
                    session_id: &tool_ctx.session_id,
                    total_tokens_used: cost_tracker.total_tokens(),
                    turn_elapsed_secs: goal_turn_start.elapsed().as_secs(),
                });
                match decision {
                    crate::continuation::ContinuationDecision::Continue { message } => {
                        if let Some(ref tx) = event_tx {
                            let _ = tx.send(QueryEvent::Status(
                                "Goal: continuing autonomously… (use /goal pause to stop)"
                                    .to_string(),
                            ));
                        }
                        if let Err(error) = append_runtime_user_message(
                            messages,
                            &mut runtime_message_ids,
                            loop_config.history_index.as_ref(),
                            message,
                            context_history::RuntimeMessageOrigin::Continuation,
                        ) {
                            return QueryOutcome::Error(ClaudeError::Config(error));
                        }
                        // Fresh per-continuation-turn budget, mirroring the old
                        // one-loop-per-goal-turn design.
                        turn = 0;
                        max_tokens_recovery_count = 0;
                        retries_left = 2;
                        goal_turn_start = std::time::Instant::now();
                        continue;
                    }
                    crate::continuation::ContinuationDecision::Stop { note } => {
                        if let Some(note) = note {
                            if let Some(ref tx) = event_tx {
                                let _ = tx.send(QueryEvent::Status(note));
                            }
                        }
                        return QueryOutcome::EndTurn {
                            message: $assistant_msg,
                            usage: $usage,
                        };
                    }
                }
            }};
        }

        // Check for cancellation
        if cancel_token.is_cancelled() {
            return QueryOutcome::Cancelled;
        }

        // Drain any pending user messages that were queued during the previous
        // tool-execution phase (e.g. commands entered while tools ran).
        // Mirrors the TS `messageQueueManager` drain between turns.
        if let Some(queue) = pending_messages.as_deref_mut() {
            let had_pending_user_work = !queue.is_empty();
            if had_pending_user_work {
                reset_for_genuine_user_work!();
            }
            for text in queue.drain(..) {
                debug!(bytes = text.len(), "Injecting pending message");
                inline_ultracode_active = claurst_core::effort::text_triggers_ultracode(&text);
                push_message_with_id(messages, Message::user(text));
            }
        }

        // T1-4: Drain the priority command queue (if wired up) and prepend any
        // resulting messages to the conversation before the API call.
        // Mirrors the TS `messageQueueManager` priority-queue drain.
        if let Some(ref cq) = config.command_queue {
            if !cq.is_empty() {
                let mut injected = Vec::new();
                let mut injected_genuine_user_work = false;
                for command in cq.drain() {
                    match command {
                        QueuedCommand::InjectUserMessage(text) => {
                            injected_genuine_user_work = true;
                            inline_ultracode_active =
                                claurst_core::effort::text_triggers_ultracode(&text);
                            injected.push(Message::user(text));
                        }
                        QueuedCommand::InjectSystemMessage(text) => {
                            injected.push(Message::user(format!("[System]: {text}")));
                        }
                        QueuedCommand::SetModel(_)
                        | QueuedCommand::Compact
                        | QueuedCommand::Clear
                        | QueuedCommand::TriggerSkill(_) => {}
                    }
                }
                if injected_genuine_user_work {
                    reset_for_genuine_user_work!();
                }
                if !injected.is_empty() {
                    debug!(count = injected.len(), "Injecting command-queue messages");
                    // Prepend so that higher-priority commands appear first.
                    let tail = std::mem::take(messages);
                    messages.extend(injected);
                    messages.extend(tail);
                }
            }
        }

        // A queued genuine user turn invalidates the cached target too. Resolve
        // it again after all queue drains, while system/synthetic messages keep
        // the current continuation and proof state intact.
        if resolved_target
            .as_ref()
            .is_none_or(|candidate| candidate.requested_model() != effective_model)
        {
            resolved_target = Some(Arc::new(
                match resolved_target::ResolvedTarget::resolve(
                    config,
                    tool_ctx,
                    &effective_model,
                    client,
                ) {
                    Ok(target) => target,
                    Err(error) => return QueryOutcome::Error(ClaudeError::Config(error)),
                },
            ));
        }
        let Some(target) = resolved_target.as_ref().cloned() else {
            return QueryOutcome::Error(ClaudeError::Config(
                "query target resolution failed after message queue drain".to_string(),
            ));
        };

        // Assign durable IDs before indexing or selecting. The canonical
        // transcript is retained in full; context planning only derives a
        // provider snapshot and never replaces tool-result content.
        ensure_message_ids(messages);
        if let Some(index) = loop_config.history_index.as_ref() {
            if let Err(error) = ingest_history_index(index, messages, &mut runtime_message_ids) {
                return QueryOutcome::Error(ClaudeError::Config(error));
            }
        }

        // Max-steps degradation: the final summary turn is dispatched with NO
        // tool definitions so the model can only produce text (issue #230).
        let api_tools: Vec<ApiToolDefinition> = if degradation_turn {
            Vec::new()
        } else {
            active_schemas.api_definitions()
        };

        let explicit_effort_level = if inline_ultracode_active {
            Some(claurst_core::effort::EffortLevel::Ultracode)
        } else {
            config.effort_level
        };
        let configured_verified_policy = config.effort_policy.verified_policy();
        let verified_route_supported = configured_verified_policy.is_some_and(|policy| {
            target
                .verified_route_pair(policy.base(), policy.continuation())
                .is_ok()
        });
        let hooks_present = tool_ctx
            .config
            .hooks
            .values()
            .any(|hooks| !hooks.is_empty())
            || claurst_plugins::global_hooks_present();
        let policy_pin_provenance = configured_verified_policy.map(|policy| {
            let base = policy.base();
            effective_pin_provenance_for_inline(
                config,
                Some(base),
                config
                    .thinking_budget
                    .or_else(|| base.thinking_budget_tokens()),
                config.temperature.or_else(|| base.temperature()),
                inline_ultracode_active,
            )
        });
        let eligible_verified_policy = configured_verified_policy.filter(|_| {
            explicit_effort_level.is_none()
                && !startup_scope_unavailable
                && verified_route_supported
                && !hooks_present
                && policy_pin_provenance
                    .is_none_or(|provenance| provenance.blocking_pin().is_none())
        });
        let adaptive_policy_active = eligible_verified_policy.is_some();

        // Proof authority is tied to the exact retained transport. A fresh
        // provider/client, endpoint, account, model, or policy revision cannot
        // inherit proof from the preceding response.
        if pending_proof.is_some()
            && last_completed_target
                .as_ref()
                .is_some_and(|previous| !target.same_transport_identity(previous))
        {
            pending_proof = None;
            continuation_state = None;
            verified_lower_used = false;
            proof_base_fallback_used = true;
            last_completed_target = None;
            provider_response_continuity = None;
            next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
        }
        let proof_is_valid = match pending_proof.as_ref() {
            Some(proof) => match proof.revalidate(tool_ctx).await {
                Ok(()) => true,
                Err(claurst_tools::ReceiptError::Cancelled) => return QueryOutcome::Cancelled,
                Err(claurst_tools::ReceiptError::Timeout) => {
                    return QueryOutcome::Error(ClaudeError::InferenceDispatchDenied(
                        claurst_core::dispatch::DispatchDenied::Timeout,
                    ));
                }
                Err(_) => false,
            },
            None => true,
        };
        if !proof_is_valid {
            pending_proof = None;
            continuation_state = None;
            verified_lower_used = false;
            proof_base_fallback_used = true;
            last_completed_target = None;
            provider_response_continuity = None;
            next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
        }
        if configured_verified_policy.is_some()
            && !verified_route_supported
            && next_selection_reason.is_none()
        {
            next_selection_reason = Some(EffortSelectionReason::UnsupportedRoute);
        }

        // Effective effort for THIS turn. The configured effort is overridden to
        // Ultracode when the latest genuine user message invokes the keyword.
        let base_effort_level = explicit_effort_level
            .or_else(|| configured_verified_policy.map(|policy| policy.base()));
        let mut effective_effort_level = base_effort_level;
        let mut using_verified_continuation = false;
        if eligible_verified_policy.is_some()
            && pending_proof.is_some()
            && !verified_lower_used
            && !proof_base_fallback_used
        {
            if let Some(policy) = eligible_verified_policy {
                effective_effort_level = Some(policy.continuation());
                using_verified_continuation = true;
            }
        }

        // Compute pin provenance before assembling the system prompt. A lower
        // verified effort must never influence prompt text or context reserve
        // when any host-side setting blocks that route.
        let preflight_thinking_budget = config
            .thinking_budget
            .or_else(|| effective_effort_level.and_then(|el| el.thinking_budget_tokens()));
        let preflight_temperature = config
            .temperature
            .or_else(|| {
                config
                    .agent_definition
                    .as_ref()
                    .and_then(|a| a.temperature)
                    .map(|t| t as f32)
            })
            .or_else(|| effective_effort_level.and_then(|el| el.temperature()));
        let preflight_pin_provenance = effective_pin_provenance_for_inline(
            config,
            effective_effort_level,
            preflight_thinking_budget,
            preflight_temperature,
            inline_ultracode_active,
        );
        if using_verified_continuation && preflight_pin_provenance.blocking_pin().is_some() {
            effective_effort_level = base_effort_level;
            using_verified_continuation = false;
        }

        // Verification nudge: if there are incomplete todos for this session
        // and the conversation has more than 2 turns, append a reminder.
        let system = {
            // Build a (possibly patched) config for system-prompt assembly.
            // Agent prompt prefix and todo nudge are both applied here.
            let mut patched = config.clone();

            // Progressive tool disclosure (issue #233 completion): populate
            // `enabled_tools` from the live tool set this run exposes so
            // `build_system_prompt` only emits per-tool guideline blocks for
            // tools that are actually loaded. This is the boundary #233 wired
            // up; sub-agents already set it explicitly, so only fill it in when
            // the caller left it unset.
            if patched.enabled_tools.is_none() {
                patched.enabled_tools = Some(
                    active_schemas
                        .definitions()
                        .map(|definition| definition.name.clone())
                        .collect(),
                );
            }

            // Apply agent system-prompt prefix: prepend before the main system prompt.
            if let Some(ref agent) = config.agent_definition {
                if let Some(ref agent_prompt) = agent.prompt {
                    patched.system_prompt = Some(match &config.system_prompt {
                        Some(existing) => format!("{}\n\n{}", agent_prompt, existing),
                        None => agent_prompt.clone(),
                    });
                }
            }

            // If managed-agent mode is active, append orchestration instructions.
            if let Some(ref ma_config) = config.managed_agents {
                if ma_config.enabled {
                    let ma_prompt =
                        crate::managed_orchestrator::managed_agent_system_prompt(ma_config);
                    patched.append_system_prompt = Some(match &patched.append_system_prompt {
                        Some(existing) => format!("{}\n\n{}", existing, ma_prompt),
                        None => ma_prompt,
                    });
                }
            }

            // Apply todo nudge on turns > 2.
            if turn > 2 {
                let nudge = build_todo_nudge(&tool_ctx.session_id);
                if !nudge.is_empty() {
                    patched.append_system_prompt = Some(match &config.append_system_prompt {
                        Some(existing) => format!("{}\n\n{}", existing, nudge),
                        None => nudge,
                    });
                }
            }

            // Goal system-prompt addendum (issue #230 / MI-3). Applied fresh
            // each turn (goal state — turns used, elapsed — changes over the
            // run) whenever goal continuation mode is active and a live goal
            // exists for this session. This relocates the addendum injection
            // from the CLI into the loop so continuation turns get it too.
            // GoalStore access here is fully synchronous (no lock held across
            // an `.await`).
            if matches!(
                config.continuation,
                crate::continuation::ContinuationMode::Goal
            ) {
                if let Some(goal) = claurst_core::GoalStore::open_default()
                    .and_then(|s| s.get_active_goal(&tool_ctx.session_id))
                {
                    let addendum = claurst_core::goal_system_prompt_addendum(&goal);
                    patched.append_system_prompt =
                        Some(match patched.append_system_prompt.take() {
                            Some(existing) => format!("{}\n{}", existing, addendum),
                            None => addendum,
                        });
                }
            }

            // Ultracode effort. When the effective effort for this turn is
            // Ultracode (set by the `ultracode` keyword or an explicit /effort
            // ultracode), inject the ultracode operating procedure as a per-turn
            // system addendum — the same injection path the goal addendum uses.
            // The keyword also raises the effort to top reasoning (see the
            // budget / provider mapping below). Applied fresh each turn so it
            // deactivates naturally, and composes with goal mode.
            if effective_effort_level == Some(claurst_core::effort::EffortLevel::Ultracode) {
                let uc_addendum = claurst_core::effort::ultracode_system_prompt_addendum();
                patched.append_system_prompt = Some(match patched.append_system_prompt.take() {
                    Some(existing) => format!("{}\n{}", existing, uc_addendum),
                    None => uc_addendum,
                });
            }

            // Output-style persona for THIS turn. An inline `rocky` / `caveman`
            // / `normal` keyword in the latest user message overrides the
            // persisted output style transiently (used for this turn, then
            // reverts); otherwise the persisted selection stands. Mirrors the
            // ultracode keyword's transient-vs-persistent behaviour above.
            let (turn_output_style, turn_output_style_prompt) =
                effective_output_style_for_turn(config, messages.as_slice());
            patched.output_style = turn_output_style;
            patched.output_style_prompt = turn_output_style_prompt;

            // The broker resolves this scope again at execution. Describe the
            // same directory to the model after EnterWorktree/ExitWorktree.
            // Keep recovery tools available when the recorded scope is unknown.
            match claurst_tools::worktree::execution_context(tool_ctx) {
                Ok(effective) => {
                    patched.working_directory = Some(effective.working_dir.display().to_string());
                }
                Err(error) => {
                    patched.working_directory =
                        Some("[unavailable: worktree recovery required]".into());
                    let notice = format!("Tool execution is blocked by worktree state: {error}. Inspect the recorded session state. ExitWorktree with action=keep can release an unconfirmed or missing worktree; corrupt records require repair.");
                    patched.append_system_prompt =
                        Some(match patched.append_system_prompt.take() {
                            Some(existing) => format!("{existing}\n{notice}"),
                            None => notice,
                        });
                }
            }

            build_system_prompt(&patched)
        };

        // Resolve effective thinking budget:
        //   1. Explicit `thinking_budget` in config takes precedence.
        //   2. Fall back to the effort level's budget when no explicit budget is set.
        let mut effective_thinking_budget = config
            .thinking_budget
            .or_else(|| effective_effort_level.and_then(|el| el.thinking_budget_tokens()));

        // Apply temperature: explicit config value takes precedence, then agent override,
        // then effort-level override.
        let mut effective_temperature = config
            .temperature
            .or_else(|| {
                config
                    .agent_definition
                    .as_ref()
                    .and_then(|a| a.temperature)
                    .map(|t| t as f32)
            })
            .or_else(|| effective_effort_level.and_then(|el| el.temperature()));
        // Preserve host-side provenance alongside normalized wire values. The
        // effort-policy owner consumes this at the provider-options boundary;
        // keeping it local here ensures each continuation turn derives from the
        // same trusted inputs without exposing provenance in model messages.
        let effective_pin_provenance = effective_pin_provenance_for_inline(
            config,
            effective_effort_level,
            effective_thinking_budget,
            effective_temperature,
            inline_ultracode_active,
        );
        if using_verified_continuation && effective_pin_provenance.blocking_pin().is_some() {
            effective_effort_level = base_effort_level;
            using_verified_continuation = false;
            effective_thinking_budget = config
                .thinking_budget
                .or_else(|| effective_effort_level.and_then(|el| el.thinking_budget_tokens()));
            effective_temperature = config
                .temperature
                .or_else(|| {
                    config
                        .agent_definition
                        .as_ref()
                        .and_then(|a| a.temperature)
                        .map(|t| t as f32)
                })
                .or_else(|| effective_effort_level.and_then(|el| el.temperature()));
        }

        let requested_mode = match config.effort_policy {
            effort_policy::EffortPolicy::Fixed => EffortSelectionMode::Fixed,
            effort_policy::EffortPolicy::Verified(_) => EffortSelectionMode::Verified,
        };
        let applied_mode = if using_verified_continuation {
            EffortSelectionMode::Verified
        } else {
            EffortSelectionMode::Fixed
        };
        let selected_effort = effective_effort_level;
        let reason = if let Some(reason) = next_selection_reason.take() {
            reason
        } else if startup_scope_unavailable {
            EffortSelectionReason::StartupScopeUnavailable
        } else if proof_base_fallback_used {
            EffortSelectionReason::ProofInvalidated
        } else if effective_pin_provenance.blocking_pin().is_some() {
            EffortSelectionReason::Pinned
        } else if configured_verified_policy.is_some() && !verified_route_supported {
            EffortSelectionReason::UnsupportedRoute
        } else if configured_verified_policy.is_some() && hooks_present {
            EffortSelectionReason::HooksConfigured
        } else if using_verified_continuation {
            EffortSelectionReason::Initial
        } else if requested_mode == EffortSelectionMode::Fixed && config.effort_level.is_none() {
            EffortSelectionReason::FixedPolicy
        } else if requested_mode == EffortSelectionMode::Fixed {
            EffortSelectionReason::Explicit
        } else {
            EffortSelectionReason::NoProof
        };
        let selection_key = (requested_mode, applied_mode, selected_effort, reason);
        if last_effort_selection != Some(selection_key) {
            if let Some(ref tx) = event_tx {
                let _ = tx.send(QueryEvent::EffortSelection(EffortSelection::new(
                    requested_mode,
                    applied_mode,
                    selected_effort,
                    reason,
                )));
            }
            last_effort_selection = Some(selection_key);
        }

        let system = match append_active_requirements(system, loop_config.history_index.as_ref()) {
            Ok(system) => system,
            Err(error) => return QueryOutcome::Error(ClaudeError::Config(error)),
        };
        let output_reserve = u64::from(config.max_tokens)
            .saturating_add(u64::from(effective_thinking_budget.unwrap_or(0)));
        if continuation_state.is_none() {
            continuation_state = Some(continuation_state::ContinuationState::new(
                target.clone(),
                continuation_state::SelectedSettings {
                    effort: effective_effort_level,
                    thinking_budget: effective_thinking_budget,
                    temperature: effective_temperature,
                    max_tokens: config.max_tokens,
                },
            ));
        }
        let (request_budget, wire_budget) = match request_budget(config, &target, output_reserve) {
            Ok(budget) => budget,
            Err(error) => return QueryOutcome::Error(context_plan_error(error)),
        };
        if request_budget.source == BudgetSource::UnknownFallback {
            if let Some(ref tx) = event_tx {
                let _ = tx.send(QueryEvent::Status(format!(
                    "Context budget unknown for {}; using finite {}-token fallback.",
                    effective_model, request_budget.context_limit_tokens
                )));
            }
        }
        let Some(current_request_id) = user_request_id(messages, &runtime_message_ids) else {
            return QueryOutcome::Error(ClaudeError::Config(
                "context planning requires a current user request".into(),
            ));
        };
        let planning_system = reserve_history_manifest(&system);
        let planning_budget = request_budget.clone();
        let mut attempts = 0usize;
        let mut max_attempts = None;
        let (plan, system) = loop {
            attempts += 1;
            let plan = match ContextAssembler::plan_with_excluded(
                messages,
                &current_request_id,
                planning_budget.clone(),
                &wire_excluded_groups,
                |candidate, omitted| {
                    measure_context_candidate(
                        target.provider_id(),
                        target.model_id(),
                        candidate,
                        omitted,
                        &planning_system,
                        &api_tools,
                    )
                },
            ) {
                Ok(plan) => plan,
                Err(error) => return QueryOutcome::Error(context_plan_error(error)),
            };
            if max_attempts.is_none() {
                max_attempts = Some(
                    wire_excluded_groups
                        .len()
                        .saturating_add(plan.optional_selected_groups.len())
                        .saturating_add(1),
                );
            }
            let omitted_refs =
                match persist_omitted_groups(loop_config.history_index.as_ref(), &plan, messages) {
                    Ok(refs) => refs,
                    Err(error) => return QueryOutcome::Error(ClaudeError::Config(error)),
                };
            let candidate_system = match append_history_manifest(system.clone(), &omitted_refs) {
                Ok(system) => system,
                Err(_) => {
                    let Some(group_id) = next_reselection_group(&plan, &wire_excluded_groups)
                    else {
                        return QueryOutcome::Error(context_plan_error(
                            ContextPlanError::CurrentRequestTooLarge {
                                input_tokens: plan.cost.input_tokens,
                                wire_bytes: plan.cost.wire_bytes,
                                budget: plan.budget.clone(),
                            },
                        ));
                    };
                    if attempts >= max_attempts.unwrap_or(1) {
                        return QueryOutcome::Error(context_plan_error(
                            ContextPlanError::CurrentRequestTooLarge {
                                input_tokens: plan.cost.input_tokens,
                                wire_bytes: plan.cost.wire_bytes,
                                budget: plan.budget.clone(),
                            },
                        ));
                    }
                    wire_excluded_groups.insert(group_id);
                    continue;
                }
            };
            let final_cost = match measure_context_candidate(
                target.provider_id(),
                target.model_id(),
                &plan.messages,
                &plan.omitted,
                &candidate_system,
                &api_tools,
            ) {
                Ok(cost) => cost,
                Err(error) => return QueryOutcome::Error(context_plan_error(error)),
            };
            if plan.budget.fits_for_query(&final_cost) {
                break (plan, candidate_system);
            }
            let Some(group_id) = next_reselection_group(&plan, &wire_excluded_groups) else {
                return QueryOutcome::Error(context_plan_error(
                    ContextPlanError::CurrentRequestTooLarge {
                        input_tokens: final_cost.input_tokens,
                        wire_bytes: final_cost.wire_bytes,
                        budget: plan.budget.clone(),
                    },
                ));
            };
            if attempts >= max_attempts.unwrap_or(1) {
                return QueryOutcome::Error(context_plan_error(
                    ContextPlanError::CurrentRequestTooLarge {
                        input_tokens: final_cost.input_tokens,
                        wire_bytes: final_cost.wire_bytes,
                        budget: plan.budget.clone(),
                    },
                ));
            }
            wire_excluded_groups.insert(group_id);
        };
        let system_for_provider = system.clone();
        let api_messages: Vec<ApiMessage> = plan.messages.iter().map(ApiMessage::from).collect();
        let request_model = target.model_id();
        let mut legacy_wire_budget = wire_budget.clone();
        if target.provider().is_none() {
            legacy_wire_budget.media = match claurst_api::MediaAccounting::from_messages(
                "anthropic",
                request_model,
                &plan.messages,
            ) {
                Ok(media) => media,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::from(
                        claurst_api::ProviderError::budget(
                            claurst_core::ProviderId::new("anthropic"),
                            error,
                        ),
                    ));
                }
            };
        }
        let dispatch_request = if using_verified_continuation {
            let Some(proof) = pending_proof.as_ref() else {
                return QueryOutcome::Error(ClaudeError::Config(
                    "verified effort proof disappeared before dispatch".into(),
                ));
            };
            let Some(state) = continuation_state.as_mut() else {
                return QueryOutcome::Error(ClaudeError::Config(
                    "verified effort continuation state missing".into(),
                ));
            };
            if !Arc::ptr_eq(state.target(), &target) {
                return QueryOutcome::Error(ClaudeError::Config(
                    "verified effort target changed within a logical request".into(),
                ));
            }
            let settings = state.settings();
            if settings.effort != effective_effort_level
                || settings.thinking_budget != effective_thinking_budget
                || settings.temperature != effective_temperature
                || settings.max_tokens != config.max_tokens
            {
                return QueryOutcome::Error(ClaudeError::Config(
                    "verified effort settings changed within a logical request".into(),
                ));
            }
            let route = match target.dispatch_route(
                effective_effort_level.unwrap_or(claurst_core::effort::EffortLevel::Medium),
            ) {
                Ok(route) => route,
                Err(error) => {
                    return QueryOutcome::Error(ClaudeError::InferenceDispatchDenied(error));
                }
            };
            let dispatch = if let Some(existing) = state.dispatch().cloned() {
                if existing.expected_route() != &route
                    || existing.logical_request_id() != state.logical_request_id()
                {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "verified effort dispatch changed during local reselection".into(),
                    ));
                }
                existing
            } else {
                let dispatch = DispatchRequest::new(
                    proof.dispatch_permit(tool_ctx, route.clone()),
                    state.logical_request_id().clone(),
                    route,
                );
                if state.set_dispatch(dispatch.clone()).is_err() {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "verified effort dispatch entered an invalid phase".into(),
                    ));
                }
                dispatch
            };
            Some(dispatch)
        } else {
            None
        };
        let provider_scope = claurst_tools::ToolCatalogScope::from_context(&loop_ctx).ok();
        let provider_cache_scope = provider_scope.as_ref().and_then(|scope| {
            claurst_api::ProviderCacheScope::for_session(
                &workspace_scope_id(scope.workspace()),
                scope.session_id(),
            )
            .ok()
        });
        let mut req_builder = CreateMessageRequest::builder(request_model, config.max_tokens)
            .messages(api_messages)
            .system(system)
            .tools(api_tools)
            .wire_budget(legacy_wire_budget);

        if target.provider_id() == "anthropic" && provider_cache_scope.is_some() {
            req_builder = req_builder.cache_control(claurst_api::types::CacheControl::ephemeral());
        }

        if let Some(dispatch) = dispatch_request.clone() {
            req_builder = req_builder.dispatch(dispatch);
        }

        if let Some(budget) = effective_thinking_budget {
            req_builder = req_builder.thinking(ThinkingConfig::enabled(budget));
        }

        if let Some(t) = effective_temperature {
            req_builder = req_builder.temperature(t);
        }

        let request = req_builder.build();

        // Create a stream handler that forwards to the event channel
        let handler: Arc<dyn StreamHandler> = if let Some(ref tx) = event_tx {
            let tx = tx.clone();
            Arc::new(ChannelStreamHandler { tx })
        } else {
            Arc::new(claurst_api::streaming::NullStreamHandler)
        };

        // ResolvedTarget has already applied provider/model precedence and
        // retained the exact provider handle used for this request.
        if let Some(state) = continuation_state.as_mut() {
            if state.begin_attempt().is_err() {
                return QueryOutcome::Error(ClaudeError::Config(
                    "query continuation entered an invalid attempt phase".into(),
                ));
            }
        }
        if let Some(provider) = target.provider().cloned() {
            let provider_id_str = target.provider_id();
            let model_id_str = target.model_id();
            debug!("Dispatching to non-Anthropic provider");

            // Notify TUI that we're calling the provider using a random spinner verb
            if let Some(ref tx) = event_tx {
                use claurst_core::sample_spinner_verb;
                let seed = provider_id_str.len() ^ model_id_str.len();
                let verb = sample_spinner_verb(seed);
                let _ = tx.send(QueryEvent::Status(format!("✳ {}…", verb)));
            }

            // Build ProviderRequest from the already-assembled request data.
            // tools comes from the api_tools we already built above.
            let caps = target
                .capabilities()
                .cloned()
                .unwrap_or_else(|| provider.capabilities());
            // Max-steps degradation (issue #230): dispatch the final
            // summary turn with no tools so the provider can only emit
            // text (opencode's `toolChoice:"none"` equivalent).
            let provider_tools: Vec<claurst_core::types::ToolDefinition> =
                if caps.tool_calling && !degradation_turn {
                    active_schemas.definitions().cloned().collect()
                } else {
                    Vec::new()
                };
            // Preserve media in the canonical provider snapshot. The
            // adapter's finite wire preparation accounts for supported
            // media or returns a visible preparation error; replacing
            // it with text would silently change the user's request.
            let provider_messages = plan.messages.clone();

            let mut provider_request = claurst_api::ProviderRequest {
                model: model_id_str.to_owned(),
                messages: provider_messages,
                system_prompt: Some(system_for_provider.clone()),
                tools: provider_tools,
                max_tokens: config.max_tokens,
                temperature: effective_temperature.map(|t| t as f64),
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
                thinking: if caps.thinking {
                    effective_thinking_budget.map(claurst_api::ThinkingConfig::enabled)
                } else {
                    None
                },
                provider_options: build_provider_options(
                    provider_id_str,
                    model_id_str,
                    effective_effort_level,
                    effective_thinking_budget,
                ),
                cache_scope: provider_cache_scope.clone(),
                response_continuation: None,
                wire_budget: Some(wire_budget.clone()),
                dispatch: dispatch_request.clone(),
            };

            let provider_supports_continuation =
                provider.response_continuation_support().is_enabled();
            if provider_supports_continuation {
                provider_request.response_continuation =
                    provider_response_continuity.as_ref().and_then(|state| {
                        provider_scope
                            .as_ref()
                            .filter(|scope| {
                                state.scope == **scope
                                    && target.same_transport_identity(&state.target)
                            })
                            .map(|_| state.anchor.clone())
                    });
                if provider_request.response_continuation.is_some()
                    && provider_request.validated_response_continuation().is_none()
                {
                    provider_request.response_continuation = None;
                }
            }
            if provider_request.response_continuation.is_none() {
                provider_response_continuity = None;
            }
            let provider_continuity_fingerprint = provider_supports_continuation
                .then(|| provider_request.continuity_fingerprint().ok())
                .flatten();

            // The inventory may change after planning but before the
            // adapter consumes this request.  Drop the unsent request
            // and any proof/permit bound to the old catalog, then
            // restart the same logical turn through the normal source
            // refresh path.  No stream has opened at this point, so
            // this cannot replay uncertain transport or tool work.
            if inventory_guard
                .as_ref()
                .is_some_and(|guard| !guard.is_current(tool_ctx))
            {
                if !take_unsent_inventory_restart(&mut unsent_inventory_restarts) {
                    return QueryOutcome::Error(ClaudeError::Config(
                        "tool inventory changed repeatedly before dispatch".to_string(),
                    ));
                }
                pending_proof = None;
                continuation_state = None;
                resolved_target = None;
                last_completed_target = None;
                provider_response_continuity = None;
                proof_base_fallback_used = false;
                wire_excluded_groups.clear();
                loop_ctx.catalog_view = None;
                next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
                turn = turn.saturating_sub(1);
                continue 'query_turn;
            }

            // Use create_message_stream so the TUI receives real-time
            // text deltas instead of waiting for the full response.
            let stream_result = tokio::select! {
                _ = cancel_token.cancelled() => return QueryOutcome::Cancelled,
                result = provider.create_message_stream(provider_request) => result,
            };
            let mut stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    if using_verified_continuation
                        && matches!(
                            &e,
                            claurst_api::ProviderError::DispatchRejected {
                                error: claurst_core::dispatch::DispatchDenied::ProofInvalidated,
                                ..
                            }
                        )
                    {
                        pending_proof = None;
                        next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
                        proof_base_fallback_used = true;
                        if let Some(state) = continuation_state.as_mut() {
                            let _ = state.use_proof_fallback();
                        }
                        continuation_state = None;
                        turn = turn.saturating_sub(1);
                        continue 'query_turn;
                    }
                    // Provider adapters perform the final, local wire
                    // preparation immediately before opening a stream.
                    // A measured overflow is the only failure eligible
                    // for same-turn context reselection; transport and
                    // stream retries must never replay this path.
                    let measured_budget_failure = matches!(
                        &e,
                        claurst_api::ProviderError::BudgetExceeded { error, .. }
                            if error.request_budget_fields().is_some()
                    );
                    if measured_budget_failure {
                        if let Some(group_id) = next_reselection_group(&plan, &wire_excluded_groups)
                        {
                            if let Some(state) = continuation_state.as_mut() {
                                let _ = state.reselection_after_local_budget();
                            }
                            next_selection_reason =
                                Some(EffortSelectionReason::LocalBudgetReselection);
                            wire_excluded_groups.insert(group_id);
                            turn = turn.saturating_sub(1);
                            continue 'query_turn;
                        }
                    }
                    error!(
                        error_kind = e.diagnostic_kind(),
                        status = ?e.diagnostic_status(),
                        "Provider stream failed"
                    );
                    return QueryOutcome::Error(claurst_core::error::ClaudeError::from(e));
                }
            };
            // Once a stream opens, the previous provider response ID
            // has been consumed. Any retry after an uncertain stream
            // must rebuild from full retained context.
            provider_response_continuity = None;
            unsent_inventory_restarts = 0;
            if let Some(state) = continuation_state.as_mut() {
                let _ = state.mark_streaming();
            }
            if using_verified_continuation {
                pending_proof = None;
                verified_lower_used = true;
            }
            // The stream owns this turn's provider handle. Drop the
            // cached target before any continuation so refreshed auth
            // and endpoint settings are observed on the next turn.
            resolved_target = None;
            wire_excluded_groups.clear();

            // Accumulators for building the final assistant message.
            let mut text_chunks: Vec<String> = Vec::new();
            // Accumulate reasoning/thinking content for providers like
            // DeepSeek that require reasoning_content to be sent back.
            let mut thinking_chunks: Vec<String> = Vec::new();
            // tool_call_blocks: index → (id, name, accumulated_json, thought_signature)
            // thought_signature carries Gemini's opaque per-call signature
            // through stream assembly so it survives into the persisted
            // ToolUse block and is echoed back next turn (#311).
            let mut tool_call_blocks: std::collections::HashMap<
                usize,
                (String, String, String, Option<String>),
            > = std::collections::HashMap::new();
            let mut usage = UsageInfo::default();
            let mut stop_str = "end_turn".to_string();
            let mut msg_id = uuid::Uuid::new_v4().to_string();

            use futures::StreamExt as ProviderStreamExt;
            let provider_stall_timeout = std::time::Duration::from_secs(45);
            let provider_stall = tokio::time::sleep(provider_stall_timeout);
            tokio::pin!(provider_stall);
            let mut provider_stream_stalled = false;
            // Set when the stream yields a mid-stream `Err`. The
            // accumulated text/tool-calls are then incomplete and MUST
            // NOT be assembled into a "completed" turn (issue #215).
            let mut provider_stream_error: Option<(String, &'static str, Option<u16>)> = None;
            let mut provider_message_stop = false;

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        return QueryOutcome::Cancelled;
                    }
                    _ = &mut provider_stall => {
                        provider_stream_stalled = true;
                        break;
                    }
                    event = stream.next() => {
                        provider_stall.as_mut().reset(tokio::time::Instant::now() + provider_stall_timeout);
                        match event {
                            None => break,
                            Some(Err(e)) => {
                                let error_kind = e.diagnostic_kind();
                                let status = e.diagnostic_status();
                                error!(
                                    error_kind,
                                    status = ?status,
                                    "Provider stream error"
                                );
                                provider_stream_error =
                                    Some((e.to_string(), error_kind, status));
                                break;
                            }
                            Some(Ok(evt)) => {
                                // Forward to TUI via AnthropicStreamEvent mapping.
                                if let Some(ref tx) = event_tx {
                                    if let Some(ae) = map_to_anthropic_event(&evt) {
                                        let _ = tx.send(QueryEvent::Stream(ae));
                                    }
                                    if let claurst_api::StreamEvent::ProviderWireWitness(witness) = &evt {
                                        let _ = tx.send(QueryEvent::ProviderWireWitness(witness.clone()));
                                    }
                                }

                                // Accumulate response data.
                                match &evt {
                                    claurst_api::StreamEvent::MessageStart { id, usage: u, .. } => {
                                        msg_id = id.clone();
                                        merge_provider_stream_usage(&mut usage, u);
                                    }
                                    claurst_api::StreamEvent::ContentBlockStart {
                                        index,
                                        content_block: ContentBlock::ToolUse { id, name, thought_signature, .. },
                                    } => {
                                        tool_call_blocks.insert(*index, (id.clone(), name.clone(), String::new(), thought_signature.clone()));
                                    }
                                    claurst_api::StreamEvent::TextDelta { text, .. } => {
                                        text_chunks.push(text.clone());
                                    }
                                    claurst_api::StreamEvent::ThinkingDelta { thinking, .. } => {
                                        thinking_chunks.push(thinking.clone());
                                    }
                                    claurst_api::StreamEvent::ReasoningDelta { reasoning, .. } => {
                                        thinking_chunks.push(reasoning.clone());
                                    }
                                    claurst_api::StreamEvent::InputJsonDelta { index, partial_json } => {
                                        if let Some((_, _, buf, _)) = tool_call_blocks.get_mut(index) {
                                            buf.push_str(partial_json);
                                        }
                                    }
                                    claurst_api::StreamEvent::MessageDelta { stop_reason, usage: u } => {
                                        stop_str = match stop_reason {
                                            Some(claurst_api::provider_types::StopReason::ToolUse) => "tool_use".to_string(),
                                            Some(claurst_api::provider_types::StopReason::MaxTokens) => "max_tokens".to_string(),
                                            Some(claurst_api::provider_types::StopReason::StopSequence) => "stop_sequence".to_string(),
                                            Some(claurst_api::provider_types::StopReason::ContentFiltered) => "content_filtered".to_string(),
                                            Some(claurst_api::provider_types::StopReason::EndTurn) => "end_turn".to_string(),
                                            Some(claurst_api::provider_types::StopReason::Other(s)) => s.clone(),
                                            None => "end_turn".to_string(),
                                        };
                                        if let Some(u) = u {
                                            merge_provider_stream_usage(&mut usage, u);
                                        }
                                    }
                                    claurst_api::StreamEvent::MessageStop => {
                                        provider_message_stop = true;
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }

            let provider_stream_uncertain = provider_stream_stalled
                || provider_stream_error.is_some()
                || !provider_message_stop;
            if adaptive_policy_active && provider_stream_uncertain {
                return QueryOutcome::Error(ClaudeError::Api(
                            "Verified effort stream ended before terminal completion; adaptive continuation aborted."
                                .to_string(),
                        ));
            }
            if provider_stream_uncertain {
                if let Some(state) = continuation_state.as_mut() {
                    state.abort();
                }
                continuation_state = None;
            }

            // If the stream stalled (no data for 45s), retry.
            if provider_stream_stalled && retries_left > 0 {
                retries_left -= 1;
                warn!(retries_left, "Provider stream stalled — retrying");
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status(format!(
                        "No response for 45s — retrying ({} left)…",
                        retries_left + 1
                    )));
                }
                turn -= 1;
                continue;
            }

            // A mid-stream error means the accumulated text and
            // tool-call JSON are incomplete/untrustworthy. Do NOT fall
            // through to assemble and execute tools from a truncated
            // stream (issue #215 — an Edit/Write could otherwise run
            // with empty `{}` args). Mirror the Anthropic branch's
            // retry semantics: retry the turn if retries remain,
            // otherwise surface the failure as a QueryOutcome::Error.
            if let Some((err, error_kind, status)) = provider_stream_error {
                if retries_left > 0 {
                    retries_left -= 1;
                    warn!(
                        retries_left,
                        error_kind,
                        status = ?status,
                        "Provider stream error — retrying turn"
                    );
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::Status(format!(
                            "Stream error — retrying ({} left)…",
                            retries_left + 1
                        )));
                    }
                    turn -= 1;
                    continue;
                }
                error!(
                    error_kind,
                    status = ?status,
                    "Provider stream error — retries exhausted; aborting turn"
                );
                return QueryOutcome::Error(ClaudeError::Api(format!(
                    "Provider '{}' stream error (model '{}'): {}",
                    provider_id_str, model_id_str, err
                )));
            }

            if let Some(state) = continuation_state.as_mut() {
                let _ = state.mark_completed();
            }
            last_completed_target = Some(target.clone());
            continuation_state = None;

            // Build the content blocks from accumulated stream data.
            let mut content_blocks: Vec<ContentBlock> = Vec::new();

            // Thinking / reasoning block — must come first so that
            // inject_reasoning_for_tool_turns can find it later.
            let combined_thinking = thinking_chunks.join("");
            if !combined_thinking.is_empty() {
                content_blocks.push(ContentBlock::Thinking {
                    thinking: combined_thinking.clone(),
                    signature: String::new(),
                });
            }

            let combined_text = text_chunks.join("");
            if !combined_text.is_empty() {
                content_blocks.push(ContentBlock::Text {
                    text: combined_text.clone(),
                });
            }

            // Reconstruct tool-use blocks (sorted by index for determinism).
            let mut tc_indices: Vec<usize> = tool_call_blocks.keys().cloned().collect();
            tc_indices.sort();
            // Tool calls whose accumulated JSON arguments failed to
            // parse. We still emit a tool_use block (so the assistant
            // message stays well-formed and every tool_use has a
            // matching tool_result), but we must NOT execute the tool
            // with empty/garbage input — instead we surface a tool
            // error to the model so it can retry (issue #215).
            let mut malformed_tool_calls: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for idx in tc_indices {
                if let Some((id, name, json_str, thought_signature)) = tool_call_blocks.remove(&idx)
                {
                    let input = match parse_tool_args(&json_str) {
                        Ok(v) => v,
                        Err(_) => {
                            warn!(
                                        "Tool-call arguments failed to parse (truncated/malformed JSON); surfacing a tool error instead of executing with empty args"
                                    );
                            malformed_tool_calls.insert(id.clone());
                            // Placeholder input — this call is never executed.
                            serde_json::json!({})
                        }
                    };
                    content_blocks.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input,
                        thought_signature,
                    });
                }
            }

            let mut assistant_msg = Message {
                role: claurst_core::types::Role::Assistant,
                content: claurst_core::types::MessageContent::Blocks(content_blocks.clone()),
                uuid: Some(msg_id.clone()),
                cost: None,
                snapshot_patch: None,
            };

            cost_tracker.add_usage(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
            );

            ensure_unique_message_id(messages, &mut assistant_msg);
            messages.push(assistant_msg.clone());

            if provider_message_stop && provider.response_continuation_support().is_enabled() {
                provider_response_continuity = provider_scope
                    .clone()
                    .zip(provider_continuity_fingerprint.clone())
                    .and_then(|(scope, fingerprint)| {
                        claurst_api::ResponseContinuation::new(
                            msg_id.clone(),
                            messages,
                            fingerprint,
                        )
                        .ok()
                        .map(|anchor| ProviderResponseContinuity {
                            target: Arc::clone(&target),
                            scope,
                            anchor,
                        })
                    });
            }

            // Handle tool-use turn: execute tools and loop.
            let tool_use_blocks: Vec<_> = content_blocks
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::ToolUse {
                        id, name, input, ..
                    } = b
                    {
                        Some((id.clone(), name.clone(), input.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            // Execute tools if any tool_use blocks were returned.
            // Note: we check the blocks themselves rather than relying
            // solely on stop_str == "tool_use" because many OpenAI-
            // compatible providers (Ollama, LM Studio, etc.) return
            // finish_reason "stop" even when tool calls are present.
            if !tool_use_blocks.is_empty() {
                let native_mcp_selections: Vec<_> = tool_use_blocks
                    .iter()
                    .map(|(_, tool_name, _)| {
                        if active_schemas.catalog_id() != catalog_view.catalog_id()
                            || !catalog_view.matches_context(tool_ctx)
                            || !active_schemas.is_disclosed(tool_name)
                        {
                            None
                        } else {
                            selected_native_mcp_identity(
                                tool_ctx.mcp_manager.as_deref(),
                                tool_name,
                                mcp_activation_binding.as_ref(),
                                &active_schemas,
                                inventory.tools(),
                            )
                        }
                    })
                    .collect();
                for (tool_id, tool_name, tool_input) in &tool_use_blocks {
                    if active_schemas.catalog_id() == catalog_view.catalog_id()
                        && catalog_view.matches_context(tool_ctx)
                        && active_schemas.is_disclosed(tool_name)
                    {
                        if let Some(ref tx) = event_tx {
                            let _ = tx.send(QueryEvent::ToolStart {
                                tool_name: tool_name.clone(),
                                tool_id: tool_id.clone(),
                                input_json: tool_input.to_string(),
                            });
                        }
                    }
                }

                let effects: Vec<ToolEffect> = tool_use_blocks
                    .iter()
                    .map(|(_, name, _)| {
                        inventory
                            .tools()
                            .iter()
                            .find(|tool| tool.name() == name)
                            .map(|tool| tool.effect())
                            .unwrap_or(ToolEffect::Unknown)
                    })
                    .collect();
                let active_schema_snapshot = active_schemas.clone();
                let catalog_snapshot = catalog_view.clone();
                let batch_tools = inventory.tools();
                let batch_inventory_guard = inventory_guard.clone();
                let dispatch_event_tx = event_tx.as_ref();
                let exec_futures: Vec<_> = tool_use_blocks
                            .iter()
                            .enumerate()
                            .map(|(index, (tool_id, tool_name, tool_input))| {
                                if malformed_tool_calls.contains(tool_id) {
                                    let message = format!(
                                        "Tool call '{}' was not executed: its arguments were malformed or truncated JSON. Retry the tool call with complete, valid JSON arguments.",
                                        tool_name
                                    );
                                    futures::future::Either::Left(async move {
                                        ToolResult::error(message)
                                    })
                                } else {
                                    let name = tool_name.clone();
                                    let input = tool_input.clone();
                                    let operation_tool_id = tool_id.clone();
                                    let selected_identity = native_mcp_selections[index].clone();
                                    let active_schema_snapshot = active_schema_snapshot.clone();
                                    let catalog_snapshot = catalog_snapshot.clone();
                                    let inventory_guard = batch_inventory_guard.clone();
                                    futures::future::Either::Right(async move {
                                        let execution = execute_tool_with_inventory(
                                            &name,
                                            &input,
                                            batch_tools,
                                            tool_ctx,
                                            &active_schema_snapshot,
                                            catalog_snapshot.as_ref(),
                                            inventory_guard,
                                        );
                                        execute_with_selected_mcp_identity(selected_identity, operation_tool_id, dispatch_event_tx, execution).await
                                    })
                                }
                            })
                            .collect();
                let effects_for_batch = effects.clone();
                let deletion_authorization = claurst_tools::DeletionAuthorization::from_user_text(
                    latest_genuine_user_text(messages, &runtime_message_ids).unwrap_or_default(),
                );
                let (mut exec_results, batch_cancelled) =
                    claurst_tools::with_deletion_authorization(
                        deletion_authorization,
                        run_tool_batch_with_effects(exec_futures, effects, &tool_ctx.cancel_token),
                    )
                    .await;
                if !batch_cancelled {
                    unsent_inventory_restarts = 0;
                }
                let discovery_receipts: Vec<_> = exec_results
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(index, result)| {
                        result
                            .take_discovery_receipt()
                            .map(|receipt| (index, receipt))
                    })
                    .collect();

                let verified_enabled = eligible_verified_policy.is_some();
                let mut retained_batch = verified_batch::RetainedToolBatch::new(
                    exec_results,
                    effects_for_batch,
                    batch_cancelled,
                    verified_enabled,
                );
                if !batch_cancelled {
                    // Provider tool rounds retain the historical fixed
                    // hook behavior. RetainedToolBatch still records
                    // hook completion and independently rejects proof
                    // whenever configured hooks are present.
                    retained_batch.mark_hooks_complete();
                }
                let (mut exec_results, proof, _batch_reason) =
                    retained_batch.finish(tool_ctx).await.into_parts();
                pending_proof = proof;
                let discovery_changed = apply_discovery_receipts(
                    &mut exec_results,
                    discovery_receipts,
                    &mut active_schemas,
                    catalog_view.as_ref(),
                    tool_ctx,
                    inventory.tools(),
                    &mut pending_proof,
                    inventory_guard.as_ref(),
                    event_tx.as_ref(),
                    &mut mcp_activation_binding,
                    config.capture_mcp_diagnostics,
                );
                match discovery_changed {
                    DiscoveryApplyOutcome::None => {}
                    DiscoveryApplyOutcome::Boundary => {}
                    DiscoveryApplyOutcome::SchemaChanged => {
                        continuation_state = None;
                        last_completed_target = None;
                        provider_response_continuity = None;
                        wire_excluded_groups.clear();
                        next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
                    }
                }
                proof_base_fallback_used = false;
                let mut tool_results = Vec::new();
                for ((tool_id, tool_name, _), result) in
                    tool_use_blocks.into_iter().zip(exec_results)
                {
                    let mut result = result;
                    let result_display = result.content.clone();
                    let was_error = result.is_error;
                    let model_content = claurst_tools::evidence::take_model_content(&mut result);
                    let result_is_error = result.is_error;
                    let result_display = match &model_content {
                        ToolResultContent::Text(error) if result_is_error && !was_error => {
                            error.clone()
                        }
                        _ => result_display,
                    };
                    let execution_witness = native_bash_execution_witness(
                        inventory.tools(),
                        &tool_name,
                        &tool_id,
                        &result,
                        &result_display,
                    );
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::ToolEnd {
                            tool_name: tool_name.clone(),
                            tool_id: tool_id.clone(),
                            result: result_display,
                            is_error: result_is_error,
                            execution_witness,
                        });
                    }
                    tool_results.push(ContentBlock::ToolResult {
                        tool_use_id: tool_id,
                        content: model_content,
                        is_error: Some(result_is_error),
                    });
                }
                push_message_with_id(
                    messages,
                    Message {
                        role: claurst_core::types::Role::User,
                        content: claurst_core::types::MessageContent::Blocks(tool_results),
                        uuid: None,
                        cost: None,
                        snapshot_patch: None,
                    },
                );
                if batch_cancelled {
                    return QueryOutcome::Cancelled;
                }
                continue; // loop for next turn
            }

            // End turn — notify TUI and return.
            // Issue #149 follow-up: providers occasionally end the
            // turn after a tool round without emitting any text or
            // tool calls, which left the user staring at a blank
            // screen ("agent randomly stops"). Surface a placeholder
            // so the user always sees *some* assistant output and
            // knows the turn really ended.
            if combined_text.is_empty() && combined_thinking.is_empty() {
                let placeholder = format!(
                    "(no response — model ended the turn with stop_reason \"{}\")",
                    stop_str
                );
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Stream(
                        AnthropicStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: claurst_api::streaming::ContentDelta::TextDelta {
                                text: placeholder.clone(),
                            },
                        },
                    ));
                }
                if let claurst_core::types::MessageContent::Blocks(ref mut blocks) =
                    assistant_msg.content
                {
                    blocks.push(ContentBlock::Text {
                        text: placeholder.clone(),
                    });
                }
                if let Some(last) = messages.last_mut() {
                    *last = assistant_msg.clone();
                }
            }

            if let Some(ref tx) = event_tx {
                let _ = tx.send(QueryEvent::TurnComplete {
                    stop_reason: stop_str.clone(),
                    turn,
                    usage: Some(usage.clone()),
                });
            }

            // Attach snapshot patch covering all file changes this query.
            if let (Some(ref snap), Some(ref hash)) = (&shadow_snap, &initial_snapshot) {
                let patch = snap.patch(hash).await;
                if !patch.files.is_empty() {
                    assistant_msg.snapshot_patch = Some(patch);
                }
            }

            continue_or_end!(assistant_msg, usage);
        }

        // Send to API
        // Recheck the source-owned inventory after all context/wire planning
        // and immediately before consuming the legacy transport.  A stale
        // unsent request is discarded and rebuilt; once stream creation starts
        // all failures remain ordinary transport/leaf outcomes.
        if inventory_guard
            .as_ref()
            .is_some_and(|guard| !guard.is_current(tool_ctx))
        {
            if !take_unsent_inventory_restart(&mut unsent_inventory_restarts) {
                return QueryOutcome::Error(ClaudeError::Config(
                    "tool inventory changed repeatedly before dispatch".to_string(),
                ));
            }
            pending_proof = None;
            continuation_state = None;
            resolved_target = None;
            last_completed_target = None;
            provider_response_continuity = None;
            proof_base_fallback_used = false;
            wire_excluded_groups.clear();
            loop_ctx.catalog_view = None;
            next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
            turn = turn.saturating_sub(1);
            continue 'query_turn;
        }
        debug!(turn, "Sending API request");
        let legacy_client = target.legacy_client().unwrap_or(client);
        let stream_result = tokio::select! {
            _ = cancel_token.cancelled() => return QueryOutcome::Cancelled,
            result = legacy_client.create_message_stream(request, handler) => result,
        };
        let mut stream_rx = match stream_result {
            Ok(rx) => rx,
            Err(e) => {
                if using_verified_continuation
                    && matches!(
                        &e,
                        ClaudeError::InferenceDispatchDenied(
                            claurst_core::dispatch::DispatchDenied::ProofInvalidated
                        )
                    )
                {
                    pending_proof = None;
                    next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
                    proof_base_fallback_used = true;
                    if let Some(state) = continuation_state.as_mut() {
                        let _ = state.use_proof_fallback();
                    }
                    continuation_state = None;
                    turn = turn.saturating_sub(1);
                    continue 'query_turn;
                }
                // The legacy adapter performs final local preparation before
                // opening HTTP. Re-select only on that typed measured error;
                // all other errors retain their existing fallback behavior.
                if matches!(&e, ClaudeError::RequestBudgetExceeded { .. }) {
                    if let Some(group_id) = next_reselection_group(&plan, &wire_excluded_groups) {
                        if let Some(state) = continuation_state.as_mut() {
                            let _ = state.reselection_after_local_budget();
                        }
                        next_selection_reason = Some(EffortSelectionReason::LocalBudgetReselection);
                        wire_excluded_groups.insert(group_id);
                        turn = turn.saturating_sub(1);
                        continue 'query_turn;
                    }
                }
                // An adaptive continuation has no safe transport retry or
                // model fallback after an unclassified leaf failure. Fixed
                // legacy behavior below retains its historical fallback.
                if adaptive_policy_active {
                    return QueryOutcome::Error(e);
                }
                // On overloaded/rate-limit errors, attempt one switch to the fallback model.
                let err_str = e.to_string().to_lowercase();
                if !used_fallback
                    && (err_str.contains("overloaded")
                        || err_str.contains("529")
                        || err_str.contains("rate_limit"))
                {
                    if let Some(ref fb) = config.fallback_model {
                        warn!("Primary model unavailable — switching to fallback");
                        if let Some(ref tx) = event_tx {
                            let _ = tx.send(QueryEvent::Status(format!(
                                "Model unavailable — switching to fallback ({})",
                                fb
                            )));
                        }
                        if let Some(state) = continuation_state.as_mut() {
                            state.abort();
                        }
                        continuation_state = None;
                        effective_model = fb.clone();
                        used_fallback = true;
                        turn -= 1; // don't count this attempt against max_turns
                        continue;
                    }
                }
                error!("API request failed");
                return QueryOutcome::Error(e);
            }
        };
        unsent_inventory_restarts = 0;
        if let Some(state) = continuation_state.as_mut() {
            let _ = state.mark_streaming();
        }
        if using_verified_continuation {
            pending_proof = None;
            verified_lower_used = true;
        }
        // The stream owns this turn's legacy client route. Re-resolve before
        // the next logical continuation so refreshed credentials/configuration
        // are observed; measured preflight failures above retain the cache.
        resolved_target = None;
        wire_excluded_groups.clear();

        // Accumulate the streamed response.
        // A stall timeout auto-retries the request if no data arrives for 45s
        // (some providers are slow; we don't want to give up too early).
        const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(45);
        let mut accumulator = StreamAccumulator::new();
        let mut legacy_stream_error = false;
        let mut legacy_message_stop = false;
        let stall_deadline = tokio::time::sleep(STALL_TIMEOUT);
        tokio::pin!(stall_deadline);

        let stream_stalled = loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    return QueryOutcome::Cancelled;
                }
                _ = &mut stall_deadline => {
                    // No data for 45s — stall detected
                    break true;
                }
                event = stream_rx.recv() => {
                    // Reset stall timer on every received event.
                    stall_deadline.as_mut().reset(tokio::time::Instant::now() + STALL_TIMEOUT);
                    match event {
                        Some(evt) => {
                            accumulator.on_event(&evt);
                            match &evt {
                                AnthropicStreamEvent::Error { error_type, .. } => {
                                    if error_type == "overloaded_error" {
                                        warn!("API overloaded");
                                    }
                                    error!("Stream error");
                                    legacy_stream_error = true;
                                    break false;
                                }
                                AnthropicStreamEvent::MessageStop => {
                                    legacy_message_stop = true;
                                    break false;
                                }
                                _ => {}
                            }
                        }
                        None => break false, // Stream ended
                    }
                }
            }
        };

        let legacy_stream_uncertain = stream_stalled || legacy_stream_error || !legacy_message_stop;
        if adaptive_policy_active && legacy_stream_uncertain {
            return QueryOutcome::Error(ClaudeError::Api(
                "Verified effort stream ended before terminal completion; adaptive continuation aborted."
                    .to_string(),
            ));
        }

        if stream_stalled && retries_left > 0 {
            if let Some(state) = continuation_state.as_mut() {
                state.abort();
            }
            continuation_state = None;
            retries_left -= 1;
            warn!(retries_left, "Stream stalled — retrying request");
            if let Some(ref tx) = event_tx {
                let _ = tx.send(QueryEvent::Status(format!(
                    "No response for 45s — retrying ({} left)…",
                    retries_left + 1
                )));
            }
            turn -= 1; // don't count this stalled attempt
            continue;
        }

        if let Some(state) = continuation_state.as_mut() {
            if legacy_stream_uncertain {
                state.abort();
            } else {
                let _ = state.mark_completed();
            }
        }
        if legacy_message_stop {
            last_completed_target = Some(target.clone());
        }
        continuation_state = None;
        let (mut assistant_msg, usage, stop_reason) = accumulator.finish();

        // Track costs
        cost_tracker.add_usage(
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
        );

        // Budget guard: abort the loop if the configured USD cap is exceeded.
        if let Some(limit) = config.max_budget_usd {
            let spent = cost_tracker.total_cost_usd();
            if spent >= limit {
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status(format!(
                        "Budget limit ${:.4} exceeded (spent ${:.4}) — stopping.",
                        limit, spent
                    )));
                }
                return QueryOutcome::BudgetExceeded {
                    cost_usd: spent,
                    limit_usd: limit,
                };
            }
        }

        // Append assistant message to conversation
        ensure_unique_message_id(messages, &mut assistant_msg);
        messages.push(assistant_msg.clone());

        // If the provider returned an unknown stop reason but the assistant
        // message contains tool_use blocks, treat it as tool_use so we don't
        // silently end the turn (issue #149: agent stops after tool call for
        // providers that emit non-standard finish reasons).
        let raw_stop = stop_reason.as_deref().unwrap_or("end_turn");
        let stop = match raw_stop {
            "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" | "content_filtered" => {
                raw_stop
            }
            _ if !assistant_msg.get_tool_use_blocks().is_empty() => {
                warn!("Unknown stop reason with tool_use blocks present; treating as tool_use");
                "tool_use"
            }
            _ => raw_stop,
        };

        // T1-3: Fire PostModelTurn hooks after the model samples a response.
        // Hooks can inject blocking errors or veto continuation entirely.
        if config.process_hooks_enabled() {
            let hook_result = fire_post_sampling_hooks(
                &assistant_msg,
                &tool_ctx.config,
                &tool_ctx.working_dir,
                tool_ctx.cancel_token.clone(),
            )
            .await;
            if !hook_result.blocking_errors.is_empty() {
                if hook_result.prevent_continuation {
                    // Hard veto: push the errors into the conversation and abort.
                    for err_msg in hook_result.blocking_errors {
                        messages.push(err_msg);
                    }
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::Status(
                            "PostModelTurn hook vetoed continuation.".to_string(),
                        ));
                    }
                    let last = messages
                        .last()
                        .cloned()
                        .unwrap_or_else(|| Message::assistant("Hook blocked continuation."));
                    return QueryOutcome::EndTurn {
                        message: last,
                        usage,
                    };
                }
                // Soft errors: inject them so the model can react next turn.
                for err_msg in hook_result.blocking_errors {
                    debug!("PostModelTurn hook injecting error message");
                    messages.push(err_msg);
                }
            }
        }

        // Resolve the effective context window ONCE per turn for the active
        // provider+model. Prefer the models.dev-backed registry value (correct
        // for every provider — 1M Gemini/GPT windows, 32k local models) and
        // fall back to the Claude-centric heuristic only when the registry has
        // no usable entry. All threshold logic below keys off this. (#216)
        let context_window = config
            .context_budget
            .context_limit_tokens
            .or_else(|| target.catalog_context_limit_tokens())
            .unwrap_or_else(|| {
                compact::resolve_context_window(
                    config.model_registry.as_deref(),
                    target.provider_id(),
                    target.model_id(),
                )
            });

        // Numerator for every threshold below: prefer the REAL context-token
        // count the provider just reported (input + cache-read + cache-creation
        // = what the model actually saw) over the chars/4 estimate. With prompt
        // caching the bare `input_tokens` field undercounts badly. Fall back to
        // the estimate only before the first response / when usage is absent. (#231)
        let real_usage = usage.total_input();
        let context_tokens =
            compact::estimate_context_tokens(messages, (real_usage > 0).then_some(real_usage));

        // Emit token warning events when approaching context limits.
        // Thresholds mirror TypeScript autoCompact.ts: 80% → Warning, 95% → Critical.
        {
            let warning_state =
                compact::calculate_token_warning_state_for_window(context_tokens, context_window);
            if warning_state != compact::TokenWarningState::Ok {
                if let Some(ref tx) = event_tx {
                    let pct_used = context_tokens as f64 / context_window as f64;
                    let _ = tx.send(QueryEvent::TokenWarning {
                        state: warning_state,
                        pct_used,
                    });
                }
            }
        }

        // Canonical history is immutable during dispatch. The assembler above
        // selects a bounded view for this request; compaction remains an explicit
        // operation and never rewrites the live transcript as a side effect of
        // a successful model turn.

        if let Some(ref tx) = event_tx {
            let _ = tx.send(QueryEvent::TurnComplete {
                turn,
                stop_reason: stop.to_string(),
                usage: Some(usage.clone()),
            });
        }

        match stop {
            "end_turn" => {
                if config.process_hooks_enabled() {
                    stop_hooks.spawn(
                        &assistant_msg,
                        &tool_ctx.config,
                        tool_ctx.working_dir.clone(),
                    );
                }

                if session_memory::extract_for_query(
                    config,
                    &target,
                    tool_ctx,
                    messages,
                    &cost_tracker,
                    std::time::Duration::from_secs(30),
                )
                .await
                {
                    return QueryOutcome::Cancelled;
                }

                if let Some(limit) = config.max_budget_usd {
                    let spent = cost_tracker.total_cost_usd();
                    if spent >= limit {
                        return QueryOutcome::BudgetExceeded {
                            cost_usd: spent,
                            limit_usd: limit,
                        };
                    }
                }

                // Trigger AutoDream consolidation check at the end of the turn.
                // maybe_trigger() checks gates + acquires lock. The nested agent
                // runs through the shared broker and is awaited here, keeping
                // its effects inside the owning cancellation/cost boundary.
                if !config.child_execution {
                    let claurst_home = claurst_core::config::Settings::config_dir();
                    let memory_dir = Some(claurst_home.join("memory"));
                    let conversations_dir = Some(claurst_home.join("conversations"));
                    if let (Some(mem), Some(conv)) = (memory_dir, conversations_dir) {
                        let dreamer = crate::auto_dream::AutoDream::new(mem, conv);
                        if let Ok(Some(task)) = dreamer.maybe_trigger().await {
                            let agent_input = serde_json::json!({
                                "description": "memory consolidation",
                                "prompt": task.prompt,
                                "max_turns": 20,
                                "system_prompt": "You are performing automatic memory consolidation. Complete the task and return a brief summary.",
                                "run_in_background": false,
                                "isolation": null
                            });
                            let agent = crate::agent_tool::AgentTool;
                            let consolidation_result = tokio::select! {
                                biased;
                                _ = tool_ctx.cancel_token.cancelled() => {
                                    ToolResult::error("AutoDream consolidation was cancelled")
                                }
                                result = claurst_tools::execute_authorized(
                                    &agent,
                                    agent_input,
                                    tool_ctx,
                                ) => result,
                            };
                            crate::auto_dream::AutoDream::finish_consolidation(
                                &task,
                                !consolidation_result.is_error,
                            )
                            .await;
                        }
                    }
                }

                // Attach snapshot patch covering all file changes this query.
                if let (Some(ref snap), Some(ref hash)) = (&shadow_snap, &initial_snapshot) {
                    let patch = snap.patch(hash).await;
                    if !patch.files.is_empty() {
                        assistant_msg.snapshot_patch = Some(patch);
                    }
                }

                continue_or_end!(assistant_msg, usage);
            }
            "max_tokens" => {
                // Mirror the TS recovery loop: inject a continuation nudge and
                // retry up to MAX_TOKENS_RECOVERY_LIMIT times before surfacing
                // the partial response as QueryOutcome::MaxTokens.
                if max_tokens_recovery_count < MAX_TOKENS_RECOVERY_LIMIT {
                    max_tokens_recovery_count += 1;
                    warn!(
                        attempt = max_tokens_recovery_count,
                        limit = MAX_TOKENS_RECOVERY_LIMIT,
                        "max_tokens hit — injecting continuation message (attempt {}/{})",
                        max_tokens_recovery_count,
                        MAX_TOKENS_RECOVERY_LIMIT,
                    );
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::Status(format!(
                            "Output token limit hit — continuing (attempt {}/{})",
                            max_tokens_recovery_count, MAX_TOKENS_RECOVERY_LIMIT
                        )));
                    }
                    // The partial assistant message must be in the history so
                    // the continuation makes sense to the model.
                    if let Err(error) = append_runtime_user_message(
                        messages,
                        &mut runtime_message_ids,
                        loop_config.history_index.as_ref(),
                        MAX_TOKENS_RECOVERY_MSG,
                        context_history::RuntimeMessageOrigin::MaxTokensRecovery,
                    ) {
                        return QueryOutcome::Error(ClaudeError::Config(error));
                    }
                    continue;
                }
                // Recovery exhausted — surface the partial response.
                warn!(
                    "max_tokens recovery exhausted after {} attempts",
                    MAX_TOKENS_RECOVERY_LIMIT
                );
                return QueryOutcome::MaxTokens {
                    partial_message: assistant_msg,
                    usage,
                };
            }
            "tool_use" => {
                // A completed tool-use turn counts as a successful recovery
                // boundary; reset the max_tokens retry counter.
                max_tokens_recovery_count = 0;
                // Extract tool calls and execute them
                let tool_blocks = assistant_msg.get_tool_use_blocks();
                if tool_blocks.is_empty() {
                    // Shouldn't happen but treat as end_turn
                    return QueryOutcome::EndTurn {
                        message: assistant_msg,
                        usage,
                    };
                }

                // ---------------------------------------------------------------------------
                // Streaming tool executor: parallel non-agent tool dispatch.
                //
                // Phase 1: Run PreToolUse hooks sequentially (they can block/deny execution
                //          and may display interactive permission dialogs).
                // Phase 2: Dispatch all non-blocked tool executions concurrently via
                //          futures::future::join_all, preserving original order.
                // Phase 3: Fire PostToolUse hooks + emit events, then collect results.
                //
                // This mirrors the TypeScript StreamingToolExecutor pattern.
                // ---------------------------------------------------------------------------

                // Intermediate record produced during Phase 1.
                struct PreparedTool {
                    id: String,
                    name: String,
                    input: Value,
                    /// The exact native MCP selection observed before hooks.
                    /// The final dispatch boundary revalidates this value;
                    /// a boolean is insufficient because catalog activation
                    /// may change while hooks are running.
                    mcp_selected_identity: Option<native_mcp_capture::NativeMcpSelectedInvocation>,
                    /// None means the pre-hook blocked execution; the String is the error reason.
                    blocked_result: Option<ToolResult>,
                }

                // Phase 1: sequential pre-hook pass.
                let mut prepared: Vec<PreparedTool> = Vec::with_capacity(tool_blocks.len());
                for block in tool_blocks {
                    if let ContentBlock::ToolUse {
                        id, name, input, ..
                    } = block
                    {
                        // Clone from the references returned by get_tool_use_blocks()
                        let id = id.clone();
                        let name = name.clone();
                        let input = input.clone();

                        // Discovery is an admission boundary, not a late
                        // execution error.  A guessed, stale, or out-of-scope
                        // name must not run hooks or reach the broker.
                        if active_schemas.catalog_id() != catalog_view.catalog_id()
                            || !catalog_view.matches_context(tool_ctx)
                            || !active_schemas.is_disclosed(&name)
                        {
                            prepared.push(PreparedTool {
                                id,
                                name,
                                input,
                                mcp_selected_identity: None,
                                blocked_result: Some(
                                    ToolResult::error(
                                        "Tool call was not run because the tool is unavailable in the current disclosed catalog.",
                                    )
                                    .with_execution_status(
                                        claurst_tools::ExecutionStatus::NotRun,
                                    ),
                                ),
                            });
                            continue;
                        }
                        if inventory_guard
                            .as_ref()
                            .is_some_and(|guard| !guard.is_current(tool_ctx))
                        {
                            prepared.push(PreparedTool {
                                id,
                                name,
                                input,
                                mcp_selected_identity: None,
                                blocked_result: Some(
                                    ToolResult::error(
                                        "Tool call was not run because the authorized tool inventory is stale.",
                                    )
                                    .with_execution_status(
                                        claurst_tools::ExecutionStatus::NotRun,
                                    ),
                                ),
                            });
                            continue;
                        }

                        // Capability and selector denials are terminal before
                        // any configured or plugin hook shell can run. An
                        // interactive Ask remains deferred to the broker.
                        if let Some(denial) = pre_hook_authorization_denial(
                            &name,
                            &input,
                            inventory.tools(),
                            tool_ctx,
                        ) {
                            prepared.push(PreparedTool {
                                id,
                                name,
                                input,
                                mcp_selected_identity: None,
                                blocked_result: Some(denial),
                            });
                            continue;
                        }

                        let mcp_selected_identity = selected_native_mcp_identity(
                            tool_ctx.mcp_manager.as_deref(),
                            &name,
                            mcp_activation_binding.as_ref(),
                            &active_schemas,
                            inventory.tools(),
                        );
                        if let Some(ref tx) = event_tx {
                            let _ = tx.send(QueryEvent::ToolStart {
                                tool_name: name.clone(),
                                tool_id: id.clone(),
                                input_json: input.to_string(),
                            });
                        }

                        let (pre_outcome, plugin_pre_outcome) = if config.process_hooks_enabled() {
                            let hooks = &tool_ctx.config.hooks;
                            let hook_ctx = claurst_core::hooks::HookContext {
                                event: "PreToolUse".to_string(),
                                tool_name: Some(name.clone()),
                                tool_input: Some(input.clone()),
                                tool_output: None,
                                is_error: None,
                                session_id: Some(tool_ctx.session_id.clone()),
                            };
                            (
                                claurst_core::hooks::run_hooks_with_cancel(
                                    hooks,
                                    claurst_core::config::HookEvent::PreToolUse,
                                    &hook_ctx,
                                    &tool_ctx.working_dir,
                                    tool_ctx.cancel_token.clone(),
                                )
                                .await,
                                claurst_plugins::run_global_pre_tool_hook(
                                    &name,
                                    &input,
                                    tool_ctx.working_dir.clone(),
                                    tool_ctx.cancel_token.clone(),
                                )
                                .await,
                            )
                        } else {
                            (
                                claurst_core::hooks::HookOutcome::Allowed,
                                claurst_plugins::HookOutcome::Allow,
                            )
                        };

                        let mut blocked_result = if let claurst_core::hooks::HookOutcome::Blocked(
                            reason,
                        ) = pre_outcome
                        {
                            warn!("PreToolUse hook blocked execution");
                            Some(claurst_tools::ToolResult::error(format!(
                                "Blocked by hook: {}",
                                reason
                            )))
                        } else if let claurst_plugins::HookOutcome::Deny(reason) =
                            plugin_pre_outcome
                        {
                            warn!("Plugin PreToolUse hook blocked execution");
                            Some(claurst_tools::ToolResult::error(format!(
                                "Blocked by plugin hook: {}",
                                reason
                            )))
                        } else {
                            None
                        };

                        // Hooks run asynchronously after the initial
                        // selection. Revalidate the full witness immediately
                        // before final dispatch; stale or changed activation
                        // becomes a blocked NotRun result with no invocation
                        // event and no task-local identity.
                        let final_mcp_selected_identity = if blocked_result.is_none() {
                            let current = selected_native_mcp_identity(
                                tool_ctx.mcp_manager.as_deref(),
                                &name,
                                mcp_activation_binding.as_ref(),
                                &active_schemas,
                                inventory.tools(),
                            );
                            if current == mcp_selected_identity {
                                mcp_selected_identity
                            } else {
                                blocked_result = Some(
                                        ToolResult::error(
                                            "Tool call was not run because the MCP activation changed before dispatch.",
                                        )
                                        .with_execution_status(
                                            claurst_tools::ExecutionStatus::NotRun,
                                        ),
                                    );
                                None
                            }
                        } else {
                            None
                        };

                        prepared.push(PreparedTool {
                            id,
                            name,
                            input,
                            mcp_selected_identity: final_mcp_selected_identity,
                            blocked_result,
                        });
                    }
                }

                // Phase 2: build execution futures for non-blocked tools and join them.
                // Blocked tools yield a ready future with the pre-computed error result.
                // Non-blocked tools execute concurrently via join_all.
                // Each async block owns its cloned name/input so there are no lifetime issues.
                let active_schema_snapshot = active_schemas.clone();
                let catalog_snapshot = catalog_view.clone();
                let batch_tools = inventory.tools();
                let batch_inventory_guard = inventory_guard.clone();
                let dispatch_event_tx = event_tx.as_ref();
                let exec_futures: Vec<_> = prepared
                    .iter()
                    .map(|p| {
                        if p.blocked_result.is_some() {
                            let r = p.blocked_result.clone().unwrap();
                            futures::future::Either::Left(async move { r })
                        } else {
                            let name = p.name.clone();
                            let input = p.input.clone();
                            let tool_id = p.id.clone();
                            let selected_identity = p.mcp_selected_identity.clone();
                            let active_schema_snapshot = active_schema_snapshot.clone();
                            let catalog_snapshot = catalog_snapshot.clone();
                            let inventory_guard = batch_inventory_guard.clone();
                            futures::future::Either::Right(async move {
                                let execution = execute_tool_with_inventory(
                                    &name,
                                    &input,
                                    batch_tools,
                                    tool_ctx,
                                    &active_schema_snapshot,
                                    catalog_snapshot.as_ref(),
                                    inventory_guard,
                                );
                                execute_with_selected_mcp_identity(
                                    selected_identity,
                                    tool_id,
                                    dispatch_event_tx,
                                    execution,
                                )
                                .await
                            })
                        }
                    })
                    .collect();

                let effects: Vec<ToolEffect> = prepared
                    .iter()
                    .map(|p| {
                        inventory
                            .tools()
                            .iter()
                            .find(|tool| tool.name() == p.name)
                            .map(|tool| tool.effect())
                            .unwrap_or(ToolEffect::Unknown)
                    })
                    .collect();
                let effects_for_batch = effects.clone();

                // Run all tool futures concurrently, but race the batch against the
                // loop's cancel token (issue #218): on cancellation the in-flight
                // tools are abandoned promptly instead of blocking until the
                // slowest one finishes, and a cancelled ToolResult is synthesized
                // for EVERY tool so each tool_use still gets a matching tool_result
                // and the message history stays well-formed.
                let deletion_authorization = claurst_tools::DeletionAuthorization::from_user_text(
                    latest_genuine_user_text(messages, &runtime_message_ids).unwrap_or_default(),
                );
                let (mut exec_results, batch_cancelled) =
                    claurst_tools::with_deletion_authorization(
                        deletion_authorization,
                        run_tool_batch_with_effects(exec_futures, effects, &tool_ctx.cancel_token),
                    )
                    .await;
                if !batch_cancelled {
                    unsent_inventory_restarts = 0;
                }
                let discovery_receipts: Vec<_> = exec_results
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(index, result)| {
                        result
                            .take_discovery_receipt()
                            .map(|receipt| (index, receipt))
                    })
                    .collect();

                // Retain private receipts before any result content moves into
                // the canonical transcript.  Hooks inspect this owner while
                // all scheduler futures have joined; proof finalization is
                // disabled for the default fixed policy.
                let verified_enabled = eligible_verified_policy.is_some();
                let mut retained_batch = verified_batch::RetainedToolBatch::new(
                    exec_results,
                    effects_for_batch,
                    batch_cancelled,
                    verified_enabled,
                );
                if !batch_cancelled {
                    if config.process_hooks_enabled() {
                        for (p, result) in prepared
                            .iter()
                            .zip(retained_batch.inspect_results_for_posthooks())
                        {
                            if p.blocked_result.is_some()
                                || result.execution_status()
                                    == Some(claurst_tools::ExecutionStatus::NotRun)
                            {
                                continue;
                            }
                            let hooks = &tool_ctx.config.hooks;
                            let post_ctx = claurst_core::hooks::HookContext {
                                event: "PostToolUse".to_string(),
                                tool_name: Some(p.name.clone()),
                                tool_input: Some(p.input.clone()),
                                tool_output: Some(result.content.clone()),
                                is_error: Some(result.is_error),
                                session_id: Some(tool_ctx.session_id.clone()),
                            };
                            claurst_core::hooks::run_hooks_with_cancel(
                                hooks,
                                claurst_core::config::HookEvent::PostToolUse,
                                &post_ctx,
                                &tool_ctx.working_dir,
                                tool_ctx.cancel_token.clone(),
                            )
                            .await;

                            claurst_plugins::run_global_post_tool_hook(
                                &p.name,
                                &p.input,
                                &result.content,
                                result.is_error,
                                tool_ctx.working_dir.clone(),
                                tool_ctx.cancel_token.clone(),
                            )
                            .await;
                        }
                    }
                    retained_batch.mark_hooks_complete();
                }
                let (mut exec_results, proof, _batch_reason) =
                    retained_batch.finish(tool_ctx).await.into_parts();
                pending_proof = proof;
                let discovery_changed = apply_discovery_receipts(
                    &mut exec_results,
                    discovery_receipts,
                    &mut active_schemas,
                    catalog_view.as_ref(),
                    tool_ctx,
                    inventory.tools(),
                    &mut pending_proof,
                    inventory_guard.as_ref(),
                    event_tx.as_ref(),
                    &mut mcp_activation_binding,
                    config.capture_mcp_diagnostics,
                );
                match discovery_changed {
                    DiscoveryApplyOutcome::None => {}
                    DiscoveryApplyOutcome::Boundary => {}
                    DiscoveryApplyOutcome::SchemaChanged => {
                        continuation_state = None;
                        last_completed_target = None;
                        provider_response_continuity = None;
                        wire_excluded_groups.clear();
                        next_selection_reason = Some(EffortSelectionReason::ProofInvalidated);
                    }
                }
                proof_base_fallback_used = false;
                let mut result_blocks: Vec<ContentBlock> = Vec::with_capacity(prepared.len());
                for (p, result) in prepared.iter().zip(exec_results) {
                    let mut result = result;
                    let result_display = result.content.clone();
                    let was_error = result.is_error;
                    let model_content = claurst_tools::evidence::take_model_content(&mut result);
                    let result_is_error = result.is_error;
                    let result_display = match &model_content {
                        ToolResultContent::Text(error) if result_is_error && !was_error => {
                            error.clone()
                        }
                        _ => result_display,
                    };
                    let execution_witness = native_bash_execution_witness(
                        inventory.tools(),
                        &p.name,
                        &p.id,
                        &result,
                        &result_display,
                    );
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::ToolEnd {
                            tool_name: p.name.clone(),
                            tool_id: p.id.clone(),
                            result: result_display,
                            is_error: result_is_error,
                            execution_witness,
                        });
                    }

                    result_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: p.id.clone(),
                        content: model_content,
                        is_error: if result_is_error { Some(true) } else { None },
                    });
                }

                // Append tool results as a user message so the history remains
                // valid (every tool_use is answered) even on cancellation.
                push_message_with_id(messages, Message::user_blocks(result_blocks));

                // If the batch was abandoned due to cancellation, stop the loop
                // now rather than sending the (cancelled) results back to the model.
                if batch_cancelled {
                    return QueryOutcome::Cancelled;
                }

                // Continue the loop to send results back to the model
                continue;
            }
            "stop_sequence" => {
                if config.process_hooks_enabled() {
                    stop_hooks.spawn(
                        &assistant_msg,
                        &tool_ctx.config,
                        tool_ctx.working_dir.clone(),
                    );
                }
                if let (Some(ref snap), Some(ref hash)) = (&shadow_snap, &initial_snapshot) {
                    let patch = snap.patch(hash).await;
                    if !patch.files.is_empty() {
                        assistant_msg.snapshot_patch = Some(patch);
                    }
                }
                continue_or_end!(assistant_msg, usage);
            }
            _ => {
                warn!("Unknown stop reason, treating as end_turn");

                if config.process_hooks_enabled() {
                    stop_hooks.spawn(
                        &assistant_msg,
                        &tool_ctx.config,
                        tool_ctx.working_dir.clone(),
                    );
                }
                if let (Some(ref snap), Some(ref hash)) = (&shadow_snap, &initial_snapshot) {
                    let patch = snap.patch(hash).await;
                    if !patch.files.is_empty() {
                        assistant_msg.snapshot_patch = Some(patch);
                    }
                }
                continue_or_end!(assistant_msg, usage);
            }
        }
    }
}

/// Stream handler that forwards through the host's per-call event sender.
struct ChannelStreamHandler {
    tx: event_channel::QueryEventSender,
}

impl StreamHandler for ChannelStreamHandler {
    fn on_event(&self, event: &AnthropicStreamEvent) {
        let _ = self.tx.send(QueryEvent::Stream(event.clone()));
    }
}

// ---------------------------------------------------------------------------
// Provider stream event mapping
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_api::SystemPrompt;

    #[test]
    fn context_preview_excludes_validated_base64_from_text_tokens() {
        let messages = vec![Message::user_blocks(vec![ContentBlock::Image {
            source: claurst_core::types::ImageSource {
                source_type: "base64".into(),
                media_type: Some("image/png".into()),
                data: Some(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
                        .into(),
                ),
                url: None,
            },
        }])];
        let system = SystemPrompt::Text(String::new());
        let tools = Vec::<ApiToolDefinition>::new();
        let omitted = Vec::<context_plan::OmittedComponent>::new();
        let cost = measure_context_candidate(
            "anthropic",
            "claude-sonnet-4-6",
            &messages,
            &omitted,
            &system,
            &tools,
        )
        .unwrap();
        let media = claurst_api::MediaAccounting::from_messages(
            "anthropic",
            "claude-sonnet-4-6",
            &messages,
        )
        .unwrap();
        let claurst_api::MediaAccounting::Images(images) = media else {
            panic!("fixture should produce image accounting");
        };
        let api_messages: Vec<ApiMessage> = messages.iter().map(ApiMessage::from).collect();
        let wire_bytes = serde_json::to_vec(&(&system, &api_messages, &tools, &omitted))
            .unwrap()
            .len() as u64;
        let expected = (wire_bytes - images.manifest().total_encoded_bytes() as u64).div_ceil(4)
            + images.tokens();
        assert_eq!(cost.wire_bytes, wire_bytes);
        assert_eq!(cost.input_tokens, expected);
        assert!(cost.input_tokens < wire_bytes.div_ceil(4));
    }

    #[tokio::test]
    async fn retained_typed_tool_text_is_bounded_and_evidence_page_is_exact() {
        let workspace = tempfile::tempdir().expect("workspace");
        let mut ctx = deny_all_context();
        ctx.working_dir = workspace.path().to_path_buf();
        ctx.session_id = format!("query-typed-page-{}", uuid::Uuid::new_v4());
        let source = format!("HEAD::{}::TAIL", "é".repeat(350 * 1024));
        let mut result = claurst_tools::evidence::retain_result(
            &ctx,
            ToolResult::success("shortened UI display")
                .with_typed_content(
                    vec![ContentBlock::Text {
                        text: source.clone(),
                    }],
                    &claurst_core::media::MediaLimits::default(),
                )
                .expect("valid typed text"),
        );
        let reference = result.metadata.as_ref().expect("metadata")["evidence"].clone();
        assert_eq!(reference["bytes"].as_u64(), Some(source.len() as u64));
        let preview = claurst_tools::evidence::take_model_content(&mut result);
        let ToolResultContent::Blocks(blocks) = preview else {
            panic!("typed preview");
        };
        let texts: Vec<&str> = blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let text = texts.join("\n");
        assert!(text.len() <= 32 * 1024);
        assert_eq!(text.matches("EvidenceRead id=").count(), 1);
        assert!(!text.contains("shortened UI display"));
        let offset = source.len() - 64;
        let mut page = claurst_tools::evidence::EvidenceReadTool
            .execute(
                serde_json::json!({"id": reference["id"], "offset": offset, "limit": 64}),
                &ctx,
            )
            .await;
        assert!(!page.is_error, "{}", page.content);
        assert_eq!(page.content, source[offset..]);
        page = claurst_tools::evidence::retain_result(&ctx, page);
        let page = claurst_tools::evidence::take_model_content(&mut page);
        let ToolResultContent::Blocks(blocks) = page else {
            panic!("typed page");
        };
        let text = blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains(&source[offset..]));
        assert!(!text.contains("HEAD::"));
        assert!(text.len() < 4096);
        assert_eq!(text.matches("EvidenceRead id=").count(), 1);
    }

    #[test]
    fn typed_tool_result_keeps_media_and_bounded_display_across_history_roundtrip() {
        let image = ContentBlock::Image {
            source: claurst_core::types::ImageSource {
                source_type: "base64".to_string(),
                media_type: Some("image/png".to_string()),
                data: Some(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
                        .to_string(),
                ),
                url: None,
            },
        };
        let mut result = ToolResult::success("bounded evidence")
            .with_typed_content(
                vec![image.clone()],
                &claurst_core::media::MediaLimits::default(),
            )
            .expect("fixture image should pass core validation");
        let content = claurst_tools::evidence::take_model_content(&mut result);
        let ToolResultContent::Blocks(blocks) = content else {
            panic!("typed result should remain block content");
        };
        assert!(matches!(&blocks[0], ContentBlock::Image { .. }));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text } if text == "bounded evidence"
        ));

        let message = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "tool-1".to_string(),
            content: ToolResultContent::Blocks(blocks),
            is_error: None,
        }]);
        let encoded = serde_json::to_vec(&message).expect("history should serialize");
        let restored: Message = serde_json::from_slice(&encoded).expect("history should restore");
        assert!(matches!(
            restored.content,
            claurst_core::types::MessageContent::Blocks(ref blocks)
                if matches!(
                    blocks.first(),
                    Some(ContentBlock::ToolResult {
                        content: ToolResultContent::Blocks(inner),
                        ..
                    }) if matches!(inner.first(), Some(ContentBlock::Image { .. }))
                )
        ));
    }

    #[test]
    fn unsent_inventory_restart_budget_is_finite_and_exact() {
        let mut restarts = 0;
        for _ in 0..MAX_UNSENT_INVENTORY_RESTARTS {
            assert!(take_unsent_inventory_restart(&mut restarts));
        }
        assert_eq!(restarts, MAX_UNSENT_INVENTORY_RESTARTS);
        assert!(!take_unsent_inventory_restart(&mut restarts));
        assert_eq!(restarts, MAX_UNSENT_INVENTORY_RESTARTS);
    }

    #[test]
    fn child_execution_disables_every_process_hook_path() {
        let parent = QueryConfig::default();
        assert!(parent.process_hooks_enabled());
        assert!(!parent.session_memory_enabled);

        let child = QueryConfig {
            child_execution: true,
            ..QueryConfig::default()
        };
        assert!(!child.process_hooks_enabled());
    }

    #[test]
    fn deletion_authority_uses_only_latest_genuine_user_text() {
        let mut earlier = Message::user("Please delete prior.txt.");
        earlier.uuid = Some("earlier".into());
        let mut current = Message::user("Inspect current.txt and report its length.");
        current.uuid = Some("current".into());
        let mut runtime = Message::user("Please delete runtime.txt.");
        runtime.uuid = Some("runtime".into());
        let mut host = Message::user("[System]: Please delete host.txt.");
        host.uuid = Some("host".into());
        let mut tool_result = Message::user_blocks(Vec::new());
        tool_result.uuid = Some("tool-result".into());
        let runtime_ids = std::collections::BTreeSet::from(["runtime".to_string()]);

        assert_eq!(
            latest_genuine_user_text(
                &[earlier, current, runtime, host, tool_result],
                &runtime_ids,
            ),
            Some("Inspect current.txt and report its length.")
        );
    }

    #[test]
    fn final_stream_usage_supplies_prompt_tokens_to_turn_usage() {
        let mut turn_usage = UsageInfo::default();
        let final_usage = UsageInfo {
            input_tokens: 1_200,
            output_tokens: 80,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 300,
        };

        merge_provider_stream_usage(&mut turn_usage, &final_usage);

        assert_eq!(turn_usage.input_tokens, 1_200);
        assert_eq!(turn_usage.cache_read_input_tokens, 300);
        assert_eq!(turn_usage.output_tokens, 80);
        let context_counter_increment = turn_usage.input_tokens
            + turn_usage.output_tokens
            + turn_usage.cache_creation_input_tokens
            + turn_usage.cache_read_input_tokens;
        assert_eq!(context_counter_increment, 1_580);
    }

    #[test]
    fn output_only_final_usage_preserves_start_input_tokens() {
        let mut turn_usage = UsageInfo {
            input_tokens: 900,
            output_tokens: 0,
            cache_creation_input_tokens: 100,
            cache_read_input_tokens: 0,
        };
        let final_usage = UsageInfo {
            output_tokens: 75,
            ..Default::default()
        };

        merge_provider_stream_usage(&mut turn_usage, &final_usage);

        assert_eq!(turn_usage.input_tokens, 900);
        assert_eq!(turn_usage.cache_creation_input_tokens, 100);
        assert_eq!(turn_usage.output_tokens, 75);
    }

    fn make_config(sys: Option<&str>, append: Option<&str>) -> QueryConfig {
        QueryConfig {
            model: "claude-sonnet-4-6".to_string(),
            capture_mcp_diagnostics: false,
            effort_policy: effort_policy::EffortPolicy::Fixed,
            pin_provenance: effort_policy::PinProvenance::default(),
            verified_startup_policy: None,
            verified_startup_binding: None,
            anthropic_client_factory: None,
            max_tokens: 4096,
            max_turns: 10,
            system_prompt: sys.map(String::from),
            append_system_prompt: append.map(String::from),
            output_style: claurst_core::system_prompt::OutputStyle::Default,
            output_style_prompt: None,
            working_directory: None,
            thinking_budget: None,
            temperature: None,
            tool_result_budget: 50_000,
            context_budget: ContextBudgetConfig::default(),
            history_index: None,
            effort_level: None,
            command_queue: None,
            skill_index: None,
            max_budget_usd: None,
            fallback_model: None,
            provider_registry: None,
            agent_name: None,
            agent_definition: None,
            model_registry: None,
            managed_agents: None,
            enabled_tools: None,
            continuation: crate::continuation::ContinuationMode::Default,
            child_execution: false,
            session_memory_enabled: false,
            session_memory_state: Arc::new(parking_lot::Mutex::new(
                session_memory::SessionMemoryState::new(),
            )),
            inventory_source: None,
            inventory_profile: ToolInventoryProfile::default(),
        }
    }

    // ---- parse_tool_args tests (issue #215) ---------------------------------

    #[test]
    fn test_parse_tool_args_valid_object() {
        // A complete JSON object parses to the same value.
        let v = parse_tool_args("{\"a\":1}").expect("valid JSON should parse");
        assert_eq!(v, serde_json::json!({ "a": 1 }));

        let v = parse_tool_args("{\"path\": \"/tmp/x\", \"content\": \"hi\"}")
            .expect("valid JSON should parse");
        assert_eq!(v["path"], "/tmp/x");
        assert_eq!(v["content"], "hi");
    }

    #[test]
    fn test_parse_tool_args_empty_is_empty_object() {
        // No-argument tool calls arrive as an empty (or whitespace-only)
        // buffer and must map to `{}` so the happy path still works.
        assert_eq!(parse_tool_args("").unwrap(), serde_json::json!({}));
        assert_eq!(parse_tool_args("   ").unwrap(), serde_json::json!({}));
        assert_eq!(parse_tool_args("\n\t ").unwrap(), serde_json::json!({}));
    }

    #[test]
    fn test_parse_tool_args_truncated_is_error_not_empty_object() {
        // The core of issue #215: a truncated/malformed stream must surface
        // an error, NOT silently become `{}` (which would run Edit/Write with
        // empty arguments).
        assert!(
            parse_tool_args("{\"a\":").is_err(),
            "truncated JSON must be an error"
        );
        assert!(
            parse_tool_args("{\"path\": \"/etc/passwd").is_err(),
            "truncated string value must be an error"
        );
        assert!(
            parse_tool_args("{not json}").is_err(),
            "invalid JSON must be an error"
        );

        // Regression guard: the failing cases must never resolve to `{}`.
        for bad in ["{\"a\":", "{\"path\": \"/etc/passwd", "{not json}"] {
            let resolved = parse_tool_args(bad).unwrap_or(serde_json::json!({}));
            // The OLD buggy behavior turned these into `{}`; assert we now
            // *detect* the error rather than relying on that fallback.
            assert!(
                parse_tool_args(bad).is_err(),
                "expected error for {:?}, but got {}",
                bad,
                resolved
            );
        }
    }

    // ---- build_system_prompt tests ------------------------------------------

    #[test]
    fn test_system_prompt_default_when_empty() {
        // The default prompt (no custom system prompt set) should include the
        // Forge attribution and standard sections.
        let cfg = make_config(None, None);
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(
                text.contains("Forge"),
                "Default prompt should contain attribution: {}",
                text
            );
            assert!(
                text.contains(claurst_core::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY),
                "Default prompt must contain the dynamic boundary marker"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_with_custom() {
        // A custom system prompt is injected into the cacheable section as
        // <custom_instructions>; the default sections are still present.
        let cfg = make_config(Some("You are a code reviewer."), None);
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(
                text.contains("You are a code reviewer."),
                "Custom prompt text should appear in the output"
            );
            assert!(
                text.contains("Forge"),
                "Default attribution should still be present"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn system_prompt_claims_only_the_enabled_tool_inventory() {
        let mut cfg = make_config(None, None);
        cfg.enabled_tools = Some(vec!["Read".to_string()]);

        let SystemPrompt::Text(text) = build_system_prompt(&cfg) else {
            panic!("Expected SystemPrompt::Text");
        };

        assert!(text.contains("Read supports text plus PNG and JPEG images"));
        assert!(!text.contains("Bash commands time out"));
        assert!(!text.contains("Use WebSearch"));
        assert!(!text.contains("use ToolSearch"));
        assert!(!text.contains("MCP servers"));
        assert!(!text.contains("Jupyter notebooks"));
        assert!(!text.contains("PDF"));
    }

    #[test]
    fn test_system_prompt_with_append() {
        // Appended text lands after the dynamic boundary.
        let cfg = make_config(Some("Base prompt."), Some("Additional context."));
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(text.contains("Base prompt."));
            assert!(text.contains("Additional context."));
            // append_system_prompt appears after the boundary
            let boundary_pos = text
                .find(claurst_core::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
                .expect("boundary must exist");
            let append_pos = text.find("Additional context.").unwrap();
            assert!(
                append_pos > boundary_pos,
                "Appended text must appear after the dynamic boundary"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_append_only() {
        // When only append is set, default sections are present plus the
        // appended text after the dynamic boundary.
        let cfg = make_config(None, Some("Appended text."));
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(
                text.contains("Appended text."),
                "Appended text must appear in the prompt"
            );
            let boundary_pos = text
                .find(claurst_core::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
                .expect("boundary must exist");
            let append_pos = text.find("Appended text.").unwrap();
            assert!(
                append_pos > boundary_pos,
                "Appended text must appear after the dynamic boundary"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_with_custom_output_style_prompt() {
        let mut cfg = make_config(None, None);
        cfg.output_style_prompt = Some("Answer like a pirate.".to_string());
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(text.contains("Answer like a pirate."));
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    // ---- QueryConfig tests --------------------------------------------------

    #[test]
    fn test_query_config_clone() {
        let cfg = make_config(Some("test"), Some("append"));
        let cloned = cfg.clone();
        assert_eq!(cloned.model, "claude-sonnet-4-6");
        assert_eq!(cloned.max_tokens, 4096);
        assert_eq!(cloned.system_prompt, Some("test".to_string()));
    }

    // ---- QueryOutcome variant tests -----------------------------------------

    #[test]
    fn test_query_outcome_debug() {
        // Ensure the enum variants can be created and debug-formatted
        let outcome = QueryOutcome::Cancelled;
        let s = format!("{:?}", outcome);
        assert!(s.contains("Cancelled"));

        let err_outcome = QueryOutcome::Error(claurst_core::error::ClaudeError::RateLimit);
        let s2 = format!("{:?}", err_outcome);
        assert!(s2.contains("Error"));
    }

    #[test]
    fn test_build_provider_options_for_google_gemini_3() {
        let options = build_provider_options(
            "google",
            "gemini-3-flash-preview",
            Some(claurst_core::effort::EffortLevel::High),
            None,
        );
        assert_eq!(
            options["thinkingConfig"]["thinkingLevel"],
            serde_json::json!("high")
        );
        assert_eq!(
            options["thinkingConfig"]["includeThoughts"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn test_build_provider_options_for_openrouter_gpt5() {
        let options = build_provider_options(
            "openrouter",
            "gpt-5.4",
            Some(claurst_core::effort::EffortLevel::Medium),
            None,
        );
        assert_eq!(options["reasoningEffort"], serde_json::json!("medium"));
        assert_eq!(options["textVerbosity"], serde_json::json!("low"));
        assert_eq!(options["usage"]["include"], serde_json::json!(true));
    }

    #[test]
    fn test_build_provider_options_codex_effort_ladder() {
        // Codex maps the lower tiers like any OpenAI reasoning model...
        for (level, expected) in [
            (claurst_core::effort::EffortLevel::Low, "low"),
            (claurst_core::effort::EffortLevel::Medium, "medium"),
            (claurst_core::effort::EffortLevel::High, "high"),
        ] {
            let options = build_provider_options("openai-codex", "gpt-5.5", Some(level), None);
            assert_eq!(options["reasoningEffort"], serde_json::json!(expected));
        }
        // ...but the top "Max" tier becomes "xhigh" (extra high) on Codex.
        let options = build_provider_options(
            "openai-codex",
            "gpt-5.5",
            Some(claurst_core::effort::EffortLevel::Max),
            None,
        );
        assert_eq!(options["reasoningEffort"], serde_json::json!("xhigh"));
        assert_eq!(options["reasoningSummary"], serde_json::json!("auto"));

        // Other OpenAI-compatible providers keep "high" for Max (no xhigh).
        let other = build_provider_options(
            "openrouter",
            "gpt-5.4",
            Some(claurst_core::effort::EffortLevel::Max),
            None,
        );
        assert_eq!(other["reasoningEffort"], serde_json::json!("high"));
    }

    #[test]
    fn test_build_provider_options_codex_gpt61_sol_low_effort() {
        for provider in ["codex", "openai-codex"] {
            let options = build_provider_options(
                provider,
                "gpt-6.1-sol",
                Some(claurst_core::effort::EffortLevel::Low),
                None,
            );
            assert_eq!(options["reasoningEffort"], serde_json::json!("low"));
            assert_eq!(options["reasoningSummary"], serde_json::json!("auto"));
        }

        // Keep future model IDs unsupported until they are explicitly allowlisted.
        let future = build_provider_options(
            "openai-codex",
            "gpt-6.2-sol",
            Some(claurst_core::effort::EffortLevel::Low),
            None,
        );
        assert!(future.is_null());
    }

    #[test]
    fn test_build_provider_options_for_bedrock_anthropic() {
        let options = build_provider_options(
            "amazon-bedrock",
            "anthropic.claude-sonnet-4-6-v1",
            Some(claurst_core::effort::EffortLevel::High),
            Some(10_000),
        );
        assert_eq!(
            options["reasoningConfig"]["budgetTokens"],
            serde_json::json!(10_000)
        );
    }

    #[test]
    fn test_alibaba_is_openaiish_provider() {
        // "alibaba" is an alias for "qwen" (Alibaba's DashScope backend);
        // both must be treated as OpenAI-compatible providers.
        assert!(is_openaiish_provider("alibaba"));
        assert!(is_openaiish_provider("qwen"));
    }

    // ---- apply_compact_result / #213 data-loss guard ------------------------

    fn sample_conversation() -> Vec<Message> {
        vec![
            Message::user("initial user request"),
            Message::assistant("assistant reply with important context"),
            Message::user("follow-up question"),
            Message::assistant("second assistant reply"),
        ]
    }

    fn texts(messages: &[Message]) -> Vec<String> {
        messages.iter().map(|m| m.get_all_text()).collect()
    }

    #[test]
    fn failed_compaction_preserves_messages() {
        // Regression test for #213: a failed compaction must NOT wipe the
        // conversation. Previously the reactive path drained `messages` with
        // std::mem::take and never restored them on error.
        let mut messages = sample_conversation();
        let before = texts(&messages);

        // Simulate a failed reactive_compact / context_collapse (API error,
        // Cancelled, empty summary all map to Err here).
        let outcome: Result<compact::CompactResult, ClaudeError> = Err(ClaudeError::Cancelled);
        let result = apply_compact_result(&mut messages, outcome);

        assert!(result.is_err(), "helper must surface the compaction error");
        assert_eq!(
            messages.len(),
            before.len(),
            "messages must not be emptied on failed compaction"
        );
        assert_eq!(
            texts(&messages),
            before,
            "message contents must be identical after failed compaction"
        );
    }

    #[test]
    fn failed_compaction_with_generic_error_preserves_messages() {
        // The helper is generic over the error type; any Err leaves messages
        // untouched.
        let mut messages = sample_conversation();
        let before = texts(&messages);

        let outcome: Result<compact::CompactResult, &str> = Err("empty summary");
        let result = apply_compact_result(&mut messages, outcome);

        assert_eq!(result, Err("empty summary"));
        assert_eq!(texts(&messages), before);
    }

    #[test]
    fn successful_compaction_replaces_messages() {
        // On success the compacted result replaces the live messages and the
        // freed-token count is returned.
        let mut messages = sample_conversation();
        let compacted = vec![
            Message::user("[summary of earlier conversation]"),
            Message::user("follow-up question"),
        ];
        let expected = texts(&compacted);

        let outcome: Result<compact::CompactResult, ClaudeError> = Ok(compact::CompactResult {
            messages: compacted,
            summary: "[summary of earlier conversation]".to_string(),
            tokens_freed: 4_096,
        });
        let result = apply_compact_result(&mut messages, outcome);

        assert_eq!(
            result.unwrap(),
            4_096,
            "tokens_freed must be surfaced on success"
        );
        assert_eq!(
            texts(&messages),
            expected,
            "messages must be replaced with the compacted result on success"
        );
    }

    // ---- Central permission backstop (issue #210) ---------------------------
    //
    // These tests pin the `execute_tool` backstop contract:
    //  (a) a non-self-gating tool at a gated level is DENIED (never executes)
    //      when the handler denies;
    //  (b) a self-gating tool is NOT gated centrally (no double-prompt) — its
    //      execute() runs even though the handler would deny;
    //  (c) a ReadOnly / None tool is never gated centrally.

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

    /// Permission handler that denies everything (returns `Ask`, which in a
    /// non-interactive context surfaces as a hard denial).
    struct DenyAllHandler;
    impl claurst_core::permissions::PermissionHandler for DenyAllHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Ask {
                reason: "denied by test handler".to_string(),
            }
        }
        fn request_permission(
            &self,
            request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            self.check_permission(request)
        }
    }

    struct ExplicitDenyHandler;
    impl claurst_core::permissions::PermissionHandler for ExplicitDenyHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Deny
        }

        fn request_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Deny
        }
    }

    struct AllowAllHandler;
    impl claurst_core::permissions::PermissionHandler for AllowAllHandler {
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

    /// A configurable mock tool that records whether its `execute()` ran.
    struct MockTool {
        name: &'static str,
        level: PermissionLevel,
        self_gates: bool,
        ran: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "mock tool for backstop tests"
        }
        fn permission_level(&self) -> PermissionLevel {
            self.level
        }
        fn self_gates(&self) -> bool {
            self.self_gates
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            self.ran.store(true, AtomicOrdering::SeqCst);
            ToolResult::success("mock ran")
        }
    }

    struct CountingTool {
        runs: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            "noop_tool"
        }
        fn description(&self) -> &str {
            "counting tool for wire retry tests"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn self_gates(&self) -> bool {
            false
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            self.runs.fetch_add(1, AtomicOrdering::SeqCst);
            ToolResult::success("counted")
        }
    }

    /// A large deferred inventory entry used by the end-to-end discovery
    /// bound.  Its schema is intentionally tiny: the regression is about the
    /// number of definitions entering the first provider request, not schema
    /// payload size.
    struct DeferredInventoryTool {
        name: String,
        runs: Arc<AtomicUsize>,
        schema_calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Tool for DeferredInventoryTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "deferred inventory fixture"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn advanced(&self) -> bool {
            true
        }
        fn input_schema(&self) -> Value {
            self.schema_calls.fetch_add(1, AtomicOrdering::SeqCst);
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            self.runs.fetch_add(1, AtomicOrdering::SeqCst);
            ToolResult::success("deferred fixture")
        }
    }

    fn deny_all_context() -> ToolContext {
        ToolContext {
            working_dir: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(DenyAllHandler),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "backstop-test".to_string(),
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

    /// (a) A tool that does NOT self-gate and requires a gated level (Execute)
    /// is blocked by the central backstop when the handler denies — and its
    /// `execute()` never runs.
    #[tokio::test]
    async fn backstop_denies_non_self_gating_gated_tool() {
        let ran = Arc::new(AtomicBool::new(false));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            name: "MockExec",
            level: PermissionLevel::Execute,
            self_gates: false,
            ran: ran.clone(),
        })];
        let ctx = deny_all_context();

        let result = execute_tool("MockExec", &serde_json::json!({}), &tools, &ctx).await;

        assert!(result.is_error, "central backstop must block a denied tool");
        assert!(
            !ran.load(AtomicOrdering::SeqCst),
            "execute() must NOT run when the backstop denies"
        );
    }

    /// (b) A self-gating tool is NOT gated by the central backstop (no double
    /// prompt): even with a deny-all handler, its `execute()` still runs
    /// because the central gate is skipped for self-gaters.
    #[tokio::test]
    async fn backstop_skips_self_gating_tool() {
        let ran = Arc::new(AtomicBool::new(false));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            name: "MockSelfGated",
            level: PermissionLevel::Execute,
            self_gates: true,
            ran: ran.clone(),
        })];
        let ctx = deny_all_context();

        let result = execute_tool("MockSelfGated", &serde_json::json!({}), &tools, &ctx).await;

        assert!(
            !result.is_error,
            "self-gating tool must not be blocked by the central backstop"
        );
        assert!(result.content.starts_with("mock ran"));
        assert!(result.content.contains("EvidenceRead id="));
        assert!(
            ran.load(AtomicOrdering::SeqCst),
            "self-gating tool's execute() must run (central gate skipped)"
        );
    }

    #[tokio::test]
    async fn capability_deny_blocks_self_gating_read_tool() {
        let ran = Arc::new(AtomicBool::new(false));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            name: "MockDeniedRead",
            level: PermissionLevel::ReadOnly,
            self_gates: true,
            ran: ran.clone(),
        })];
        let mut ctx = deny_all_context();
        ctx.permission_handler = Arc::new(ExplicitDenyHandler);

        let result = execute_tool("MockDeniedRead", &serde_json::json!({}), &tools, &ctx).await;

        assert!(result.is_error);
        assert!(!ran.load(AtomicOrdering::SeqCst));
    }

    #[tokio::test]
    async fn configured_deny_blocks_read_with_permissive_handler() {
        let ran = Arc::new(AtomicBool::new(false));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            name: "Read",
            level: PermissionLevel::ReadOnly,
            self_gates: true,
            ran: ran.clone(),
        })];
        let mut ctx = deny_all_context();
        ctx.permission_handler = Arc::new(AllowAllHandler);
        ctx.config.disallowed_tools = vec!["Read".to_string()];

        let result =
            claurst_tools::execute_authorized(tools[0].as_ref(), serde_json::json!({}), &ctx).await;

        assert!(result.is_error);
        assert!(result.content.contains("disallowed"));
        assert!(!ran.load(AtomicOrdering::SeqCst));
    }

    #[tokio::test]
    async fn configured_allowlist_blocks_unlisted_none_tools() {
        for name in ["Config", "memory"] {
            let ran = Arc::new(AtomicBool::new(false));
            let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
                name,
                level: PermissionLevel::None,
                self_gates: true,
                ran: ran.clone(),
            })];
            let mut ctx = deny_all_context();
            ctx.permission_handler = Arc::new(AllowAllHandler);
            ctx.config.allowed_tools = vec!["Read".to_string()];

            let result = execute_tool(name, &serde_json::json!({}), &tools, &ctx).await;

            assert!(result.is_error, "{name} must be blocked by the allowlist");
            assert!(!ran.load(AtomicOrdering::SeqCst));
        }
    }

    #[tokio::test]
    async fn real_mutators_are_denied_before_execution_and_policy_cannot_be_bypassed() {
        // These are real inventory tools, rather than mock tools, so this
        // locks the permission declarations at the broker boundary.
        const MUTATORS: &[&str] = &[
            "RemoteTrigger",
            "TodoWrite",
            "CronDelete",
            "GoalComplete",
            "TaskCreate",
            "TaskUpdate",
            "SendMessage",
            "mcp__auth",
            "Agent",
        ];

        let mut tools = claurst_tools::all_tools();
        tools.push(Box::new(AgentTool));

        for name in MUTATORS {
            let mut ctx = deny_all_context();
            ctx.permission_handler = Arc::new(ExplicitDenyHandler);
            let result = execute_tool(name, &serde_json::json!({}), &tools, &ctx).await;

            assert!(result.is_error, "{name} must be denied before execute()");
            assert_eq!(
                result.metadata.as_ref().unwrap()["execution_status"],
                "not_run",
                "{name} denial must certify that no side effect was started"
            );

            // An allow-all permission handler must not bypass an explicit
            // configured selector deny.
            let mut ctx = deny_all_context();
            ctx.permission_handler = Arc::new(AllowAllHandler);
            ctx.config.disallowed_tools = vec![(*name).to_string()];
            let result = execute_tool(name, &serde_json::json!({}), &tools, &ctx).await;

            assert!(result.is_error, "{name} must honor configured denial");
            assert_eq!(
                result.metadata.as_ref().unwrap()["execution_status"],
                "not_run",
                "{name} configured denial must certify that no side effect was started"
            );
            assert!(result.content.contains("disallowed"));
        }
    }

    #[tokio::test]
    async fn authorized_broker_skips_cancelled_nested_tool() {
        let ran = Arc::new(AtomicBool::new(false));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            name: "NestedAgent",
            level: PermissionLevel::None,
            self_gates: false,
            ran: ran.clone(),
        })];
        let ctx = deny_all_context();
        ctx.cancel_token.cancel();

        let result =
            claurst_tools::execute_authorized(tools[0].as_ref(), serde_json::json!({}), &ctx).await;

        assert!(result.is_error);
        assert!(result.content.contains("cancelled"));
        assert!(!ran.load(AtomicOrdering::SeqCst));
    }

    /// (c) ReadOnly and None tools are never gated centrally, so they run even
    /// under a deny-all handler.
    #[tokio::test]
    async fn backstop_skips_read_only_and_none_tools() {
        for level in [PermissionLevel::ReadOnly, PermissionLevel::None] {
            let ran = Arc::new(AtomicBool::new(false));
            let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
                name: "MockSafe",
                level,
                self_gates: false,
                ran: ran.clone(),
            })];
            let ctx = deny_all_context();

            let result = execute_tool("MockSafe", &serde_json::json!({}), &tools, &ctx).await;

            assert!(
                !result.is_error,
                "{:?} tool must not be gated centrally",
                level
            );
            assert!(
                ran.load(AtomicOrdering::SeqCst),
                "{:?} tool's execute() must run",
                level
            );
        }
    }

    #[test]
    fn backstop_permission_level_gating_matrix() {
        assert!(!permission_level_is_gated(PermissionLevel::None));
        assert!(!permission_level_is_gated(PermissionLevel::ReadOnly));
        assert!(permission_level_is_gated(PermissionLevel::Write));
        assert!(permission_level_is_gated(PermissionLevel::Execute));
        assert!(permission_level_is_gated(PermissionLevel::Dangerous));
        assert!(permission_level_is_gated(PermissionLevel::Forbidden));
    }

    // ---- Issue #218: cancellation plumbing ---------------------------------

    /// (a) The parallel tool executor (`run_tool_batch`, the exact code the query
    /// loop runs) must abandon a long-running tool the moment the cancel token
    /// fires: with a tool future that never completes and a pre-cancelled token,
    /// the batch returns promptly instead of blocking, reports cancellation, and
    /// still yields one cancelled `ToolResult` per tool so every `tool_use` can
    /// be answered and the message history stays valid.
    #[tokio::test]
    async fn executor_abandons_in_flight_tools_on_cancel() {
        use std::future::Future;
        use std::pin::Pin;

        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel(); // pre-cancelled

        // Two tool futures: one that never completes (a long-running tool) and
        // one that would succeed. Boxed so they share a concrete type.
        let never: Pin<Box<dyn Future<Output = ToolResult> + Send>> =
            Box::pin(std::future::pending());
        let quick: Pin<Box<dyn Future<Output = ToolResult> + Send>> =
            Box::pin(async { ToolResult::success("done") });

        // If the executor blocked on the never-completing tool this would time
        // out; it must return promptly instead.
        let (results, cancelled) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_tool_batch(vec![never, quick], &cancel),
        )
        .await
        .expect("executor must return promptly, not block on the pending tool");

        assert!(cancelled, "batch must report that it was cancelled");
        assert_eq!(
            results.len(),
            2,
            "every tool_use must still receive a tool_result"
        );
        assert!(
            results.iter().all(|r| r.is_error),
            "cancelled tool results are errors"
        );
        assert!(
            results[0].content.contains("cancelled"),
            "cancelled result should say so, got: {}",
            results[0].content
        );
    }

    /// The happy path is unchanged: with a live (never-cancelled) token the batch
    /// runs the futures to completion and returns their real results in order.
    #[tokio::test]
    async fn executor_runs_to_completion_without_cancel() {
        let cancel = tokio_util::sync::CancellationToken::new();
        // `std::future::ready` gives both futures the same concrete type so they
        // share a Vec (mirroring the Either-unified futures the real loop builds).
        let f1 = std::future::ready(ToolResult::success("a"));
        let f2 = std::future::ready(ToolResult::error("b"));

        let (results, cancelled) = run_tool_batch(vec![f1, f2], &cancel).await;

        assert!(!cancelled, "no cancellation should have occurred");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "a");
        assert!(!results[0].is_error);
        assert_eq!(results[1].content, "b");
        assert!(results[1].is_error);
    }

    #[tokio::test]
    async fn scheduler_caps_contiguous_observations_and_preserves_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let futures: Vec<_> = (0..6)
            .map(|index| {
                let active = active.clone();
                let maximum = maximum.clone();
                async move {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    ToolResult::success(index.to_string())
                }
            })
            .collect();
        let effects = vec![ToolEffect::Observation; 6];
        let (results, cancelled) = run_tool_batch_with_effects(
            futures,
            effects,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await;

        assert!(!cancelled);
        assert_eq!(maximum.load(Ordering::SeqCst), DEFAULT_TOOL_OBSERVATION_CAP);
        assert_eq!(
            results
                .iter()
                .map(|result| result.content.as_str())
                .collect::<Vec<_>>(),
            vec!["0", "1", "2", "3", "4", "5"]
        );
    }

    #[tokio::test]
    async fn scheduler_barriers_mutations_and_unknown_calls() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let phase = Arc::new(AtomicUsize::new(0));
        let first_phase = phase.clone();
        let first: Pin<Box<dyn Future<Output = ToolResult> + Send>> = Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            first_phase.store(1, Ordering::SeqCst);
            ToolResult::success("read")
        });
        let mutation_phase = phase.clone();
        let mutation: Pin<Box<dyn Future<Output = ToolResult> + Send>> = Box::pin(async move {
            assert_eq!(mutation_phase.load(Ordering::SeqCst), 1);
            mutation_phase.store(2, Ordering::SeqCst);
            ToolResult::success("mutated")
        });
        let last_phase = phase.clone();
        let last: Pin<Box<dyn Future<Output = ToolResult> + Send>> = Box::pin(async move {
            assert_eq!(last_phase.load(Ordering::SeqCst), 2);
            ToolResult::success("after")
        });
        let (results, cancelled) = run_tool_batch_with_effects(
            vec![first, mutation, last],
            vec![
                ToolEffect::Observation,
                ToolEffect::Mutation,
                ToolEffect::Unknown,
            ],
            &tokio_util::sync::CancellationToken::new(),
        )
        .await;

        assert!(!cancelled);
        assert_eq!(
            results
                .iter()
                .map(|result| result.content.as_str())
                .collect::<Vec<_>>(),
            vec!["read", "mutated", "after"]
        );
    }

    #[tokio::test]
    async fn scheduler_retains_completed_mutation_on_cancellation() {
        use std::future::Future;
        use std::pin::Pin;
        let cancel = tokio_util::sync::CancellationToken::new();
        let trigger = cancel.clone();
        let completed: Pin<Box<dyn Future<Output = ToolResult> + Send>> = Box::pin(async move {
            trigger.cancel();
            ToolResult::success("committed")
        });
        let unfinished: Pin<Box<dyn Future<Output = ToolResult> + Send>> =
            Box::pin(std::future::pending());
        let (results, cancelled) = run_tool_batch_with_effects(
            vec![completed, unfinished],
            vec![ToolEffect::Mutation, ToolEffect::Mutation],
            &cancel,
        )
        .await;

        assert!(cancelled);
        assert_eq!(results[0].content, "committed");
        assert!(results[1].content.contains("not run"));
        assert!(!results[1].content.contains("uncertain"));
        assert_eq!(
            results[1].metadata.as_ref().unwrap()["execution_status"],
            "not_run"
        );
    }

    #[tokio::test]
    async fn scheduler_marks_started_unfinished_mutation_uncertain() {
        use std::future::Future;
        use std::pin::Pin;

        let cancel = tokio_util::sync::CancellationToken::new();
        let trigger = cancel.clone();
        let started: Pin<Box<dyn Future<Output = ToolResult> + Send>> = Box::pin(async move {
            trigger.cancel();
            std::future::pending::<ToolResult>().await
        });
        let (results, cancelled) =
            run_tool_batch_with_effects(vec![started], vec![ToolEffect::Mutation], &cancel).await;

        assert!(cancelled);
        assert!(results[0].content.contains("uncertain"));
        assert!(!results[0].content.contains("not run"));
        assert_eq!(
            results[0].metadata.as_ref().unwrap()["execution_status"],
            "unknown"
        );
    }

    #[tokio::test]
    async fn scheduler_marks_cancelled_observation_without_unknown_effect() {
        let cancel = tokio_util::sync::CancellationToken::new();
        let trigger = cancel.clone();
        let observation = async move {
            trigger.cancel();
            std::future::pending::<ToolResult>().await
        };
        let (results, cancelled) =
            run_tool_batch_with_effects(vec![observation], vec![ToolEffect::Observation], &cancel)
                .await;
        assert!(cancelled);
        assert_eq!(
            results[0].metadata.as_ref().unwrap()["execution_status"],
            "cancelled"
        );
    }

    /// (b) A sub-agent receives a CHILD of the parent's cancel token — exactly
    /// how `AgentTool` derives it from `ctx.cancel_token` — so cancelling the
    /// parent query propagates into the sub-agent. `ToolContext` now exposes the
    /// token, and cancelling it must flip the child.
    #[test]
    fn subagent_child_token_propagates_parent_cancel() {
        let ctx = deny_all_context();
        // AgentTool spawns each sub-agent with a token derived exactly this way.
        let child = ctx.cancel_token.child_token();

        assert!(!child.is_cancelled(), "child starts live");
        ctx.cancel_token.cancel();
        assert!(
            child.is_cancelled(),
            "cancelling the parent's token must cancel the sub-agent's child token"
        );
    }

    // ---- Issue #230 (MI-3): in-loop continuation + max-steps degradation -----

    use std::sync::Mutex as StdMutex;

    /// A provider double that records, per request, whether the tool set was
    /// empty (i.e. tools were disabled — the max-steps degradation turn) and
    /// replays a scripted response. Drives `run_query_loop` end-to-end.
    struct RecordingProvider {
        id: claurst_core::provider_id::ProviderId,
        /// One entry per request: `true` when its tool set was empty.
        tools_empty_per_request: Arc<StdMutex<Vec<bool>>>,
        /// Number of schemas actually sent on each request.
        tool_counts_per_request: Arc<StdMutex<Vec<usize>>>,
        /// The normalized model identifier sent to the provider adapter.
        models_per_request: Arc<StdMutex<Vec<String>>>,
        /// The context limit selected for each normalized provider request.
        wire_context_limits_per_request: Arc<StdMutex<Vec<u64>>>,
        /// Thinking budgets and host dispatch-envelope presence observed on
        /// the final provider request.  These are test-only wire assertions.
        thinking_budgets_per_request: Arc<StdMutex<Vec<Option<u32>>>>,
        dispatch_per_request: Arc<StdMutex<Vec<bool>>>,
        /// When true, always end the turn with text (ignores tools). Otherwise
        /// emit a `tool_use` while tools are present and end the turn once
        /// they're gone (so the degradation turn ends the loop).
        always_end_turn: bool,
        /// Number of local final-wire budget failures to return before an
        /// accepted stream. These failures represent adapter preparation and
        /// therefore must not count as inference calls.
        budget_failures_left: Arc<std::sync::atomic::AtomicUsize>,
        accepted_requests: Arc<StdMutex<usize>>,
        message_ids_per_request: Arc<StdMutex<Vec<Vec<String>>>>,
        tool_uses_left: Arc<std::sync::atomic::AtomicUsize>,
        scripted_verified_flow: bool,
        deferred_discovery_flow: bool,
    }

    /// A provider whose initial stream creation stays pending until released.
    /// Cancellation must drop that future before it can become an inference
    /// stream or allow a later tool round to run.
    struct GatedStreamProvider {
        id: claurst_core::provider_id::ProviderId,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        completed: Arc<AtomicBool>,
        calls: Arc<AtomicUsize>,
    }

    type ContinuityRequestRecord = (Option<String>, Option<String>, Option<usize>);

    struct ContinuityRecordingProvider {
        id: claurst_core::provider_id::ProviderId,
        calls: AtomicUsize,
        requests: Arc<StdMutex<Vec<ContinuityRequestRecord>>>,
    }

    struct CodexLoopbackProvider {
        id: claurst_core::provider_id::ProviderId,
        inner: claurst_api::providers::CodexProvider,
    }

    #[async_trait::async_trait]
    impl claurst_api::LlmProvider for GatedStreamProvider {
        fn id(&self) -> &claurst_core::provider_id::ProviderId {
            &self.id
        }

        fn name(&self) -> &str {
            "gated-stream-mock"
        }

        async fn create_message(
            &self,
            _request: claurst_api::ProviderRequest,
        ) -> Result<claurst_api::ProviderResponse, claurst_api::ProviderError> {
            unimplemented!("the cancellation gate only exercises streaming")
        }

        async fn create_message_stream(
            &self,
            _request: claurst_api::ProviderRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<claurst_api::StreamEvent, claurst_api::ProviderError>,
                        > + Send,
                >,
            >,
            claurst_api::ProviderError,
        > {
            self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            self.entered.notify_one();
            self.release.notified().await;
            self.completed.store(true, AtomicOrdering::SeqCst);
            Ok(Box::pin(futures::stream::empty()))
        }

        async fn health_check(
            &self,
        ) -> Result<claurst_api::ProviderStatus, claurst_api::ProviderError> {
            Ok(claurst_api::ProviderStatus::Healthy)
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

    #[async_trait::async_trait]
    impl claurst_api::LlmProvider for ContinuityRecordingProvider {
        fn id(&self) -> &claurst_core::provider_id::ProviderId {
            &self.id
        }

        fn name(&self) -> &str {
            "continuity-recording-mock"
        }

        async fn create_message(
            &self,
            _request: claurst_api::ProviderRequest,
        ) -> Result<claurst_api::ProviderResponse, claurst_api::ProviderError> {
            unimplemented!("continuity fixture uses streaming")
        }

        async fn create_message_stream(
            &self,
            request: claurst_api::ProviderRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<claurst_api::StreamEvent, claurst_api::ProviderError>,
                        > + Send,
                >,
            >,
            claurst_api::ProviderError,
        > {
            let request_index = self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            self.requests.lock().unwrap().push((
                request
                    .cache_scope
                    .as_ref()
                    .map(|scope| scope.key().to_string()),
                request
                    .response_continuation
                    .as_ref()
                    .map(|continuation| continuation.previous_response_id().to_string()),
                request
                    .response_continuation
                    .as_ref()
                    .map(claurst_api::ResponseContinuation::message_prefix_len),
            ));

            let response_id = format!("continuity-response-{request_index}");
            let mut events = vec![Ok(claurst_api::StreamEvent::MessageStart {
                id: response_id,
                model: request.model,
                usage: UsageInfo::default(),
            })];
            if request_index == 0 {
                events.extend([
                    Ok(claurst_api::StreamEvent::ContentBlockStart {
                        index: 0,
                        content_block: ContentBlock::ToolUse {
                            id: "continuity-tool-0".into(),
                            name: "noop_tool".into(),
                            input: serde_json::json!({}),
                            thought_signature: None,
                        },
                    }),
                    Ok(claurst_api::StreamEvent::InputJsonDelta {
                        index: 0,
                        partial_json: "{}".into(),
                    }),
                    Ok(claurst_api::StreamEvent::MessageDelta {
                        stop_reason: Some(claurst_api::StopReason::ToolUse),
                        usage: Some(UsageInfo::default()),
                    }),
                    Ok(claurst_api::StreamEvent::MessageStop),
                ]);
            } else {
                events.extend([
                    Ok(claurst_api::StreamEvent::TextDelta {
                        index: 0,
                        text: "done".into(),
                    }),
                    Ok(claurst_api::StreamEvent::MessageDelta {
                        stop_reason: Some(claurst_api::StopReason::EndTurn),
                        usage: Some(UsageInfo::default()),
                    }),
                    Ok(claurst_api::StreamEvent::MessageStop),
                ]);
            }
            Ok(Box::pin(futures::stream::iter(events)))
        }

        async fn health_check(
            &self,
        ) -> Result<claurst_api::ProviderStatus, claurst_api::ProviderError> {
            Ok(claurst_api::ProviderStatus::Healthy)
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
                caching: true,
                structured_output: false,
                system_prompt_style: claurst_api::SystemPromptStyle::TopLevel,
            }
        }

        fn response_continuation_support(&self) -> claurst_api::ResponseContinuationSupport {
            claurst_api::ResponseContinuationSupport::FullContextFallback
        }

        fn response_continuation_route_identity(&self, model: &str) -> Option<String> {
            Some(format!("continuity-recording:{model}"))
        }
    }

    #[async_trait::async_trait]
    impl claurst_api::LlmProvider for CodexLoopbackProvider {
        fn id(&self) -> &claurst_core::provider_id::ProviderId {
            &self.id
        }

        fn name(&self) -> &str {
            "codex-loopback-continuation"
        }

        async fn create_message(
            &self,
            request: claurst_api::ProviderRequest,
        ) -> Result<claurst_api::ProviderResponse, claurst_api::ProviderError> {
            claurst_api::LlmProvider::create_message(&self.inner, request).await
        }

        async fn create_message_stream(
            &self,
            request: claurst_api::ProviderRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<claurst_api::StreamEvent, claurst_api::ProviderError>,
                        > + Send,
                >,
            >,
            claurst_api::ProviderError,
        > {
            claurst_api::LlmProvider::create_message_stream(&self.inner, request).await
        }

        async fn health_check(
            &self,
        ) -> Result<claurst_api::ProviderStatus, claurst_api::ProviderError> {
            claurst_api::LlmProvider::health_check(&self.inner).await
        }

        fn capabilities(&self) -> claurst_api::ProviderCapabilities {
            claurst_api::LlmProvider::capabilities(&self.inner)
        }

        fn response_continuation_support(&self) -> claurst_api::ResponseContinuationSupport {
            claurst_api::LlmProvider::response_continuation_support(&self.inner)
        }

        fn response_continuation_route_identity(&self, model: &str) -> Option<String> {
            claurst_api::LlmProvider::response_continuation_route_identity(&self.inner, model)
        }
    }

    #[async_trait::async_trait]
    impl claurst_api::LlmProvider for RecordingProvider {
        fn id(&self) -> &claurst_core::provider_id::ProviderId {
            &self.id
        }
        fn name(&self) -> &str {
            "recording-mock"
        }

        async fn create_message(
            &self,
            _request: claurst_api::ProviderRequest,
        ) -> Result<claurst_api::ProviderResponse, claurst_api::ProviderError> {
            unimplemented!("these tests only use create_message_stream")
        }

        async fn create_message_stream(
            &self,
            request: claurst_api::ProviderRequest,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<claurst_api::StreamEvent, claurst_api::ProviderError>,
                        > + Send,
                >,
            >,
            claurst_api::ProviderError,
        > {
            use claurst_api::provider_types::StopReason;
            use claurst_api::StreamEvent;

            let tools_empty = request.tools.is_empty();
            self.tools_empty_per_request
                .lock()
                .unwrap()
                .push(tools_empty);
            self.tool_counts_per_request
                .lock()
                .unwrap()
                .push(request.tools.len());
            self.models_per_request
                .lock()
                .unwrap()
                .push(request.model.clone());
            self.wire_context_limits_per_request.lock().unwrap().push(
                request
                    .wire_budget
                    .as_ref()
                    .map_or(0, |budget| budget.context_limit_tokens),
            );
            self.thinking_budgets_per_request.lock().unwrap().push(
                request
                    .thinking
                    .as_ref()
                    .map(|thinking| thinking.budget_tokens),
            );
            self.dispatch_per_request
                .lock()
                .unwrap()
                .push(request.dispatch.is_some());
            self.message_ids_per_request.lock().unwrap().push(
                request
                    .messages
                    .iter()
                    .filter_map(|message| message.uuid.clone())
                    .collect(),
            );
            if self
                .budget_failures_left
                .try_update(
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                    |remaining| remaining.checked_sub(1),
                )
                .is_ok()
            {
                return Err(claurst_api::ProviderError::budget(
                    self.id.clone(),
                    claurst_api::WireBudgetError::WireOverflow {
                        wire_bytes: 9_000,
                        max_wire_bytes: 8_000,
                        estimated_input_tokens: 2_250,
                        input_limit_tokens: 2_000,
                        context_limit_tokens: 32_000,
                        output_reserve_tokens: 4_096,
                        policy_revision: "test".into(),
                        budget_source: "test".into(),
                    },
                ));
            }
            let accepted_index = {
                let mut accepted = self.accepted_requests.lock().unwrap();
                let index = *accepted;
                *accepted += 1;
                index
            };

            let msg_id = uuid::Uuid::new_v4().to_string();
            let scripted_tool = if self.deferred_discovery_flow && !tools_empty {
                match accepted_index {
                    0 => Some((
                        "ToolSearch",
                        serde_json::json!({"query": "select:DeferredInventory42"}),
                    )),
                    1 => Some(("DeferredInventory42", serde_json::json!({}))),
                    _ => None,
                }
            } else if self.scripted_verified_flow && !tools_empty {
                match accepted_index {
                    0 => Some((
                        "EditCheck",
                        serde_json::json!({
                            "operation": "Write",
                            "edit": {"file_path": "target", "content": "after"},
                            "check": {"tool": "Bash", "input": {
                                "command": "test \"$(cat target)\" = after; printf x >> checker-runs",
                                "timeout": 5000
                            }}
                        }),
                    )),
                    1 => Some(("noop_tool", serde_json::json!({}))),
                    _ => None,
                }
            } else {
                None
            };
            let emit_tool_use = scripted_tool.is_some()
                || (!self.always_end_turn
                    && !tools_empty
                    && self
                        .tool_uses_left
                        .try_update(
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                            |remaining| remaining.checked_sub(1),
                        )
                        .is_ok());

            let events: Vec<Result<StreamEvent, claurst_api::ProviderError>> = if emit_tool_use {
                let tool_id = uuid::Uuid::new_v4().to_string();
                let (tool_name, tool_input) =
                    scripted_tool.unwrap_or(("noop_tool", serde_json::json!({})));
                vec![
                    Ok(StreamEvent::MessageStart {
                        id: msg_id,
                        model: "mock-model".to_string(),
                        usage: UsageInfo::default(),
                    }),
                    Ok(StreamEvent::ContentBlockStart {
                        index: 0,
                        content_block: ContentBlock::ToolUse {
                            id: tool_id,
                            name: tool_name.to_string(),
                            input: tool_input.clone(),
                            thought_signature: None,
                        },
                    }),
                    Ok(StreamEvent::InputJsonDelta {
                        index: 0,
                        partial_json: tool_input.to_string(),
                    }),
                    Ok(StreamEvent::MessageDelta {
                        stop_reason: Some(StopReason::ToolUse),
                        usage: Some(UsageInfo::default()),
                    }),
                    Ok(StreamEvent::MessageStop),
                ]
            } else {
                vec![
                    Ok(StreamEvent::MessageStart {
                        id: msg_id,
                        model: "mock-model".to_string(),
                        usage: UsageInfo::default(),
                    }),
                    Ok(StreamEvent::TextDelta {
                        index: 0,
                        text: "Progress summary.".to_string(),
                    }),
                    Ok(StreamEvent::MessageDelta {
                        stop_reason: Some(StopReason::EndTurn),
                        usage: Some(UsageInfo::default()),
                    }),
                    Ok(StreamEvent::MessageStop),
                ]
            };

            Ok(Box::pin(futures::stream::iter(events)))
        }

        async fn health_check(
            &self,
        ) -> Result<claurst_api::ProviderStatus, claurst_api::ProviderError> {
            Ok(claurst_api::ProviderStatus::Healthy)
        }

        fn capabilities(&self) -> claurst_api::ProviderCapabilities {
            claurst_api::ProviderCapabilities {
                streaming: true,
                tool_calling: true,
                thinking: true,
                image_input: false,
                pdf_input: false,
                audio_input: false,
                video_input: false,
                caching: false,
                structured_output: false,
                system_prompt_style: claurst_api::SystemPromptStyle::TopLevel,
            }
        }

        fn dispatch_route(
            &self,
            model: &str,
            effort: claurst_core::effort::EffortLevel,
        ) -> Result<claurst_core::dispatch::DispatchRoute, claurst_core::dispatch::DispatchDenied>
        {
            claurst_core::dispatch::DispatchRoute::new(claurst_core::dispatch::DispatchRouteSpec {
                provider_id: self.id.to_string(),
                provider_instance_id: "query-test-instance".into(),
                account_identity: "query-test-account".into(),
                model_id: model.to_string(),
                endpoint_identity: "query-test-endpoint".into(),
                effort: effort.as_str().into(),
                policy_revision: 1,
            })
            .map_err(|_| claurst_core::dispatch::DispatchDenied::UnsupportedRoute)
        }
    }

    fn noop_tools() -> Vec<Box<dyn Tool>> {
        vec![Box::new(MockTool {
            name: "noop_tool",
            level: PermissionLevel::ReadOnly,
            self_gates: false,
            ran: Arc::new(AtomicBool::new(false)),
        })]
    }

    /// Drive `run_query_loop` against the recording provider. Returns the
    /// outcome, the per-request "tools were empty" record, and the final
    /// message history.
    async fn drive_loop_with_mock(
        always_end_turn: bool,
        max_turns: u32,
        tools: Vec<Box<dyn Tool>>,
        continuation: crate::continuation::ContinuationMode,
    ) -> (QueryOutcome, Vec<bool>, Vec<Message>) {
        let recorded = Arc::new(StdMutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider {
            id: claurst_core::provider_id::ProviderId::new("mockprov"),
            tools_empty_per_request: recorded.clone(),
            tool_counts_per_request: Arc::new(StdMutex::new(Vec::new())),
            models_per_request: Arc::new(StdMutex::new(Vec::new())),
            wire_context_limits_per_request: Arc::new(StdMutex::new(Vec::new())),
            thinking_budgets_per_request: Arc::new(StdMutex::new(Vec::new())),
            dispatch_per_request: Arc::new(StdMutex::new(Vec::new())),
            always_end_turn,
            budget_failures_left: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            accepted_requests: Arc::new(StdMutex::new(0)),
            message_ids_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_uses_left: Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX)),
            scripted_verified_flow: false,
            deferred_discovery_flow: false,
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);
        let registry = Arc::new(registry);

        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        })
        .expect("build test client");

        let mut ctx = deny_all_context();
        ctx.session_id = "loop-test".to_string();
        ctx.config.provider = Some("mockprov".to_string());

        let mut config = make_config(None, None);
        config.model = "mock-model".to_string();
        config.max_turns = max_turns;
        config.provider_registry = Some(registry);
        config.continuation = continuation;

        let cost = claurst_core::cost::CostTracker::new();
        let cancel = tokio_util::sync::CancellationToken::new();
        let mut messages = vec![Message::user("start")];

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            run_query_loop(
                &client,
                &mut messages,
                &tools,
                &ctx,
                &config,
                cost,
                None,
                cancel,
                None,
            ),
        )
        .await
        .expect("loop must not hang");

        let recorded = recorded.lock().unwrap().clone();
        (outcome, recorded, messages)
    }

    #[tokio::test]
    async fn provider_continuation_reuses_exact_completed_prefix_for_tool_round() {
        let recorded = Arc::new(StdMutex::new(Vec::new()));
        let provider = Arc::new(ContinuityRecordingProvider {
            id: claurst_core::provider_id::ProviderId::new("continuity-mock"),
            calls: AtomicUsize::new(0),
            requests: Arc::clone(&recorded),
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            ..Default::default()
        })
        .expect("build test client");
        let mut ctx = deny_all_context();
        ctx.session_id = "continuity-session".into();
        ctx.config.provider = Some("continuity-mock".into());
        let mut config = make_config(None, None);
        config.model = "continuity-model".into();
        config.max_turns = 3;
        config.provider_registry = Some(Arc::new(registry));
        config.context_budget.wire_byte_limit = 1_000_000;
        let mut messages = vec![Message::user("start")];
        let tools = noop_tools();

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(10),
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
        .expect("continuity loop timeout");

        assert!(
            matches!(outcome, QueryOutcome::EndTurn { .. }),
            "{outcome:?}"
        );
        let requests = recorded.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].1, None);
        assert_eq!(requests[0].2, None);
        assert_eq!(requests[1].1.as_deref(), Some("continuity-response-0"));
        assert_eq!(requests[1].2, Some(2));
        assert_eq!(requests[0].0, requests[1].0);
        assert_eq!(requests[0].0.as_ref().map(String::len), Some(64));
    }

    #[tokio::test]
    async fn codex_continuation_fallback_retries_inference_once_without_replaying_tool() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        async fn read_json_body(socket: &mut tokio::net::TcpStream) -> Value {
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
                let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                else {
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
                    return serde_json::from_slice(
                        &request[body_start..body_start + content_length],
                    )
                    .expect("request JSON");
                }
            }
        }

        fn response_sse(events: &[Value]) -> Vec<u8> {
            let mut body = Vec::new();
            for event in events {
                body.extend_from_slice(b"data: ");
                body.extend_from_slice(event.to_string().as_bytes());
                body.extend_from_slice(b"\n\n");
            }
            body
        }

        async fn write_sse(socket: &mut tokio::net::TcpStream, body: &[u8]) {
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .expect("success headers");
            socket.write_all(body).await.expect("success body");
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener");
        let address = listener.local_addr().expect("listener address");
        let tool_body = response_sse(&[
            serde_json::json!({"type":"response.created","response":{"id":"resp_tool","model":"fixture-model"}}),
            serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"id":"fc_tool","call_id":"call_tool","type":"function_call","name":"noop_tool"}}),
            serde_json::json!({"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_tool","arguments":"{\"fixture\":\"final-only\"}"}),
            serde_json::json!({"type":"response.output_item.done","output_index":0}),
            serde_json::json!({"type":"response.completed","response":{"id":"resp_tool","model":"fixture-model","usage":{"input_tokens":4,"output_tokens":1}}}),
        ]);
        let final_body = response_sse(&[
            serde_json::json!({"type":"response.created","response":{"id":"resp_final","model":"fixture-model"}}),
            serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"id":"msg_final","type":"message"}}),
            serde_json::json!({"type":"response.content_part.added","output_index":0,"part":{"type":"output_text"}}),
            serde_json::json!({"type":"response.output_text.delta","output_index":0,"delta":"done"}),
            serde_json::json!({"type":"response.output_item.done","output_index":0}),
            serde_json::json!({"type":"response.completed","response":{"id":"resp_final","model":"fixture-model","usage":{"input_tokens":7,"output_tokens":1}}}),
        ]);
        let mut server = tokio::spawn(async move {
            let mut requests = Vec::new();
            for index in 0..3 {
                let (mut socket, _) = listener.accept().await.expect("loopback accept");
                requests.push(read_json_body(&mut socket).await);
                match index {
                    0 => write_sse(&mut socket, &tool_body).await,
                    1 => {
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
                    }
                    _ => write_sse(&mut socket, &final_body).await,
                }
            }
            requests
        });

        let provider = Arc::new(CodexLoopbackProvider {
            id: claurst_core::provider_id::ProviderId::new("codex-loopback"),
            inner: claurst_api::providers::CodexProvider::new(
                claurst_core::oauth_config::CodexTokens {
                    access_token: "synthetic-codex-token".into(),
                    refresh_token: None,
                    account_id: Some("synthetic-account".into()),
                    expires_at: None,
                },
            )
            .with_base_url(format!("http://{address}/responses"))
            .with_response_continuation(),
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            ..Default::default()
        })
        .expect("build test client");
        let runs = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool { runs: runs.clone() })];
        let mut ctx = deny_all_context();
        ctx.session_id = "codex-fallback-no-replay".into();
        ctx.config.provider = Some("codex-loopback".into());
        let mut config = make_config(None, None);
        config.model = "fixture-model".into();
        config.max_turns = 3;
        config.provider_registry = Some(Arc::new(registry));
        config.context_budget.wire_byte_limit = 1_000_000;
        let mut messages = vec![Message::user("run once")];

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(10),
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
        .expect("query loop timeout");
        assert!(
            matches!(outcome, QueryOutcome::EndTurn { .. }),
            "{outcome:?}"
        );
        assert_eq!(runs.load(AtomicOrdering::SeqCst), 1);

        let requests = tokio::time::timeout(std::time::Duration::from_secs(5), &mut server)
            .await
            .expect("server timeout")
            .expect("loopback server");
        assert_eq!(requests.len(), 3);
        assert!(requests[0].get("previous_response_id").is_none());
        assert_eq!(requests[1]["previous_response_id"], "resp_tool");
        assert!(requests[2].get("previous_response_id").is_none());
        assert_eq!(
            requests[1]["input"].as_array().map(Vec::len),
            Some(1),
            "continuation attempt carries only the appended tool result"
        );
        assert_eq!(
            requests[2]["input"].as_array().map(Vec::len),
            Some(3),
            "fallback rebuilds full retained user/tool context"
        );
        let retained_call = requests[2]["input"]
            .as_array()
            .expect("full retained input")
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("retained function call");
        assert_eq!(retained_call["call_id"], "call_tool");
        let retained_arguments: Value = serde_json::from_str(
            retained_call["arguments"]
                .as_str()
                .expect("retained argument string"),
        )
        .expect("retained argument object");
        assert_eq!(
            retained_arguments,
            serde_json::json!({"fixture":"final-only"})
        );
        assert_eq!(
            requests[0]["prompt_cache_key"],
            requests[1]["prompt_cache_key"]
        );
        assert_eq!(
            requests[1]["prompt_cache_key"],
            requests[2]["prompt_cache_key"]
        );
    }

    type ProviderSelectionFixture = (
        Arc<RecordingProvider>,
        Arc<StdMutex<Vec<String>>>,
        Arc<StdMutex<Vec<u64>>>,
    );

    fn precedence_provider(id: &'static str) -> ProviderSelectionFixture {
        let models = Arc::new(StdMutex::new(Vec::new()));
        let wire_context_limits = Arc::new(StdMutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider {
            id: claurst_core::provider_id::ProviderId::new(id),
            tools_empty_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_counts_per_request: Arc::new(StdMutex::new(Vec::new())),
            models_per_request: models.clone(),
            wire_context_limits_per_request: wire_context_limits.clone(),
            thinking_budgets_per_request: Arc::new(StdMutex::new(Vec::new())),
            dispatch_per_request: Arc::new(StdMutex::new(Vec::new())),
            always_end_turn: true,
            budget_failures_left: Arc::new(AtomicUsize::new(0)),
            accepted_requests: Arc::new(StdMutex::new(0)),
            message_ids_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_uses_left: Arc::new(AtomicUsize::new(0)),
            scripted_verified_flow: false,
            deferred_discovery_flow: false,
        });
        (provider, models, wire_context_limits)
    }

    async fn run_provider_selection_case(
        mut config: QueryConfig,
        ctx: ToolContext,
        registry: Arc<claurst_api::ProviderRegistry>,
    ) -> QueryOutcome {
        config.provider_registry = Some(registry);
        config.max_turns = 1;
        config.context_budget.wire_byte_limit = 1_000_000;
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        })
        .expect("build test client");
        let mut messages = vec![Message::user("record provider resolution")];
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_query_loop(
                &client,
                &mut messages,
                &[],
                &ctx,
                &config,
                claurst_core::cost::CostTracker::new(),
                None,
                tokio_util::sync::CancellationToken::new(),
                None,
            ),
        )
        .await
        .expect("provider selection regression must finish")
    }

    struct FailingInventorySource;

    #[async_trait::async_trait]
    impl ToolInventorySource for FailingInventorySource {
        async fn snapshot(
            &self,
            _: &InventoryRequest,
            _: &ToolContext,
        ) -> Result<Arc<ToolInventorySnapshot>, InventoryError> {
            Err(InventoryError::Unavailable(
                "source fixture unavailable".to_string(),
            ))
        }

        fn current_generation(
            &self,
            _: &InventoryRequest,
            _: &ToolContext,
        ) -> Result<InventoryGeneration, InventoryError> {
            Ok(1)
        }
    }

    #[tokio::test]
    async fn source_inventory_failure_stops_before_provider_send() {
        let (provider, _, _) = precedence_provider("source-failure");
        let accepted = Arc::clone(&provider.accepted_requests);
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let mut ctx = deny_all_context();
        ctx.config.provider = Some("source-failure".to_string());
        let mut config = make_config(None, None);
        config.model = "source-failure-model".to_string();
        config.inventory_source = Some(Arc::new(FailingInventorySource));

        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;
        assert!(matches!(outcome, QueryOutcome::Error(_)));
        assert_eq!(*accepted.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn verified_effort_query_requires_receipt_before_lower_wire() {
        let root = tempfile::tempdir().expect("query fixture root");
        std::fs::write(root.path().join("target"), b"before").expect("target");
        let command = "test \"$(cat target)\" = after; printf x >> checker-runs";

        let mut ctx = deny_all_context();
        ctx.working_dir = root.path().to_path_buf();
        ctx.session_id = "verified-query-wire".into();
        ctx.config.provider = Some("mockprov".into());
        ctx.permission_handler = Arc::new(AllowAllHandler);
        let invocation = claurst_tools::TrustedCheckerInvocation::foreground(
            claurst_tools::TrustedCheckerShell::Bash,
            command,
            root.path(),
            std::time::Duration::from_secs(5),
        )
        .expect("checker invocation");
        let registration =
            claurst_tools::TrustedCheckerRegistration::new(claurst_tools::RegistrationSpec {
                id: "verified-query-check".into(),
                revision: 1,
                source: claurst_tools::TrustedCheckerSource::Application,
                workspace: root.path().to_path_buf(),
                session_id: ctx.session_id.clone(),
                target: root.path().join("target"),
                operation: claurst_tools::TrustedCheckerOperation::Write,
                invocation,
            })
            .expect("checker registration");
        let mut catalog = claurst_tools::TrustedCheckerCatalogBuilder::new();
        catalog.register(registration).expect("unique checker");
        ctx.trusted_checker_catalog = Arc::new(catalog.build());

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(claurst_tools::EditCheckTool),
            Box::new(CountingTool {
                runs: Arc::new(AtomicUsize::new(0)),
            }),
        ];
        let thinking = Arc::new(StdMutex::new(Vec::new()));
        let dispatch = Arc::new(StdMutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider {
            id: claurst_core::provider_id::ProviderId::new("mockprov"),
            tools_empty_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_counts_per_request: Arc::new(StdMutex::new(Vec::new())),
            models_per_request: Arc::new(StdMutex::new(Vec::new())),
            wire_context_limits_per_request: Arc::new(StdMutex::new(Vec::new())),
            thinking_budgets_per_request: thinking.clone(),
            dispatch_per_request: dispatch.clone(),
            always_end_turn: false,
            budget_failures_left: Arc::new(AtomicUsize::new(0)),
            accepted_requests: Arc::new(StdMutex::new(0)),
            message_ids_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_uses_left: Arc::new(AtomicUsize::new(0)),
            scripted_verified_flow: true,
            deferred_discovery_flow: false,
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            ..Default::default()
        })
        .expect("test client");
        let mut config = make_config(None, None);
        config.model = "mock-model".into();
        config.max_turns = 4;
        config.provider_registry = Some(Arc::new(registry));
        config.effort_policy = effort_policy::EffortPolicy::verified(
            claurst_core::effort::EffortLevel::High,
            claurst_core::effort::EffortLevel::Medium,
        )
        .expect("verified ladder");
        config.context_budget.wire_byte_limit = 1_000_000;

        let mut messages = vec![Message::user("perform the verified edit")];
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(10),
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
        .expect("verified query fixture must finish");

        assert!(
            matches!(outcome, QueryOutcome::EndTurn { .. }),
            "{outcome:?}"
        );
        #[cfg(unix)]
        {
            assert_eq!(
                *thinking.lock().unwrap(),
                vec![Some(10_000), Some(5_000), Some(10_000)]
            );
            assert_eq!(*dispatch.lock().unwrap(), vec![false, true, false]);
        }
        #[cfg(not(unix))]
        {
            assert_eq!(*thinking.lock().unwrap(), vec![Some(10_000); 3]);
            assert_eq!(*dispatch.lock().unwrap(), vec![false; 3]);
        }
        assert_eq!(std::fs::read(root.path().join("target")).unwrap(), b"after");
        #[cfg(unix)]
        assert_eq!(
            std::fs::read(root.path().join("checker-runs")).unwrap(),
            b"x"
        );
        #[cfg(not(unix))]
        assert!(!root.path().join("checker-runs").exists());
    }

    #[tokio::test]
    async fn provider_selection_explicit_provider_wins_over_model_prefix() {
        let (selected, selected_models, _) = precedence_provider("mockprov");
        let (shadowed, shadowed_models, _) = precedence_provider("openrouter");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(selected);
        registry.register(shadowed);

        let mut ctx = deny_all_context();
        ctx.config.provider = Some("mockprov".to_string());
        let mut config = make_config(None, None);
        config.model = "openrouter/pinned-model".to_string();
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let selected_models = selected_models.lock().unwrap();
        assert_eq!(selected_models.len(), 1);
        assert_eq!(selected_models[0], "openrouter/pinned-model");
        assert!(shadowed_models.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn provider_selection_normalizes_known_provider_model() {
        let (provider, models, _) = precedence_provider("openrouter");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let ctx = deny_all_context();
        let mut config = make_config(None, None);
        config.model = "openrouter/pinned-model".to_string();
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let models = models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "pinned-model");
    }

    #[tokio::test]
    async fn provider_selection_preserves_namespaced_model_under_explicit_provider() {
        let (provider, models, _) = precedence_provider("openrouter");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let mut ctx = deny_all_context();
        ctx.config.provider = Some("openrouter".to_string());
        let mut config = make_config(None, None);
        config.model = "meta-llama/Llama-3.3-70B-Instruct".to_string();
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let models = models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "meta-llama/Llama-3.3-70B-Instruct");
    }

    #[tokio::test]
    async fn provider_selection_uses_model_registry_fallback_for_bare_model() {
        let (provider, models, wire_context_limits) = precedence_provider("google");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let ctx = deny_all_context();
        let mut config = make_config(None, None);
        config.model = "gemini-3-flash-preview".to_string();
        let model_registry = Arc::new(claurst_api::ModelRegistry::new());
        let expected_context_limit = u64::from(
            model_registry
                .get("google", "gemini-3-flash-preview")
                .expect("the bundled Google catalog entry must exist")
                .info
                .context_window,
        );
        assert!(expected_context_limit > config.context_budget.unknown_fallback_tokens);
        config.model_registry = Some(model_registry);
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let models = models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "gemini-3-flash-preview");
        let wire_context_limits = wire_context_limits.lock().unwrap();
        assert_eq!(wire_context_limits.len(), 1);
        assert_eq!(wire_context_limits[0], expected_context_limit);
    }

    #[tokio::test]
    async fn provider_selection_agent_model_overrides_base_model() {
        let (provider, models, _) = precedence_provider("openrouter");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let ctx = deny_all_context();
        let mut config = make_config(None, None);
        config.model = "openrouter/base-model".to_string();
        config.agent_definition = Some(claurst_core::AgentDefinition {
            model: Some("openrouter/agent-model".to_string()),
            ..Default::default()
        });
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let models = models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "agent-model");
    }

    #[tokio::test]
    async fn provider_selection_managed_manager_model_overrides_base_model() {
        let (provider, models, _) = precedence_provider("openrouter");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let ctx = deny_all_context();
        let mut config = make_config(None, None);
        config.model = "openrouter/base-model".to_string();
        config.managed_agents = Some(claurst_core::ManagedAgentConfig {
            enabled: true,
            manager_model: "openrouter/manager-model".to_string(),
            executor_model: "openrouter/executor-model".to_string(),
            executor_max_turns: 10,
            max_concurrent_executors: 4,
            budget_split: claurst_core::BudgetSplitPolicy::SharedPool,
            total_budget_usd: None,
            preset_name: None,
            executor_isolation: false,
        });
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let models = models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "manager-model");
    }

    #[tokio::test]
    async fn provider_selection_keeps_explicit_canonical_model_pin() {
        let (provider, models, _) = precedence_provider("openrouter");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let ctx = deny_all_context();
        let mut config = make_config(None, None);
        config.model = "openrouter/my-explicit-model".to_string();
        config.model_registry = Some(Arc::new(claurst_api::ModelRegistry::new()));
        let outcome = run_provider_selection_case(config, ctx, Arc::new(registry)).await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let models = models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "my-explicit-model");
    }

    async fn run_inline_queue_case(command: QueuedCommand) -> (Vec<Message>, EffortSelection) {
        let (provider, _, _) = precedence_provider("mockprov");
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let queue = CommandQueue::new();
        queue.push(command, CommandPriority::High);
        let mut ctx = deny_all_context();
        ctx.config.provider = Some("mockprov".into());
        let mut config = make_config(None, None);
        config.model = "mock-model".into();
        config.max_turns = 1;
        config.provider_registry = Some(Arc::new(registry));
        config.command_queue = Some(queue);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let mut messages = vec![Message::user("please use ultracode for this work")];
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            ..Default::default()
        })
        .expect("build test client");

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_query_loop(
                &client,
                &mut messages,
                &[],
                &ctx,
                &config,
                claurst_core::cost::CostTracker::new(),
                Some(event_tx),
                tokio_util::sync::CancellationToken::new(),
                None,
            ),
        )
        .await
        .expect("queued query must finish");
        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));

        let selection = std::iter::from_fn(|| event_rx.try_recv().ok())
            .find_map(|event| match event {
                QueryEvent::EffortSelection(selection) => Some(selection),
                _ => None,
            })
            .expect("query must emit effort selection");
        (messages, selection)
    }

    #[tokio::test]
    async fn genuine_queue_work_resets_inline_pin_but_system_message_preserves_it() {
        let (system_messages, system_selection) =
            run_inline_queue_case(QueuedCommand::InjectSystemMessage("host status".into())).await;
        assert!(system_messages
            .iter()
            .any(|message| message.get_all_text().contains("[System]: host status")));
        assert_eq!(
            system_selection.selected(),
            Some(claurst_core::effort::EffortLevel::Ultracode)
        );

        let (user_messages, user_selection) =
            run_inline_queue_case(QueuedCommand::InjectUserMessage("new genuine work".into()))
                .await;
        assert!(user_messages
            .iter()
            .any(|message| message.get_all_text() == "new genuine work"));
        assert_eq!(user_selection.selected(), None);
    }

    #[tokio::test]
    async fn cancellation_during_initial_stream_creation_runs_no_tools() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(GatedStreamProvider {
            id: claurst_core::provider_id::ProviderId::new("gated"),
            entered: entered.clone(),
            release,
            completed: completed.clone(),
            calls: calls.clone(),
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        })
        .expect("build test client");
        let mut ctx = deny_all_context();
        ctx.session_id = "stream-cancel-test".to_string();
        ctx.config.provider = Some("gated".to_string());
        let mut config = make_config(None, None);
        config.model = "gated-model".to_string();
        config.max_turns = 1;
        config.provider_registry = Some(Arc::new(registry));
        config.context_budget.wire_byte_limit = 1_000_000;
        let tool_runs = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool {
            runs: tool_runs.clone(),
        })];
        let cancel = tokio_util::sync::CancellationToken::new();
        let cancel_after_entry = cancel.clone();
        let cancel_waiter = tokio::spawn(async move {
            entered.notified().await;
            cancel_after_entry.cancel();
        });
        let mut messages = vec![Message::user("cancel while provider starts")];
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(2),
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
        .expect("cancellation must join while stream creation is pending");
        cancel_waiter.await.expect("cancellation waiter");
        assert!(matches!(outcome, QueryOutcome::Cancelled));
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
        assert!(!completed.load(AtomicOrdering::SeqCst));
        assert_eq!(tool_runs.load(AtomicOrdering::SeqCst), 0);
    }

    /// (a) A non-goal turn that ends with `end_turn` stops after exactly one
    /// turn — the default `StopPolicy` never continues the loop.
    #[tokio::test]
    async fn non_goal_turn_stops_after_one_turn() {
        let (outcome, recorded, _msgs) = drive_loop_with_mock(
            true,
            5,
            noop_tools(),
            crate::continuation::ContinuationMode::Default,
        )
        .await;

        assert!(
            matches!(outcome, QueryOutcome::EndTurn { .. }),
            "a completed turn must yield EndTurn"
        );
        assert_eq!(
            recorded.len(),
            1,
            "a non-goal end_turn must stop after exactly one request/turn, got {:?}",
            recorded
        );
    }

    #[tokio::test]
    async fn deferred_inventory_stays_out_of_initial_provider_request() {
        let tool_counts = Arc::new(StdMutex::new(Vec::new()));
        let deferred_runs = Arc::new(AtomicUsize::new(0));
        let schema_calls = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(RecordingProvider {
            id: claurst_core::provider_id::ProviderId::new("deferred-inventory-mock"),
            tools_empty_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_counts_per_request: tool_counts.clone(),
            models_per_request: Arc::new(StdMutex::new(Vec::new())),
            wire_context_limits_per_request: Arc::new(StdMutex::new(Vec::new())),
            thinking_budgets_per_request: Arc::new(StdMutex::new(Vec::new())),
            dispatch_per_request: Arc::new(StdMutex::new(Vec::new())),
            always_end_turn: true,
            budget_failures_left: Arc::new(AtomicUsize::new(0)),
            accepted_requests: Arc::new(StdMutex::new(0)),
            message_ids_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_uses_left: Arc::new(AtomicUsize::new(0)),
            scripted_verified_flow: false,
            deferred_discovery_flow: true,
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);

        let mut tools: Vec<Box<dyn Tool>> = vec![Box::new(claurst_tools::ToolSearchTool)];
        tools.extend((0..1_000).map(|index| {
            Box::new(DeferredInventoryTool {
                name: format!("DeferredInventory{index}"),
                runs: deferred_runs.clone(),
                schema_calls: schema_calls.clone(),
            }) as Box<dyn Tool>
        }));

        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        })
        .expect("build test client");
        let mut ctx = deny_all_context();
        ctx.session_id = "deferred-inventory-query".to_string();
        ctx.permission_handler = Arc::new(AllowAllHandler);
        ctx.config.provider = Some("deferred-inventory-mock".to_string());

        let mut config = make_config(None, None);
        config.model = "deferred-inventory-model".to_string();
        config.max_turns = 3;
        config.context_budget.wire_byte_limit = 1_000_000;
        config.provider_registry = Some(Arc::new(registry));
        let mut messages = vec![Message::user("discover only when requested")];
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(10),
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
        .expect("deferred inventory query must finish");

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        let counts = tool_counts.lock().unwrap();
        assert_eq!(&*counts, &[1, 2, 2]);
        assert_eq!(deferred_runs.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(schema_calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn provider_local_wire_budget_reselects_one_complete_group_before_inference() {
        let attempted = Arc::new(StdMutex::new(Vec::new()));
        let accepted = Arc::new(StdMutex::new(0usize));
        let ids = Arc::new(StdMutex::new(Vec::<Vec<String>>::new()));
        let provider = Arc::new(RecordingProvider {
            id: claurst_core::provider_id::ProviderId::new("mockprov"),
            tools_empty_per_request: attempted.clone(),
            tool_counts_per_request: Arc::new(StdMutex::new(Vec::new())),
            models_per_request: Arc::new(StdMutex::new(Vec::new())),
            wire_context_limits_per_request: Arc::new(StdMutex::new(Vec::new())),
            thinking_budgets_per_request: Arc::new(StdMutex::new(Vec::new())),
            dispatch_per_request: Arc::new(StdMutex::new(Vec::new())),
            always_end_turn: true,
            budget_failures_left: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
            accepted_requests: accepted.clone(),
            message_ids_per_request: ids.clone(),
            tool_uses_left: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            scripted_verified_flow: false,
            deferred_discovery_flow: false,
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            ..Default::default()
        })
        .unwrap();
        let mut ctx = deny_all_context();
        ctx.session_id = "wire-reselect-test".into();
        ctx.config.provider = Some("mockprov".into());
        let mut config = make_config(None, None);
        config.model = "mock-model".into();
        config.provider_registry = Some(Arc::new(registry));
        config.context_budget.wire_byte_limit = 1_000_000;
        let old = Message::user("old complete request");
        let current = Message::user("current request");
        let mut messages = vec![old, Message::assistant("old complete response"), current];
        let tools = noop_tools();
        let outcome = run_query_loop(
            &client,
            &mut messages,
            &tools,
            &ctx,
            &config,
            claurst_core::cost::CostTracker::new(),
            None,
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        assert_eq!(*accepted.lock().unwrap(), 1);
        let requests = ids.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].len(), 3);
        assert_eq!(requests[1].len(), 1);
        assert_eq!(attempted.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn provider_wire_reselection_runs_tools_only_after_acceptance() {
        let attempted = Arc::new(StdMutex::new(Vec::new()));
        let accepted = Arc::new(StdMutex::new(0usize));
        let runs = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(RecordingProvider {
            id: claurst_core::provider_id::ProviderId::new("mockprov"),
            tools_empty_per_request: attempted.clone(),
            tool_counts_per_request: Arc::new(StdMutex::new(Vec::new())),
            models_per_request: Arc::new(StdMutex::new(Vec::new())),
            wire_context_limits_per_request: Arc::new(StdMutex::new(Vec::new())),
            thinking_budgets_per_request: Arc::new(StdMutex::new(Vec::new())),
            dispatch_per_request: Arc::new(StdMutex::new(Vec::new())),
            always_end_turn: false,
            budget_failures_left: Arc::new(AtomicUsize::new(1)),
            accepted_requests: accepted.clone(),
            message_ids_per_request: Arc::new(StdMutex::new(Vec::new())),
            tool_uses_left: Arc::new(AtomicUsize::new(1)),
            scripted_verified_flow: false,
            deferred_discovery_flow: false,
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider);
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            ..Default::default()
        })
        .unwrap();
        let mut ctx = deny_all_context();
        ctx.session_id = "wire-side-effect-test".into();
        ctx.config.provider = Some("mockprov".into());
        let mut config = make_config(None, None);
        config.model = "mock-model".into();
        config.max_turns = 1;
        config.provider_registry = Some(Arc::new(registry));
        config.context_budget.wire_byte_limit = 1_000_000;
        let mut messages = vec![
            Message::user("old complete request"),
            Message::assistant("old complete response"),
            Message::user("current request"),
        ];
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool { runs: runs.clone() })];

        let outcome = run_query_loop(
            &client,
            &mut messages,
            &tools,
            &ctx,
            &config,
            claurst_core::cost::CostTracker::new(),
            None,
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        assert!(matches!(outcome, QueryOutcome::EndTurn { .. }));
        assert_eq!(runs.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(*accepted.lock().unwrap(), 2);
        assert_eq!(attempted.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn legacy_anthropic_budget_rejection_is_local_and_does_not_open_transport() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let accepted_server = accepted.clone();
        let server = tokio::spawn(async move {
            let _ = tokio::time::timeout(std::time::Duration::from_millis(100), async move {
                if listener.accept().await.is_ok() {
                    accepted_server.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            })
            .await;
        });

        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "test-key".into(),
            api_base: format!("http://{}", address),
            ..Default::default()
        })
        .unwrap();
        let wire_budget = claurst_api::WireBudget::finite(
            32_000,
            1,
            1,
            1_000_000,
            claurst_api::MediaAccounting::None,
        )
        .unwrap();
        let request = claurst_api::CreateMessageRequest::builder("mock-model", 1)
            .messages(vec![claurst_api::ApiMessage::from(&Message::user(
                "legacy local preflight canary",
            ))])
            .wire_budget(wire_budget)
            .build();
        let result = client
            .create_message_stream(request, Arc::new(claurst_api::streaming::NullStreamHandler))
            .await;
        assert!(matches!(
            result,
            Err(ClaudeError::RequestBudgetExceeded { .. })
        ));
        server.await.unwrap();
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// (c) Hitting `effective_max_turns` runs ONE final turn with tools disabled
    /// (graceful degradation) rather than returning cold: the last request has
    /// an empty tool set and the loop then ends.
    #[tokio::test]
    async fn max_steps_runs_tool_less_summary_turn_then_ends() {
        // max_turns = 2: turns 1 & 2 are tool_use turns, turn 3 exceeds the cap
        // and triggers the tool-less summary turn.
        let (outcome, recorded, msgs) = drive_loop_with_mock(
            false,
            2,
            noop_tools(),
            crate::continuation::ContinuationMode::Default,
        )
        .await;

        assert!(
            matches!(outcome, QueryOutcome::EndTurn { .. }),
            "the loop must end after the degradation summary turn"
        );
        assert_eq!(
            recorded.len(),
            3,
            "expected 2 tool turns + 1 degradation turn, got {:?}",
            recorded
        );
        assert!(
            *recorded.last().unwrap(),
            "the final (summary) turn must be dispatched with tools DISABLED: {:?}",
            recorded
        );
        assert!(
            recorded[..recorded.len() - 1].iter().all(|&empty| !empty),
            "only the degradation turn disables tools: {:?}",
            recorded
        );
        assert!(
            msgs.iter()
                .any(|m| m.get_all_text().contains("maximum number of steps")),
            "the tool-less summary prompt must be injected into the history"
        );
    }

    /// (b) The goal continuation guards, exercised against an in-memory store:
    /// an active goal within its guards continues (recording the turn), while
    /// the soft-budget and runaway guards each stop with the same paused
    /// outcome as before.
    #[test]
    fn goal_policy_continues_while_active_and_stops_on_guards() {
        use crate::goal_loop::{decide_goal_continuation, GoalContinuation, StopReason};

        let store =
            claurst_core::GoalStore::open(std::path::Path::new(":memory:")).expect("open store");

        // Active goal, guards allow → continue with the goal continuation message.
        store.set_goal("live", "ship the feature", None).unwrap();
        match decide_goal_continuation(&store, "live", 0, 1) {
            GoalContinuation::Continue { message } => {
                assert!(
                    message.contains("Goal continuation"),
                    "unexpected continuation message: {}",
                    message
                );
            }
            _ => panic!("an active goal within its guards must continue"),
        }
        // The turn was recorded in the store.
        assert_eq!(store.get_goal("live").unwrap().turns_used, 1);

        // Soft token budget tripped → budget-limited (paused) outcome.
        store.set_goal("budget", "big task", Some(100)).unwrap();
        match decide_goal_continuation(&store, "budget", 500, 1) {
            GoalContinuation::Stop {
                reason: StopReason::BudgetLimited,
            } => {}
            _ => panic!("an over-budget goal must stop budget-limited"),
        }
        assert_eq!(
            store.get_goal("budget").unwrap().status,
            claurst_core::GoalStatus::BudgetLimited,
            "over-budget goal must be persisted as budget-limited"
        );

        // Runaway guard tripped → paused outcome (same as the cross-turn design).
        store.set_goal("runaway", "endless", None).unwrap();
        for _ in 0..claurst_core::MAX_GOAL_TURNS {
            store.record_turn("runaway", 0).unwrap();
        }
        match decide_goal_continuation(&store, "runaway", 0, 1) {
            GoalContinuation::Stop {
                reason: StopReason::RunawayGuard { turns_used },
            } => {
                assert_eq!(turns_used, claurst_core::MAX_GOAL_TURNS);
            }
            _ => panic!("a runaway goal must pause"),
        }
        assert_eq!(
            store.get_goal("runaway").unwrap().status,
            claurst_core::GoalStatus::Paused,
            "runaway goal must be persisted as paused"
        );
    }

    // ---- ultracode activation (effort) ----------------------------------

    #[test]
    fn ultracode_keyword_raises_effort_to_ultracode() {
        use claurst_core::effort::EffortLevel;
        let msgs = vec![Message::user("please ultracode this refactor")];
        // Even with no configured effort, the keyword forces Ultracode.
        assert_eq!(
            effective_effort_for_turn(None, &msgs),
            Some(EffortLevel::Ultracode)
        );
        // ...and it overrides a lower configured effort for the turn.
        assert_eq!(
            effective_effort_for_turn(Some(EffortLevel::Low), &msgs),
            Some(EffortLevel::Ultracode)
        );
    }

    #[test]
    fn no_keyword_keeps_configured_effort() {
        use claurst_core::effort::EffortLevel;
        let msgs = vec![Message::user("please refactor this module")];
        assert_eq!(effective_effort_for_turn(None, &msgs), None);
        assert_eq!(
            effective_effort_for_turn(Some(EffortLevel::High), &msgs),
            Some(EffortLevel::High)
        );
    }

    #[test]
    fn ultracode_effort_checks_only_the_last_user_message() {
        // Keyword in an earlier turn does not keep ultracode active on a later
        // plain turn.
        let msgs = vec![
            Message::user("ultracode kick things off"),
            Message::assistant("working on it"),
            Message::user("now just tidy up the docs"),
        ];
        assert_eq!(effective_effort_for_turn(None, &msgs), None);
    }

    #[test]
    fn ultracode_addendum_flows_into_built_system_prompt() {
        use claurst_core::effort::EffortLevel;
        // Mirrors the loop wiring: when the effective effort is Ultracode the
        // procedure addendum is threaded through `append_system_prompt` into the
        // assembled system prompt.
        let msgs = vec![Message::user("ultracode audit the query loop")];
        assert_eq!(
            effective_effort_for_turn(None, &msgs),
            Some(EffortLevel::Ultracode)
        );
        let addendum = claurst_core::effort::ultracode_system_prompt_addendum();
        let opts = claurst_core::system_prompt::SystemPromptOptions {
            append_system_prompt: Some(addendum),
            skip_env_info: true,
            ..Default::default()
        };
        let prompt = claurst_core::system_prompt::build_system_prompt(&opts);
        assert!(prompt.contains("Ultracode Mode"));
        assert!(prompt.contains("TeamCreate"));

        // Absent path: no keyword -> configured effort stays, no ultracode text.
        assert_eq!(
            effective_effort_for_turn(None, &[Message::user("hi there")]),
            None
        );
        let plain = claurst_core::system_prompt::build_system_prompt(
            &claurst_core::system_prompt::SystemPromptOptions {
                skip_env_info: true,
                ..Default::default()
            },
        );
        assert!(!plain.contains("Ultracode Mode"));
    }

    #[test]
    fn query_config_captures_raw_model_settings_provenance() {
        let mut cfg = Config::default();
        let default_query = QueryConfig::from_config(&cfg);
        assert_eq!(
            default_query.pin_provenance.model_source().source(),
            effort_policy::ValueSource::Default
        );
        assert!(!default_query.pin_provenance.model_source().raw_present());

        cfg.model = Some("openrouter/pinned-model".to_string());
        let configured_query = QueryConfig::from_config(&cfg);
        assert_eq!(
            configured_query.pin_provenance.model_source().source(),
            effort_policy::ValueSource::Settings
        );
        assert!(configured_query.pin_provenance.model_source().raw_present());
    }

    #[test]
    fn turn_provenance_overlays_agent_temperature_and_inline_ultracode() {
        let config = QueryConfig {
            agent_definition: Some(claurst_core::AgentDefinition {
                temperature: Some(0.2),
                ..Default::default()
            }),
            ..Default::default()
        };
        let provenance = effective_pin_provenance(
            &config,
            Some(claurst_core::effort::EffortLevel::Ultracode),
            Some(20_000),
            Some(0.2),
            &[Message::user("ultracode this request")],
        );
        assert_eq!(
            provenance.temperature().source(),
            effort_policy::ValueSource::Agent
        );
        assert_eq!(
            provenance.effort().source(),
            effort_policy::ValueSource::InlineKeyword
        );
    }

    #[test]
    fn effort_derived_wire_values_keep_default_and_verified_sources() {
        let config = QueryConfig::default();
        let default_provenance = effective_pin_provenance(
            &config,
            Some(claurst_core::effort::EffortLevel::Medium),
            Some(5_000),
            Some(0.3),
            &[Message::user("ordinary request")],
        );
        assert_eq!(default_provenance.blocking_pin(), None);

        let mut verified_config = QueryConfig::default();
        verified_config.pin_provenance =
            verified_config
                .pin_provenance
                .with_effort(effort_policy::ValueProvenance::new(
                    effort_policy::ValueSource::VerifiedBase,
                    true,
                ));
        let verified_provenance = effective_pin_provenance(
            &verified_config,
            Some(claurst_core::effort::EffortLevel::High),
            Some(10_000),
            Some(0.2),
            &[Message::user("ordinary request")],
        );
        assert_eq!(verified_provenance.blocking_pin(), None);
    }

    #[test]
    fn raw_unmarked_wire_overrides_pin_conservatively() {
        let config = QueryConfig {
            thinking_budget: Some(4_000),
            ..Default::default()
        };
        let provenance = effective_pin_provenance(
            &config,
            None,
            Some(4_000),
            None,
            &[Message::user("ordinary request")],
        );
        assert_eq!(
            provenance.blocking_pin(),
            Some(effort_policy::PinReason::UnknownWireOverride)
        );
    }

    // ---- persona output-style (transient vs persistent) ------------------

    #[test]
    fn inline_persona_keyword_applies_transiently_for_the_turn() {
        // No persisted persona; an inline `rocky` selects the rocky prompt for
        // this turn only.
        let cfg = QueryConfig::default();
        let msgs = vec![Message::user("please rocky explain this borrow error")];
        let (_style, prompt) = effective_output_style_for_turn(&cfg, &msgs);
        let prompt = prompt.expect("inline rocky should resolve a persona prompt");
        assert!(prompt.contains("Project Hail Mary"));

        // Caveman likewise.
        let msgs = vec![Message::user("caveman summarize the diff")];
        let (_s, prompt) = effective_output_style_for_turn(&cfg, &msgs);
        assert!(prompt.unwrap().contains("UNCHANGED"));
    }

    #[test]
    fn persona_only_checks_the_last_user_message() {
        // A persona keyword in an earlier turn does not linger onto a later
        // plain turn (transient, like ultracode).
        let cfg = QueryConfig::default();
        let msgs = vec![
            Message::user("rocky kick things off"),
            Message::assistant("good good good"),
            Message::user("now just tidy the docs"),
        ];
        let (_style, prompt) = effective_output_style_for_turn(&cfg, &msgs);
        assert!(
            prompt.is_none(),
            "persona should not persist to a plain turn"
        );
    }

    #[test]
    fn persisted_persona_stands_without_an_inline_keyword() {
        // A persona chosen via /rocky or /output-style lives in the config and
        // persists across plain turns.
        let cfg = QueryConfig {
            output_style_prompt: Some("PERSISTED PERSONA".to_string()),
            ..QueryConfig::default()
        };
        let msgs = vec![Message::user("just a plain request here please")];
        let (_style, prompt) = effective_output_style_for_turn(&cfg, &msgs);
        assert_eq!(prompt.as_deref(), Some("PERSISTED PERSONA"));
    }

    #[test]
    fn inline_normal_resets_a_persisted_persona_for_the_turn() {
        // With a persona persisted, an inline `normal` clears it for this turn.
        let cfg = QueryConfig {
            output_style_prompt: Some("PERSISTED PERSONA".to_string()),
            ..QueryConfig::default()
        };
        let msgs = vec![Message::user("back to normal for this one please")];
        let (style, prompt) = effective_output_style_for_turn(&cfg, &msgs);
        assert!(prompt.is_none(), "inline normal should reset the persona");
        assert_eq!(style, claurst_core::system_prompt::OutputStyle::Default);
    }

    #[test]
    fn inline_persona_overrides_a_different_persisted_persona() {
        // Persisted caveman, but this turn asks for rocky inline → rocky wins
        // transiently.
        let cfg = QueryConfig {
            output_style_prompt: Some(
                claurst_core::output_styles::OutputStyleDef::builtin_caveman().prompt,
            ),
            ..QueryConfig::default()
        };
        let msgs = vec![Message::user("rocky, review this function")];
        let (_style, prompt) = effective_output_style_for_turn(&cfg, &msgs);
        assert!(prompt.unwrap().contains("Project Hail Mary"));
    }

    #[test]
    fn history_manifest_has_bounded_reference_count_and_bytes() {
        let reference = |index: usize| context_history::HistoryRef {
            id: format!("href-{index:032x}"),
            group_id: format!("group-{index}"),
            sequence: index as u64,
            source_range: context_history::SourceRange {
                message_id: format!("message-{index}"),
                revision: index as u64,
            },
            source_message_ids: vec![format!("message-{index}")],
            reason: context_history::OmitReason::Budget,
            authority: context_history::HistoryAuthority::Historical,
        };
        let too_many = (0..=MAX_HISTORY_MANIFEST_REFS)
            .map(reference)
            .collect::<Vec<_>>();
        let manifest =
            append_history_manifest(SystemPrompt::Text("system".into()), &too_many).unwrap();
        let SystemPrompt::Text(manifest) = manifest else {
            panic!("test manifest should be text");
        };
        assert!(manifest.len() <= MAX_HISTORY_MANIFEST_BYTES + 256);
        assert!(manifest.contains("remaining_count"));
        assert!(manifest.contains("index_search"));

        let oversized = vec![reference(0); 1];
        let oversized = oversized
            .into_iter()
            .map(|mut reference| {
                reference.source_message_ids = vec!["x".repeat(MAX_HISTORY_MANIFEST_BYTES)];
                reference
            })
            .collect::<Vec<_>>();
        let manifest =
            append_history_manifest(SystemPrompt::Text("system".into()), &oversized).unwrap();
        let SystemPrompt::Text(manifest) = manifest else {
            panic!("test manifest should be text");
        };
        assert!(manifest.contains("remaining_count"));
    }

    #[test]
    fn reselection_never_targets_a_group_already_omitted_by_the_plan() {
        let plan = context_plan::ContextPlan {
            messages: Vec::new(),
            selected_groups: vec!["selected".into()],
            optional_selected_groups: vec!["selected".into()],
            omitted: vec![context_plan::OmittedComponent {
                group_id: "already-omitted".into(),
                message_ids: Vec::new(),
                reason: context_plan::OmissionReason::Budget,
            }],
            cost: context_plan::CandidateCost {
                input_tokens: 0,
                wire_bytes: 0,
            },
            budget: RequestBudget::resolve(Some(10), None, None, None, 1, 10).unwrap(),
            measurements: 1,
        };
        let excluded = std::collections::BTreeSet::new();
        assert_eq!(
            next_reselection_group(&plan, &excluded).as_deref(),
            Some("selected")
        );
    }

    #[test]
    fn tool_roundtrip_messages_have_ids_before_context_selection() {
        let mut messages = vec![Message::user("current request")];
        messages[0].uuid = Some("request".into());

        let mut assistant = Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "tool-call".into(),
            name: "noop_tool".into(),
            input: serde_json::json!({}),
            thought_signature: None,
        }]);
        ensure_message_id(&mut assistant);
        messages.push(assistant);

        push_message_with_id(
            &mut messages,
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "tool-call".into(),
                content: claurst_core::types::ToolResultContent::Text("done".into()),
                is_error: None,
            }]),
        );
        let ids: Vec<String> = messages
            .iter()
            .map(|message| message.uuid.clone().unwrap())
            .collect();
        assert!(ids.iter().all(|id| !id.is_empty()));

        let budget = RequestBudget::resolve(Some(1_000), None, None, None, 100, 100_000).unwrap();
        let plan =
            context_plan::ContextAssembler::plan(&messages, "request", budget, |candidate, _| {
                Ok(context_plan::CandidateCost {
                    input_tokens: candidate.len() as u64,
                    wire_bytes: serde_json::to_vec(candidate).unwrap().len() as u64,
                })
            })
            .unwrap();
        assert_eq!(plan.messages.len(), 3);
        assert_eq!(
            ids,
            plan.messages
                .iter()
                .map(|message| message.uuid.clone().unwrap())
                .collect::<Vec<_>>()
        );

        let mut repeated_provider_response = Message::assistant("next response");
        repeated_provider_response.uuid = Some("request".into());
        ensure_unique_message_id(&messages, &mut repeated_provider_response);
        assert_ne!(repeated_provider_response.uuid.as_deref(), Some("request"));
        assert!(repeated_provider_response.uuid.is_some());
    }
    #[cfg(unix)]
    mod resolver_binding_regressions {
        include!("query_resolver_bind_tests.rs");
    }
    include!("session_memory_query_tests.rs");
}
