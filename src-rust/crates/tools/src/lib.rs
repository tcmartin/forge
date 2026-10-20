// claurst-tools: All tool implementations for Claurst.
//
// Each tool maps to a capability the LLM can invoke: running shell commands,
// reading/writing/editing files, searching codebases, fetching web pages, etc.

// type_complexity: the REPL tool holds a boxed session-callback map whose full
// type is unwieldy; a type alias would not meaningfully improve readability.
#![allow(clippy::type_complexity)]

use async_trait::async_trait;
use claurst_core::config::PermissionMode;
use claurst_core::cost::CostTracker;
use claurst_core::media::{MediaError, MediaLimits, MediaManifest};
use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
use claurst_core::types::{ContentBlock, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Sub-modules – each contains a full tool implementation.
pub mod agent_tasks;
pub mod apply_patch;
pub mod ask_user;
mod background_processes;
pub mod batch_edit;
pub mod brief;
pub mod bundled_skills;
mod command_capture;
pub mod computer_use;
pub mod config_tool;
pub mod cron;
pub mod deletion_intent;
pub mod enter_plan_mode;
pub mod evidence;
pub mod execution;
pub mod exit_plan_mode;
pub mod file_edit;
pub mod file_read;
pub mod file_write;
pub mod formatter;
pub mod glob_tool;
pub mod goal_complete;
pub mod grep_tool;
pub mod line_endings;
pub mod lsp_tool;
pub mod mcp_auth_tool;
pub mod mcp_content;
pub mod mcp_resources;
pub mod monitor_tool;
pub mod mutation;
pub mod notebook_edit;
pub mod powershell;
pub mod pty_bash;
pub mod remote_trigger;
pub mod repl_tool;
pub mod send_message;
pub mod skill_tool;
pub mod sleep;
pub mod synthetic_output;
pub mod tasks;
pub mod team_tool;
#[cfg(test)]
pub(crate) mod test_support;
pub mod todo_write;
pub mod tool_catalog;
pub mod tool_policy;
pub mod tool_search;
pub mod trusted_checker;
pub(crate) mod trusted_checker_execution;
#[cfg(all(test, unix))]
mod trusted_dispatch_tests;
pub mod verification_receipts;
pub mod web_fetch;
pub mod web_search;
pub mod worktree;

// Re-exports for convenience.
pub use claurst_mcp::{
    with_mcp_query_operation_identity, with_mcp_query_operation_observer, McpOperationBackend,
    McpOperationContext, McpOperationObserver, McpQueryOperationIdentity,
};

pub use apply_patch::ApplyPatchTool;
pub use ask_user::AskUserQuestionTool;
pub use batch_edit::BatchEditTool;
pub use brief::BriefTool;
pub use computer_use::ComputerUseTool;
pub use config_tool::ConfigTool;
pub use cron::{CronCreateTool, CronDeleteTool, CronListTool};
pub use deletion_intent::{with_deletion_authorization, DeletionAuthorization};
pub use enter_plan_mode::EnterPlanModeTool;
pub use evidence::EvidenceReadTool;
pub use execution::{authorization_preflight, execute_authorized, ExecutionStatus};
// Inventory hosts use the MCP boundary already owned by this crate.
pub use claurst_mcp::{
    mcp_result_to_string, CatalogRevisionHeader, CombinedCatalogSnapshot, McpCatalogError,
    McpManager, McpToolDescriptor,
};
pub use exit_plan_mode::ExitPlanModeTool;
pub use file_edit::FileEditTool;
pub use file_read::FileReadTool;
pub use file_write::FileWriteTool;
pub(crate) use formatter::try_format_file;
pub use glob_tool::GlobTool;
pub use goal_complete::GoalCompleteTool;
pub use grep_tool::GrepTool;
pub use lsp_tool::LspTool;
pub use mcp_auth_tool::McpAuthTool;
pub use mcp_content::mcp_result_to_tool_result;
pub use mcp_resources::{ListMcpResourcesTool, ReadMcpResourceTool};
pub use monitor_tool::MonitorTool;
pub use mutation::EditCheckTool;
pub use notebook_edit::NotebookEditTool;
pub use powershell::PowerShellTool;
pub use pty_bash::PtyBashTool;
pub use remote_trigger::RemoteTriggerTool;
pub use repl_tool::ReplTool;
pub use send_message::{drain_inbox, peek_inbox, SendMessageTool};
pub use skill_tool::SkillTool;
pub use sleep::SleepTool;
pub use synthetic_output::SyntheticOutputTool;
pub use tasks::{
    Task, TaskAcceptTool, TaskCreateTool, TaskGetTool, TaskListTool, TaskOutputTool, TaskStatus,
    TaskStopTool, TaskUpdateTool, TASK_STORE,
};
pub use team_tool::{register_agent_runner, AgentRunFn, TeamCreateTool, TeamDeleteTool};
pub use todo_write::TodoWriteTool;
pub use tool_catalog::{
    DiscoveryReceipt, ToolCatalogEntry, ToolCatalogError, ToolCatalogScope, ToolCatalogSource,
    ToolCatalogView,
};
pub use tool_search::ToolSearchTool;
pub use trusted_checker::{
    RegistrationSpec, TrustedCheckerCatalog, TrustedCheckerCatalogBuilder, TrustedCheckerError,
    TrustedCheckerInvocation, TrustedCheckerOperation, TrustedCheckerRegistration,
    TrustedCheckerShell, TrustedCheckerSource,
};
pub use verification_receipts::{
    finalize_proof, CheckerReceipt, MutationReceipt, ReceiptBundle, ReceiptError,
    TrustedVerificationProof,
};
pub use web_fetch::WebFetchTool;
pub use web_search::WebSearchTool;
pub use worktree::{EnterWorktreeTool, ExitWorktreeTool};

// ---------------------------------------------------------------------------
// AskUser question channel
// ---------------------------------------------------------------------------

/// Event sent through the TUI side-channel when the `AskUserQuestion` tool
/// needs to pause the query loop and collect a response from the user.
pub struct UserQuestionEvent {
    /// The question text to display.
    pub question: String,
    /// Optional predefined choices (for multiple-choice questions).
    pub options: Option<Vec<String>>,
    /// Send the user's answer back through this channel to resume execution.
    pub reply_tx: tokio::sync::oneshot::Sender<String>,
}

// ---------------------------------------------------------------------------
// Core trait & types
// ---------------------------------------------------------------------------

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Content to send back to the model as the tool result.
    pub content: String,
    /// Whether this invocation was an error.
    pub is_error: bool,
    /// Optional structured metadata (for the TUI to render diffs, etc.).
    pub metadata: Option<Value>,
    application_receipts: Option<ReceiptBundle>,
    discovery_receipt: Option<DiscoveryReceipt>,
    execution_status: Option<ExecutionStatus>,
    typed_content: Option<Arc<ValidatedToolContent>>,
    /// Host-owned evidence authority.  Model-visible metadata never mints
    /// this state; it is populated only by the retention/read paths.
    evidence_state: Option<EvidenceState>,
}

/// Native Bash process outcome allowed to cross the query event boundary.
///
/// This is deliberately narrower than `ToolResult::metadata`: only the
/// built-in PTY Bash implementation can construct it through the `Tool` hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeBashCommandStatus {
    Completed,
    Unknown,
    NotRun,
    Cancelled,
    TimedOut,
}

/// Content-free, bounded evidence for one built-in PTY Bash result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NativeBashExecutionWitness {
    /// Exact `ToolStart`/`ToolEnd` identity; never a command or tool name.
    pub tool_id: String,
    /// SHA-256 over the exact UTF-8 bytes emitted in the `ToolEnd.result`.
    pub displayed_result_sha256: String,
    pub command_status: NativeBashCommandStatus,
    pub exit_code: Option<i32>,
    pub partial: bool,
}

impl NativeBashExecutionWitness {
    /// Parse only the allowlisted fields written by `PtyBashTool` and bind the
    /// digest to the final display string before metadata is discarded.
    pub(crate) fn from_trusted_result(
        tool_id: &str,
        result: &ToolResult,
        emitted_result: &str,
    ) -> Option<Self> {
        if tool_id.is_empty() || tool_id.len() > 256 {
            return None;
        }
        let metadata = result.metadata.as_ref()?.as_object()?;
        let command_status = match metadata.get("command_status")?.as_str()? {
            "completed" => NativeBashCommandStatus::Completed,
            "unknown" => NativeBashCommandStatus::Unknown,
            "not_run" => NativeBashCommandStatus::NotRun,
            "cancelled" => NativeBashCommandStatus::Cancelled,
            "timed_out" => NativeBashCommandStatus::TimedOut,
            _ => return None,
        };
        let exit_code = match metadata.get("exit_code")? {
            Value::Null => None,
            Value::Number(number) => {
                let code = number.as_i64()?.try_into().ok()?;
                Some(code)
            }
            _ => return None,
        };
        let partial = metadata.get("partial")?.as_bool()?;
        Some(Self {
            tool_id: tool_id.to_string(),
            displayed_result_sha256: claurst_core::crypto_utils::sha256_hex(
                emitted_result.as_bytes(),
            ),
            command_status,
            exit_code,
            partial,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EvidenceSelection {
    InitialPreview,
    Page { offset: usize, end: usize },
    Query,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvidenceState {
    pub(crate) id: String,
    pub(crate) bytes: u64,
    pub(crate) partial: bool,
    pub(crate) workspace: Option<PathBuf>,
    pub(crate) session: Option<String>,
    pub(crate) selection: EvidenceSelection,
    pub(crate) fallback: Option<String>,
    pub(crate) next_offset: Option<usize>,
    pub(crate) done: Option<bool>,
}

/// Private role indices for MCP-created resource provenance text blocks.
/// Indices are retained separately from payload and are never inferred from
/// rendered text or caller metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct TypedProvenance {
    #[serde(default)]
    pub(crate) resource_text_indices: Vec<usize>,
}

impl TypedProvenance {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    fn validate(&self, blocks: &[ContentBlock]) -> Result<(), String> {
        let mut previous = None;
        for &index in &self.resource_text_indices {
            if previous.is_some_and(|previous| index <= previous)
                || !matches!(blocks.get(index), Some(ContentBlock::Text { .. }))
            {
                return Err("MCP provenance indices are invalid".to_string());
            }
            previous = Some(index);
        }
        Ok(())
    }
}

/// Validated typed blocks retained alongside the bounded display string.
///
/// This remains private so callers can only create it through
/// `ToolResult::with_typed_content`, which validates every image occurrence
/// against the shared media policy before the payload becomes sendable.
#[derive(Debug, Clone)]
struct ValidatedToolContent {
    blocks: Vec<ContentBlock>,
    manifest: MediaManifest,
    limits: MediaLimits,
    provenance: TypedProvenance,
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            metadata: None,
            application_receipts: None,
            discovery_receipt: None,
            execution_status: None,
            typed_content: None,
            evidence_state: None,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            metadata: None,
            application_receipts: None,
            discovery_receipt: None,
            execution_status: None,
            typed_content: None,
            evidence_state: None,
        }
    }

    pub fn with_metadata(mut self, meta: Value) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Attach a validated typed content payload without copying it into the
    /// display summary. Image and nested tool-result blocks are checked by the
    /// shared core media validator before they become sendable.
    pub fn with_typed_content(
        mut self,
        blocks: Vec<ContentBlock>,
        limits: &MediaLimits,
    ) -> Result<Self, MediaError> {
        // An uncertain execution outcome is a terminal revocation.  Keep this
        // constructor order-safe: callers cannot reattach sendable media by
        // adding typed blocks after recording NotRun/Cancelled/Unknown.
        if self.execution_status.is_some_and(|status| {
            matches!(
                status,
                ExecutionStatus::NotRun | ExecutionStatus::Cancelled | ExecutionStatus::Unknown
            )
        }) {
            return Ok(self);
        }
        let manifest = MediaManifest::from_blocks(&blocks, limits)?;
        self.typed_content = Some(Arc::new(ValidatedToolContent {
            blocks,
            manifest,
            limits: *limits,
            provenance: TypedProvenance::empty(),
        }));
        Ok(self)
    }

    pub(crate) fn with_mcp_typed_content(
        mut self,
        blocks: Vec<ContentBlock>,
        provenance: TypedProvenance,
        limits: &MediaLimits,
    ) -> Result<Self, String> {
        if self.execution_status.is_some_and(|status| {
            matches!(
                status,
                ExecutionStatus::NotRun | ExecutionStatus::Cancelled | ExecutionStatus::Unknown
            )
        }) {
            return Ok(self);
        }
        provenance.validate(&blocks)?;
        let manifest =
            MediaManifest::from_blocks(&blocks, limits).map_err(|error| error.to_string())?;
        self.typed_content = Some(Arc::new(ValidatedToolContent {
            blocks,
            manifest,
            limits: *limits,
            provenance,
        }));
        Ok(self)
    }

    /// Return the validated typed blocks without exposing the private payload
    /// container or its mutable backing storage.
    pub fn typed_blocks(&self) -> Option<&[ContentBlock]> {
        self.typed_content
            .as_ref()
            .map(|content| content.blocks.as_slice())
    }

    /// Return the validated media metadata associated with the typed blocks.
    pub fn typed_manifest(&self) -> Option<&MediaManifest> {
        self.typed_content.as_ref().map(|content| &content.manifest)
    }

    /// Remove and return the validated typed blocks, if present.
    pub fn take_typed_content(&mut self) -> Option<Vec<ContentBlock>> {
        self.take_validated_typed_content()
            .map(|(blocks, _, _, _)| blocks)
    }

    pub(crate) fn take_validated_typed_content(
        &mut self,
    ) -> Option<(
        Vec<ContentBlock>,
        MediaManifest,
        MediaLimits,
        TypedProvenance,
    )> {
        let content = self.typed_content.take()?;
        match Arc::try_unwrap(content) {
            Ok(content) => Some((
                content.blocks,
                content.manifest,
                content.limits,
                content.provenance,
            )),
            Err(content) => Some((
                content.blocks.clone(),
                content.manifest.clone(),
                content.limits,
                content.provenance.clone(),
            )),
        }
    }

    pub(crate) fn evidence_state(&self) -> Option<&EvidenceState> {
        self.evidence_state.as_ref()
    }

    pub(crate) fn take_evidence_state(&mut self) -> Option<EvidenceState> {
        self.evidence_state.take()
    }

    pub(crate) fn with_evidence_state(mut self, state: EvidenceState) -> Self {
        self.evidence_state = Some(state);
        self
    }

    pub(crate) fn evidence_scope_matches(&self, ctx: &ToolContext) -> bool {
        let Some(state) = self.evidence_state.as_ref() else {
            return false;
        };
        let Ok(workspace) = crate::worktree::workspace_root(ctx).canonicalize() else {
            return false;
        };
        state.workspace.as_ref() == Some(&workspace)
            && state.session.as_deref() == Some(ctx.session_id.as_str())
    }

    /// Return the host-owned exceptional completion state, when one was
    /// recorded.  This is deliberately separate from model-visible metadata:
    /// callers cannot clear it by replacing or editing that metadata.
    pub fn execution_status(&self) -> Option<ExecutionStatus> {
        self.execution_status
    }

    /// Take the private host-only receipt bundle attached by an authorized
    /// mutator. Model-visible metadata never contains this value.
    pub fn take_application_receipts(&mut self) -> Option<ReceiptBundle> {
        self.application_receipts.take()
    }

    /// Take the private host-only discovery receipt attached by ToolSearch.
    /// The receipt is never serialized into model-visible metadata.
    pub fn take_discovery_receipt(&mut self) -> Option<DiscoveryReceipt> {
        self.discovery_receipt.take()
    }

    pub(crate) fn with_discovery_receipt(mut self, receipt: DiscoveryReceipt) -> Self {
        self.discovery_receipt = Some(receipt);
        self
    }

    pub(crate) fn with_application_receipts(mut self, receipts: ReceiptBundle) -> Self {
        self.application_receipts = Some(receipts);
        self
    }

    pub(crate) fn with_mutation_receipt(self, receipt: MutationReceipt) -> Self {
        self.with_application_receipts(ReceiptBundle::new(receipt))
    }

    /// Add an execution outcome without discarding evidence or tool metadata.
    /// Uncertain outcomes also revoke any private application receipt.
    pub fn with_execution_status(mut self, status: ExecutionStatus) -> Self {
        self.execution_status = Some(status);
        if matches!(
            status,
            ExecutionStatus::NotRun | ExecutionStatus::Cancelled | ExecutionStatus::Unknown
        ) {
            self.application_receipts = None;
            self.discovery_receipt = None;
            self.typed_content = None;
            self.evidence_state = None;
        }
        let mut metadata = match self.metadata.take() {
            Some(Value::Object(metadata)) => metadata,
            Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
            None => serde_json::Map::new(),
        };
        metadata.insert(
            "execution_status".into(),
            Value::String(status.as_str().into()),
        );
        self.metadata = Some(Value::Object(metadata));
        self
    }
}

/// Permission level required by a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionLevel {
    /// No permission needed (read-only, purely informational).
    None,
    /// Read-only access to the filesystem or network.
    ReadOnly,
    /// Write access to the filesystem.
    Write,
    /// Arbitrary command execution.
    Execute,
    /// Potentially dangerous (e.g., bypass sandbox).
    Dangerous,
    /// Unconditionally forbidden — the action must never be executed regardless
    /// of permission mode.  Used by the bash tool (`PtyBashTool`) when the
    /// classifier identifies a `Critical`-risk command (e.g. `rm -rf /`,
    /// fork-bomb, `dd if=…`).
    Forbidden,
}

/// Trusted execution-effect metadata used by the query scheduler.
///
/// Tools must opt into `Observation` explicitly. The default is `Unknown`,
/// because permission level and tool names do not prove that a call is pure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEffect {
    /// The invocation only observes state and may overlap another observation.
    Observation,
    /// The invocation may change state or perform an external side effect.
    Mutation,
    /// The effect cannot be established; schedule it as a barrier.
    Unknown,
}

/// Host-authored capability for mutation tools exposed to an isolated child.
///
/// Names and [`ToolEffect`] values are insufficient: a dynamic tool can reuse
/// a built-in name or claim mutation metadata while still spawning arbitrary
/// processes. The default is therefore unavailable. Only native file mutators
/// that accept the query layer's worktree path wrapper opt in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChildWriterCapability {
    #[default]
    Unavailable,
    ScopedFilesystem,
}

#[derive(Debug)]
pub struct PendingPermissionRequest {
    pub tool_use_id: String,
    pub request: claurst_core::permissions::PermissionRequest,
    pub reason: String,
    pub decision_tx: Option<tokio::sync::oneshot::Sender<PermissionDecision>>,
}

#[derive(Default)]
pub struct PendingPermissionStore {
    pub queue: VecDeque<PendingPermissionRequest>,
    pub waiting: HashMap<String, PendingPermissionRequest>,
    closed: bool,
    max_queued: Option<usize>,
}

impl PendingPermissionStore {
    /// Reopen only after the previous serialized turn has joined its drainer.
    pub fn reopen(&mut self, max_queued: Option<usize>) {
        self.closed = false;
        self.max_queued = max_queued;
    }

    /// Atomically reject late producers and resolve retained requests as denied.
    pub fn close(&mut self) {
        self.closed = true;
        for pending in self
            .queue
            .drain(..)
            .chain(self.waiting.drain().map(|(_, request)| request))
        {
            if let Some(sender) = pending.decision_tx {
                let _ = sender.send(PermissionDecision::Deny);
            }
        }
    }
}

/// Persistent shell state shared across Bash tool invocations within one session.
///
/// The bash tool (`PtyBashTool`) reads and writes this state on every call so
/// that `cd` and `export` commands persist across separate tool invocations, matching the
/// mental model described in the tool description ("the working directory
/// persists between commands").
#[derive(Debug, Clone, Default)]
pub struct ShellState {
    /// Current working directory as tracked by the shell state.
    /// Starts as the session's `working_dir`; updated after each `cd` command.
    pub cwd: Option<PathBuf>,
    /// Environment variable overrides exported by previous commands.
    pub env_vars: HashMap<String, String>,
}

impl ShellState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Process-global registry of shell states keyed by session_id.
/// This lets us persist cwd/env across Bash invocations without changing
/// the `ToolContext` struct (which is constructed in places we cannot modify).
#[derive(Clone, PartialEq, Eq, Hash)]
enum ShellScope {
    Legacy(String),
    Context {
        workspace: PathBuf,
        session: String,
        execution_root: PathBuf,
    },
}

static SHELL_STATE_REGISTRY: once_cell::sync::Lazy<
    dashmap::DashMap<ShellScope, Arc<parking_lot::Mutex<ShellState>>>,
> = once_cell::sync::Lazy::new(dashmap::DashMap::new);

/// Return the persistent `ShellState` for the given session, creating one if needed.
pub fn session_shell_state(session_id: &str) -> Arc<parking_lot::Mutex<ShellState>> {
    SHELL_STATE_REGISTRY
        .entry(ShellScope::Legacy(session_id.to_string()))
        .or_insert_with(|| Arc::new(parking_lot::Mutex::new(ShellState::new())))
        .clone()
}

/// Shell state is owned by a workspace/session and the active execution root.
/// Switching worktrees cannot carry a previous `cd` back into the main tree.
fn shell_scope(ctx: &ToolContext) -> ShellScope {
    ShellScope::Context {
        workspace: crate::worktree::workspace_root(ctx),
        session: ctx.session_id.clone(),
        execution_root: ctx
            .working_dir
            .canonicalize()
            .unwrap_or_else(|_| ctx.working_dir.clone()),
    }
}

pub fn context_shell_state(ctx: &ToolContext) -> Arc<parking_lot::Mutex<ShellState>> {
    SHELL_STATE_REGISTRY
        .entry(shell_scope(ctx))
        .or_insert_with(|| Arc::new(parking_lot::Mutex::new(ShellState::new())))
        .clone()
}

/// Exit resets only the abandoned execution root; the parent shell remains intact.
pub fn clear_context_shell_state(ctx: &ToolContext) {
    SHELL_STATE_REGISTRY.remove(&shell_scope(ctx));
}

/// Remove every execution root belonging to this exact session identifier.
pub fn clear_session_shell_state(session_id: &str) {
    SHELL_STATE_REGISTRY.retain(|key, _| match key {
        ShellScope::Legacy(id) | ShellScope::Context { session: id, .. } => id != session_id,
    });
}

/// Return the `ShadowSnapshot` for `working_dir`, creating it on first call.
/// Returns `None` when git is unavailable or the directory is not in a git repo.
pub fn session_shadow(
    working_dir: &std::path::Path,
) -> Option<Arc<claurst_core::snapshot::ShadowSnapshot>> {
    claurst_core::snapshot::get_or_create(working_dir)
}

/// Drop the cached shadow snapshot for `working_dir` (e.g. when a session ends).
pub fn clear_session_shadow(working_dir: &std::path::Path) {
    claurst_core::snapshot::remove(working_dir);
}

/// Write `contents` to `path` atomically: write to a temp file in the same
/// directory, then rename over the destination. A crash or disk-full mid-write
/// can never leave the destination truncated or half-written.
pub(crate) async fn write_atomic(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    let path = path.to_path_buf();
    let contents = contents.to_vec();
    let commit_lease = crate::mutation::owned_current_lease();
    // Once scheduled, the owned blocking transaction finishes even if the
    // caller is dropped. Cancellation after admission is therefore uncertain
    // until destination readback; it never strands a half-written scratch file.
    tokio::task::spawn_blocking(move || {
        let _commit_lease = commit_lease;
        use std::io::Write;
        let parent = path.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "destination has no parent",
            )
        })?;
        let file_name = path.file_name().unwrap_or_default().to_string_lossy();
        let mut temporary = tempfile::Builder::new()
            .prefix(&format!(".{file_name}.claurst-tmp-"))
            .tempfile_in(parent)?;
        temporary.write_all(&contents)?;
        if let Ok(metadata) = std::fs::metadata(&path) {
            temporary
                .as_file()
                .set_permissions(metadata.permissions())?;
        }
        temporary.as_file().sync_all()?;
        // Windows cannot replace a destination while another writer still
        // holds its handle. Retry only that transient rename failure within a
        // short bound; all other errors remain hard failures and the test's
        // atomic-write assertion stays intact.
        #[cfg(windows)]
        {
            let retry_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                match temporary.persist(&path) {
                    Ok(_) => break,
                    Err(error)
                        if error.error.kind() == std::io::ErrorKind::PermissionDenied
                            && std::time::Instant::now() < retry_deadline =>
                    {
                        temporary = error.file;
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(error) => return Err(error.error),
                }
            }
        }
        #[cfg(not(windows))]
        temporary.persist(&path).map_err(|error| error.error)?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    })
    .await
    .map_err(std::io::Error::other)?
}

/// A cloneable handle for injecting notification messages into the next agent turn.
/// Used by background tasks with `notify_on_complete` to signal completion without polling.
#[derive(Clone)]
pub struct CompletionNotifier(Arc<dyn Fn(String) + Send + Sync>);

impl CompletionNotifier {
    pub fn new(f: impl Fn(String) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
    pub fn notify(&self, msg: String) {
        (self.0)(msg);
    }
}

/// Shared context passed to every tool invocation.
#[derive(Clone)]
pub struct ToolContext {
    pub working_dir: PathBuf,
    pub permission_mode: PermissionMode,
    pub permission_handler: Arc<dyn PermissionHandler>,
    pub cost_tracker: Arc<CostTracker>,
    pub session_id: String,
    pub file_history: Arc<parking_lot::Mutex<claurst_core::file_history::FileHistory>>,
    pub current_turn: Arc<AtomicUsize>,
    /// If true, suppress interactive prompts (batch / CI mode).
    pub non_interactive: bool,
    /// Optional MCP manager for ListMcpResources / ReadMcpResource tools.
    pub mcp_manager: Option<Arc<claurst_mcp::McpManager>>,
    /// Configured event hooks (PreToolUse, PostToolUse, etc.).
    pub config: claurst_core::config::Config,
    /// Managed agent (manager-executor) configuration, if active.
    pub managed_agent_config: Option<claurst_core::config::ManagedAgentConfig>,
    /// Optional notifier for injecting completion messages into the next agent turn.
    /// Set when the query loop has a command queue wired up.
    pub completion_notifier: Option<CompletionNotifier>,
    /// Queue used by interactive mode to surface permission dialogs to the TUI.
    pub pending_permissions: Option<Arc<parking_lot::Mutex<PendingPermissionStore>>>,
    /// Shared permission manager so the interactive loop can record session/persistent approvals.
    pub permission_manager:
        Option<Arc<std::sync::Mutex<claurst_core::permissions::PermissionManager>>>,
    /// Channel for the `AskUserQuestion` tool to send questions to the TUI and
    /// receive the user's typed answer.  `None` in headless / non-interactive mode.
    pub user_question_tx: Option<tokio::sync::mpsc::UnboundedSender<UserQuestionEvent>>,
    /// Cancellation token for the owning query loop (issue #218). The parallel
    /// tool executor selects on this to abandon in-flight tools when the user
    /// cancels, and long-running tools may observe it to bail out early. Defaults
    /// to a fresh disconnected token; `run_query_loop` rebinds it to the loop's
    /// actual token so cancellation propagates into tools and sub-agents.
    pub cancel_token: tokio_util::sync::CancellationToken,
    /// Immutable host-owned trusted checker registrations. An empty catalog
    /// preserves the default adaptive behavior until the host opts in.
    pub trusted_checker_catalog: Arc<trusted_checker::TrustedCheckerCatalog>,
    /// Host-built immutable metadata view for authorized lazy tool discovery.
    /// `None` preserves legacy behavior until the query host binds a view.
    pub catalog_view: Option<Arc<tool_catalog::ToolCatalogView>>,
}

impl ToolContext {
    /// Resolve a potentially relative path against the working directory.
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            self.working_dir.join(p)
        }
    }

    fn permission_allowed_roots(&self) -> Vec<PathBuf> {
        let mut roots = self.config.workspace_paths.clone();
        roots.extend(self.config.additional_dirs.clone());
        roots
    }

    fn build_permission_request(
        &self,
        tool_name: &str,
        description: &str,
        details: Option<String>,
        is_read_only: bool,
        path: Option<PathBuf>,
    ) -> PermissionRequest {
        PermissionRequest {
            tool_name: tool_name.to_string(),
            description: description.to_string(),
            details,
            is_read_only,
            path: path.map(|p| p.display().to_string()),
            working_dir: Some(self.working_dir.clone()),
            allowed_roots: self.permission_allowed_roots(),
            context_description: None,
        }
    }

    fn request_permission_inner(
        &self,
        request: PermissionRequest,
    ) -> Result<(), claurst_core::error::ClaudeError> {
        let interactive_reason = request.details.clone();
        let decision = self.permission_handler.request_permission(&request);
        match decision {
            PermissionDecision::Allow | PermissionDecision::AllowPermanently => Ok(()),
            PermissionDecision::Ask { reason } if self.non_interactive => {
                Err(claurst_core::error::ClaudeError::PermissionDenied(format!(
                    "Permission denied for tool '{}': {}",
                    request.tool_name,
                    interactive_reason.unwrap_or(reason)
                )))
            }
            PermissionDecision::Ask { reason } => {
                let Some(queue) = &self.pending_permissions else {
                    return Err(claurst_core::error::ClaudeError::PermissionDenied(format!(
                        "Permission denied for tool '{}'",
                        request.tool_name
                    )));
                };

                let (tx, rx) = tokio::sync::oneshot::channel();
                let mut store = queue.lock();
                if store.closed
                    || self.cancel_token.is_cancelled()
                    || store
                        .max_queued
                        .is_some_and(|limit| store.queue.len() >= limit)
                {
                    return Err(claurst_core::error::ClaudeError::PermissionDenied(
                        "Permission request is unavailable".to_string(),
                    ));
                }
                store.queue.push_back(PendingPermissionRequest {
                    tool_use_id: format!(
                        "perm-{}-{}",
                        self.session_id,
                        self.current_turn.fetch_add(1, Ordering::Relaxed)
                    ),
                    request,
                    reason: interactive_reason.unwrap_or(reason),
                    decision_tx: Some(tx),
                });
                drop(store);

                let decision = tokio::task::block_in_place(|| rx.blocking_recv());
                match decision {
                    Ok(PermissionDecision::Allow | PermissionDecision::AllowPermanently) => Ok(()),
                    _ => Err(claurst_core::error::ClaudeError::PermissionDenied(
                        "Permission denied by user".to_string(),
                    )),
                }
            }
            _ => Err(claurst_core::error::ClaudeError::PermissionDenied(format!(
                "Permission denied for tool '{}'",
                request.tool_name
            ))),
        }
    }

    /// Check permissions for a tool invocation.
    pub fn check_permission(
        &self,
        tool_name: &str,
        description: &str,
        is_read_only: bool,
    ) -> Result<(), claurst_core::error::ClaudeError> {
        let request =
            self.build_permission_request(tool_name, description, None, is_read_only, None);
        self.request_permission_inner(request)
    }

    /// Perform the non-interactive allow/deny capability check used by the
    /// query execution backstop. Self-gating tools still receive `Ask` here so
    /// their existing path-aware prompt and hook behavior remains authoritative;
    /// an explicit deny always stops execution before the tool is called.
    pub fn check_capability(
        &self,
        tool_name: &str,
        description: &str,
        is_read_only: bool,
    ) -> PermissionDecision {
        let request =
            self.build_permission_request(tool_name, description, None, is_read_only, None);
        self.permission_handler.check_permission(&request)
    }

    pub fn check_permission_for_path(
        &self,
        tool_name: &str,
        description: &str,
        path: PathBuf,
        is_read_only: bool,
    ) -> Result<(), claurst_core::error::ClaudeError> {
        let request =
            self.build_permission_request(tool_name, description, None, is_read_only, Some(path));
        self.request_permission_inner(request)
    }

    /// Like `check_permission` but also passes structured `details` text
    /// (e.g. a risk explanation) that the TUI permission dialog can display.
    pub fn check_permission_with_details(
        &self,
        tool_name: &str,
        description: &str,
        details: &str,
        is_read_only: bool,
    ) -> Result<(), claurst_core::error::ClaudeError> {
        let request = self.build_permission_request(
            tool_name,
            description,
            Some(details.to_string()),
            is_read_only,
            None,
        );
        self.request_permission_inner(request).map_err(|_| {
            claurst_core::error::ClaudeError::PermissionDenied(format!(
                "Permission denied for tool '{}': {}",
                tool_name, details
            ))
        })
    }

    pub fn path_is_within_workspace(&self, path: &std::path::Path) -> bool {
        let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut roots =
            vec![std::fs::canonicalize(&self.working_dir)
                .unwrap_or_else(|_| self.working_dir.clone())];
        roots.extend(
            self.permission_allowed_roots()
                .into_iter()
                .map(|root| std::fs::canonicalize(&root).unwrap_or(root)),
        );
        roots.iter().any(|root| resolved.starts_with(root))
    }

    pub fn check_permission_with_details_and_path(
        &self,
        tool_name: &str,
        description: &str,
        details: &str,
        path: PathBuf,
        is_read_only: bool,
    ) -> Result<(), claurst_core::error::ClaudeError> {
        let request = self.build_permission_request(
            tool_name,
            description,
            Some(details.to_string()),
            is_read_only,
            Some(path),
        );
        self.request_permission_inner(request).map_err(|_| {
            claurst_core::error::ClaudeError::PermissionDenied(format!(
                "Permission denied for tool '{}': {}",
                tool_name, details
            ))
        })
    }

    pub fn current_turn_index(&self) -> usize {
        self.current_turn.load(Ordering::Relaxed)
    }

    pub fn record_file_change(
        &self,
        path: PathBuf,
        before_content: &[u8],
        after_content: &[u8],
        tool_name: &str,
    ) {
        self.file_history.lock().record_modification(
            path,
            before_content,
            after_content,
            self.current_turn_index(),
            tool_name,
        );
    }
}

/// The trait every tool must implement.
// async-trait injects #[must_use] on boxed futures; Rust 1.99 flags that generated attribute.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait Tool: Send + Sync + std::any::Any {
    /// Human-readable name (matches the constant in claurst_core::constants).
    fn name(&self) -> &str;

    /// One-line description shown to the LLM.
    fn description(&self) -> &str;

    /// The permission level the tool requires.
    fn permission_level(&self) -> PermissionLevel;

    /// Trusted scheduler metadata. Unknown is conservative and serializes the
    /// call until a tool explicitly proves observation-only behavior.
    fn effect(&self) -> ToolEffect {
        ToolEffect::Unknown
    }

    /// Whether this exact handle may be wrapped as a path-scoped child writer.
    ///
    /// Child execution also clears formatters, hooks, extra directories and
    /// automatic commits. Implementations must not spawn another writer or
    /// interpret commands from their input.
    fn child_writer_capability(&self) -> ChildWriterCapability {
        ChildWriterCapability::Unavailable
    }

    /// Whether this tool performs its own permission gating inside `execute()`.
    ///
    /// - `false` (default): the central backstop in `execute_tool` is
    ///   responsible for gating this tool. Secure by default — a tool that
    ///   forgets to call `ctx.check_permission*` is still caught by the backstop
    ///   whenever its `permission_level()` is a gated level.
    /// - `true`: the tool already prompts for permission internally (it calls
    ///   `ctx.check_permission*` in `execute()`), so the central gate must NOT
    ///   also prompt — this prevents double-prompting.
    fn self_gates(&self) -> bool {
        false
    }

    /// Whether this tool is "advanced" / rarely used and a candidate for
    /// deferred (on-demand) disclosure rather than being sent in every request.
    ///
    /// Defaults to `false` (initially disclosed). When authorized ToolSearch
    /// is available, advanced schemas are disclosed only after a scoped host
    /// discovery receipt activates them. Execution policy remains independent.
    fn advanced(&self) -> bool {
        false
    }

    /// Trusted host metadata for discovery. Source identity never derives
    /// from a displayed name and does not grant execution authority.
    fn catalog_source(&self) -> ToolCatalogSource {
        if self.advanced() {
            ToolCatalogSource::DeferredNative
        } else {
            ToolCatalogSource::Native
        }
    }

    /// Return a witness only for a concrete built-in native Bash capability.
    /// The default is intentionally absent, so same-name MCP/plugin tools
    /// cannot turn arbitrary metadata into execution authority.
    fn native_bash_execution_witness(
        &self,
        _tool_id: &str,
        _result: &ToolResult,
        _emitted_result: &str,
    ) -> Option<NativeBashExecutionWitness> {
        None
    }

    /// JSON Schema describing the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Execute the tool with the given JSON input.
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult;

    /// Produce a `ToolDefinition` suitable for sending to the API.
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// Return all built-in tools (excluding AgentTool, which lives in cc-query).
pub fn all_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(PtyBashTool),
        Box::new(FileReadTool),
        Box::new(EvidenceReadTool),
        Box::new(FileEditTool),
        Box::new(FileWriteTool),
        Box::new(EditCheckTool),
        Box::new(BatchEditTool),
        Box::new(ApplyPatchTool),
        Box::new(GlobTool),
        Box::new(GrepTool),
        Box::new(WebFetchTool),
        Box::new(WebSearchTool),
        Box::new(NotebookEditTool),
        Box::new(TaskCreateTool),
        Box::new(TaskGetTool),
        Box::new(TaskUpdateTool),
        Box::new(TaskListTool),
        Box::new(TaskStopTool),
        Box::new(TaskOutputTool),
        Box::new(TaskAcceptTool),
        Box::new(TodoWriteTool),
        Box::new(AskUserQuestionTool),
        Box::new(EnterPlanModeTool),
        Box::new(ExitPlanModeTool),
        Box::new(PowerShellTool),
        Box::new(SleepTool),
        Box::new(CronCreateTool),
        Box::new(CronDeleteTool),
        Box::new(CronListTool),
        Box::new(EnterWorktreeTool),
        Box::new(ExitWorktreeTool),
        Box::new(ListMcpResourcesTool),
        Box::new(ReadMcpResourceTool),
        Box::new(ToolSearchTool),
        Box::new(BriefTool),
        Box::new(ConfigTool),
        Box::new(SendMessageTool),
        Box::new(SkillTool),
        Box::new(LspTool),
        Box::new(ReplTool),
        Box::new(TeamCreateTool),
        Box::new(TeamDeleteTool),
        Box::new(SyntheticOutputTool),
        Box::new(McpAuthTool),
        Box::new(RemoteTriggerTool),
        Box::new(MonitorTool),
        Box::new(GoalCompleteTool),
        // Computer Use is only available when compiled with the feature flag.
        #[cfg(feature = "computer-use")]
        Box::new(computer_use::ComputerUseTool),
    ]
}

/// Find a tool by name (case-sensitive).
pub fn find_tool(name: &str) -> Option<Box<dyn Tool>> {
    all_tools().into_iter().find(|t| t.name() == name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct AskPermissionHandler {
        reason: String,
    }

    impl claurst_core::permissions::PermissionHandler for AskPermissionHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Ask {
                reason: self.reason.clone(),
            }
        }

        fn request_permission(
            &self,
            request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            self.check_permission(request)
        }
    }

    fn test_tool_context(
        handler: Arc<dyn claurst_core::permissions::PermissionHandler>,
    ) -> ToolContext {
        use claurst_core::config::Config;

        ToolContext {
            working_dir: PathBuf::from("/workspace"),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "test".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            trusted_checker_catalog: Arc::new(trusted_checker::TrustedCheckerCatalog::empty()),
            catalog_view: None,
        }
    }

    #[test]
    fn closed_or_cancelled_permission_producers_never_block_or_enqueue() {
        for scenario in ["closed", "cancelled", "full"] {
            let store = Arc::new(parking_lot::Mutex::new(PendingPermissionStore::default()));
            let mut context = test_tool_context(Arc::new(AskPermissionHandler {
                reason: "approval".to_string(),
            }));
            context.non_interactive = false;
            context.pending_permissions = Some(store.clone());
            if scenario == "closed" {
                store.lock().close();
            } else if scenario == "cancelled" {
                context.cancel_token.cancel();
            } else {
                let mut store = store.lock();
                store.reopen(Some(1));
                store.queue.push_back(PendingPermissionRequest {
                    tool_use_id: "retained".to_string(),
                    request: claurst_core::permissions::PermissionRequest {
                        tool_name: "Write".to_string(),
                        description: String::new(),
                        details: None,
                        is_read_only: false,
                        path: None,
                        working_dir: None,
                        allowed_roots: Vec::new(),
                        context_description: None,
                    },
                    reason: String::new(),
                    decision_tx: None,
                });
            }
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                let denied = context.check_permission("Write", "fixture", false).is_err();
                let _ = sender.send(denied);
            });
            assert!(receiver
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("inactive permission producer blocked"));
            worker.join().unwrap();
            assert_eq!(store.lock().queue.len(), usize::from(scenario == "full"));
        }
    }

    #[tokio::test]
    async fn permission_store_closure_denies_retained_waiters_and_can_reopen() {
        let mut store = PendingPermissionStore::default();
        for _ in 0..2 {
            store.reopen(None);
            let (sender, receiver) = tokio::sync::oneshot::channel();
            store.queue.push_back(PendingPermissionRequest {
                tool_use_id: "fixture".to_string(),
                request: claurst_core::permissions::PermissionRequest {
                    tool_name: "Write".to_string(),
                    description: String::new(),
                    details: None,
                    is_read_only: false,
                    path: None,
                    working_dir: None,
                    allowed_roots: Vec::new(),
                    context_description: None,
                },
                reason: String::new(),
                decision_tx: Some(sender),
            });
            store.close();
            assert!(matches!(receiver.await.unwrap(), PermissionDecision::Deny));
            assert!(store.queue.is_empty() && store.waiting.is_empty());
        }
    }

    #[test]
    fn typed_execution_status_survives_metadata_replacement() {
        let result = ToolResult::error("uncertain")
            .with_execution_status(ExecutionStatus::Unknown)
            .with_metadata(serde_json::json!({"execution_status": "completed"}));

        assert_eq!(result.execution_status(), Some(ExecutionStatus::Unknown));
        assert_eq!(
            result.metadata.as_ref().unwrap()["execution_status"],
            "completed"
        );
    }

    #[test]
    fn model_metadata_cannot_mint_discovery_authority() {
        let mut result = ToolResult::success("search").with_metadata(serde_json::json!({
            "discovery_receipt": {"catalog_id": 1, "selected_names": ["Read"]}
        }));
        assert!(result.take_discovery_receipt().is_none());
    }

    // ---- Tool registry tests ------------------------------------------------

    #[test]
    fn test_all_tools_non_empty() {
        let tools = all_tools();
        assert!(
            !tools.is_empty(),
            "all_tools() must return at least one tool"
        );
    }

    #[test]
    fn test_all_tools_have_unique_names() {
        let tools = all_tools();
        let mut names = std::collections::HashSet::new();
        for tool in &tools {
            assert!(
                names.insert(tool.name().to_string()),
                "Duplicate tool name: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_all_tools_have_non_empty_descriptions() {
        for tool in all_tools() {
            assert!(
                !tool.description().is_empty(),
                "Tool '{}' has empty description",
                tool.name()
            );
        }
    }

    #[test]
    fn test_all_tools_have_valid_input_schema() {
        for tool in all_tools() {
            let schema = tool.input_schema();
            assert!(
                schema.is_object(),
                "Tool '{}' input_schema must be a JSON object",
                tool.name()
            );
            assert!(
                schema.get("type").is_some() || schema.get("properties").is_some(),
                "Tool '{}' schema missing type or properties",
                tool.name()
            );
        }
    }

    #[test]
    fn test_find_tool_found() {
        let tool = find_tool("Bash");
        assert!(tool.is_some(), "Should find the Bash tool");
        assert_eq!(tool.unwrap().name(), "Bash");
    }

    #[test]
    fn test_find_tool_not_found() {
        assert!(find_tool("NonExistentTool12345").is_none());
    }

    #[test]
    fn test_find_tool_case_sensitive() {
        // Tool names are case-sensitive — "bash" should not match "Bash"
        assert!(find_tool("bash").is_none());
        assert!(find_tool("Bash").is_some());
    }

    #[test]
    fn test_core_tools_present() {
        let expected = [
            "Bash",
            "Read",
            "Edit",
            "Write",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "Skill",
        ];
        for name in &expected {
            assert!(
                find_tool(name).is_some(),
                "Expected tool '{}' not found in all_tools()",
                name
            );
        }
    }

    // ---- ToolResult tests ---------------------------------------------------

    #[test]
    fn test_tool_result_success() {
        let r = ToolResult::success("done");
        assert!(!r.is_error);
        assert_eq!(r.content, "done");
        assert!(r.metadata.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let r = ToolResult::error("something went wrong");
        assert!(r.is_error);
        assert_eq!(r.content, "something went wrong");
    }

    #[test]
    fn test_tool_result_with_metadata() {
        let r = ToolResult::success("ok")
            .with_metadata(serde_json::json!({"file": "foo.rs", "lines": 10}));
        assert!(r.metadata.is_some());
        let meta = r.metadata.unwrap();
        assert_eq!(meta["file"], "foo.rs");
    }

    #[test]
    fn typed_tool_content_validates_and_preserves_blocks_separately() {
        let blocks = vec![
            claurst_core::types::ContentBlock::Text {
                text: "original text".to_string(),
            },
            claurst_core::types::ContentBlock::Image {
                source: claurst_core::types::ImageSource {
                    source_type: "base64".to_string(),
                    media_type: Some("image/png".to_string()),
                    data: Some(
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
                            .to_string(),
                    ),
                    url: None,
                },
            },
        ];
        let result = ToolResult::success("bounded display summary")
            .with_typed_content(blocks.clone(), &claurst_core::media::MediaLimits::default())
            .expect("valid typed image");
        assert_eq!(result.content, "bounded display summary");
        assert_eq!(result.typed_blocks().unwrap().len(), blocks.len());
        assert_eq!(result.typed_manifest().unwrap().items().len(), 1);
        assert_eq!(
            serde_json::to_vec(result.typed_blocks().unwrap()).unwrap(),
            serde_json::to_vec(&blocks).unwrap()
        );
    }

    #[test]
    fn typed_tool_content_rejects_urls_and_uncertain_status_drops_media() {
        let blocks = vec![claurst_core::types::ContentBlock::Image {
            source: claurst_core::types::ImageSource {
                source_type: "url".to_string(),
                media_type: Some("image/png".to_string()),
                data: None,
                url: Some("https://example.invalid/image.png".to_string()),
            },
        }];
        assert!(matches!(
            ToolResult::success("summary")
                .with_typed_content(blocks, &claurst_core::media::MediaLimits::default()),
            Err(claurst_core::media::MediaError::UrlUnsupported)
        ));

        let image = claurst_core::types::ContentBlock::Image {
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
        let result = ToolResult::success("summary")
            .with_typed_content(vec![image], &claurst_core::media::MediaLimits::default())
            .expect("valid typed image")
            .with_execution_status(ExecutionStatus::Cancelled);
        assert!(result.typed_blocks().is_none());

        let image = claurst_core::types::ContentBlock::Image {
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
        for status in [
            ExecutionStatus::NotRun,
            ExecutionStatus::Cancelled,
            ExecutionStatus::Unknown,
        ] {
            let result = ToolResult::success("summary")
                .with_execution_status(status)
                .with_typed_content(
                    vec![image.clone()],
                    &claurst_core::media::MediaLimits::default(),
                )
                .expect("uncertain result's typed attachment is a documented revocation");
            assert!(result.typed_blocks().is_none());
        }
    }

    // ---- ToolContext::resolve_path tests ------------------------------------

    #[test]
    fn test_resolve_path_absolute() {
        use claurst_core::permissions::AutoPermissionHandler;

        let handler = Arc::new(AutoPermissionHandler {
            mode: claurst_core::config::PermissionMode::Default,
        });
        let ctx = test_tool_context(handler);

        // Absolute paths pass through unchanged
        let resolved = ctx.resolve_path("/absolute/path/file.rs");
        assert_eq!(resolved, PathBuf::from("/absolute/path/file.rs"));
    }

    #[test]
    fn test_resolve_path_relative() {
        use claurst_core::permissions::AutoPermissionHandler;

        let handler = Arc::new(AutoPermissionHandler {
            mode: claurst_core::config::PermissionMode::Default,
        });
        let ctx = test_tool_context(handler);

        // Relative paths get joined with working_dir
        let resolved = ctx.resolve_path("src/main.rs");
        assert_eq!(resolved, PathBuf::from("/workspace/src/main.rs"));
    }

    #[test]
    fn test_request_permission_uses_details_for_non_interactive_errors() {
        let ctx = test_tool_context(Arc::new(AskPermissionHandler {
            reason: "generic reason".to_string(),
        }));
        let request = ctx.build_permission_request(
            "PowerShell",
            "[High risk] set execution policy",
            Some("[High risk] This may modify system-wide security policy.".to_string()),
            false,
            Some(PathBuf::from("Set-ExecutionPolicy RemoteSigned")),
        );

        let error = ctx
            .request_permission_inner(request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("[High risk] This may modify system-wide security policy."));
        assert!(!error.contains("generic reason"));
    }

    #[test]
    fn test_request_permission_falls_back_to_handler_reason_without_details() {
        let ctx = test_tool_context(Arc::new(AskPermissionHandler {
            reason: "generic reason".to_string(),
        }));
        let request = ctx.build_permission_request(
            "Bash",
            "run ls",
            None,
            false,
            Some(PathBuf::from("ls -la")),
        );

        let error = ctx
            .request_permission_inner(request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("generic reason"));
    }

    // ---- PermissionLevel tests ---------------------------------------------

    #[test]
    fn test_permission_level_order() {
        // Just verify the variants exist and are distinct
        assert_ne!(PermissionLevel::None, PermissionLevel::ReadOnly);
        assert_ne!(PermissionLevel::Write, PermissionLevel::Execute);
        assert_ne!(PermissionLevel::Execute, PermissionLevel::Dangerous);
    }

    #[test]
    fn test_bash_tool_permission_level() {
        assert_eq!(PtyBashTool.permission_level(), PermissionLevel::Execute);
    }

    #[test]
    fn test_file_read_permission_level() {
        assert_eq!(FileReadTool.permission_level(), PermissionLevel::ReadOnly);
    }

    #[test]
    fn test_file_edit_permission_level() {
        assert_eq!(FileEditTool.permission_level(), PermissionLevel::Write);
    }

    #[test]
    fn test_file_write_permission_level() {
        assert_eq!(FileWriteTool.permission_level(), PermissionLevel::Write);
    }

    #[test]
    fn scheduler_effect_metadata_is_conservative() {
        assert_eq!(FileReadTool.effect(), ToolEffect::Observation);
        assert_eq!(GlobTool.effect(), ToolEffect::Observation);
        assert_eq!(GrepTool.effect(), ToolEffect::Observation);
        assert_eq!(FileWriteTool.effect(), ToolEffect::Mutation);
        assert_eq!(PtyBashTool.effect(), ToolEffect::Unknown);
    }

    // ---- Tool to_definition tests ------------------------------------------

    #[test]
    fn test_tool_to_definition() {
        let def = PtyBashTool.to_definition();
        assert_eq!(def.name, "Bash");
        assert!(!def.description.is_empty());
        assert!(def.input_schema.is_object());
    }

    // ---- write_atomic tests -------------------------------------------------
    //
    // `write_atomic` is the single atomic-write path that ApplyPatch, BatchEdit,
    // NotebookEdit and the cron store (#226) all route through. These tests pin
    // its contract: it writes the exact bytes and never leaves a temp file
    // behind on success — the guarantee that makes those tools crash-safe.

    /// Count the `.claurst-tmp-*` scratch files left in `dir`.
    fn count_atomic_tmp_files(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".claurst-tmp-"))
            .count()
    }

    #[tokio::test]
    async fn write_atomic_writes_content_and_leaves_no_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");

        // Fresh file.
        write_atomic(&path, b"hello\nworld\n").await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello\nworld\n");
        assert_eq!(count_atomic_tmp_files(dir.path()), 0, "no tmp after create");

        // Overwrite an existing file (the crash-truncation scenario #226 fixes).
        write_atomic(&path, b"replaced").await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replaced");
        assert_eq!(
            count_atomic_tmp_files(dir.path()),
            0,
            "no tmp after overwrite"
        );
    }

    #[tokio::test]
    async fn write_atomic_concurrent_writers_use_distinct_scratch_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.txt");
        let writes = (0..16).map(|index| {
            let path = path.clone();
            async move {
                let content = vec![b'a' + index; 64 * 1024];
                write_atomic(&path, &content).await
            }
        });
        for result in futures::future::join_all(writes).await {
            assert!(
                result.is_ok(),
                "concurrent atomic commit failed: {result:?}"
            );
        }
        let final_bytes = std::fs::read(&path).unwrap();
        assert_eq!(final_bytes.len(), 64 * 1024);
        assert!(final_bytes.iter().all(|byte| *byte == final_bytes[0]));
        assert_eq!(count_atomic_tmp_files(dir.path()), 0);
    }

    #[test]
    fn write_atomic_finishes_owned_commit_after_caller_is_dropped() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("owned.txt");
        runtime.block_on(async {
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                ready_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            ready_rx.await.unwrap();
            let destination = path.clone();
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let writer = tokio::spawn(async move {
                entered_tx.send(()).unwrap();
                write_atomic(&destination, b"whole committed content").await
            });
            entered_rx.await.unwrap();
            tokio::task::yield_now().await;
            writer.abort();
            assert!(writer.await.unwrap_err().is_cancelled());
            release_tx.send(()).unwrap();
            blocker.await.unwrap();
            // FIFO blocking-pool fence: the owned commit has finished even
            // though its awaiting task was cancelled.
            tokio::task::spawn_blocking(|| ()).await.unwrap();
        });
        assert_eq!(std::fs::read(&path).unwrap(), b"whole committed content");
        assert_eq!(count_atomic_tmp_files(dir.path()), 0);
    }

    #[test]
    fn write_atomic_holds_scope_until_cancelled_callers_commit_finishes() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ctx = crate::test_support::allow_all_context(dir.path().to_path_buf());
        let path = dir.path().join("leased.txt");
        runtime.block_on(async {
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                ready_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            ready_rx.await.unwrap();
            let destination = path.clone();
            let writer_ctx = ctx.clone();
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let writer = tokio::spawn(async move {
                let lease = match crate::mutation::admit(&writer_ctx, &FileWriteTool).unwrap() {
                    crate::mutation::MutationAdmission::Acquired(lease) => lease,
                    _ => panic!("expected fresh mutation lease"),
                };
                crate::mutation::with_lease(&lease, async move {
                    entered_tx.send(()).unwrap();
                    write_atomic(&destination, b"owned lease commit").await
                })
                .await
            });
            entered_rx.await.unwrap();
            tokio::task::yield_now().await;
            writer.abort();
            assert!(writer.await.unwrap_err().is_cancelled());
            assert!(
                crate::mutation::admit(&ctx, &FileWriteTool).is_err(),
                "in-flight owned commit must retain exclusion"
            );
            release_tx.send(()).unwrap();
            blocker.await.unwrap();
            tokio::task::spawn_blocking(|| ()).await.unwrap();
            assert!(crate::mutation::admit(&ctx, &FileWriteTool).is_ok());
        });
        assert_eq!(std::fs::read(&path).unwrap(), b"owned lease commit");
        assert_eq!(count_atomic_tmp_files(dir.path()), 0);
    }

    /// The executable bit (and other permissions) must survive an atomic
    /// overwrite, since we rename a fresh temp file over the destination.
    #[cfg(unix)]
    #[tokio::test]
    async fn write_atomic_preserves_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("script.sh");

        std::fs::write(&path, b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();

        write_atomic(&path, b"#!/bin/sh\necho hi\n").await.unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755, "executable bit preserved");
        assert_eq!(count_atomic_tmp_files(dir.path()), 0);
    }
}
