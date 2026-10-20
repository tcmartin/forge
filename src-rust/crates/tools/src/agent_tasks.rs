//! Scoped lifecycle service for background AgentTool children.
//!
//! AgentTool lives in `claurst-query`, while monitor and task tools live in
//! `claurst-tools`.  Keeping this registry here gives both sides one owner for
//! child cancellation, terminal results, and evidence without exposing the
//! process-global core task mirror as an authority.

use crate::background_processes::{self, BackgroundCleanup};
use crate::command_capture::{capture_owned_with_cancel, ProcessCleanup, ProcessGroupGuard};
use crate::evidence::{retain_output, retain_typed_output};
use crate::mutation::{self, PendingMutationTicket};
use crate::{ToolContext, ToolResult, TrustedCheckerRegistration, TrustedCheckerShell};
use claurst_core::crypto_utils::sha256_hex;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

pub const MAX_AGENT_TASKS: usize = 64;
const TASK_OUTPUT_WAIT: Duration = Duration::from_secs(3);
const TASK_VALUE_BUDGET_BYTES: usize = 24 * 1024;
const TASK_INNER_BUDGET_BYTES: usize = TASK_VALUE_BUDGET_BYTES - 2 * 1024;
const TASK_OUTPUT_PREVIEW_BYTES: usize = 8 * 1024;
const TASK_ID_PREVIEW_BYTES: usize = 512;
const TASK_PATH_PREVIEW_BYTES: usize = 1024;
const TASK_TEXT_PREVIEW_BYTES: usize = 1024;
const ACCEPTANCE_COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
const ACCEPTANCE_CONFIRM_TIMEOUT: Duration = Duration::from_secs(6);
const MAX_ACCEPTANCE_FILES: usize = 4096;
const MAX_ACCEPTANCE_PATH_BYTES: usize = 4096;
const MAX_ACCEPTANCE_FILE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_ACCEPTANCE_TOTAL_BYTES: u64 = 64 * 1024 * 1024;

/// The query loop's terminal outcome.  Keep cancellation and incomplete
/// outcomes distinct from a successful end turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTaskStatus {
    Completed,
    CompletedAfterCancellationRequest,
    AwaitingParentAcceptance,
    MaxTokens,
    Cancelled,
    Failed,
    BudgetExceeded,
}

impl AgentTaskStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::CompletedAfterCancellationRequest => "completed_after_cancellation_request",
            Self::AwaitingParentAcceptance => "awaiting_parent_acceptance",
            Self::MaxTokens => "max_tokens",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::BudgetExceeded => "budget_exceeded",
        }
    }

    fn partial(self) -> bool {
        !matches!(
            self,
            Self::Completed
                | Self::CompletedAfterCancellationRequest
                | Self::AwaitingParentAcceptance
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentProcessStatus {
    Running,
    CancellationRequested,
    Terminal(AgentTaskStatus),
}

impl AgentProcessStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::CancellationRequested => "cancellation_requested",
            Self::Terminal(status) => status.as_str(),
        }
    }

    fn terminal(self) -> bool {
        matches!(self, Self::Terminal(_))
    }
}

struct AgentProcessState {
    status: AgentProcessStatus,
    result: Option<ToolResult>,
    cancellation_requested: bool,
    pending_acceptance_status: Option<AgentTaskStatus>,
}

struct AgentProcess {
    id: String,
    description: String,
    workspace: PathBuf,
    parent_working_dir: PathBuf,
    parent_session: String,
    child_session: String,
    working_dir: PathBuf,
    worktree_path: Option<PathBuf>,
    created_at: Instant,
    cancel: CancellationToken,
    pending: Option<PendingMutationTicket>,
    parent_pending: Option<PendingMutationTicket>,
    state: Mutex<AgentProcessState>,
    acceptance: Mutex<()>,
    changed: Notify,
}

/// Immutable completion authority captured before a parent worktree can be
/// exited. Live monitor reads use the caller's current context instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTaskOwner {
    workspace: PathBuf,
    session_id: String,
}

static AGENT_PROCESSES: Lazy<DashMap<String, Arc<AgentProcess>>> = Lazy::new(DashMap::new);
static REGISTRATION: Lazy<parking_lot::Mutex<()>> = Lazy::new(|| parking_lot::Mutex::new(()));

fn canonical_scope(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn same_scope(process: &AgentProcess, ctx: &ToolContext) -> bool {
    process.parent_session == ctx.session_id
        && process.workspace == canonical_scope(&crate::worktree::workspace_root(ctx))
}

fn unavailable_scope() -> String {
    "Background agent is not available in this session or workspace".to_string()
}

fn bounded_preview(text: &str, max_bytes: usize, notice: &str) -> String {
    let mut end = text.len().min(max_bytes);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut excerpt = text[..end].to_string();
    if end < text.len() {
        excerpt.push_str(&format!(
            "\n[{notice}; use TaskOutput/EvidenceRead for exact output]"
        ));
    }
    excerpt
}

fn compact_metadata(metadata: Option<&Value>) -> Option<Value> {
    let object = metadata?.as_object()?;
    let mut compact = serde_json::Map::new();
    if let Some(value) = object.get("evidence") {
        if let Some(evidence) = value.as_object() {
            let mut reference = serde_json::Map::new();
            for key in ["id", "bytes", "partial"] {
                if let Some(value) = evidence.get(key) {
                    reference.insert(key.to_string(), value.clone());
                }
            }
            compact.insert("evidence".to_string(), Value::Object(reference));
        }
    }
    for key in [
        "evidence_unavailable",
        "agent_outcome",
        "execution_status",
        "pending_parent_acceptance",
        "parent_acceptance",
    ] {
        if let Some(value) = object.get(key) {
            compact.insert(key.to_string(), value.clone());
        }
    }
    if let Some(Value::Object(provenance)) = object.get("child_provenance") {
        let mut bounded = serde_json::Map::new();
        for key in [
            "child_agent_id",
            "child_session_id",
            "working_dir",
            "worktree_path",
            "child_outcome",
        ] {
            if let Some(value) = provenance.get(key) {
                let value = value
                    .as_str()
                    .map(|value| {
                        bounded_preview(
                            value,
                            TASK_PATH_PREVIEW_BYTES,
                            "provenance preview shortened",
                        )
                    })
                    .map(Value::String)
                    .unwrap_or_else(|| value.clone());
                bounded.insert(key.to_string(), value);
            }
        }
        compact.insert("child_provenance".to_string(), Value::Object(bounded));
    }
    Some(Value::Object(compact))
}

fn retain_agent_result(ctx: &ToolContext, mut result: ToolResult, partial: bool) -> ToolResult {
    let execution_status = result.execution_status();
    let application_receipts = result.take_application_receipts();
    let discovery_receipt = result.take_discovery_receipt();
    let selected_evidence = result.take_evidence_state().map(|state| state.selection);
    let original_metadata = result.metadata.take();
    let typed = if partial
        || execution_status.is_some_and(|status| {
            matches!(
                status,
                crate::ExecutionStatus::NotRun
                    | crate::ExecutionStatus::Cancelled
                    | crate::ExecutionStatus::Unknown
            )
        }) {
        result.take_validated_typed_content();
        None
    } else {
        result.take_validated_typed_content()
    };
    let mut retained = match typed {
        Some((blocks, _, limits, provenance)) => retain_typed_output(
            ctx,
            result.content,
            result.is_error,
            partial,
            blocks,
            provenance,
            &limits,
        ),
        None => retain_output(ctx, result.content, result.is_error, partial),
    };
    if let Some(selection) = selected_evidence {
        if !matches!(selection, crate::EvidenceSelection::InitialPreview) {
            if let Some(state) = retained.take_evidence_state() {
                let (next_offset, done) = match selection {
                    crate::EvidenceSelection::Page { offset: _, end } => {
                        (Some(end), Some(end >= state.bytes as usize))
                    }
                    crate::EvidenceSelection::Query => (None, None),
                    crate::EvidenceSelection::InitialPreview => (None, None),
                };
                retained = retained.with_evidence_state(crate::EvidenceState {
                    selection,
                    fallback: None,
                    next_offset,
                    done,
                    ..state
                });
            }
        }
    }
    if let Some(original) = original_metadata {
        let mut metadata = match original {
            Value::Object(metadata) => metadata,
            other => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
        };
        // Existing evidence may point at a child scope or a shortened
        // preview. Replace it with a fresh parent-scoped capture.
        metadata.remove("evidence");
        metadata.remove("evidence_unavailable");
        if let Some(Value::Object(evidence)) = retained.metadata.take() {
            metadata.extend(evidence);
        }
        retained.metadata = Some(Value::Object(metadata));
    }
    if let Some(receipts) = application_receipts {
        retained = retained.with_application_receipts(receipts);
    }
    if let Some(receipt) = discovery_receipt {
        retained = retained.with_discovery_receipt(receipt);
    }
    if let Some(status) = execution_status {
        retained = retained.with_execution_status(status);
    }
    retained
}

fn evict_oldest_terminal() -> bool {
    let candidate = AGENT_PROCESSES
        .iter()
        .filter_map(|entry| {
            let state = entry.state.try_lock().ok()?;
            state
                .status
                .terminal()
                .then(|| (entry.key().clone(), entry.created_at))
        })
        .min_by_key(|(_, created_at)| *created_at)
        .map(|(id, _)| id);
    candidate
        .map(|id| AGENT_PROCESSES.remove(&id).is_some())
        .unwrap_or(false)
}

/// Return whether an ID belongs to this owned child registry.
pub fn contains(id: &str) -> bool {
    AGENT_PROCESSES.contains_key(id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildCleanup {
    ConfirmedClean,
    CleanupUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ChildShutdown {
    pub repl_sessions: usize,
    pub background_processes: usize,
    pub cleanup: ChildCleanup,
}

/// Stop exact child-owned REPL and background Bash services under one finite
/// deadline. A timeout keeps the caller fail-closed for worktree/pending use.
pub async fn shutdown_child_services(ctx: &ToolContext, timeout: Duration) -> ChildShutdown {
    let deadline = tokio::time::Instant::now() + timeout;
    let repl = match tokio::time::timeout_at(deadline, crate::repl_tool::shutdown_for_context(ctx))
        .await
    {
        Ok(result) => result,
        Err(_) => crate::repl_tool::ReplShutdown {
            sessions: 0,
            cleanup: crate::repl_tool::ReplCleanup::CleanupUnknown,
        },
    };
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    let background = background_processes::shutdown_for_context(ctx, remaining).await;
    let cleanup = if matches!(repl.cleanup, crate::repl_tool::ReplCleanup::ConfirmedClean)
        && matches!(background.cleanup, BackgroundCleanup::ConfirmedClean)
    {
        ChildCleanup::ConfirmedClean
    } else {
        ChildCleanup::CleanupUnknown
    };
    ChildShutdown {
        repl_sessions: repl.sessions,
        background_processes: background.processes,
        cleanup,
    }
}

/// Register a background AgentTool child before spawning its query loop.
struct Registration {
    id: String,
    description: String,
    ctx: ToolContext,
    child_session: String,
    working_dir: PathBuf,
    worktree_path: Option<PathBuf>,
    cancel: CancellationToken,
}

fn register_with_profile(
    registration: Registration,
    pending_writer: bool,
) -> Result<AgentTaskOwner, String> {
    let Registration {
        id,
        description,
        ctx,
        child_session,
        working_dir,
        worktree_path,
        cancel,
    } = registration;
    let _registration = REGISTRATION.lock();
    if AGENT_PROCESSES.contains_key(&id) {
        return Err("Background agent ID collision".to_string());
    }
    if AGENT_PROCESSES.len() >= MAX_AGENT_TASKS && !evict_oldest_terminal() {
        return Err(format!(
            "Background agent limit reached ({} active tasks)",
            MAX_AGENT_TASKS
        ));
    }
    let workspace = canonical_scope(&crate::worktree::workspace_root(&ctx));
    let parent_working_dir = canonical_scope(&ctx.working_dir);
    // A shared child charges the parent scope. An isolated worktree gets an
    // independent scope while remaining a conservative pending writer.
    let pending_scope = worktree_path
        .as_ref()
        .map(|path| mutation::scope_for_workspace(path.clone(), child_session.clone()))
        .unwrap_or_else(|| {
            mutation::scope_for_workspace(workspace.clone(), ctx.session_id.clone())
        });
    let parent_scope = mutation::scope_for_workspace(workspace.clone(), ctx.session_id.clone());
    let pending = pending_writer
        .then(|| {
            mutation::register_pending(pending_scope.clone(), format!("background agent {id}"))
        })
        .transpose()?;
    let parent_pending = if pending_writer && pending_scope != parent_scope {
        match mutation::register_pending(
            parent_scope,
            format!("background agent {id} child-scope barrier"),
        ) {
            Ok(ticket) => Some(ticket),
            Err(error) => {
                if let Some(ticket) = pending.as_ref() {
                    let _ = ticket.confirm_terminal();
                }
                return Err(error);
            }
        }
    } else {
        None
    };
    let owner = AgentTaskOwner {
        workspace: workspace.clone(),
        session_id: ctx.session_id.clone(),
    };
    let process = Arc::new(AgentProcess {
        id: id.clone(),
        description,
        workspace,
        parent_working_dir,
        parent_session: ctx.session_id.clone(),
        child_session,
        working_dir,
        worktree_path,
        created_at: Instant::now(),
        cancel,
        pending,
        parent_pending,
        state: Mutex::new(AgentProcessState {
            status: AgentProcessStatus::Running,
            result: None,
            cancellation_requested: false,
            pending_acceptance_status: None,
        }),
        acceptance: Mutex::new(()),
        changed: Notify::new(),
    });
    use dashmap::mapref::entry::Entry;
    match AGENT_PROCESSES.entry(id) {
        Entry::Occupied(_) => return Err("Background agent ID collision".to_string()),
        Entry::Vacant(entry) => {
            entry.insert(process);
        }
    }
    Ok(owner)
}

/// Register a background child conservatively as a potential writer.
pub fn register(
    id: String,
    description: String,
    ctx: &ToolContext,
    child_session: String,
    working_dir: PathBuf,
    worktree_path: Option<PathBuf>,
    cancel: CancellationToken,
) -> Result<AgentTaskOwner, String> {
    register_with_profile(
        Registration {
            id,
            description,
            ctx: ctx.clone(),
            child_session,
            working_dir,
            worktree_path,
            cancel,
        },
        true,
    )
}

/// Register a child under an app-enforced observation-only profile. Such a
/// child has no pending writer ticket, while its lifecycle remains monitorable.
pub fn register_observation(
    id: String,
    description: String,
    ctx: &ToolContext,
    child_session: String,
    working_dir: PathBuf,
    worktree_path: Option<PathBuf>,
    cancel: CancellationToken,
) -> Result<AgentTaskOwner, String> {
    register_with_profile(
        Registration {
            id,
            description,
            ctx: ctx.clone(),
            child_session,
            working_dir,
            worktree_path,
            cancel,
        },
        false,
    )
}

fn task_value(
    process: &AgentProcess,
    status: AgentProcessStatus,
    result: Option<&ToolResult>,
    cancellation_requested: bool,
    retrieval_status: &str,
) -> Value {
    let source_output = result.map(|result| result.content.as_str());
    let mut output_limit = source_output.map_or(0, |_| TASK_OUTPUT_PREVIEW_BYTES);
    let mut metadata = result.and_then(|result| result.metadata.clone());
    let mut metadata_preview = false;

    for _ in 0..16 {
        let output = source_output
            .map(|output| bounded_preview(output, output_limit, "output preview shortened"));
        let value = json!({
            "id": bounded_preview(&process.id, TASK_ID_PREVIEW_BYTES, "task ID preview shortened"),
            "subject": bounded_preview(&process.description, TASK_TEXT_PREVIEW_BYTES, "subject preview shortened"),
            "description": bounded_preview(&process.description, TASK_TEXT_PREVIEW_BYTES, "description preview shortened"),
            "status": status.as_str(),
            "cancellation_requested": cancellation_requested,
            "owner": bounded_preview(&process.parent_session, TASK_ID_PREVIEW_BYTES, "owner preview shortened"),
            "child_session_id": bounded_preview(&process.child_session, TASK_ID_PREVIEW_BYTES, "child session preview shortened"),
            "working_dir": bounded_preview(&process.working_dir.to_string_lossy(), TASK_PATH_PREVIEW_BYTES, "working directory preview shortened"),
            "worktree_path": process.worktree_path.as_ref().map(|path| bounded_preview(&path.to_string_lossy(), TASK_PATH_PREVIEW_BYTES, "worktree path preview shortened")),
            "metadata": metadata.clone(),
            "metadata_preview": metadata_preview,
            "output": output,
            "output_preview": result.is_some(),
        });
        let envelope = json!({
            "retrieval_status": retrieval_status,
            "task": value.clone(),
        });
        let fits_budget = serde_json::to_vec(&envelope)
            .ok()
            .is_some_and(|serialized| serialized.len() <= TASK_INNER_BUDGET_BYTES)
            && serde_json::to_vec_pretty(&envelope)
                .ok()
                .is_some_and(|serialized| serialized.len() <= TASK_VALUE_BUDGET_BYTES)
            && serde_json::to_string_pretty(&envelope)
                .ok()
                .is_some_and(|content| transport_fits(&process.id, &content));
        if fits_budget {
            return value;
        }
        if output_limit > 256 {
            output_limit /= 2;
        } else if !metadata_preview {
            metadata = compact_metadata(metadata.as_ref());
            metadata_preview = true;
        } else {
            output_limit = 0;
        }
    }
    json!({
        "id": bounded_preview(&process.id, 128, "task ID preview shortened"),
        "subject": "[preview unavailable]",
        "description": "[preview unavailable]",
        "status": status.as_str(),
        "cancellation_requested": cancellation_requested,
        "owner": bounded_preview(&process.parent_session, 128, "owner preview shortened"),
        "metadata": compact_metadata(metadata.as_ref()),
        "metadata_preview": true,
        "output": result.map(|_| "[output preview unavailable]"),
        "output_preview": result.is_some(),
    })
}

fn transport_fits(tool_call_id: &str, content: &str) -> bool {
    let wrapper = json!({
        "role": "tool",
        "tool_call_id": bounded_preview(tool_call_id, TASK_ID_PREVIEW_BYTES, "tool call ID preview shortened"),
        "content": content,
    });
    serde_json::to_vec(&wrapper)
        .ok()
        .is_some_and(|serialized| serialized.len() <= TASK_VALUE_BUDGET_BYTES)
        && serde_json::to_vec_pretty(&wrapper)
            .ok()
            .is_some_and(|serialized| serialized.len() <= TASK_VALUE_BUDGET_BYTES)
}

fn bounded_tool_content(content: &str) -> String {
    let mut limit = TASK_OUTPUT_PREVIEW_BYTES;
    for _ in 0..16 {
        let candidate = bounded_preview(content, limit, "agent output preview shortened");
        if transport_fits("", &candidate) {
            return candidate;
        }
        if limit == 0 {
            break;
        }
        limit /= 2;
    }
    "[output preview unavailable; use EvidenceRead for exact output]".to_string()
}

fn visible_evidence_notice(metadata: Option<&Value>) -> &'static str {
    if metadata
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("evidence_unavailable"))
        .is_some()
    {
        "[Exact evidence was not retained; the original child output may be unavailable.]

"
    } else {
        ""
    }
}

fn visible_evidence_reference(metadata: Option<&Value>) -> String {
    let Some(Value::Object(evidence)) = metadata
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("evidence"))
    else {
        return visible_evidence_notice(metadata).to_string();
    };
    let Some(id) = evidence.get("id").and_then(Value::as_str) else {
        return visible_evidence_notice(metadata).to_string();
    };
    let bytes = evidence
        .get("bytes")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let partial = evidence
        .get("partial")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    format!(
        "[EvidenceRead id={} retrieves {} retained UTF-8 bytes; partial={}]

",
        bounded_preview(id, TASK_ID_PREVIEW_BYTES, "evidence ID preview shortened"),
        bytes,
        partial
    )
}

/// Serialize the TaskOutput envelope and enforce the same bound after the
/// provider escapes its content as a tool-result JSON string.
pub fn serialize_task_output(task_id: &str, output: Value) -> String {
    let content = serde_json::to_string_pretty(&output).unwrap_or_default();
    if transport_fits(task_id, &content) {
        return content;
    }

    let retrieval_status = output
        .get("retrieval_status")
        .and_then(Value::as_str)
        .unwrap_or("success");
    let metadata = output
        .get("task")
        .and_then(|task| task.get("metadata"))
        .and_then(|metadata| compact_metadata(Some(metadata)));
    let fallback = json!({
        "retrieval_status": retrieval_status,
        "task": {
            "id": bounded_preview(task_id, 128, "task ID preview shortened"),
            "status": "preview_unavailable",
            "metadata": metadata,
            "metadata_preview": true,
            "output": "[output preview unavailable; use EvidenceRead for exact output]",
            "output_preview": true,
        },
    });
    let content = serde_json::to_string_pretty(&fallback).unwrap_or_else(|_| {
        "{\"retrieval_status\":\"success\",\"task\":{\"status\":\"preview_unavailable\"}}"
            .to_string()
    });
    debug_assert!(transport_fits(task_id, &content));
    content
}

async fn snapshot(process: &AgentProcess) -> (AgentProcessStatus, Option<ToolResult>, bool) {
    let state = process.state.lock().await;
    (
        state.status,
        state.result.clone(),
        state.cancellation_requested,
    )
}

/// Publish the typed query outcome and retain its full result in the parent
/// EvidenceStore scope before the child worktree may be removed.
pub async fn finish(
    id: &str,
    owner: &AgentTaskOwner,
    ctx: &ToolContext,
    outcome: AgentTaskStatus,
    cleanup: ChildCleanup,
    result: ToolResult,
) -> Result<(), String> {
    let process = AGENT_PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if process.parent_session != owner.session_id || process.workspace != owner.workspace {
        return Err(unavailable_scope());
    }
    let mut evidence_ctx = ctx.clone();
    // The live parent context may still point into a worktree that has just
    // been removed. Evidence belongs to the immutable registration workspace.
    evidence_ctx.working_dir = process.workspace.clone();
    let mut state = process.state.lock().await;
    // Terminal results are immutable. A duplicate completion can race with a
    // late query callback, but it must not overwrite evidence or lifecycle
    // status that the first completion published.
    if state.status.terminal() {
        return Ok(());
    }
    let cancellation_requested = state.cancellation_requested || ctx.cancel_token.is_cancelled();
    let pending_parent_acceptance = result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("pending_parent_acceptance"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let requires_parent_acceptance = pending_parent_acceptance
        && matches!(
            outcome,
            AgentTaskStatus::Completed | AgentTaskStatus::CompletedAfterCancellationRequest
        )
        && matches!(cleanup, ChildCleanup::ConfirmedClean);
    let mut result = retain_agent_result(
        &evidence_ctx,
        result,
        outcome.partial() || !matches!(cleanup, ChildCleanup::ConfirmedClean),
    );
    let cleanup_error = if matches!(cleanup, ChildCleanup::ConfirmedClean) {
        let mut errors = Vec::new();
        let tickets = if requires_parent_acceptance {
            [process.pending.as_ref(), None]
        } else {
            [process.pending.as_ref(), process.parent_pending.as_ref()]
        };
        for ticket in tickets.into_iter().flatten() {
            if let Err(error) = ticket.confirm_terminal() {
                errors.push(error);
            }
        }
        (!errors.is_empty()).then(|| errors.join("; "))
    } else {
        Some("Child-owned service cleanup remains unknown".to_string())
    };
    let completed_status = if cleanup_error.is_some() {
        // A completed child with unconfirmed owned cleanup cannot be exposed
        // as completed. Keep the pending ticket as a fail-closed barrier and
        // publish a failed lifecycle status until durable cleanup is proven.
        AgentTaskStatus::Failed
    } else if cancellation_requested && outcome == AgentTaskStatus::Completed {
        AgentTaskStatus::CompletedAfterCancellationRequest
    } else {
        outcome
    };
    let pending_acceptance_status = (requires_parent_acceptance
        && matches!(
            completed_status,
            AgentTaskStatus::Completed | AgentTaskStatus::CompletedAfterCancellationRequest
        ))
    .then_some(completed_status);
    let terminal_status = if pending_acceptance_status.is_some() {
        AgentTaskStatus::AwaitingParentAcceptance
    } else {
        completed_status
    };
    if completed_status == AgentTaskStatus::CompletedAfterCancellationRequest {
        let mut metadata = match result.metadata.take() {
            Some(Value::Object(metadata)) => metadata,
            Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
            None => serde_json::Map::new(),
        };
        metadata.insert("cancellation_requested".into(), Value::Bool(true));
        result.metadata = Some(Value::Object(metadata));
    }
    if let Some(error) = cleanup_error {
        let mut metadata = match result.metadata.take() {
            Some(Value::Object(metadata)) => metadata,
            Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
            None => serde_json::Map::new(),
        };
        metadata.insert("cleanup_status".into(), Value::String("unknown".into()));
        metadata.insert("cleanup_error".into(), Value::String(error));
        metadata.insert(
            "requested_status".into(),
            Value::String(outcome.as_str().into()),
        );
        result.metadata = Some(Value::Object(metadata));
        result.is_error = true;
        result = result.with_execution_status(crate::ExecutionStatus::Unknown);
    }
    state.status = AgentProcessStatus::Terminal(terminal_status);
    state.pending_acceptance_status = pending_acceptance_status;
    state.result = Some(result);
    drop(state);
    process.changed.notify_waiters();
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ArtifactEntry {
    path: String,
    kind: &'static str,
    sha256: String,
    bytes: u64,
    executable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactSnapshot {
    entries: Vec<ArtifactEntry>,
    digest: String,
    bytes: u64,
}

fn validate_relative_artifact_path(path: &str) -> Result<PathBuf, String> {
    if path.is_empty() || path.len() > MAX_ACCEPTANCE_PATH_BYTES {
        return Err("Child artifact path is empty or exceeds the acceptance limit".into());
    }
    let path = PathBuf::from(path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("Child artifact path is outside the isolated worktree".into());
    }
    Ok(path)
}

fn parse_nul_paths(bytes: &[u8]) -> Result<Vec<PathBuf>, String> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if !bytes.ends_with(&[0]) {
        return Err("Git artifact inventory was truncated".into());
    }
    bytes[..bytes.len() - 1]
        .split(|byte| *byte == 0)
        .map(|raw| {
            let path = std::str::from_utf8(raw)
                .map_err(|_| "Non-UTF-8 child artifact paths are unsupported".to_string())?;
            validate_relative_artifact_path(path)
        })
        .collect()
}

fn ensure_no_symlink_ancestors(root: &Path, relative: &Path) -> Result<(), String> {
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for component in components.iter().take(components.len().saturating_sub(1)) {
        let Component::Normal(component) = component else {
            return Err("Invalid artifact path component".into());
        };
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err("Artifact path traverses a symbolic link".into())
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(format!("Could not inspect artifact path: {error}")),
        }
    }
    Ok(())
}

fn executable(metadata: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

fn read_bounded_file(path: &Path, expected: &std::fs::Metadata) -> Result<Vec<u8>, String> {
    use std::io::Read;

    if expected.len() > MAX_ACCEPTANCE_FILE_BYTES {
        return Err("Child artifact exceeds the per-file acceptance limit".into());
    }
    let file =
        std::fs::File::open(path).map_err(|error| format!("Could not open artifact: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("Could not inspect opened artifact: {error}"))?;
    if !opened.is_file() || opened.len() > MAX_ACCEPTANCE_FILE_BYTES {
        return Err("Artifact changed type or size while being inspected".into());
    }
    let mut bytes = Vec::with_capacity(opened.len().min(MAX_ACCEPTANCE_FILE_BYTES) as usize);
    file.take(MAX_ACCEPTANCE_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Could not read artifact: {error}"))?;
    if bytes.len() as u64 > MAX_ACCEPTANCE_FILE_BYTES {
        return Err("Artifact grew beyond the per-file acceptance limit".into());
    }
    Ok(bytes)
}

fn artifact_entry(root: &Path, relative: &Path) -> Result<ArtifactEntry, String> {
    ensure_no_symlink_ancestors(root, relative)?;
    let path = root.join(relative);
    let display = relative
        .to_str()
        .ok_or_else(|| "Non-UTF-8 artifact path is unsupported".to_string())?
        .to_string();
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ArtifactEntry {
                path: display,
                kind: "deleted",
                sha256: sha256_hex(&[]),
                bytes: 0,
                executable: false,
            })
        }
        Err(error) => return Err(format!("Could not inspect artifact: {error}")),
    };
    if metadata.file_type().is_symlink() {
        let target = std::fs::read_link(&path)
            .map_err(|error| format!("Could not read artifact symbolic link: {error}"))?;
        let encoded = target.as_os_str().as_encoded_bytes();
        if encoded.len() as u64 > MAX_ACCEPTANCE_FILE_BYTES {
            return Err("Artifact symbolic link exceeds the acceptance limit".into());
        }
        return Ok(ArtifactEntry {
            path: display,
            kind: "symlink",
            sha256: sha256_hex(encoded),
            bytes: encoded.len() as u64,
            executable: false,
        });
    }
    if !metadata.is_file() {
        return Err("Child artifact type is unsupported".into());
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("Could not canonicalize artifact: {error}"))?;
    if !canonical.starts_with(root) {
        return Err("Artifact resolves outside the accepted workspace".into());
    }
    let bytes = read_bounded_file(&path, &metadata)?;
    Ok(ArtifactEntry {
        path: display,
        kind: "file",
        sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
        executable: executable(&metadata),
    })
}

async fn owned_git_stdout(
    cwd: &Path,
    arguments: &[&str],
    cancel: &CancellationToken,
) -> Result<Vec<u8>, String> {
    let git = which::which("git")
        .map_err(|_| "Git is unavailable for parent artifact verification".to_string())?
        .canonicalize()
        .map_err(|_| "Git executable identity could not be captured".to_string())?;
    let mut argv = Vec::with_capacity(arguments.len() + 1);
    argv.push(git.to_string_lossy().into_owned());
    argv.extend(arguments.iter().map(|argument| (*argument).to_string()));
    let (child, mut group) = crate::command_capture::spawn_owned_argv(&argv, cwd)
        .map_err(|error| format!("Could not start Git artifact inventory: {error}"))?;
    let captured = capture_owned_with_cancel(child, ACCEPTANCE_COMMAND_TIMEOUT, cancel, &mut group)
        .await
        .map_err(|error| format!("Could not capture Git artifact inventory: {error}"))?;
    if captured
        .status
        .as_ref()
        .is_none_or(|status| !status.success())
        || captured.partial
        || captured.timed_out
        || captured.cancelled
        || captured.error.is_some()
        || captured.cleanup != ProcessCleanup::ConfirmedClean
    {
        return Err("Git artifact inventory did not complete cleanly".into());
    }
    Ok(captured.stdout)
}

async fn artifact_snapshot(
    child: &Path,
    parent: &Path,
    cancel: &CancellationToken,
) -> Result<ArtifactSnapshot, String> {
    let child = child
        .canonicalize()
        .map_err(|_| "Child worktree is unavailable".to_string())?;
    let parent = parent
        .canonicalize()
        .map_err(|_| "Parent workspace is unavailable".to_string())?;
    if child == parent {
        return Err("Child worktree is not isolated from the parent workspace".into());
    }
    let tracked = owned_git_stdout(
        &child,
        &["diff", "--name-only", "-z", "--no-renames", "HEAD", "--"],
        cancel,
    )
    .await?;
    let untracked = owned_git_stdout(
        &child,
        &["ls-files", "--others", "--exclude-standard", "-z"],
        cancel,
    )
    .await?;
    let mut paths = BTreeSet::new();
    paths.extend(parse_nul_paths(&tracked)?);
    paths.extend(parse_nul_paths(&untracked)?);
    if paths.is_empty() {
        return Err("Child produced no changed artifacts to accept".into());
    }
    if paths.len() > MAX_ACCEPTANCE_FILES {
        return Err("Child artifact count exceeds the acceptance limit".into());
    }
    let mut entries = Vec::with_capacity(paths.len());
    let mut total = 0_u64;
    for relative in paths {
        let child_entry = artifact_entry(&child, &relative)?;
        let parent_entry = artifact_entry(&parent, &relative)?;
        if child_entry != parent_entry {
            return Err(format!(
                "Parent artifact does not match the isolated child: {}",
                child_entry.path
            ));
        }
        total = total
            .checked_add(child_entry.bytes)
            .ok_or_else(|| "Child artifact byte count overflowed".to_string())?;
        if total > MAX_ACCEPTANCE_TOTAL_BYTES {
            return Err("Child artifacts exceed the aggregate acceptance limit".into());
        }
        entries.push(child_entry);
    }
    let serialized = serde_json::to_vec(&entries)
        .map_err(|error| format!("Could not serialize child artifact manifest: {error}"))?;
    Ok(ArtifactSnapshot {
        entries,
        digest: sha256_hex(&serialized),
        bytes: total,
    })
}

fn acceptance_checker(
    process: &AgentProcess,
    ctx: &ToolContext,
    command: &str,
) -> Result<TrustedCheckerRegistration, String> {
    let mut matches = ctx
        .trusted_checker_catalog
        .registrations()
        .filter(|registration| {
            registration.workspace() == process.workspace
                && registration.session_id() == process.parent_session
                && registration.invocation().command() == command
                && registration
                    .invocation()
                    .cwd()
                    .starts_with(&process.parent_working_dir)
                && registration.invocation().is_foreground()
                && registration.invocation().is_available()
        });
    let first = matches.next().cloned().ok_or_else(|| {
        "Verification command is not registered for this parent scope".to_string()
    })?;
    if matches.any(|candidate| candidate.invocation() != first.invocation()) {
        return Err("Verification command is ambiguous in the parent checker catalog".into());
    }
    Ok(first)
}

async fn run_acceptance_checker(
    registration: &TrustedCheckerRegistration,
    ctx: &ToolContext,
) -> Result<crate::command_capture::CapturedCommand, String> {
    if ctx.cancel_token.is_cancelled() {
        return Err("Parent acceptance was cancelled before verification".into());
    }
    let invocation = registration.invocation();
    let program = invocation
        .program()
        .ok_or_else(|| "Registered verification interpreter is unavailable".to_string())?;
    let mut command = Command::new(program);
    match invocation.shell() {
        TrustedCheckerShell::Bash => {
            command.args(["--noprofile", "--norc", "-c"]);
        }
        TrustedCheckerShell::PowerShell => {
            command.args(["-NoProfile", "-NonInteractive", "-Command"]);
        }
    }
    command
        .arg(invocation.command())
        .current_dir(invocation.cwd())
        .env_clear()
        .envs(invocation.environment())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let child = command
        .spawn()
        .map_err(|error| format!("Could not start registered parent verification: {error}"))?;
    let mut group = ProcessGroupGuard::new(child.id());
    capture_owned_with_cancel(child, invocation.timeout(), &ctx.cancel_token, &mut group)
        .await
        .map_err(|error| format!("Could not capture registered parent verification: {error}"))
}

fn checker_output(captured: &crate::command_capture::CapturedCommand) -> String {
    let mut output = String::from_utf8_lossy(&captured.stdout).into_owned();
    if !captured.stderr.is_empty() {
        output.push_str("\n[stderr]\n");
        output.push_str(&String::from_utf8_lossy(&captured.stderr));
    }
    if output.is_empty() {
        output.push_str("(no output)");
    }
    output
}

/// Accept one isolated mutating child only after its changed artifacts match
/// the parent workspace and a fresh host-registered checker succeeds there.
/// Failed attempts never clear the durable parent mutation barrier.
pub async fn accept(
    id: &str,
    verification_command: &str,
    ctx: &ToolContext,
) -> Result<ToolResult, String> {
    let process = AGENT_PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{id}' not found"))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }
    let _acceptance = process.acceptance.lock().await;
    let accepted_status = {
        let state = process.state.lock().await;
        if state.status != AgentProcessStatus::Terminal(AgentTaskStatus::AwaitingParentAcceptance) {
            return Err(format!(
                "Task '{id}' is not awaiting parent acceptance (status: {})",
                state.status.as_str()
            ));
        }
        state
            .pending_acceptance_status
            .ok_or_else(|| "Task acceptance status is unavailable".to_string())?
    };
    let worktree = process
        .worktree_path
        .as_deref()
        .ok_or_else(|| "Task has no isolated worktree to verify".to_string())?;
    if process.parent_pending.is_none() {
        return Err("Task has no parent acceptance barrier".into());
    }
    let registration = acceptance_checker(&process, ctx, verification_command)?;
    let before =
        artifact_snapshot(worktree, &process.parent_working_dir, &ctx.cancel_token).await?;
    let captured = run_acceptance_checker(&registration, ctx).await?;
    let exit_code = captured
        .status
        .as_ref()
        .and_then(std::process::ExitStatus::code);
    let clean = captured.cleanup == ProcessCleanup::ConfirmedClean;
    let complete = clean
        && !captured.partial
        && !captured.timed_out
        && !captured.cancelled
        && captured.error.is_none();
    let command_succeeded = complete && exit_code == Some(0);
    let mut verification = retain_output(
        ctx,
        checker_output(&captured),
        !command_succeeded,
        !complete,
    );
    let evidence_retained = verification.metadata.as_ref().is_some_and(|metadata| {
        metadata.get("evidence").is_some_and(|evidence| {
            evidence.get("id").and_then(Value::as_str).is_some()
                && evidence.get("partial").and_then(Value::as_bool) == Some(false)
        })
    });
    let success = command_succeeded && evidence_retained;
    if !success {
        let mut metadata = match verification.metadata.take() {
            Some(Value::Object(metadata)) => metadata,
            Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
            None => serde_json::Map::new(),
        };
        metadata.insert(
            "parent_acceptance".into(),
            json!({
                "status": "verification_failed",
                "task_id": id,
                "checker_id": registration.id(),
                "checker_revision": registration.revision(),
                "checker_digest": registration.digest(),
                "exit_code": exit_code,
                "process_cleanup_confirmed": clean,
                "partial": !complete,
                "evidence_retained": evidence_retained,
                "artifact_sha256": before.digest,
            }),
        );
        verification.metadata = Some(Value::Object(metadata));
        verification.is_error = true;
        if !complete {
            verification = verification.with_execution_status(crate::ExecutionStatus::Unknown);
        }
        return Ok(verification);
    }
    let after = artifact_snapshot(worktree, &process.parent_working_dir, &ctx.cancel_token).await?;
    if after != before {
        verification.is_error = true;
        verification.content = "Child or parent artifacts changed during verification".into();
        verification = verification.with_execution_status(crate::ExecutionStatus::Unknown);
        return Ok(verification);
    }
    let confirm_process = Arc::clone(&process);
    tokio::time::timeout(
        ACCEPTANCE_CONFIRM_TIMEOUT,
        tokio::task::spawn_blocking(move || {
            confirm_process
                .parent_pending
                .as_ref()
                .ok_or_else(|| "Task parent acceptance barrier disappeared".to_string())?
                .confirm_terminal()
        }),
    )
    .await
    .map_err(|_| "Parent acceptance confirmation timed out".to_string())?
    .map_err(|_| "Parent acceptance confirmation task failed".to_string())??;

    let evidence = verification
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("evidence"))
        .cloned();
    let acceptance = json!({
        "status": "accepted",
        "task_id": id,
        "artifact_count": after.entries.len(),
        "artifact_bytes": after.bytes,
        "artifact_sha256": after.digest,
        "checker_id": registration.id(),
        "checker_revision": registration.revision(),
        "checker_digest": registration.digest(),
        "verification_evidence": evidence,
    });
    {
        let mut state = process.state.lock().await;
        if state.status != AgentProcessStatus::Terminal(AgentTaskStatus::AwaitingParentAcceptance)
            || state.pending_acceptance_status != Some(accepted_status)
        {
            return Err("Task acceptance state changed during verification".into());
        }
        let result = state
            .result
            .as_mut()
            .ok_or_else(|| "Task result is unavailable".to_string())?;
        let mut metadata = match result.metadata.take() {
            Some(Value::Object(metadata)) => metadata,
            Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
            None => serde_json::Map::new(),
        };
        metadata.insert("pending_parent_acceptance".into(), Value::Bool(false));
        metadata.insert("parent_acceptance".into(), acceptance.clone());
        result.metadata = Some(Value::Object(metadata));
        state.status = AgentProcessStatus::Terminal(accepted_status);
        state.pending_acceptance_status = None;
    }
    process.changed.notify_waiters();

    verification.content = serde_json::to_string_pretty(&acceptance).unwrap_or_default();
    verification.is_error = false;
    let mut metadata = match verification.metadata.take() {
        Some(Value::Object(metadata)) => metadata,
        Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
        None => serde_json::Map::new(),
    };
    metadata.insert("parent_acceptance".into(), acceptance);
    verification.metadata = Some(Value::Object(metadata));
    Ok(verification)
}

/// Retain foreground AgentTool output in the parent scope and cap the visible
/// content while preserving the exact result in EvidenceStore.
pub fn retain_foreground_result(
    ctx: &ToolContext,
    outcome: AgentTaskStatus,
    result: ToolResult,
) -> ToolResult {
    let mut retained = retain_agent_result(ctx, result, outcome.partial());
    let evidence_reference = visible_evidence_reference(retained.metadata.as_ref());
    // Keep the returned ToolResult within the same metadata contract as
    // TaskOutput. The complete child content remains in EvidenceStore.
    if let Some(metadata) = retained.metadata.take() {
        let mut metadata = compact_metadata(Some(&metadata)).unwrap_or_else(|| json!({}));
        if outcome == AgentTaskStatus::CompletedAfterCancellationRequest {
            if let Some(object) = metadata.as_object_mut() {
                object.insert("cancellation_requested".into(), Value::Bool(true));
            }
        }
        retained.metadata = Some(metadata);
    }
    retained.content = bounded_tool_content(&format!("{}{}", evidence_reference, retained.content));
    retained
}

/// Return a scoped child output. Blocking is bounded to avoid hanging a query.
pub async fn output(id: &str, ctx: &ToolContext, block: bool) -> Result<Value, String> {
    let process = AGENT_PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }
    let deadline = Instant::now() + TASK_OUTPUT_WAIT;
    loop {
        let (status, result, cancellation_requested) = snapshot(&process).await;
        if status.terminal() || !block {
            return Ok(json!({
                "retrieval_status": if status.terminal() { "success" } else { "not_ready" },
                "task": task_value(
                    &process,
                    status,
                    result.as_ref(),
                    cancellation_requested,
                    if status.terminal() { "success" } else { "not_ready" },
                ),
            }));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(json!({
                "retrieval_status": "not_ready",
                "task": task_value(
                    &process,
                    status,
                    result.as_ref(),
                    cancellation_requested,
                    "not_ready",
                ),
            }));
        }
        let notified = process.changed.notified();
        if tokio::time::timeout(remaining, notified).await.is_err() {
            let (status, result, cancellation_requested) = snapshot(&process).await;
            return Ok(json!({
                "retrieval_status": if status.terminal() { "success" } else { "not_ready" },
                "task": task_value(
                    &process,
                    status,
                    result.as_ref(),
                    cancellation_requested,
                    if status.terminal() { "success" } else { "not_ready" },
                ),
            }));
        }
    }
}

pub async fn status(id: &str, ctx: &ToolContext) -> Result<Value, String> {
    let process = AGENT_PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }
    let (status, result, cancellation_requested) = snapshot(&process).await;
    Ok(task_value(
        &process,
        status,
        result.as_ref(),
        cancellation_requested,
        "success",
    ))
}

pub async fn list(ctx: &ToolContext) -> Vec<Value> {
    let mut tasks = Vec::new();
    for entry in AGENT_PROCESSES.iter() {
        let process = entry.value().clone();
        drop(entry);
        if !same_scope(&process, ctx) {
            continue;
        }
        let (status, result, cancellation_requested) = snapshot(&process).await;
        tasks.push((
            process.created_at,
            task_value(
                &process,
                status,
                result.as_ref(),
                cancellation_requested,
                "success",
            ),
        ));
    }
    tasks.sort_by_key(|(created_at, _)| *created_at);
    tasks.into_iter().map(|(_, task)| task).collect()
}

/// Request cancellation through the exact child token after checking scope.
/// The task remains `cancellation_requested` until the query loop publishes a
/// typed `Cancelled` outcome.
pub async fn stop(id: &str, ctx: &ToolContext) -> Result<(), String> {
    let process = AGENT_PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }
    let mut state = process.state.lock().await;
    if state.status.terminal() {
        return Err(format!(
            "Task '{}' is already {}",
            id,
            state.status.as_str()
        ));
    }
    if !state.cancellation_requested {
        process.cancel.cancel();
        state.cancellation_requested = true;
        state.status = AgentProcessStatus::CancellationRequested;
    }
    drop(state);
    process.changed.notify_waiters();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use crate::{
        RegistrationSpec, Tool, TrustedCheckerCatalogBuilder, TrustedCheckerInvocation,
        TrustedCheckerOperation, TrustedCheckerSource,
    };
    use std::process::Command as StdCommand;
    use tempfile::tempdir;

    fn child_registration(
        id: &str,
        ctx: &ToolContext,
        cancel: CancellationToken,
    ) -> Result<AgentTaskOwner, String> {
        register(
            id.to_string(),
            "test child".to_string(),
            ctx,
            format!("child-{id}"),
            ctx.working_dir.clone(),
            None,
            cancel,
        )
    }

    fn run_git(cwd: &Path, arguments: &[&str]) {
        let output = StdCommand::new("git")
            .args(arguments)
            .current_dir(cwd)
            .output()
            .expect("run git fixture command");
        assert!(
            output.status.success(),
            "git fixture command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn acceptance_fixture(command: &str) -> (tempfile::TempDir, PathBuf, PathBuf, ToolContext) {
        let root = tempdir().expect("temporary fixture root");
        let parent = root.path().join("parent");
        let child = root.path().join("child");
        std::fs::create_dir(&parent).expect("create parent repository");
        run_git(&parent, &["init", "-q"]);
        std::fs::write(parent.join("artifact.txt"), "before\n").expect("write base artifact");
        run_git(&parent, &["add", "artifact.txt"]);
        run_git(
            &parent,
            &[
                "-c",
                "user.name=Forge Tests",
                "-c",
                "user.email=forge@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
        );
        run_git(
            &parent,
            &[
                "worktree",
                "add",
                "--detach",
                child.to_str().expect("UTF-8 fixture path"),
                "HEAD",
            ],
        );
        let mut ctx = allow_all_context(parent.clone());
        ctx.session_id = format!("agent-parent-accept-{}", uuid::Uuid::new_v4());
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            command,
            &parent,
            Duration::from_secs(2),
        )
        .expect("registered checker invocation");
        let registration = crate::TrustedCheckerRegistration::new(RegistrationSpec {
            id: "agent-parent-acceptance".to_string(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: parent.clone(),
            session_id: ctx.session_id.clone(),
            target: parent.join("artifact.txt"),
            operation: TrustedCheckerOperation::Edit,
            invocation,
        })
        .expect("registered checker");
        let mut catalog = TrustedCheckerCatalogBuilder::new();
        catalog.register(registration).expect("checker catalog");
        ctx.trusted_checker_catalog = Arc::new(catalog.build());
        (root, parent, child, ctx)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn child_shutdown_joins_owned_background_service_before_completion() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = format!("child-shutdown-{}", uuid::Uuid::new_v4());
        let id = background_processes::spawn(
            "sleep 30".to_string(),
            root.path().to_path_buf(),
            30_000,
            &ctx,
        )
        .expect("spawn owned child service");
        let shutdown = shutdown_child_services(&ctx, Duration::from_secs(2)).await;
        assert_eq!(shutdown.background_processes, 1);
        assert_eq!(shutdown.cleanup, ChildCleanup::ConfirmedClean);
        let task = background_processes::output(&id, &ctx, true)
            .await
            .expect("retrieve stopped child service");
        assert_eq!(task["task"]["status"], "cancelled");
    }

    #[tokio::test]
    async fn owned_lifecycle_retains_parent_evidence_and_typed_status() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = "agent-task-lifecycle".to_string();
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &ctx, CancellationToken::new()).expect("register child");

        let partial = "partial 🦀\u{0001}".repeat(20_000);
        let result = ToolResult::error(partial).with_metadata(json!({
            "adversarial": "\u{0001}".repeat(100_000),
            "child_provenance": {
                "child_agent_id": "agent-".repeat(2_000),
                "working_dir": "/tmp/".to_string() + &"path/".repeat(2_000),
            },
        }));
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::MaxTokens,
            ChildCleanup::ConfirmedClean,
            result,
        )
        .await
        .expect("publish child result");
        let value = output(&id, &ctx, true)
            .await
            .expect("retrieve child result");
        assert_eq!(value["retrieval_status"], "success");
        assert_eq!(value["task"]["status"], "max_tokens");
        assert!(
            serde_json::to_vec(&value).expect("serialize task").len() <= TASK_VALUE_BUDGET_BYTES
        );
        assert!(
            serde_json::to_vec_pretty(&value)
                .expect("pretty serialize task")
                .len()
                <= TASK_VALUE_BUDGET_BYTES
        );
        let evidence_id = value["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("parent evidence ID");
        let (_, exact) = crate::evidence::EvidenceStore::for_context(&ctx)
            .expect("parent evidence store")
            .read(evidence_id)
            .expect("parent evidence remains readable");
        assert!(exact.starts_with("partial 🦀"));

        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn completed_child_retains_typed_blocks_and_exact_display_text() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = format!("agent-typed-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &ctx, CancellationToken::new()).expect("register child");
        let blocks = vec![
            claurst_core::types::ContentBlock::Text {
                text: "original typed text".to_string(),
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
        let result = ToolResult::success("display summary")
            .with_typed_content(blocks.clone(), &claurst_core::media::MediaLimits::default())
            .expect("validated typed blocks");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            result,
        )
        .await
        .expect("publish child result");
        let process = AGENT_PROCESSES.get(&id).expect("registered process");
        let retained = process.state.lock().await.result.clone().expect("result");
        assert!(retained
            .content
            .starts_with("display summary\n\n[EvidenceRead id="));
        assert_eq!(
            serde_json::to_vec(retained.typed_blocks().unwrap()).unwrap(),
            serde_json::to_vec(&blocks).unwrap()
        );
        let evidence_id = retained.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .expect("evidence ID");
        let (_, exact, restored) = crate::evidence::EvidenceStore::for_context(&ctx)
            .expect("evidence store")
            .read_with_typed(evidence_id)
            .expect("typed evidence");
        assert_eq!(exact, "original typed text");
        assert_eq!(
            serde_json::to_vec(&restored.unwrap().blocks).unwrap(),
            serde_json::to_vec(&blocks).unwrap()
        );
        drop(process);
        AGENT_PROCESSES.remove(&id);
    }

    #[test]
    fn child_retention_carries_private_page_selection_to_parent_reference() {
        let root = tempdir().expect("temporary workspace");
        let ctx = allow_all_context(root.path().to_path_buf());
        let result = ToolResult::success("display")
            .with_typed_content(
                vec![claurst_core::types::ContentBlock::Text {
                    text: "full child payload".to_string(),
                }],
                &claurst_core::media::MediaLimits::default(),
            )
            .expect("validated typed blocks")
            .with_evidence_state(crate::EvidenceState {
                id: "child-reference".to_string(),
                bytes: 18,
                partial: false,
                workspace: None,
                session: None,
                selection: crate::EvidenceSelection::Page { offset: 0, end: 5 },
                fallback: None,
                next_offset: Some(5),
                done: Some(false),
            });
        let retained = retain_agent_result(&ctx, result, false);
        let state = retained.evidence_state().expect("parent evidence state");
        assert_eq!(
            state.selection,
            crate::EvidenceSelection::Page { offset: 0, end: 5 }
        );
        assert_eq!(state.next_offset, Some(5));
        assert_eq!(state.done, Some(false));
    }

    #[test]
    fn child_retention_preserves_scoped_discovery_receipt() {
        let root = tempdir().expect("temporary workspace");
        let ctx = allow_all_context(root.path().to_path_buf());
        let tools = crate::all_tools();
        let view = crate::ToolCatalogView::from_tools(&ctx, &tools).expect("catalog");
        let receipt = crate::DiscoveryReceipt::new(&view, vec!["Read".to_string()]);
        let mut retained = retain_agent_result(
            &ctx,
            ToolResult::success("child discovery").with_discovery_receipt(receipt),
            false,
        );
        let retained_receipt = retained
            .take_discovery_receipt()
            .expect("child evidence keeps receipt");
        assert_eq!(
            retained_receipt.resolve_names(&view, &ctx).unwrap(),
            vec!["Read"]
        );
    }

    #[test]
    fn child_retention_preserves_typed_exceptional_status_and_revokes_receipts() {
        let root = tempdir().expect("temporary workspace");
        let ctx = allow_all_context(root.path().to_path_buf());
        let tools = crate::all_tools();
        let view = crate::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        for status in [
            crate::ExecutionStatus::NotRun,
            crate::ExecutionStatus::Cancelled,
            crate::ExecutionStatus::Unknown,
        ] {
            let receipt = crate::DiscoveryReceipt::new(&view, vec!["Read".to_string()]);
            let mut retained = retain_agent_result(
                &ctx,
                ToolResult::error("uncertain")
                    .with_execution_status(status)
                    .with_discovery_receipt(receipt),
                false,
            );
            assert_eq!(retained.execution_status(), Some(status));
            assert!(retained.take_discovery_receipt().is_none());
        }
    }

    #[tokio::test]
    async fn isolated_background_child_holds_parent_scope_ticket_until_acceptance() {
        let parent = tempdir().expect("parent workspace");
        let child = tempdir().expect("child worktree");
        let mut ctx = allow_all_context(parent.path().to_path_buf());
        ctx.session_id = format!("agent-parent-ticket-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = register(
            id.clone(),
            "isolated child".to_string(),
            &ctx,
            format!("child-{id}"),
            child.path().to_path_buf(),
            Some(child.path().to_path_buf()),
            CancellationToken::new(),
        )
        .expect("register isolated child");
        let process = AGENT_PROCESSES.get(&id).expect("registered process");
        assert!(process.parent_pending.is_some());
        let child_ctx = {
            let mut child_ctx = ctx.clone();
            child_ctx.working_dir = child.path().to_path_buf();
            child_ctx.session_id = format!("child-{id}");
            child_ctx
        };
        assert!(mutation::pending_for_scope(&ctx));
        assert!(mutation::pending_for_scope(&child_ctx));
        drop(process);
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("child complete").with_metadata(json!({
                "pending_parent_acceptance": true,
            })),
        )
        .await
        .expect("publish child completion");
        let awaiting = output(&id, &ctx, false).await.expect("task output");
        assert_eq!(
            awaiting["task"]["status"],
            AgentTaskStatus::AwaitingParentAcceptance.as_str()
        );
        assert!(mutation::pending_for_scope(&ctx));
        assert!(!mutation::pending_for_scope(&child_ctx));
        let process = AGENT_PROCESSES.remove(&id).expect("remove process").1;
        if let Some(ticket) = process.parent_pending.as_ref() {
            ticket.confirm_terminal().expect("clear parent ticket");
        }
    }

    #[tokio::test]
    async fn parent_acceptance_requires_matching_artifacts_and_fresh_registered_check() {
        let (_root, parent, child, mut ctx) = acceptance_fixture("test -f artifact.txt");
        std::fs::write(child.join("artifact.txt"), "accepted\n").expect("write child artifact");
        std::fs::write(parent.join("artifact.txt"), "accepted\n")
            .expect("integrate child artifact in parent");
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let child_session = format!("child-{id}");
        let owner = register(
            id.clone(),
            "isolated child".to_string(),
            &ctx,
            child_session,
            child.clone(),
            Some(child),
            CancellationToken::new(),
        )
        .expect("register isolated child");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("child complete").with_metadata(json!({
                "pending_parent_acceptance": true,
            })),
        )
        .await
        .expect("publish child completion");

        let accepted = crate::tasks::TaskAcceptTool
            .execute(
                json!({
                    "task_id": id,
                    "verification_command": "test -f artifact.txt",
                }),
                &ctx,
            )
            .await;
        assert!(!accepted.is_error, "{accepted:?}");
        let acceptance: Value = serde_json::from_str(&accepted.content).expect("acceptance JSON");
        assert_eq!(acceptance["status"], "accepted");
        assert_eq!(acceptance["artifact_count"], 1);
        assert_eq!(acceptance["artifact_bytes"], 9);
        let evidence_id = accepted.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .expect("fresh parent verification evidence");
        let (_, evidence) = crate::evidence::EvidenceStore::for_context(&ctx)
            .expect("parent evidence store")
            .read(evidence_id)
            .expect("parent checker evidence");
        assert_eq!(evidence, "(no output)");
        let completed = output(&id, &ctx, false)
            .await
            .expect("accepted task output");
        assert_eq!(completed["task"]["status"], "completed");
        assert_eq!(
            completed["task"]["metadata"]["pending_parent_acceptance"],
            false
        );
        assert!(!mutation::pending_for_scope(&ctx));

        let duplicate = crate::tasks::TaskAcceptTool
            .execute(
                json!({
                    "task_id": id,
                    "verification_command": "test -f artifact.txt",
                }),
                &ctx,
            )
            .await;
        assert!(duplicate.is_error);
        AGENT_PROCESSES.remove(&id);
        ctx.trusted_checker_catalog = Arc::new(crate::TrustedCheckerCatalog::empty());
    }

    #[tokio::test]
    async fn mismatched_parent_artifact_keeps_acceptance_barrier() {
        let (_root, _parent, child, ctx) = acceptance_fixture("true");
        std::fs::write(child.join("artifact.txt"), "not integrated\n")
            .expect("write child artifact");
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = register(
            id.clone(),
            "isolated child".to_string(),
            &ctx,
            format!("child-{id}"),
            child.clone(),
            Some(child),
            CancellationToken::new(),
        )
        .expect("register isolated child");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("child complete").with_metadata(json!({
                "pending_parent_acceptance": true,
            })),
        )
        .await
        .expect("publish child completion");

        let rejected = crate::tasks::TaskAcceptTool
            .execute(json!({"task_id": id, "verification_command": "true"}), &ctx)
            .await;
        assert!(rejected.is_error);
        assert!(rejected.content.contains("does not match"));
        let awaiting = output(&id, &ctx, false).await.expect("task output");
        assert_eq!(awaiting["task"]["status"], "awaiting_parent_acceptance");
        assert!(mutation::pending_for_scope(&ctx));
        let process = AGENT_PROCESSES.remove(&id).expect("remove process").1;
        process
            .parent_pending
            .as_ref()
            .unwrap()
            .confirm_terminal()
            .expect("clear parent test barrier");
    }

    #[tokio::test]
    async fn failed_isolated_child_is_not_accepted_and_releases_parent_barrier() {
        let parent = tempdir().expect("parent workspace");
        let child = tempdir().expect("child worktree");
        let mut ctx = allow_all_context(parent.path().to_path_buf());
        ctx.session_id = format!("agent-parent-failed-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = register(
            id.clone(),
            "failed isolated child".to_string(),
            &ctx,
            format!("child-{id}"),
            child.path().to_path_buf(),
            Some(child.path().to_path_buf()),
            CancellationToken::new(),
        )
        .expect("register isolated child");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Failed,
            ChildCleanup::ConfirmedClean,
            ToolResult::error("child failed").with_metadata(json!({
                "pending_parent_acceptance": true,
            })),
        )
        .await
        .expect("publish failed child");

        let value = output(&id, &ctx, false).await.expect("failed child output");
        assert_eq!(value["task"]["status"], "failed");
        assert!(!mutation::pending_for_scope(&ctx));
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn failed_parent_checker_retains_barrier_and_exact_evidence() {
        let (_root, parent, child, ctx) = acceptance_fixture("printf verifier-failed; exit 7");
        std::fs::write(child.join("artifact.txt"), "integrated\n").expect("write child artifact");
        std::fs::write(parent.join("artifact.txt"), "integrated\n")
            .expect("integrate child artifact");
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = register(
            id.clone(),
            "isolated child".to_string(),
            &ctx,
            format!("child-{id}"),
            child.clone(),
            Some(child),
            CancellationToken::new(),
        )
        .expect("register isolated child");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("child complete").with_metadata(json!({
                "pending_parent_acceptance": true,
            })),
        )
        .await
        .expect("publish child completion");

        let failed = crate::tasks::TaskAcceptTool
            .execute(
                json!({
                    "task_id": id,
                    "verification_command": "printf verifier-failed; exit 7",
                }),
                &ctx,
            )
            .await;
        assert!(failed.is_error);
        assert_eq!(
            failed.metadata.as_ref().unwrap()["parent_acceptance"]["status"],
            "verification_failed"
        );
        let evidence_id = failed.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .expect("checker evidence");
        let (_, evidence) = crate::evidence::EvidenceStore::for_context(&ctx)
            .expect("parent evidence store")
            .read(evidence_id)
            .expect("failed checker evidence");
        assert_eq!(evidence, "verifier-failed");
        assert!(mutation::pending_for_scope(&ctx));
        let process = AGENT_PROCESSES.remove(&id).expect("remove process").1;
        process
            .parent_pending
            .as_ref()
            .unwrap()
            .confirm_terminal()
            .expect("clear parent test barrier");
    }

    #[tokio::test]
    async fn confirmed_cleanup_clears_pending_ticket_for_non_success() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = format!("agent-clean-failure-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &ctx, CancellationToken::new()).expect("register child");
        assert!(mutation::pending_for_scope(&ctx));
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Failed,
            ChildCleanup::ConfirmedClean,
            ToolResult::error("child failed after cleanup"),
        )
        .await
        .expect("publish failed child");
        assert!(!mutation::pending_for_scope(&ctx));
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn unknown_cleanup_retains_pending_ticket_for_non_success() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = format!("agent-unknown-failure-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &ctx, CancellationToken::new()).expect("register child");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Failed,
            ChildCleanup::CleanupUnknown,
            ToolResult::error("child failed before cleanup proof"),
        )
        .await
        .expect("publish uncertain child");
        let value = output(&id, &ctx, true).await.expect("retrieve child");
        assert_eq!(value["task"]["metadata"]["cleanup_status"], "unknown");
        assert!(mutation::pending_for_scope(&ctx));
        let process = AGENT_PROCESSES.remove(&id).expect("remove process").1;
        process
            .pending
            .as_ref()
            .unwrap()
            .confirm_terminal()
            .expect("clear test ticket");
    }

    #[tokio::test]
    async fn terminal_agent_result_is_immutable_after_duplicate_finish() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = format!("agent-double-finish-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &ctx, CancellationToken::new()).expect("register child");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("first result"),
        )
        .await
        .expect("publish first result");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Failed,
            ChildCleanup::ConfirmedClean,
            ToolResult::error("late duplicate"),
        )
        .await
        .expect("ignore duplicate result");
        let output = output(&id, &ctx, true).await.expect("retrieve result");
        assert_eq!(output["task"]["status"], "completed");
        assert!(output["task"]["output"]
            .as_str()
            .unwrap()
            .contains("first result"));
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn cleanup_persist_failure_is_published_as_failed_unknown() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = format!("agent-cleanup-failure-{}", uuid::Uuid::new_v4());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &ctx, CancellationToken::new()).expect("register child");
        let ticket_id = AGENT_PROCESSES
            .get(&id)
            .and_then(|process| {
                process
                    .pending
                    .as_ref()
                    .map(|ticket| ticket.id().to_string())
            })
            .expect("registered pending writer ticket");
        let durable_record = claurst_core::config::Settings::config_dir()
            .join("mutation-pending")
            .join(format!("{ticket_id}.json"));
        std::fs::remove_file(&durable_record)
            .expect("remove durable record to force cleanup failure");

        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("child completed before durable cleanup failed"),
        )
        .await
        .expect("publish failed cleanup status");
        let value = output(&id, &ctx, true)
            .await
            .expect("retrieve failed result");
        assert_eq!(value["task"]["status"], "failed");
        assert_eq!(value["task"]["metadata"]["cleanup_status"], "unknown");
        assert_eq!(value["task"]["metadata"]["requested_status"], "completed");
        assert_eq!(value["task"]["metadata"]["execution_status"], "unknown");
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn task_output_execute_bounds_escaped_tool_result_and_keeps_evidence() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = "agent-task-escaped-output".to_string();
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = register(
            id.clone(),
            "subject \u{0001} ".repeat(4_000),
            &ctx,
            "child-output".to_string(),
            PathBuf::from(format!("/tmp/{}", "deep/".repeat(2_000))),
            Some(PathBuf::from(format!("/tmp/{}", "worktree/".repeat(2_000)))),
            CancellationToken::new(),
        )
        .expect("register child");
        let exact_output = "quote=\" slash=\\ control=\u{0001} ".repeat(20_000);
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::MaxTokens,
            ChildCleanup::ConfirmedClean,
            ToolResult::error(exact_output.clone()).with_metadata(json!({
                "adversarial": "\u{0001}".repeat(100_000),
                "child_provenance": {
                    "working_dir": "path/".repeat(4_000),
                    "worktree_path": "worktree/".repeat(4_000),
                },
            })),
        )
        .await
        .expect("publish child result");

        let result = crate::tasks::TaskOutputTool
            .execute(json!({"task_id": id, "block": true}), &ctx)
            .await;
        assert!(!result.is_error);
        let output: Value = serde_json::from_str(&result.content).expect("TaskOutput JSON");
        assert_eq!(output["retrieval_status"], "success");
        let evidence_id = output["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("evidence ID");
        assert!(
            serde_json::to_vec(&output).expect("compact output").len() <= TASK_VALUE_BUDGET_BYTES
        );
        assert!(
            serde_json::to_vec_pretty(&output)
                .expect("pretty output")
                .len()
                <= TASK_VALUE_BUDGET_BYTES
        );
        let tool_result = json!({
            "role": "tool",
            "tool_call_id": "call_agent_output",
            "content": result.content,
        });
        assert!(
            serde_json::to_vec(&tool_result)
                .expect("escaped tool result")
                .len()
                <= TASK_VALUE_BUDGET_BYTES
        );
        assert!(
            serde_json::to_vec_pretty(&tool_result)
                .expect("pretty escaped tool result")
                .len()
                <= TASK_VALUE_BUDGET_BYTES
        );
        let (_, exact) = crate::evidence::EvidenceStore::for_context(&ctx)
            .expect("parent evidence store")
            .read(evidence_id)
            .expect("exact evidence");
        assert_eq!(exact, exact_output);
        AGENT_PROCESSES.remove(&id);
    }

    #[test]
    fn foreground_result_bounds_escaped_content_and_keeps_exact_evidence() {
        let root = tempdir().expect("temporary workspace");
        let ctx = allow_all_context(root.path().to_path_buf());
        let exact_output = "quote=\" slash=\\ control=\u{0001} ".repeat(20_000);
        let result = retain_foreground_result(
            &ctx,
            AgentTaskStatus::MaxTokens,
            ToolResult::error(exact_output.clone()),
        );
        let tool_result = json!({
            "role": "tool",
            "tool_call_id": "call_foreground_agent",
            "content": result.content,
        });
        assert!(
            serde_json::to_vec(&tool_result)
                .expect("escaped foreground result")
                .len()
                <= TASK_VALUE_BUDGET_BYTES
        );
        assert!(
            serde_json::to_vec_pretty(&tool_result)
                .expect("pretty escaped foreground result")
                .len()
                <= TASK_VALUE_BUDGET_BYTES
        );
        let evidence_id = result.metadata.as_ref().expect("metadata")["evidence"]["id"]
            .as_str()
            .expect("evidence ID");
        assert!(result
            .content
            .contains(&format!("EvidenceRead id={evidence_id}")));
        let (_, exact) = crate::evidence::EvidenceStore::for_context(&ctx)
            .expect("parent evidence store")
            .read(evidence_id)
            .expect("exact evidence");
        assert_eq!(exact, exact_output);
    }

    #[test]
    fn unavailable_evidence_is_visible_in_foreground_prefix() {
        let notice = visible_evidence_reference(Some(&json!({
            "evidence_unavailable": true,
        })));
        assert!(notice.contains("Exact evidence was not retained"));
    }

    #[tokio::test]
    async fn completion_uses_registered_parent_after_live_worktree_disappears() {
        let parent = tempdir().expect("parent workspace");
        let live_child = tempdir().expect("live worktree");
        let mut parent_ctx = allow_all_context(parent.path().to_path_buf());
        parent_ctx.session_id = "agent-task-worktree-exit".to_string();
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner =
            child_registration(&id, &parent_ctx, CancellationToken::new()).expect("register child");

        let mut live_ctx = parent_ctx.clone();
        live_ctx.working_dir = live_child.path().join("removed-worktree");
        // The parent ExitWorktree path leaves the captured background context
        // pointing at a directory that no longer exists.
        assert!(!live_ctx.working_dir.exists());
        finish(
            &id,
            &owner,
            &live_ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("finished after parent exit"),
        )
        .await
        .expect("finish against immutable parent owner");

        let value = output(&id, &parent_ctx, true)
            .await
            .expect("retrieve result");
        let evidence_id = value["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("parent evidence ID");
        let (_, exact) = crate::evidence::EvidenceStore::for_context(&parent_ctx)
            .expect("parent evidence store")
            .read(evidence_id)
            .expect("parent evidence remains readable");
        assert_eq!(exact, "finished after parent exit");
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn scope_denies_foreign_session_and_workspace() {
        let workspace = tempdir().expect("temporary workspace");
        let other_workspace = tempdir().expect("other workspace");
        let mut owner = allow_all_context(workspace.path().to_path_buf());
        owner.session_id = "agent-owner".to_string();
        let mut foreign_session = allow_all_context(workspace.path().to_path_buf());
        foreign_session.session_id = "agent-foreign".to_string();
        let mut foreign_workspace = allow_all_context(other_workspace.path().to_path_buf());
        foreign_workspace.session_id = owner.session_id.clone();
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let completion_owner =
            child_registration(&id, &owner, CancellationToken::new()).expect("register child");
        finish(
            &id,
            &completion_owner,
            &owner,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("done"),
        )
        .await
        .expect("finish child");

        assert!(output(&id, &foreign_session, false)
            .await
            .expect_err("foreign session denied")
            .contains("not available"));
        assert!(status(&id, &foreign_workspace)
            .await
            .expect_err("foreign workspace denied")
            .contains("not available"));
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn cancellation_request_is_distinct_until_cancelled_outcome() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = "agent-task-cancel".to_string();
        let cancel = CancellationToken::new();
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = child_registration(&id, &ctx, cancel.clone()).expect("register child");
        stop(&id, &ctx).await.expect("request cancellation");
        assert!(cancel.is_cancelled());
        assert_eq!(
            status(&id, &ctx).await.expect("status")["status"],
            "cancellation_requested"
        );
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Cancelled,
            ChildCleanup::ConfirmedClean,
            ToolResult::error("cancelled"),
        )
        .await
        .expect("publish cancellation");
        assert_eq!(
            status(&id, &ctx).await.expect("status")["status"],
            "cancelled"
        );
        assert!(!mutation::pending_for_scope(&ctx));
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn late_success_after_cancellation_request_is_explicit() {
        let root = tempdir().expect("temporary workspace");
        let mut ctx = allow_all_context(root.path().to_path_buf());
        ctx.session_id = "agent-task-late-cancel".to_string();
        let cancel = CancellationToken::new();
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let owner = child_registration(&id, &ctx, cancel).expect("register child");
        stop(&id, &ctx).await.expect("request cancellation");
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("late success"),
        )
        .await
        .expect("publish late success");
        let task = status(&id, &ctx).await.expect("status");
        assert_eq!(task["status"], "completed_after_cancellation_request");
        assert_eq!(task["metadata"]["cancellation_requested"], true);
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn duplicate_registration_preserves_original_owner_and_token() {
        let root = tempdir().expect("temporary workspace");
        let ctx = allow_all_context(root.path().to_path_buf());
        let id = format!("agent-test-{}", uuid::Uuid::new_v4());
        let first_cancel = CancellationToken::new();
        let second_cancel = CancellationToken::new();
        let owner = child_registration(&id, &ctx, first_cancel.clone()).expect("register child");
        let duplicate = register(
            id.clone(),
            "replacement child".to_string(),
            &ctx,
            "replacement-session".to_string(),
            ctx.working_dir.clone(),
            None,
            second_cancel.clone(),
        );
        assert!(duplicate.is_err());
        assert!(!first_cancel.is_cancelled());
        assert!(!second_cancel.is_cancelled());
        assert_eq!(
            status(&id, &ctx).await.expect("status")["status"],
            "running"
        );
        finish(
            &id,
            &owner,
            &ctx,
            AgentTaskStatus::Completed,
            ChildCleanup::ConfirmedClean,
            ToolResult::success("original remains"),
        )
        .await
        .expect("finish original");
        AGENT_PROCESSES.remove(&id);
    }

    #[tokio::test]
    async fn terminal_cap_evicts_oldest_record() {
        let root = tempdir().expect("temporary workspace");
        let ctx = allow_all_context(root.path().to_path_buf());
        let prefix = format!("agent-cap-{}-", uuid::Uuid::new_v4());
        let mut ids = Vec::new();
        for index in 0..MAX_AGENT_TASKS {
            let id = format!("{prefix}{index}");
            let owner = child_registration(&id, &ctx, CancellationToken::new())
                .expect("register within cap");
            finish(
                &id,
                &owner,
                &ctx,
                AgentTaskStatus::Completed,
                ChildCleanup::ConfirmedClean,
                ToolResult::success("done"),
            )
            .await
            .expect("finish within cap");
            ids.push(id);
        }
        let extra = format!("{prefix}extra");
        child_registration(&extra, &ctx, CancellationToken::new()).expect("evict terminal");
        assert!(!contains(&ids[0]));
        assert!(contains(&extra));
        for id in ids.into_iter().skip(1) {
            AGENT_PROCESSES.remove(&id);
        }
        AGENT_PROCESSES.remove(&extra);
    }
}
