//! Owned lifecycle for background shell processes.
//!
//! The core task registry is a display mirror only.  This module owns the
//! cancellation handle, output/evidence, and scope of each process so task
//! control cannot cross session or workspace boundaries.

#[cfg(test)]
use crate::command_capture::capture;
use crate::command_capture::{
    capture_owned_with_cancel, spawn_owned_argv, ProcessCleanup, ProcessGroupGuard,
};
use crate::evidence::retain_output;
use crate::mutation::{self, PendingMutationTicket};
use crate::{ToolContext, ToolResult};
use claurst_core::tasks::{global_registry, BackgroundTask, TaskStatus as CoreTaskStatus};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Child;
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Maximum number of retained background processes, including completed ones.
/// Keeping terminal entries bounded preserves TaskOutput retrieval without
/// permitting unbounded process metadata growth.
pub const MAX_BACKGROUND_PROCESSES: usize = 64;

// Capture cleanup can drain inherited pipes for up to two seconds after a
// cancellation or deadline. Keep retrieval finite while allowing that result
// to be published before returning not_ready.
const TASK_OUTPUT_WAIT: Duration = Duration::from_secs(3);
const DISPLAY_OUTPUT_CHARS: usize = 4096;
// TaskOutput is itself serialized JSON. Keep its nested previews well below
// the generic evidence display limit so shortening never cuts an escaped JSON
// string in half. Exact output remains available through the nested evidence
// reference in task.metadata.
const TASK_OUTPUT_PREVIEW_BYTES: usize = 8 * 1024;
const TASK_COMMAND_PREVIEW_BYTES: usize = 2 * 1024;
const TASK_VALUE_BUDGET_BYTES: usize = 24 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundCleanup {
    ConfirmedClean,
    CleanupUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackgroundShutdown {
    pub processes: usize,
    pub cleanup: BackgroundCleanup,
}

impl ProcessStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn terminal(self) -> bool {
        !matches!(self, Self::Running)
    }
}

struct ProcessState {
    status: ProcessStatus,
    result: Option<ToolResult>,
    cleanup_confirmed: bool,
}

struct OwnedProcess {
    id: String,
    command: String,
    workspace: PathBuf,
    cwd: PathBuf,
    session_id: String,
    created_at: Instant,
    cancel: CancellationToken,
    pending: PendingMutationTicket,
    state: Mutex<ProcessState>,
    changed: Notify,
    #[cfg(test)]
    cleanup_wait_gate: Option<Arc<CleanupWaitGate>>,
}

#[cfg(test)]
struct CleanupWaitGate {
    reached: Notify,
    release: tokio::sync::Semaphore,
}

static PROCESSES: Lazy<DashMap<String, Arc<OwnedProcess>>> = Lazy::new(DashMap::new);
static PROCESS_REGISTRATION: Lazy<parking_lot::Mutex<()>> =
    Lazy::new(|| parking_lot::Mutex::new(()));

fn canonical_scope(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn same_scope(process: &OwnedProcess, ctx: &ToolContext) -> bool {
    process.session_id == ctx.session_id
        && process.workspace == canonical_scope(&crate::worktree::workspace_root(ctx))
}

fn unavailable_scope() -> String {
    "Background task is not available in this session or workspace".to_string()
}

/// Evict the oldest terminal process when the retained-process cap is full.
/// Running processes are never evicted; callers hold PROCESS_REGISTRATION.
fn evict_oldest_terminal() -> bool {
    let candidate = PROCESSES
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
        .map(|id| {
            let removed = PROCESSES.remove(&id).is_some();
            if removed {
                global_registry().remove(&id);
            }
            removed
        })
        .unwrap_or(false)
}

/// Return whether an ID belongs to the owned background-process registry.
/// Callers use this to distinguish those IDs from manual TASK_STORE entries.
pub fn contains(id: &str) -> bool {
    PROCESSES.contains_key(id)
}

fn display_excerpt(text: &str) -> String {
    bounded_preview(text, DISPLAY_OUTPUT_CHARS, "display output shortened")
}

fn bounded_preview(text: &str, max_bytes: usize, notice: &str) -> String {
    let mut end = text.len().min(max_bytes);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut excerpt = text[..end].to_string();
    if end < text.len() {
        excerpt.push_str(&format!("\n[{notice}; use EvidenceRead for exact output]"));
    }
    excerpt
}

fn compact_metadata(metadata: Option<&Value>) -> Option<Value> {
    let object = metadata?.as_object()?;
    let mut compact = serde_json::Map::new();
    if let Some(evidence) = object.get("evidence") {
        let mut reference = serde_json::Map::new();
        if let Some(evidence_object) = evidence.as_object() {
            for key in ["id", "bytes", "partial"] {
                if let Some(value) = evidence_object.get(key) {
                    reference.insert(key.to_string(), value.clone());
                }
            }
        }
        if !reference.is_empty() {
            compact.insert("evidence".to_string(), Value::Object(reference));
        }
    }
    if let Some(status) = object.get("execution_status") {
        compact.insert("execution_status".to_string(), status.clone());
    }
    Some(Value::Object(compact))
}

fn child_for(command: &str, cwd: &Path) -> std::io::Result<(Child, ProcessGroupGuard)> {
    let argv = if cfg!(windows) {
        vec!["cmd".to_string(), "/C".to_string(), command.to_string()]
    } else {
        vec!["bash".to_string(), "-c".to_string(), command.to_string()]
    };
    spawn_owned_argv(&argv, cwd)
}

fn normalized_capture(
    stdout: &[u8],
    stderr: &[u8],
    status: Option<&std::process::ExitStatus>,
    timed_out: bool,
    cancelled: bool,
    partial: bool,
    capture_error: Option<&str>,
) -> (String, bool, bool) {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    let mut output = stdout.into_owned();
    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("[stderr]\n");
        output.push_str(&stderr);
    }
    if output.is_empty() {
        output.push_str("(no output)");
    }

    let failed = timed_out
        || cancelled
        || capture_error.is_some()
        || status.is_none_or(|status| !status.success());
    if cancelled {
        output = format!("Command cancelled; capture incomplete\n{output}");
    } else if timed_out {
        output = format!("Command timed out; capture incomplete\n{output}");
    } else if let Some(error) = capture_error {
        output = format!("Command output capture failed: {error}; capture incomplete\n{output}");
    } else if let Some(status) = status.filter(|status| !status.success()) {
        output = format!(
            "Command exited with code {}\n{output}",
            status.code().unwrap_or(-1)
        );
    }
    (
        output,
        failed,
        partial || timed_out || cancelled || capture_error.is_some() || status.is_none(),
    )
}

fn mirror_terminal(id: &str, status: ProcessStatus, result: &ToolResult) {
    global_registry().append_output(id, &display_excerpt(&result.content));
    let core_status = match status {
        ProcessStatus::Running => CoreTaskStatus::Running,
        ProcessStatus::Completed => CoreTaskStatus::Completed,
        ProcessStatus::Failed => CoreTaskStatus::Failed("background command failed".into()),
        ProcessStatus::Cancelled => CoreTaskStatus::Cancelled,
    };
    global_registry().update_status(id, core_status);
}

fn cleanup_unknown(mut result: ToolResult, error: Option<String>) -> ToolResult {
    let mut metadata = match result.metadata.take() {
        Some(Value::Object(metadata)) => metadata,
        Some(other) => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
        None => serde_json::Map::new(),
    };
    metadata.insert("cleanup_status".into(), Value::String("unknown".into()));
    if let Some(error) = error {
        metadata.insert("cleanup_error".into(), Value::String(error));
    }
    result.metadata = Some(Value::Object(metadata));
    result.with_execution_status(crate::ExecutionStatus::Unknown)
}

async fn run_process(process: Arc<OwnedProcess>, ctx: ToolContext, timeout: Duration) {
    let result = match child_for(&process.command, &process.cwd) {
        Ok((child, mut group)) => {
            match capture_owned_with_cancel(child, timeout, &process.cancel, &mut group).await {
                Ok(captured) => {
                    let mut cleanup_error = None;
                    let cleanup_confirmed = captured.cleanup == ProcessCleanup::ConfirmedClean;
                    if cleanup_confirmed {
                        if let Err(error) = process.pending.confirm_terminal() {
                            cleanup_error = Some(error);
                        }
                    }
                    if captured.timed_out || captured.cancelled || captured.error.is_some() {
                        group.kill_now();
                    } else {
                        group.disarm();
                    }
                    let (text, is_error, partial) = normalized_capture(
                        &captured.stdout,
                        &captured.stderr,
                        captured.status.as_ref(),
                        captured.timed_out,
                        captured.cancelled,
                        captured.partial,
                        captured.error.as_deref(),
                    );
                    let result = retain_output(&ctx, text, is_error, partial);
                    let cleanup_failed = cleanup_error.is_some();
                    let terminal_cleanup = cleanup_confirmed && !cleanup_failed;
                    let result = if terminal_cleanup {
                        result
                    } else {
                        cleanup_unknown(result, cleanup_error)
                    };
                    if captured.cancelled {
                        (
                            ProcessStatus::Cancelled,
                            result.with_execution_status(crate::ExecutionStatus::Unknown),
                            terminal_cleanup,
                        )
                    } else if captured.timed_out
                        || captured.error.is_some()
                        || captured.status.is_none()
                        || !terminal_cleanup
                    {
                        (
                            ProcessStatus::Failed,
                            result.with_execution_status(crate::ExecutionStatus::Unknown),
                            terminal_cleanup,
                        )
                    } else if is_error {
                        (ProcessStatus::Failed, result, terminal_cleanup)
                    } else {
                        (ProcessStatus::Completed, result, terminal_cleanup)
                    }
                }
                Err(error) => {
                    group.kill_now();
                    let result = retain_output(
                        &ctx,
                        format!("Failed to capture background command output: {error}"),
                        true,
                        true,
                    );
                    (
                        ProcessStatus::Failed,
                        cleanup_unknown(result, Some(error.to_string())),
                        false,
                    )
                }
            }
        }
        Err(error) => {
            let mut cleanup_error = None;
            if let Err(confirm_error) = process.pending.confirm_terminal() {
                cleanup_error = Some(confirm_error);
            }
            let terminal_cleanup = cleanup_error.is_none();
            let result = retain_output(
                &ctx,
                format!("Failed to spawn background command: {error}"),
                true,
                true,
            );
            (
                ProcessStatus::Failed,
                cleanup_unknown(result, cleanup_error),
                terminal_cleanup,
            )
        }
    };

    {
        let mut state = process.state.lock().await;
        state.status = result.0;
        state.result = Some(result.1.clone());
        state.cleanup_confirmed = result.2;
    }
    mirror_terminal(&process.id, result.0, &result.1);
    process.changed.notify_waiters();
}

/// Register and launch a background process under the caller's session/workspace.
pub fn spawn(
    command: String,
    cwd: PathBuf,
    timeout_ms: u64,
    ctx: &ToolContext,
) -> Result<String, String> {
    #[cfg(windows)]
    {
        return Err(
            "Background shell processes are unavailable on Windows until Job Object cleanup is supported"
                .to_string(),
        );
    }

    let _registration = PROCESS_REGISTRATION.lock();
    if PROCESSES.len() >= MAX_BACKGROUND_PROCESSES && !evict_oldest_terminal() {
        return Err(format!(
            "Background process limit reached ({} active tasks)",
            MAX_BACKGROUND_PROCESSES
        ));
    }

    let workspace = canonical_scope(&crate::worktree::workspace_root(ctx));
    let cwd = canonical_scope(&cwd);
    let id = format!("bg-{}", Uuid::new_v4());
    // Arbitrary background shell is a potential writer. Register before
    // spawning so a foreground check cannot accept a proof during the race.
    // The ticket remains pending unless owned process-group cleanup is proven.
    let pending = mutation::register_pending(
        mutation::scope_for_workspace(workspace.clone(), ctx.session_id.clone()),
        format!("background shell {id}"),
    )?;
    let process = Arc::new(OwnedProcess {
        id: id.clone(),
        command: command.clone(),
        workspace: workspace.clone(),
        cwd,
        session_id: ctx.session_id.clone(),
        created_at: Instant::now(),
        cancel: ctx.cancel_token.child_token(),
        pending,
        state: Mutex::new(ProcessState {
            status: ProcessStatus::Running,
            result: None,
            cleanup_confirmed: false,
        }),
        changed: Notify::new(),
        #[cfg(test)]
        cleanup_wait_gate: None,
    });
    PROCESSES.insert(id.clone(), process.clone());

    let mut mirror = BackgroundTask::new(format!("bg: {}", display_excerpt(&command)));
    mirror.id = id.clone();
    mirror.cancel_token = Some(process.cancel.clone());
    global_registry().register(mirror);

    let mut evidence_ctx = ctx.clone();
    evidence_ctx.working_dir = workspace;
    tokio::spawn(run_process(
        process,
        evidence_ctx,
        Duration::from_millis(timeout_ms),
    ));
    Ok(id)
}

fn task_value(process: &OwnedProcess, status: ProcessStatus, result: Option<&ToolResult>) -> Value {
    let source_output = result.map(|result| result.content.as_str());
    let mut output_limit = source_output.map_or(0, |_| TASK_OUTPUT_PREVIEW_BYTES);
    let mut command_limit = TASK_COMMAND_PREVIEW_BYTES;
    let mut metadata = result.and_then(|result| result.metadata.clone());
    let mut metadata_preview = false;

    for _ in 0..16 {
        let output = source_output
            .map(|output| bounded_preview(output, output_limit, "output preview shortened"));
        let command = bounded_preview(&process.command, command_limit, "command preview shortened");
        let value = json!({
            "id": process.id,
            "subject": format!("bg: {command}"),
            "description": command,
            "status": status.as_str(),
            "owner": bounded_preview(&process.session_id, 512, "owner preview shortened"),
            "metadata": metadata.clone(),
            "metadata_preview": metadata_preview,
            "output": output,
            "output_preview": result.is_some(),
            "workspace": bounded_preview(
                &process.workspace.to_string_lossy(),
                1024,
                "workspace preview shortened"
            ),
        });
        if serde_json::to_vec(&value)
            .map(|serialized| serialized.len() <= TASK_VALUE_BUDGET_BYTES)
            .unwrap_or(false)
        {
            return value;
        }

        if output_limit > 256 {
            output_limit /= 2;
        } else if command_limit > 128 {
            command_limit /= 2;
        } else if !metadata_preview {
            metadata = compact_metadata(metadata.as_ref());
            metadata_preview = true;
        } else {
            output_limit = 0;
            command_limit = 0;
        }
    }

    // The fields above are all bounded, so this fallback is only defensive.
    // Keep the evidence reference and status metadata if a future field changes
    // the budget calculation.
    json!({
        "id": process.id,
        "subject": "bg: [preview unavailable]",
        "description": "[preview unavailable]",
        "status": status.as_str(),
        "owner": bounded_preview(&process.session_id, 128, "owner preview shortened"),
        "metadata": compact_metadata(metadata.as_ref()),
        "metadata_preview": true,
        "output": result.map(|_| "[output preview unavailable]"),
        "output_preview": result.is_some(),
        "workspace": bounded_preview(
            &process.workspace.to_string_lossy(),
            128,
            "workspace preview shortened"
        ),
    })
}

async fn snapshot(process: &OwnedProcess) -> (ProcessStatus, Option<ToolResult>) {
    let state = process.state.lock().await;
    (state.status, state.result.clone())
}

async fn wait_for_terminal_cleanup(process: &OwnedProcess, deadline: Instant) -> bool {
    loop {
        // notify_waiters retains a wakeup for an already-created future. Create
        // it before reading state to cover completion between the read and wait.
        let notified = process.changed.notified();
        {
            let state = process.state.lock().await;
            if state.status.terminal() {
                return state.cleanup_confirmed;
            }
        }
        #[cfg(test)]
        if let Some(gate) = &process.cleanup_wait_gate {
            gate.reached.notify_one();
            gate.release.acquire().await.unwrap().forget();
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return false;
        }
        if tokio::time::timeout(remaining, notified).await.is_err() {
            return false;
        }
    }
}

/// Cancel and join background processes owned by this exact workspace/session.
/// The timeout is one total deadline for the whole scope; a timeout or an
/// unconfirmed owned group remains a cleanup barrier.
pub async fn shutdown_for_context(ctx: &ToolContext, timeout: Duration) -> BackgroundShutdown {
    let owned: Vec<Arc<OwnedProcess>> = PROCESSES
        .iter()
        .filter(|entry| same_scope(entry.value(), ctx))
        .map(|entry| entry.value().clone())
        .collect();
    for process in &owned {
        process.cancel.cancel();
        global_registry().update_status(&process.id, CoreTaskStatus::Cancelled);
    }
    let deadline = Instant::now() + timeout;
    let mut clean = true;
    for process in &owned {
        clean &= wait_for_terminal_cleanup(process, deadline).await;
    }
    BackgroundShutdown {
        processes: owned.len(),
        cleanup: if clean {
            BackgroundCleanup::ConfirmedClean
        } else {
            BackgroundCleanup::CleanupUnknown
        },
    }
}

/// Return a scoped task output. Blocking is bounded to avoid hanging a query.
pub async fn output(id: &str, ctx: &ToolContext, block: bool) -> Result<Value, String> {
    let process = PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }

    let deadline = Instant::now() + TASK_OUTPUT_WAIT;
    loop {
        let (status, result) = snapshot(&process).await;
        if status.terminal() || !block {
            return Ok(json!({
                "retrieval_status": if status.terminal() { "success" } else { "not_ready" },
                "task": task_value(&process, status, result.as_ref()),
            }));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(json!({
                "retrieval_status": "not_ready",
                "task": task_value(&process, status, result.as_ref()),
            }));
        }
        let notified = process.changed.notified();
        if tokio::time::timeout(remaining, notified).await.is_err() {
            let (status, result) = snapshot(&process).await;
            return Ok(json!({
                "retrieval_status": if status.terminal() { "success" } else { "not_ready" },
                "task": task_value(&process, status, result.as_ref()),
            }));
        }
    }
}

/// List only records owned by the caller's session and workspace.
pub async fn list(ctx: &ToolContext) -> Vec<Value> {
    let mut tasks = Vec::new();
    for entry in PROCESSES.iter() {
        let process = entry.value().clone();
        if !same_scope(&process, ctx) {
            continue;
        }
        let (status, result) = snapshot(&process).await;
        tasks.push((
            process.created_at,
            task_value(&process, status, result.as_ref()),
        ));
    }
    tasks.sort_by_key(|(created_at, _)| *created_at);
    tasks.into_iter().map(|(_, task)| task).collect()
}

/// Read one owned task's current status without consulting the display mirror.
pub async fn status(id: &str, ctx: &ToolContext) -> Result<Value, String> {
    let process = PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }
    let (status, result) = snapshot(&process).await;
    Ok(task_value(&process, status, result.as_ref()))
}

/// Request cancellation through the process-owned token after checking scope.
pub fn stop(id: &str, ctx: &ToolContext) -> Result<(), String> {
    let process = PROCESSES
        .get(id)
        .map(|entry| entry.value().clone())
        .ok_or_else(|| format!("Task '{}' not found", id))?;
    if !same_scope(&process, ctx) {
        return Err(unavailable_scope());
    }
    if process.cancel.is_cancelled() {
        return Ok(());
    }
    process.cancel.cancel();
    global_registry().update_status(id, CoreTaskStatus::Cancelled);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::EvidenceStore;
    use crate::{execute_authorized, TaskOutputTool, Tool, ToolContext};
    use claurst_core::config::PermissionMode;
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
    use std::sync::Arc;

    struct AllowAll;

    impl PermissionHandler for AllowAll {
        fn check_permission(&self, _: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }

        fn request_permission(&self, _: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    fn test_context(workspace: &Path, session_id: &str) -> ToolContext {
        ToolContext {
            working_dir: workspace.to_path_buf(),
            permission_mode: PermissionMode::Default,
            permission_handler: Arc::new(AllowAll),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: session_id.to_string(),
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
            trusted_checker_catalog: Default::default(),
            catalog_view: None,
        }
    }

    async fn wait_for_file(path: &Path, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while !path.exists() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(path.exists(), "timed out waiting for {}", path.display());
    }

    async fn wait_for_terminal(id: &str, ctx: &ToolContext) -> Value {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let value = output(id, ctx, false).await.unwrap();
            if value["task"]["status"] != "running" {
                return value;
            }
            assert!(
                Instant::now() < deadline,
                "task {id} did not reach terminal status"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    #[cfg(unix)]
    fn process_alive(pid: i32) -> bool {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
    }

    #[cfg(unix)]
    async fn assert_process_group_gone(pid_file: &Path) {
        let pids: Vec<i32> = std::fs::read_to_string(pid_file)
            .expect("owned process pid record")
            .split_whitespace()
            .map(|pid| pid.parse().expect("numeric owned process pid"))
            .collect();
        assert!(pids.len() >= 2, "recorded parent and descendant pids");
        let deadline = Instant::now() + Duration::from_secs(2);
        while pids.iter().any(|pid| process_alive(*pid)) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            pids.iter().all(|pid| !process_alive(*pid)),
            "owned process group members remain alive: {pids:?}"
        );
        let group = -pids[0];
        assert!(
            !process_alive(group),
            "owned process group remains alive: {group}"
        );
    }

    #[test]
    fn command_labels_are_unicode_safe_and_bounded() {
        let command = "🦀".repeat(100);
        let excerpt = display_excerpt(&command);
        assert!(excerpt.is_char_boundary(excerpt.len()));
        assert!(excerpt.len() <= DISPLAY_OUTPUT_CHARS + 64);
    }

    #[test]
    fn task_value_budget_handles_control_character_metadata() {
        let workspace = tempfile::tempdir().unwrap();
        let workspace_path = workspace.path().to_path_buf();
        let process = OwnedProcess {
            id: "bg-control-budget".to_string(),
            command: "printf '\\001'".repeat(10_000),
            workspace: workspace_path.clone(),
            cwd: workspace_path.clone(),
            session_id: "control-budget".to_string(),
            created_at: Instant::now(),
            cancel: CancellationToken::new(),
            pending: mutation::register_pending(
                mutation::MutationScope {
                    workspace: workspace_path,
                    session: "background-control-budget".into(),
                },
                "test process",
            )
            .unwrap(),
            state: Mutex::new(ProcessState {
                status: ProcessStatus::Completed,
                result: None,
                cleanup_confirmed: true,
            }),
            changed: Notify::new(),
            cleanup_wait_gate: None,
        };
        let result = ToolResult::success("\u{1}".repeat(30_000)).with_metadata(json!({
            "evidence": {"id": "control-evidence", "bytes": 30_000, "partial": false},
            "extra": "\u{1}".repeat(30_000),
        }));
        let value = task_value(&process, ProcessStatus::Completed, Some(&result));
        let serialized = serde_json::to_vec_pretty(&value).unwrap();
        assert!(serialized.len() <= TASK_VALUE_BUDGET_BYTES);
        assert_eq!(value["output_preview"], true);
        assert_eq!(value["metadata_preview"], true);
        assert_eq!(value["metadata"]["evidence"]["id"], "control-evidence");
        let _: Value = serde_json::from_slice(&serialized).unwrap();
        process.pending.confirm_terminal().unwrap();
    }

    #[test]
    fn capture_errors_keep_available_streams() {
        let (text, is_error, partial) = normalized_capture(
            b"stdout-before-pipe-error",
            b"stderr-before-pipe-error",
            None,
            false,
            false,
            true,
            Some("pipe closed"),
        );
        assert!(is_error);
        assert!(partial);
        assert!(text.contains("Command output capture failed: pipe closed"));
        assert!(text.contains("stdout-before-pipe-error"));
        assert!(text.contains("stderr-before-pipe-error"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cap_admits_after_evicting_oldest_terminal_entry() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-cap-{}", Uuid::new_v4()),
        );
        let mut ids = Vec::new();
        for _ in 0..MAX_BACKGROUND_PROCESSES {
            let Ok(id) = spawn("true".into(), workspace.path().into(), 30_000, &ctx) else {
                break;
            };
            let _ = output(&id, &ctx, true).await;
            ids.push(id);
        }
        assert!(!ids.is_empty(), "cap test could not register a process");

        let extra = spawn("true".into(), workspace.path().into(), 30_000, &ctx)
            .expect("a completed process should be evicted at the cap");
        assert!(contains(&extra));
        let evicted = ids
            .iter()
            .find(|id| !contains(id))
            .expect("an oldest terminal process should be evicted");
        assert!(global_registry().get(evicted).is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn completed_owned_process_clears_only_after_group_cleanup() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-pending-{}", Uuid::new_v4()),
        );
        let id = spawn(
            "sleep 0.15; printf done".into(),
            workspace.path().into(),
            30_000,
            &ctx,
        )
        .unwrap();
        assert!(mutation::pending_for_scope(&ctx));
        let result = wait_for_terminal(&id, &ctx).await;
        assert_eq!(result["task"]["status"], "completed");
        assert!(!mutation::pending_for_scope(&ctx));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn scoped_shutdown_cancels_owned_process_and_confirms_cleanup() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-shutdown-{}", Uuid::new_v4()),
        );
        let marker = workspace.path().join("late-shutdown-marker");
        let ready = workspace.path().join("shutdown-ready");
        let id = spawn(
            format!(
                "sleep 30 & child=$!; printf ready > '{}'; wait; printf late > '{}'",
                ready.display(),
                marker.display()
            ),
            workspace.path().into(),
            30_000,
            &ctx,
        )
        .unwrap();
        wait_for_file(&ready, Duration::from_secs(2)).await;
        // Group cancellation may spend two seconds joining the child and
        // inherited pipes, then two more seconds proving the group is gone.
        // Keep the one total shutdown deadline above that bounded cleanup
        // budget while retaining the confirmed-clean assertion.
        let shutdown = shutdown_for_context(&ctx, Duration::from_secs(5)).await;
        assert_eq!(shutdown.processes, 1);
        assert_eq!(shutdown.cleanup, BackgroundCleanup::ConfirmedClean);
        let result = wait_for_terminal(&id, &ctx).await;
        assert_eq!(result["task"]["status"], "cancelled");
        assert!(!marker.exists(), "cancelled owned process continued");
    }

    #[tokio::test]
    async fn cleanup_completion_between_state_check_and_wait_is_not_lost() {
        let workspace = tempfile::tempdir().unwrap();
        let session = format!("background-wakeup-{}", Uuid::new_v4());
        let gate = Arc::new(CleanupWaitGate {
            reached: Notify::new(),
            release: tokio::sync::Semaphore::new(0),
        });
        // Model the owned completion boundary without spawning an OS process.
        let process = Arc::new(OwnedProcess {
            id: session.clone(),
            command: String::new(),
            workspace: workspace.path().to_path_buf(),
            cwd: workspace.path().to_path_buf(),
            session_id: session.clone(),
            created_at: Instant::now(),
            cancel: CancellationToken::new(),
            pending: mutation::register_pending(
                mutation::scope_for_workspace(workspace.path().to_path_buf(), session),
                "controlled cleanup completion",
            )
            .unwrap(),
            state: Mutex::new(ProcessState {
                status: ProcessStatus::Running,
                result: None,
                cleanup_confirmed: false,
            }),
            changed: Notify::new(),
            cleanup_wait_gate: Some(gate.clone()),
        });
        let owned = process.clone();
        let waiter = tokio::spawn(async move {
            wait_for_terminal_cleanup(&owned, Instant::now() + Duration::from_millis(200)).await
        });
        tokio::time::timeout(Duration::from_secs(1), gate.reached.notified())
            .await
            .unwrap();
        process.pending.confirm_terminal().unwrap();
        {
            let mut state = process.state.lock().await;
            state.status = ProcessStatus::Cancelled;
            state.cleanup_confirmed = true;
        }
        process.changed.notify_waiters();
        gate.release.add_permits(1);
        assert!(
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .unwrap()
                .unwrap(),
            "confirmed cleanup notification was lost between state check and wait"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn scoped_shutdown_does_not_cancel_foreign_session() {
        let workspace = tempfile::tempdir().unwrap();
        let owner = test_context(
            workspace.path(),
            &format!("background-shutdown-owner-{}", Uuid::new_v4()),
        );
        let foreign = test_context(
            workspace.path(),
            &format!("background-shutdown-foreign-{}", Uuid::new_v4()),
        );
        let id = spawn("sleep 30".into(), workspace.path().into(), 30_000, &owner).unwrap();
        let shutdown = shutdown_for_context(&foreign, Duration::from_millis(100)).await;
        assert_eq!(shutdown.processes, 0);
        assert_eq!(shutdown.cleanup, BackgroundCleanup::ConfirmedClean);
        assert_eq!(
            output(&id, &owner, false).await.unwrap()["task"]["status"],
            "running"
        );
        stop(&id, &owner).unwrap();
        let _ = wait_for_terminal(&id, &owner).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawn_failure_clears_its_pending_owner() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-spawn-failure-{}", Uuid::new_v4()),
        );
        let missing_cwd = workspace.path().join("missing-cwd");
        let id = spawn("true".into(), missing_cwd, 30_000, &ctx).unwrap();
        let result = wait_for_terminal(&id, &ctx).await;
        assert_eq!(result["task"]["status"], "failed");
        assert_eq!(result["task"]["metadata"]["execution_status"], "unknown");
        assert!(!mutation::pending_for_scope(&ctx));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn unavailable_windows_background_launch_leaves_no_task_or_pending_owner() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("windows-background-unavailable-{}", Uuid::new_v4()),
        );
        let marker = workspace.path().join("must-not-launch.txt");
        assert!(list(&ctx).await.is_empty());
        assert!(!mutation::pending_for_scope(&ctx));
        let error = spawn(
            format!("echo forbidden > \"{}\"", marker.display()),
            workspace.path().into(),
            30_000,
            &ctx,
        )
        .unwrap_err();
        assert!(error.contains("unavailable on Windows until Job Object cleanup is supported"));
        assert!(list(&ctx).await.is_empty());
        assert!(!mutation::pending_for_scope(&ctx));
        assert!(!marker.exists());
    }

    #[test]
    fn cleanup_unknown_is_explicit_in_result_metadata() {
        let result = cleanup_unknown(ToolResult::success("done"), None);
        let metadata = result.metadata.unwrap();
        assert_eq!(metadata["cleanup_status"], "unknown");
        assert_eq!(metadata["execution_status"], "unknown");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_capture_keeps_both_streams() {
        let (child, mut group) = child_for(
            "printf 'out-middle-🦀'; printf 'err-tail-🦀' >&2; sleep 10",
            Path::new("/tmp"),
        )
        .unwrap();
        let captured = capture(child, Duration::from_millis(100)).await.unwrap();
        if captured.timed_out {
            group.kill_now();
        } else {
            group.disarm();
        }
        assert!(captured.timed_out && captured.partial);
        assert!(String::from_utf8_lossy(&captured.stdout).contains("out-middle"));
        assert!(String::from_utf8_lossy(&captured.stderr).contains("err-tail"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn owned_stop_is_scoped_and_retrievable() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(workspace.path(), "background-owner");
        let foreign = test_context(workspace.path(), "background-foreign");
        let marker = workspace.path().join("completed-after-cancel");
        let ready = workspace.path().join("ready-to-cancel");
        let pid_file = workspace.path().join("owned-pids");
        let command = format!(
            "printf before-stop; sleep 30 & child=$!; printf '%s %s' $$ $child > '{}'; printf ready > '{}'; wait; printf done > '{}'",
            pid_file.display(),
            ready.display(),
            marker.display()
        );
        let id = spawn(command, workspace.path().into(), 30_000, &ctx).unwrap();

        assert!(output(&id, &foreign, false).await.is_err());
        assert!(stop(&id, &foreign).is_err());
        wait_for_file(&ready, Duration::from_secs(2)).await;
        stop(&id, &ctx).unwrap();
        let result = wait_for_terminal(&id, &ctx).await;
        assert_eq!(result["task"]["status"], "cancelled");
        assert_eq!(result["task"]["metadata"]["execution_status"], "unknown");
        assert!(
            !marker.exists(),
            "cancelled child continued after ownership stop"
        );
        assert_process_group_gone(&pid_file).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn parent_cancellation_retains_partial_output_and_kills_group() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-parent-cancel-{}", Uuid::new_v4()),
        );
        let marker = workspace.path().join("completed-after-parent-cancel");
        let ready = workspace.path().join("ready-for-parent-cancel");
        let pid_file = workspace.path().join("parent-cancel-pids");
        let id = spawn(
            format!(
                "printf before-parent-cancel; sleep 30 & child=$!; printf '%s %s' $$ $child > '{}'; printf ready > '{}'; wait; printf done > '{}'",
                pid_file.display(),
                ready.display(),
                marker.display()
            ),
            workspace.path().into(),
            30_000,
            &ctx,
        )
        .unwrap();
        wait_for_file(&ready, Duration::from_secs(2)).await;
        ctx.cancel_token.cancel();
        let result = wait_for_terminal(&id, &ctx).await;
        assert_eq!(result["task"]["status"], "cancelled");
        assert_eq!(result["task"]["metadata"]["execution_status"], "unknown");
        let evidence_id = result["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("parent cancellation evidence id");
        let (_, text) = EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(evidence_id)
            .unwrap();
        assert!(text.contains("before-parent-cancel"));
        assert!(!marker.exists(), "parent cancellation left a child running");
        assert_process_group_gone(&pid_file).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_result_retains_both_streams_as_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-timeout-{}", Uuid::new_v4()),
        );
        let marker = workspace.path().join("completed-after-timeout");
        let ready = workspace.path().join("ready-for-timeout");
        let pid_file = workspace.path().join("timeout-pids");
        let id = spawn(
            format!(
                "printf 'stdout-middle-🦀'; printf 'stderr-tail-🦀' >&2; sleep 30 & child=$!; printf '%s %s' $$ $child > '{}'; printf ready > '{}'; wait; printf done > '{}'",
                pid_file.display(),
                ready.display(),
                marker.display()
            ),
            workspace.path().into(),
            1_000,
            &ctx,
        )
        .unwrap();
        wait_for_file(&ready, Duration::from_secs(2)).await;
        let result = wait_for_terminal(&id, &ctx).await;
        assert_eq!(result["task"]["status"], "failed");
        assert_eq!(result["task"]["metadata"]["execution_status"], "unknown");
        let evidence_id = result["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("timeout evidence id");
        let (_, text) = EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(evidence_id)
            .unwrap();
        assert!(text.contains("stdout-middle-🦀"));
        assert!(text.contains("stderr-tail-🦀"));
        assert!(!marker.exists(), "timed-out child continued after deadline");
        assert_process_group_gone(&pid_file).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn large_both_stream_output_keeps_middle_in_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-large-{}", Uuid::new_v4()),
        );
        let command = "awk 'BEGIN { for (i=0; i<20000; i++) { if (i==10000) print \"distinctive-middle-fact\"; print \"stdout-🦀abcdefgh\"; print \"stderr-🦀abcdefgh\" > \"/dev/stderr\" } }'";
        let id = spawn(command.into(), workspace.path().into(), 30_000, &ctx).unwrap();
        let result = output(&id, &ctx, true).await.unwrap();
        assert_eq!(result["task"]["status"], "completed");
        let evidence_id = result["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("large output evidence id");
        let (_, text) = EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(evidence_id)
            .unwrap();
        assert!(text.contains("distinctive-middle-fact"));
        assert!(text.contains("stdout-🦀abcdefgh"));
        assert!(text.contains("stderr-🦀abcdefgh"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn task_output_preview_is_parseable_and_keeps_source_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = test_context(
            workspace.path(),
            &format!("background-task-output-{}", Uuid::new_v4()),
        );
        let id = spawn(
            "awk 'BEGIN { for (i=0; i<30000; i++) printf \"🦀\" }'".into(),
            workspace.path().into(),
            30_000,
            &ctx,
        )
        .unwrap();
        let _ = wait_for_terminal(&id, &ctx).await;

        let task_output = TaskOutputTool.execute(json!({"task_id": id}), &ctx).await;
        assert!(!task_output.is_error);
        assert!(task_output.content.len() < 24 * 1024);
        let parsed: Value = serde_json::from_str(&task_output.content)
            .expect("TaskOutput content remains complete JSON");
        assert_eq!(parsed["task"]["output_preview"], true);
        let source_id = parsed["task"]["metadata"]["evidence"]["id"]
            .as_str()
            .expect("source evidence ID in TaskOutput metadata")
            .to_string();
        let (_, source) = EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(&source_id)
            .unwrap();
        assert_eq!(source.chars().count(), 30_000);

        let authorized = execute_authorized(&TaskOutputTool, json!({"task_id": id}), &ctx).await;
        assert!(!authorized.is_error);
        assert!(authorized.content.len() < 24 * 1024);
        let authorized_json = authorized
            .content
            .split_once("\n\n[EvidenceRead id=")
            .map(|(json, _)| json)
            .expect("authorized TaskOutput retains a parseable JSON prefix");
        let _: Value = serde_json::from_str(authorized_json)
            .expect("authorized TaskOutput JSON prefix parses");
    }
}
