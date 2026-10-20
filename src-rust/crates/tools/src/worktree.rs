// Worktree tools: create and exit git worktrees for isolated work sessions.
//
// EnterWorktreeTool – create a new git worktree with an optional branch name,
//                     switching the session's working directory to it.
// ExitWorktreeTool  – exit the current worktree, optionally removing it, and
//                     restore the original working directory.
//
// These tools mirror the TypeScript EnterWorktreeTool / ExitWorktreeTool.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use tracing::debug;

// ---------------------------------------------------------------------------
// Session-level state: only one active worktree per session.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeSession {
    pub original_cwd: PathBuf,
    pub worktree_path: PathBuf,
    pub branch: Option<String>,
    pub original_head: Option<String>,
    /// Git registration identity captured when the worktree was created.
    /// Older records omit this field and remain eligible for `keep`, but can
    /// never authorize destructive removal.
    #[serde(default)]
    identity: Option<WorktreeIdentity>,
    phase: WorktreePhase,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WorktreeIdentity {
    common_dir: PathBuf,
    git_dir: PathBuf,
    head: String,
    branch: String,
    #[serde(default)]
    owner_nonce: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum WorktreePhase {
    Creating,
    Active,
    Unknown,
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct SessionKey {
    workspace: PathBuf,
    session: String,
}

static WORKTREE_SESSIONS: Lazy<RwLock<HashMap<SessionKey, WorktreeSession>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
// Only transitions in the same workspace/session serialize. Weak entries do
// not retain sessions; remove expired entries on the next transition.
static TRANSITION_GATES: Lazy<
    parking_lot::Mutex<HashMap<SessionKey, Weak<tokio::sync::Mutex<()>>>>,
> = Lazy::new(|| parking_lot::Mutex::new(HashMap::new()));

fn transition_gate(key: &SessionKey) -> Arc<tokio::sync::Mutex<()>> {
    let mut gates = TRANSITION_GATES.lock();
    gates.retain(|_, gate| gate.strong_count() > 0);
    if let Some(gate) = gates.get(key).and_then(Weak::upgrade) {
        return gate;
    }
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    gates.insert(key.clone(), Arc::downgrade(&gate));
    gate
}

#[derive(Serialize, Deserialize)]
struct SessionRecord {
    owner: SessionKey,
    state: WorktreeSession,
}

fn record_path(key: &SessionKey) -> PathBuf {
    // Persistent identity must survive Rust compiler/hash implementation updates.
    let identity = format!(
        "forge/worktree-session/v1\0{}\0{}",
        key.workspace.display(),
        key.session
    );
    let hash = claurst_core::crypto_utils::sha256_hex(identity.as_bytes());
    claurst_core::config::Settings::config_dir()
        .join("worktree-sessions")
        .join(format!("{hash}.json"))
}

fn persist_session(key: &SessionKey, state: &WorktreeSession) -> Result<(), String> {
    let path = record_path(key);
    let parent = path.parent().ok_or("Invalid worktree record path")?;
    #[cfg(unix)]
    let created_directories: Vec<PathBuf> = parent
        .ancestors()
        .take_while(|directory| !directory.exists())
        .map(|directory| directory.to_path_buf())
        .collect();
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(parent).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        for directory in created_directories.iter().rev() {
            let containing_directory = directory
                .parent()
                .ok_or("Invalid worktree state directory")?;
            std::fs::File::open(containing_directory)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| error.to_string())?;
        }
    }
    let bytes = serde_json::to_vec(&SessionRecord {
        owner: key.clone(),
        state: state.clone(),
    })
    .map_err(|error| error.to_string())?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    file.as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    file.persist(&path).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        std::fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn load_session(key: &SessionKey) -> Result<Option<WorktreeSession>, String> {
    if let Some(state) = WORKTREE_SESSIONS.read().get(key).cloned() {
        return Ok(Some(state));
    }
    let path = record_path(key);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 64 * 1024 {
        return Err("Invalid worktree session record".into());
    }
    let record: SessionRecord =
        serde_json::from_slice(&std::fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if record.owner != *key
        || record.state.original_cwd.canonicalize().ok().as_ref() != Some(&key.workspace)
    {
        return Err("Worktree session ownership mismatch".into());
    }
    WORKTREE_SESSIONS
        .write()
        .insert(key.clone(), record.state.clone());
    Ok(Some(record.state))
}

fn forget_session(
    key: &SessionKey,
    ctx: &ToolContext,
    state: &WorktreeSession,
) -> Result<(), String> {
    match std::fs::remove_file(record_path(key)) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let mut old_root = ctx.clone();
    old_root.working_dir = state.worktree_path.clone();
    crate::clear_context_shell_state(&old_root);
    WORKTREE_SESSIONS.write().remove(key);
    Ok(())
}

const MAX_WORKTREE_SESSIONS: usize = 128;
const WORKTREE_OWNER_MARKER: &str = ".forge-worktree-owner";

/// Artifact and process ownership remains attached to the original workspace
/// while execution moves into a worktree. Never use process-global cwd.
pub fn workspace_root(ctx: &ToolContext) -> PathBuf {
    let cwd = ctx
        .working_dir
        .canonicalize()
        .unwrap_or_else(|_| ctx.working_dir.clone());
    let sessions = WORKTREE_SESSIONS.read();
    sessions
        .iter()
        .find_map(|(key, session)| {
            (key.session == ctx.session_id && session.worktree_path == cwd)
                .then(|| key.workspace.clone())
        })
        .unwrap_or(cwd)
}

fn session_key(ctx: &ToolContext) -> SessionKey {
    SessionKey {
        workspace: workspace_root(ctx),
        session: ctx.session_id.clone(),
    }
}

/// Used by the broker on every call, including nested invocation. A missing
/// active worktree is an error rather than a silent return to the main checkout.
pub fn execution_context(ctx: &ToolContext) -> Result<ToolContext, String> {
    let key = session_key(ctx);
    let session = load_session(&key)?;
    let mut effective = ctx.clone();
    if let Some(session) = session {
        if session.phase != WorktreePhase::Active {
            return Err(format!("Worktree creation is unconfirmed ({:?}); inspect the recorded path or use ExitWorktree keep before running tools", session.phase));
        }
        if !session.worktree_path.is_dir() {
            return Err(
                "Active worktree is unavailable; exit or repair it before running tools".into(),
            );
        }
        effective.working_dir = session.worktree_path;
    }
    Ok(effective)
}

// ---------------------------------------------------------------------------
// EnterWorktreeTool
// ---------------------------------------------------------------------------

pub struct EnterWorktreeTool;

#[derive(Debug, Deserialize)]
struct EnterWorktreeInput {
    /// Optional branch name. If omitted, a timestamped branch is created.
    #[serde(default)]
    branch: Option<String>,
    /// Sub-path under the repo root where the worktree will be created.
    /// Defaults to `.worktrees/<branch>`.
    #[serde(default)]
    path: Option<String>,
    /// Optional shell command to run inside the new worktree directory after creation.
    /// Example: "npm install" or "cargo build".
    #[serde(default)]
    post_create_command: Option<String>,
}

#[async_trait]
impl Tool for EnterWorktreeTool {
    // Gates itself: prompts for worktree creation AND for the post_create_command
    // in `execute()` (#210). The central backstop must not also prompt.
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "EnterWorktree"
    }

    fn description(&self) -> &str {
        "Create a new git worktree and switch the session's working directory to it. \
         This gives you an isolated environment to experiment or work on a feature \
         without affecting the main working tree. \
         Use ExitWorktree to return to the original directory."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Branch name to create. Defaults to a timestamped name like claurst-20240101-120000."
                },
                "path": {
                    "type": "string",
                    "description": "Optional path for the worktree directory. Defaults to .worktrees/<branch>."
                },
                "post_create_command": {
                    "type": "string",
                    "description": "Optional command to run inside the new worktree after creation (e.g. 'npm install')."
                }
            }
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: EnterWorktreeInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let key = session_key(ctx);
        let gate = transition_gate(&key);
        let _transition = gate.lock().await;
        if let Err(error) = load_session(&key) {
            return ToolResult::error(error);
        }
        // Session identity includes both workspace and conversation.
        {
            let sessions = WORKTREE_SESSIONS.read();
            if sessions.len() >= MAX_WORKTREE_SESSIONS {
                return ToolResult::error(
                    "Worktree session capacity reached; exit an existing session first",
                );
            }
            if sessions.contains_key(&key) {
                return ToolResult::error(
                    "Already in a worktree session. Call ExitWorktree first.".to_string(),
                );
            }
        }

        if let Err(e) = ctx.check_permission(self.name(), "Create a git worktree", false) {
            return ToolResult::error(e.to_string());
        }

        // Determine branch name — use a human-readable timestamp if none supplied
        let branch = params.branch.clone().unwrap_or_else(|| {
            // Format: claurst-YYYYMMDD-HHMMSS
            use std::time::{SystemTime, UNIX_EPOCH};
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            // Manual UTC decomposition (no chrono dep in this crate)
            let s = secs % 60;
            let m = (secs / 60) % 60;
            let h = (secs / 3600) % 24;
            let days = secs / 86400;
            // Approximate Gregorian calendar for branch name purposes
            let year = 1970 + days / 365;
            let day_of_year = days % 365;
            let month = day_of_year / 30 + 1;
            let day = day_of_year % 30 + 1;
            format!(
                "claurst-{:04}{:02}{:02}-{:02}{:02}{:02}",
                year, month, day, h, m, s
            )
        });

        // Determine worktree path
        let worktree_path = if let Some(p) = params.path {
            ctx.working_dir.join(p)
        } else {
            ctx.working_dir.join(".worktrees").join(&branch)
        };

        // Verify we are inside a git repository before attempting worktree creation
        let head_result = run_git(&ctx.working_dir, &["rev-parse", "HEAD"]).await;
        let original_head = match &head_result {
            Ok(h) => Some(h.trim().to_string()),
            Err(e) => {
                let msg = e.to_lowercase();
                if msg.contains("not a git repository") || msg.contains("fatal") {
                    return ToolResult::error(format!(
                        "Cannot create worktree: the current directory '{}' is not inside a git repository.",
                        ctx.working_dir.display()
                    ));
                }
                None
            }
        };

        // Check if the target path already exists
        if worktree_path.exists() {
            return ToolResult::error(format!(
                "Cannot create worktree: the path '{}' already exists.                  Provide a different 'path' argument or remove the existing directory.",
                worktree_path.display()
            ));
        }

        // Persist ownership reservation before Git can mutate. Cancellation or
        // a crash leaves an explicit unconfirmed record rather than a fallback.
        let reservation = WorktreeSession {
            original_cwd: ctx.working_dir.clone(),
            worktree_path: worktree_path.clone(),
            branch: None,
            original_head: original_head.clone(),
            identity: None,
            phase: WorktreePhase::Creating,
        };
        if let Err(error) = persist_session(&key, &reservation) {
            return ToolResult::error(format!(
                "Worktree was not started: cannot persist reservation: {error}"
            ));
        }
        WORKTREE_SESSIONS.write().insert(key.clone(), reservation);

        // Create the worktree
        let worktree_str = worktree_path.to_string_lossy().to_string();
        let result = run_git(
            &ctx.working_dir,
            &["worktree", "add", "-b", &branch, &worktree_str],
        )
        .await;

        match result {
            Err(e) => {
                if let Some(state) = WORKTREE_SESSIONS.write().get_mut(&key) {
                    state.phase = WorktreePhase::Unknown;
                    let _ = persist_session(&key, state);
                }
                let msg = e.trim().to_string();
                let friendly = if msg.to_lowercase().contains("already exists") {
                    format!(
                        "Failed to create worktree: branch '{}' already exists.                          Use a different branch name or delete the existing branch first.",
                        branch
                    )
                } else if msg.to_lowercase().contains("not a git repository") {
                    format!(
                        "Failed to create worktree: '{}' is not inside a git repository.",
                        ctx.working_dir.display()
                    )
                } else {
                    format!("Failed to create worktree: {}", msg)
                };
                ToolResult::error(friendly)
                    .with_execution_status(crate::ExecutionStatus::Unknown)
                    .with_metadata(
                        json!({"execution_status":"unknown", "worktree_path":worktree_path,
                        "worktree_status":"unconfirmed", "requested_branch":branch}),
                    )
            }
            Ok(_) => {
                debug!(
                    branch = %branch,
                    path = %worktree_path.display(),
                    "Created worktree"
                );

                // Save session state
                let canonical = worktree_path.canonicalize();
                let identity = match canonical.as_ref() {
                    Ok(path) => {
                        match capture_worktree_identity(&ctx.working_dir, path, &branch).await {
                            Ok(identity) => Some(identity),
                            Err(error) => {
                                let state = WorktreeSession {
                                    original_cwd: ctx.working_dir.clone(),
                                    worktree_path: path.clone(),
                                    branch: Some(branch.clone()),
                                    original_head: original_head.clone(),
                                    identity: None,
                                    phase: WorktreePhase::Unknown,
                                };
                                WORKTREE_SESSIONS.write().insert(key.clone(), state.clone());
                                let _ = persist_session(&key, &state);
                                return ToolResult::error(format!(
                                "Worktree created but Git identity could not be captured: {error}. Inspect {}",
                                path.display()
                            ))
                            .with_execution_status(crate::ExecutionStatus::Unknown);
                            }
                        }
                    }
                    Err(_) => None,
                };
                let state = WorktreeSession {
                    original_cwd: ctx.working_dir.clone(),
                    worktree_path: canonical.as_ref().unwrap_or(&worktree_path).clone(),
                    branch: Some(branch.clone()),
                    original_head,
                    identity,
                    phase: if canonical.is_ok() {
                        WorktreePhase::Active
                    } else {
                        WorktreePhase::Unknown
                    },
                };
                WORKTREE_SESSIONS.write().insert(key.clone(), state.clone());
                if let Err(error) = persist_session(&key, &state) {
                    if let Some(saved) = WORKTREE_SESSIONS.write().get_mut(&key) {
                        saved.phase = WorktreePhase::Unknown;
                    }
                    return ToolResult::error(format!(
                        "Worktree created; durable state update failed: {error}. Inspect {}",
                        worktree_path.display()
                    ))
                    .with_execution_status(crate::ExecutionStatus::Unknown);
                }
                let worktree_path = match canonical {
                    Ok(path) => path,
                    Err(error) => return ToolResult::error(format!("Worktree created but path is unavailable: {error}. Reservation retained at {}", worktree_path.display()))
                        .with_execution_status(crate::ExecutionStatus::Unknown),
                };

                // Run optional post-create command in the new worktree directory.
                //
                // Security (#210): the post_create_command is model-supplied and
                // was previously run via raw `sh -c` behind only the generic
                // "create a git worktree" prompt — the command itself was never
                // gated or surfaced. Gate it specifically here: classify it,
                // hard-block Critical-risk commands, and prompt with the ACTUAL
                // command text so the user sees exactly what will run.
                let mut post_create_status = "not_run";
                let mut post_create_evidence = None;
                let post_create_output = if let Some(cmd) = params.post_create_command {
                    let risk = claurst_core::bash_classifier::classify_bash_command(&cmd);
                    if risk == claurst_core::bash_classifier::BashRiskLevel::Critical {
                        format!(
                            "\nPost-create command '{}' was BLOCKED: classified as Critical risk \
                             and never executed. Re-create the worktree without it, or run a safer command.",
                            cmd
                        )
                    } else if let Err(e) = ctx.check_permission(
                        self.name(),
                        &format!("Run post-create command in the new worktree: {}", cmd),
                        false,
                    ) {
                        format!("\nPost-create command '{}' was not run: {}", cmd, e)
                    } else {
                        let result = crate::execute_authorized(
                            &crate::PtyBashTool,
                            json!({"command":cmd, "cwd":worktree_path,
                                "description":"Run post-create command in the new worktree"}),
                            ctx,
                        )
                        .await;
                        post_create_status = match result
                            .metadata
                            .as_ref()
                            .and_then(|metadata| metadata["execution_status"].as_str())
                        {
                            Some("not_run") => "not_run",
                            Some("unknown") => "unknown",
                            Some("cancelled") => "cancelled",
                            _ if result.is_error => "failed",
                            _ => "passed",
                        };
                        post_create_evidence = result
                            .metadata
                            .as_ref()
                            .and_then(|metadata| metadata.get("evidence"))
                            .cloned();
                        format!(
                            "\nPost-create command status: {}.\n{}",
                            post_create_status, result.content
                        )
                    }
                } else {
                    String::new()
                };

                ToolResult::success(format!(
                    "Created worktree at {} on branch '{}'.\n\
                     The working directory is now {}.\n\
                     Use ExitWorktree to return to {}.{}",
                    worktree_path.display(),
                    branch,
                    worktree_path.display(),
                    ctx.working_dir.display(),
                    post_create_output,
                ))
                .with_metadata(json!({
                    "worktree_path": worktree_path.to_string_lossy(),
                    "branch": branch,
                    "original_cwd": ctx.working_dir.to_string_lossy(),
                    "post_create_status":post_create_status,
                    "post_create_evidence":post_create_evidence,
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExitWorktreeTool
// ---------------------------------------------------------------------------

pub struct ExitWorktreeTool;

#[derive(Debug, Deserialize)]
struct ExitWorktreeInput {
    /// "keep" = leave the worktree on disk; "remove" = delete it.
    #[serde(default = "default_action")]
    action: String,
    /// Required if action=="remove" and there are uncommitted changes.
    #[serde(default)]
    discard_changes: bool,
}

fn default_action() -> String {
    "keep".to_string()
}

#[async_trait]
impl Tool for ExitWorktreeTool {
    fn name(&self) -> &str {
        "ExitWorktree"
    }

    fn description(&self) -> &str {
        "Exit the current worktree session created by EnterWorktree and restore the \
         original working directory. Use action='keep' to preserve the worktree on \
         disk, or action='remove' to delete it. Only operates on worktrees created \
         by EnterWorktree in this session."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["keep", "remove"],
                    "description": "\"keep\" leaves the worktree on disk; \"remove\" deletes it and its branch."
                },
                "discard_changes": {
                    "type": "boolean",
                    "description": "Set true when action=remove and the worktree has uncommitted/unmerged work to discard."
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ExitWorktreeInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        if !matches!(params.action.as_str(), "keep" | "remove") {
            return ToolResult::error("Unknown action. Use 'keep' or 'remove'.");
        }
        let key = session_key(ctx);
        let gate = transition_gate(&key);
        let _transition = gate.lock().await;
        let session = match load_session(&key) {
            Ok(session) => session,
            Err(error) => return ToolResult::error(error),
        };
        let Some(session) = session else {
            return ToolResult::error("No active worktree belongs to this workspace and session.");
        };

        let worktree_str = session.worktree_path.to_string_lossy().to_string();

        if params.action == "remove" && session.phase != WorktreePhase::Active {
            return ToolResult::error("Worktree ownership is unconfirmed; automatic removal is blocked. Inspect the recorded path. ExitWorktree keep clears the reservation and preserves any work.");
        }

        // Stop interpreters rooted in this worktree before restoring the parent
        // context or deleting the directory. The workspace/session scope keeps
        // a same-id interpreter in another workspace out of this teardown.
        let mut repl_ctx = ctx.clone();
        repl_ctx.working_dir = session.worktree_path.clone();
        let repl_shutdown = crate::repl_tool::shutdown_for_context(&repl_ctx).await;
        if matches!(
            repl_shutdown.cleanup,
            crate::repl_tool::ReplCleanup::CleanupUnknown
        ) {
            let message = if params.action == "remove" {
                "REPL cleanup is unconfirmed; refusing to remove the worktree."
            } else {
                "REPL cleanup is unconfirmed; keeping the worktree session active."
            };
            return ToolResult::error(message)
                .with_execution_status(crate::ExecutionStatus::Unknown)
                .with_metadata(json!({
                    "repl_sessions": repl_shutdown.sessions,
                    "repl_cleanup": "unknown",
                    "execution_status": "unknown",
                }));
        }

        // Reconcile durable mutation tickets after interpreter shutdown and
        // before any worktree removal or session-state deletion. Check both
        // the parent workspace and the execution root so an empty in-process
        // REPL registry after restart cannot masquerade as quiescence.
        let pending_mutation = crate::mutation::pending_for_scope(ctx)
            || crate::mutation::pending_for_scope(&repl_ctx);
        if pending_mutation && params.action == "remove" {
            return ToolResult::error(
                "Pending mutation cleanup is unconfirmed; refusing to remove the worktree.",
            )
            .with_execution_status(crate::ExecutionStatus::Unknown)
            .with_metadata(json!({
                "repl_sessions": repl_shutdown.sessions,
                "repl_cleanup": "confirmed_clean",
                "pending_mutation": true,
                "cleanup_status": "unknown",
                "execution_status": "unknown",
            }));
        }

        if params.action == "remove" {
            if let Err(error) = validate_worktree_identity(&session).await {
                return ToolResult::error(format!(
                    "Worktree identity is unconfirmed; refusing removal: {error}"
                ))
                .with_execution_status(crate::ExecutionStatus::Unknown)
                .with_metadata(json!({
                    "repl_sessions": repl_shutdown.sessions,
                    "repl_cleanup": "confirmed_clean",
                    "pending_mutation": pending_mutation,
                    "cleanup_status": "unknown",
                    "execution_status": "unknown",
                }));
            }
        }

        // If action is "remove", check for uncommitted changes
        if params.action == "remove" && !params.discard_changes {
            let status = match run_git(&session.worktree_path, &["status", "--porcelain"]).await {
                Ok(status) => status,
                Err(error) => {
                    return ToolResult::error(format!(
                        "Cannot inspect worktree; nothing removed: {error}"
                    ))
                }
            };
            let changed_files = status.lines().filter(|l| !l.trim().is_empty()).count();

            let commit_count = if let Some(ref head) = session.original_head {
                let rev = match run_git(
                    &session.worktree_path,
                    &["rev-list", "--count", &format!("{}..HEAD", head)],
                )
                .await
                {
                    Ok(rev) => rev,
                    Err(error) => {
                        return ToolResult::error(format!(
                            "Cannot inspect worktree commits; nothing removed: {error}"
                        ))
                    }
                };
                match rev.trim().parse::<usize>() {
                    Ok(count) => count,
                    Err(_) => {
                        return ToolResult::error("Invalid worktree commit count; nothing removed")
                    }
                }
            } else {
                0
            };

            if changed_files > 0 || commit_count > 0 {
                let mut parts = Vec::new();
                if changed_files > 0 {
                    parts.push(format!("{} uncommitted file(s)", changed_files));
                }
                if commit_count > 0 {
                    parts.push(format!("{} commit(s) on the worktree branch", commit_count));
                }
                return ToolResult::error(format!(
                    "Worktree has {}. Removing will discard this work permanently. \
                     Confirm with the user, then re-invoke with discard_changes=true — \
                     or use action=\"keep\" to preserve the worktree.",
                    parts.join(" and ")
                ));
            }
        }

        match params.action.as_str() {
            "keep" => {
                // Preserve the registered worktree and its branch. Returning to
                // the parent does not require deleting or locking either one.
                if let Err(error) = forget_session(&key, ctx, &session) {
                    return ToolResult::error(format!(
                        "Could not clear worktree session; state retained: {error}"
                    ));
                }

                ToolResult::success(format!(
                    "Exited worktree. Work preserved at {} on branch {}. \
                     Session is now back in {}.",
                    session.worktree_path.display(),
                    session.branch.as_deref().unwrap_or("(unknown)"),
                    session.original_cwd.display(),
                ))
                .with_metadata(json!({
                    "repl_sessions": repl_shutdown.sessions,
                    "repl_cleanup": "confirmed_clean",
                    "pending_mutation": pending_mutation,
                    "cleanup_status": if pending_mutation { "unknown" } else { "confirmed_clean" },
                }))
            }
            "remove" => {
                // Remove the worktree
                if let Err(error) = run_git(
                    &session.original_cwd,
                    &["worktree", "remove", "--force", &worktree_str],
                )
                .await
                {
                    return ToolResult::error(format!(
                        "Worktree removal failed; session retained: {error}"
                    ))
                    .with_execution_status(crate::ExecutionStatus::Unknown);
                }
                if let Err(error) = forget_session(&key, ctx, &session) {
                    return ToolResult::error(format!(
                        "Worktree removed but session record remains: {error}"
                    ))
                    .with_execution_status(crate::ExecutionStatus::Unknown);
                }

                // Delete the branch if we created it
                if let Some(ref branch) = session.branch {
                    if let Err(error) =
                        run_git(&session.original_cwd, &["branch", "-D", branch]).await
                    {
                        return ToolResult::error(format!("Worktree removed and session restored, but branch was retained: {error}"))
                            .with_metadata(json!({"worktree_removed":true,"session_restored":true,"branch_retained":branch}));
                    }
                }

                ToolResult::success(format!(
                    "Exited and removed worktree at {}. \
                     Session is now back in {}.",
                    session.worktree_path.display(),
                    session.original_cwd.display(),
                ))
                .with_metadata(json!({
                    "repl_sessions": repl_shutdown.sessions,
                    "repl_cleanup": "confirmed_clean",
                    "pending_mutation": pending_mutation,
                    "cleanup_status": if pending_mutation { "unknown" } else { "confirmed_clean" },
                }))
            }
            other => ToolResult::error(format!(
                "Unknown action '{}'. Use 'keep' or 'remove'.",
                other
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

async fn run_git(cwd: &std::path::Path, args: &[&str]) -> Result<String, String> {
    #[cfg(windows)]
    {
        let mut command = tokio::process::Command::new("git");
        command
            .args(args)
            .current_dir(cwd)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        let (child, mut group) = match crate::command_capture::spawn_owned_command_checked(command)
        {
            Ok(pair) => pair,
            Err(crate::command_capture::SpawnOwnedError::Spawn(error)) => {
                return Err(error.to_string());
            }
            Err(crate::command_capture::SpawnOwnedError::Arm { child, error }) => {
                let cleanup = crate::command_capture::cleanup_unarmed_child(child).await;
                return Err(format!(
                    "Git Job Object arm failed: {error}; cleanup remains unknown ({cleanup:?})"
                ));
            }
        };
        let captured = crate::command_capture::capture_owned_with_cancel(
            child,
            std::time::Duration::from_secs(30),
            &tokio_util::sync::CancellationToken::new(),
            &mut group,
        )
        .await
        .map_err(|error| format!("Git output capture failed: {error}"))?;
        let capture_complete = !captured.partial
            && !captured.timed_out
            && !captured.cancelled
            && captured.error.is_none();
        if captured.cleanup != crate::command_capture::ProcessCleanup::ConfirmedClean
            || !capture_complete
        {
            return Err("Git process-tree/output cleanup remains unknown".to_string());
        }
        if captured.status.is_some_and(|status| status.success()) {
            return Ok(String::from_utf8_lossy(&captured.stdout).to_string());
        }
        return Err(String::from_utf8_lossy(&captured.stderr).to_string());
    }
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        tokio::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| "Git deadline exceeded; worktree state may be unknown".to_string())?
    .map_err(|error| error.to_string())?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn resolved_git_path(cwd: &std::path::Path, value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value.trim());
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    path.canonicalize().map_err(|error| {
        format!(
            "Cannot resolve Git identity path '{}': {error}",
            path.display()
        )
    })
}

#[derive(Debug, PartialEq, Eq)]
struct WorktreeRegistration {
    path: PathBuf,
    head: String,
    branch: Option<String>,
}

fn parse_worktree_list(output: &str) -> Vec<WorktreeRegistration> {
    let mut registrations = Vec::new();
    let mut current: Option<WorktreeRegistration> = None;
    for line in output.lines() {
        if let Some(path) = line.strip_prefix("worktree ") {
            if let Some(previous) = current.take() {
                registrations.push(previous);
            }
            current = Some(WorktreeRegistration {
                path: PathBuf::from(path),
                head: String::new(),
                branch: None,
            });
        } else if let Some(head) = line.strip_prefix("HEAD ") {
            if let Some(registration) = current.as_mut() {
                registration.head = head.to_string();
            }
        } else if let Some(branch) = line.strip_prefix("branch ") {
            if let Some(registration) = current.as_mut() {
                registration.branch = Some(branch.to_string());
            }
        }
    }
    if let Some(previous) = current {
        registrations.push(previous);
    }
    registrations
}

fn create_worktree_owner_marker(git_dir: &std::path::Path) -> Result<String, String> {
    let nonce = uuid::Uuid::new_v4().to_string();
    let marker = git_dir.join(WORKTREE_OWNER_MARKER);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker)
        .map_err(|error| format!("Cannot create worktree ownership marker: {error}"))?;
    file.write_all(nonce.as_bytes())
        .map_err(|error| format!("Cannot write worktree ownership marker: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("Cannot persist worktree ownership marker: {error}"))?;
    #[cfg(unix)]
    std::fs::File::open(git_dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("Cannot persist worktree Git registration: {error}"))?;
    Ok(nonce)
}

fn validate_worktree_owner_marker(
    git_dir: &std::path::Path,
    expected_nonce: &str,
) -> Result<(), String> {
    if expected_nonce.is_empty() {
        return Err("Worktree ownership nonce is unavailable; refusing removal".into());
    }
    let marker = git_dir.join(WORKTREE_OWNER_MARKER);
    let metadata = std::fs::symlink_metadata(&marker)
        .map_err(|error| format!("Cannot inspect worktree ownership marker: {error}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("Worktree ownership marker is not a regular file; refusing removal".into());
    }
    let actual_nonce = std::fs::read_to_string(&marker)
        .map_err(|error| format!("Cannot read worktree ownership marker: {error}"))?;
    if actual_nonce != expected_nonce {
        return Err("Worktree ownership marker changed; refusing removal".into());
    }
    Ok(())
}

async fn capture_worktree_identity(
    original_cwd: &std::path::Path,
    worktree_path: &std::path::Path,
    branch: &str,
) -> Result<WorktreeIdentity, String> {
    let canonical_worktree = worktree_path
        .canonicalize()
        .map_err(|error| format!("Cannot resolve created worktree: {error}"))?;
    if std::fs::symlink_metadata(&canonical_worktree)
        .map_err(|error| format!("Cannot inspect created worktree: {error}"))?
        .file_type()
        .is_symlink()
    {
        return Err("Created worktree path is a symlink".into());
    }
    let common_dir = resolved_git_path(
        original_cwd,
        &run_git(original_cwd, &["rev-parse", "--git-common-dir"]).await?,
    )?;
    let worktree_common_dir = resolved_git_path(
        &canonical_worktree,
        &run_git(&canonical_worktree, &["rev-parse", "--git-common-dir"]).await?,
    )?;
    if common_dir != worktree_common_dir {
        return Err("Created worktree Git common directory does not match its repository".into());
    }
    let git_dir = resolved_git_path(
        &canonical_worktree,
        &run_git(&canonical_worktree, &["rev-parse", "--git-dir"]).await?,
    )?;
    let owner_nonce = create_worktree_owner_marker(&git_dir)?;
    let head = run_git(&canonical_worktree, &["rev-parse", "HEAD"])
        .await?
        .trim()
        .to_string();
    let registrations =
        parse_worktree_list(&run_git(original_cwd, &["worktree", "list", "--porcelain"]).await?);
    let expected_branch = format!("refs/heads/{branch}");
    let registration = registrations
        .into_iter()
        .find(|registration| {
            registration
                .path
                .canonicalize()
                .ok()
                .is_some_and(|path| path == canonical_worktree)
        })
        .ok_or_else(|| "Created worktree is not registered by Git".to_string())?;
    if registration.head != head || registration.branch.as_deref() != Some(expected_branch.as_str())
    {
        return Err("Created worktree Git registration does not match its branch or HEAD".into());
    }
    Ok(WorktreeIdentity {
        common_dir,
        git_dir,
        head,
        branch: expected_branch,
        owner_nonce,
    })
}

async fn validate_worktree_identity(state: &WorktreeSession) -> Result<(), String> {
    let identity = state.identity.as_ref().ok_or_else(|| {
        "Worktree identity is unavailable; refusing destructive removal".to_string()
    })?;
    let original_cwd = state
        .original_cwd
        .canonicalize()
        .map_err(|error| format!("Cannot resolve original repository: {error}"))?;
    let worktree_metadata = std::fs::symlink_metadata(&state.worktree_path)
        .map_err(|error| format!("Cannot inspect registered worktree path: {error}"))?;
    if worktree_metadata.file_type().is_symlink() {
        return Err("Registered worktree path is now a symlink; refusing removal".into());
    }
    let worktree_path = state
        .worktree_path
        .canonicalize()
        .map_err(|error| format!("Cannot resolve registered worktree path: {error}"))?;
    if worktree_path != state.worktree_path {
        return Err("Registered worktree path changed; refusing removal".into());
    }
    let common_dir = resolved_git_path(
        &original_cwd,
        &run_git(&original_cwd, &["rev-parse", "--git-common-dir"]).await?,
    )?;
    if common_dir != identity.common_dir {
        return Err("Original repository Git identity changed; refusing removal".into());
    }
    let worktree_common_dir = resolved_git_path(
        &worktree_path,
        &run_git(&worktree_path, &["rev-parse", "--git-common-dir"]).await?,
    )?;
    if worktree_common_dir != identity.common_dir {
        return Err("Worktree Git common directory changed; refusing removal".into());
    }
    let git_dir = resolved_git_path(
        &worktree_path,
        &run_git(&worktree_path, &["rev-parse", "--git-dir"]).await?,
    )?;
    if git_dir != identity.git_dir {
        return Err("Worktree Git registration changed; refusing removal".into());
    }
    validate_worktree_owner_marker(&git_dir, &identity.owner_nonce)?;
    let registration =
        parse_worktree_list(&run_git(&original_cwd, &["worktree", "list", "--porcelain"]).await?)
            .into_iter()
            .find(|registration| {
                registration
                    .path
                    .canonicalize()
                    .ok()
                    .is_some_and(|path| path == worktree_path)
            })
            .ok_or_else(|| {
                "Registered worktree is absent from Git; refusing removal".to_string()
            })?;
    let expected_branch = state
        .branch
        .as_deref()
        .map(|branch| format!("refs/heads/{branch}"));
    let current_head = run_git(&worktree_path, &["rev-parse", "HEAD"])
        .await?
        .trim()
        .to_string();
    if registration.branch.as_deref() != Some(identity.branch.as_str())
        || registration.branch.as_deref() != expected_branch.as_deref()
        || registration.head.is_empty()
        || registration.head != current_head
        || identity.head.is_empty()
    {
        return Err("Registered worktree branch or HEAD changed; refusing removal".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};

    static TEST_LOCK: Lazy<tokio::sync::Mutex<()>> = Lazy::new(|| tokio::sync::Mutex::new(()));

    /// Handler that allows worktree creation but denies (via `Ask`) any request
    /// whose description mentions the post-create command.
    struct DenyPostCreateHandler;
    impl PermissionHandler for DenyPostCreateHandler {
        fn check_permission(&self, request: &PermissionRequest) -> PermissionDecision {
            if request.description.contains("post-create command") {
                PermissionDecision::Ask {
                    reason: "denied in test".to_string(),
                }
            } else {
                PermissionDecision::Allow
            }
        }
        fn request_permission(&self, request: &PermissionRequest) -> PermissionDecision {
            self.check_permission(request)
        }
    }

    async fn init_git_repo(dir: &std::path::Path) {
        run_git(dir, &["init"]).await.expect("git init");
        run_git(dir, &["config", "user.email", "t@example.com"])
            .await
            .expect("git config email");
        run_git(dir, &["config", "user.name", "Test"])
            .await
            .expect("git config name");
        std::fs::write(dir.join("README.md"), b"hi").unwrap();
        run_git(dir, &["add", "."]).await.expect("git add");
        run_git(dir, &["commit", "-m", "init"])
            .await
            .expect("git commit");
    }

    async fn reset_session() {
        let keys: Vec<_> = WORKTREE_SESSIONS.read().keys().cloned().collect();
        for key in keys {
            let _ = std::fs::remove_file(record_path(&key));
        }
        WORKTREE_SESSIONS.write().clear();
    }

    /// A Critical-classified post_create_command is BLOCKED and never executed,
    /// while the worktree itself is still created.
    #[tokio::test]
    async fn worktree_post_create_critical_command_is_blocked() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;

        // `dd if=…` is classified Critical, yet writing /dev/null → marker is a
        // harmless no-op if it ever ran — so this test is regression-safe.
        let marker = tmp.path().join("critical_ran");
        let cmd = format!("dd if=/dev/null of={}", marker.display());

        let ctx = allow_all_context(tmp.path().to_path_buf());
        let result = EnterWorktreeTool
            .execute(json!({ "post_create_command": cmd }), &ctx)
            .await;

        assert!(!result.is_error, "worktree creation should still succeed");
        assert!(
            result.content.contains("BLOCKED"),
            "Critical post-create command must be reported as blocked: {}",
            result.content
        );
        assert!(
            !marker.exists(),
            "Critical post-create command must NOT run"
        );
        reset_session().await;
    }

    /// A (non-Critical) post_create_command is not executed when permission is
    /// denied; the worktree is still created.
    #[tokio::test]
    async fn worktree_post_create_command_denied_is_not_run() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;

        let marker = tmp.path().join("denied_ran");
        let cmd = format!("touch {}", marker.display());

        let mut ctx = allow_all_context(tmp.path().to_path_buf());
        ctx.permission_handler = Arc::new(DenyPostCreateHandler);

        let result = EnterWorktreeTool
            .execute(json!({ "post_create_command": cmd }), &ctx)
            .await;

        assert!(!result.is_error, "worktree creation should still succeed");
        assert!(
            result.content.contains("was not run"),
            "denied post-create command must be reported as not run: {}",
            result.content
        );
        assert!(!marker.exists(), "denied post-create command must NOT run");
        reset_session().await;
    }

    #[tokio::test]
    async fn nested_command_cannot_bypass_configured_bash_denial() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.config.disallowed_tools = vec!["Bash".into()];
        let result = EnterWorktreeTool
            .execute(
                json!({
                    "branch":"broker-denial", "post_create_command":"touch forbidden"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "the worktree itself was created");
        assert_eq!(
            result.metadata.as_ref().unwrap()["post_create_status"],
            "not_run"
        );
        assert!(result
            .content
            .contains("disallowed by configured tool policy"));
        assert!(!tmp
            .path()
            .join(".worktrees/broker-denial/forbidden")
            .exists());
        reset_session().await;
    }

    #[tokio::test]
    async fn nested_command_runs_in_new_worktree_and_retains_evidence() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let post_create_command = if cfg!(windows) {
            r#"powershell -NoProfile -NonInteractive -Command "Set-Content -NoNewline -LiteralPath marker -Value 'worktree-marker'; Write-Output verified-worktree""#
        } else {
            "printf worktree-marker > marker; printf verified-worktree"
        };
        let result = EnterWorktreeTool
            .execute(
                json!({
                    "branch":"broker-cwd", "post_create_command":post_create_command
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert_eq!(
            result.metadata.as_ref().unwrap()["post_create_status"],
            "passed"
        );
        assert!(!tmp.path().join("marker").exists());
        assert_eq!(
            std::fs::read_to_string(tmp.path().join(".worktrees/broker-cwd/marker")).unwrap(),
            "worktree-marker"
        );
        let id = result.metadata.as_ref().unwrap()["post_create_evidence"]["id"]
            .as_str()
            .unwrap();
        let (_, text) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert!(text.contains("verified-worktree"));
        reset_session().await;
    }
    #[tokio::test]
    async fn broker_switches_cwd_without_losing_evidence_on_exit() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered =
            crate::execute_authorized(&EnterWorktreeTool, json!({"branch":"scoped-cwd"}), &ctx)
                .await;
        assert!(!entered.is_error, "{}", entered.content);
        let wt = tmp.path().join(".worktrees/scoped-cwd");
        let written = crate::execute_authorized(
            &crate::FileWriteTool,
            json!({"file_path":"marker", "content":"worktree-only"}),
            &ctx,
        )
        .await;
        assert!(!written.is_error, "{}", written.content);
        assert_eq!(
            std::fs::read_to_string(wt.join("marker")).unwrap(),
            "worktree-only"
        );
        assert!(!tmp.path().join("marker").exists());
        let shell_command = if cfg!(windows) {
            r#"powershell -NoProfile -NonInteractive -Command "Set-Content -NoNewline -LiteralPath shell-file -Value 'shell-marker'; Write-Output worktree-evidence""#
        } else {
            "printf shell-marker > shell-file; printf worktree-evidence"
        };
        let command =
            crate::execute_authorized(&crate::PtyBashTool, json!({"command":shell_command}), &ctx)
                .await;
        assert!(!command.is_error, "{}", command.content);
        assert!(wt.join("shell-file").exists());
        assert!(!tmp.path().join("shell-file").exists());
        let id = command.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap();
        let exited =
            crate::execute_authorized(&ExitWorktreeTool, json!({"action":"keep"}), &ctx).await;
        assert!(!exited.is_error, "{}", exited.content);
        let evidence =
            crate::execute_authorized(&crate::EvidenceReadTool, json!({"id":id}), &ctx).await;
        assert!(!evidence.is_error, "{}", evidence.content);
        assert!(evidence.content.contains("worktree-evidence"));
        let parent_command = if cfg!(windows) {
            r#"powershell -NoProfile -NonInteractive -Command "Set-Content -NoNewline -LiteralPath parent-file -Value 'parent-marker'""#
        } else {
            "printf parent-marker > parent-file"
        };
        let command =
            crate::execute_authorized(&crate::PtyBashTool, json!({"command":parent_command}), &ctx)
                .await;
        assert!(!command.is_error, "{}", command.content);
        assert!(tmp.path().join("parent-file").exists());
        assert!(!wt.join("parent-file").exists());
        reset_session().await;
    }

    #[tokio::test]
    async fn exit_worktree_waits_for_scoped_repl_before_restoring_root() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.session_id = uuid::Uuid::new_v4().to_string();

        let entered =
            crate::execute_authorized(&EnterWorktreeTool, json!({"branch":"repl-teardown"}), &ctx)
                .await;
        assert!(!entered.is_error, "{}", entered.content);

        let repl_input = if cfg!(windows) {
            json!({"language":"python", "code":"print('persistent-repl')"})
        } else {
            json!({"code":"printf persistent-repl"})
        };
        let repl = crate::execute_authorized(&crate::ReplTool, repl_input, &ctx).await;
        assert!(!repl.is_error, "{}", repl.content);

        let exited =
            crate::execute_authorized(&ExitWorktreeTool, json!({"action":"keep"}), &ctx).await;
        assert!(!exited.is_error, "{}", exited.content);
        assert_eq!(
            exited
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("repl_cleanup")),
            Some(&json!("confirmed_clean"))
        );

        // The worktree key is gone, but the workspace/session shutdown still
        // finds any leaked execution-root session if ExitWorktree failed to
        // terminate it before restoring the parent root.
        let leftover = crate::repl_tool::shutdown_for_workspace(&ctx).await;
        assert_eq!(leftover.sessions, 0);
        reset_session().await;
    }

    #[tokio::test]
    async fn persisted_pending_writer_blocks_remove_after_empty_repl_registry() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered = EnterWorktreeTool
            .execute(json!({"branch": "persisted-pending"}), &ctx)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        let worktree = tmp.path().join(".worktrees/persisted-pending");
        let ticket = crate::mutation::register_pending(
            crate::mutation::scope_for_context(&ctx),
            "orphaned writer after restart",
        )
        .unwrap();

        // The worktree record survives while the in-process REPL registry does
        // not, matching a process restart with a durable pending writer.
        WORKTREE_SESSIONS.write().clear();
        let removed = ExitWorktreeTool
            .execute(json!({"action": "remove", "discard_changes": true}), &ctx)
            .await;
        assert!(removed.is_error);
        assert_eq!(removed.metadata.as_ref().unwrap()["pending_mutation"], true);
        assert_eq!(
            removed.metadata.as_ref().unwrap()["execution_status"],
            "unknown"
        );
        assert!(worktree.exists(), "pending writer must block removal");

        let kept = ExitWorktreeTool
            .execute(json!({"action": "keep"}), &ctx)
            .await;
        assert!(!kept.is_error, "{}", kept.content);
        assert_eq!(kept.metadata.as_ref().unwrap()["pending_mutation"], true);
        assert_eq!(kept.metadata.as_ref().unwrap()["cleanup_status"], "unknown");

        // Explicitly retire only this test ticket after the keep path has
        // restored the parent; the worktree itself remains preserved.
        ticket.confirm_terminal().unwrap();
        assert!(worktree.exists());
        reset_session().await;
    }

    #[tokio::test]
    async fn workspace_and_session_ownership_and_invalid_exit_are_enforced() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        init_git_repo(a.path()).await;
        init_git_repo(b.path()).await;
        let mut owner = allow_all_context(a.path().into());
        owner.session_id = "same-session-id".into();
        let mut foreign_workspace = allow_all_context(b.path().into());
        foreign_workspace.session_id = owner.session_id.clone();
        let mut foreign_session = owner.clone();
        foreign_session.session_id = "other-session-id".into();
        let entered = EnterWorktreeTool
            .execute(json!({"branch":"owner"}), &owner)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        for foreign in [&foreign_workspace, &foreign_session] {
            let result = ExitWorktreeTool
                .execute(json!({"action":"remove","discard_changes":true}), foreign)
                .await;
            assert!(result.is_error);
        }
        let invalid = ExitWorktreeTool
            .execute(json!({"action":"invalid"}), &owner)
            .await;
        assert!(invalid.is_error);
        assert_eq!(
            execution_context(&owner).unwrap().working_dir,
            a.path().join(".worktrees/owner").canonicalize().unwrap()
        );
        let other = EnterWorktreeTool
            .execute(json!({"branch":"other"}), &foreign_workspace)
            .await;
        assert!(!other.is_error, "{}", other.content);
        let exited = ExitWorktreeTool
            .execute(json!({"action":"keep"}), &foreign_workspace)
            .await;
        assert!(!exited.is_error);
        assert_ne!(
            execution_context(&owner).unwrap().working_dir,
            owner.working_dir
        );
        reset_session().await;
    }

    #[tokio::test]
    async fn missing_worktree_fails_closed_and_failed_removal_preserves_state() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let ctx = allow_all_context(tmp.path().into());
        let entered = EnterWorktreeTool
            .execute(json!({"branch":"missing"}), &ctx)
            .await;
        assert!(!entered.is_error);
        let wt = tmp.path().join(".worktrees/missing");
        // A locked worktree cannot be removed by a single --force.
        run_git(tmp.path(), &["worktree", "lock", wt.to_str().unwrap()])
            .await
            .unwrap();
        let failed = ExitWorktreeTool
            .execute(json!({"action":"remove","discard_changes":true}), &ctx)
            .await;
        assert!(failed.is_error);
        assert_eq!(
            execution_context(&ctx).unwrap().working_dir,
            wt.canonicalize().unwrap()
        );
        std::fs::remove_dir_all(&wt).unwrap();
        let denied = crate::execute_authorized(
            &crate::FileWriteTool,
            json!({"file_path":"must-not-fall-back","content":"no"}),
            &ctx,
        )
        .await;
        assert!(denied.is_error);
        assert_eq!(denied.metadata.unwrap()["execution_status"], "not_run");
        assert!(!tmp.path().join("must-not-fall-back").exists());
        let inspection = ExitWorktreeTool
            .execute(json!({"action":"remove"}), &ctx)
            .await;
        assert!(inspection.is_error);
        assert!(WORKTREE_SESSIONS.read().contains_key(&session_key(&ctx)));
        let kept = ExitWorktreeTool
            .execute(json!({"action":"keep"}), &ctx)
            .await;
        assert!(!kept.is_error);
        reset_session().await;
    }

    #[tokio::test]
    async fn interrupted_creation_reservation_survives_restart_without_fallback() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let key = session_key(&ctx);
        let pending = WorktreeSession {
            original_cwd: ctx.working_dir.clone(),
            worktree_path: tmp.path().join("interrupted"),
            branch: None,
            original_head: None,
            identity: None,
            phase: WorktreePhase::Creating,
        };
        persist_session(&key, &pending).unwrap();
        WORKTREE_SESSIONS.write().clear(); // Simulate process restart.
        let result = crate::execute_authorized(
            &crate::FileWriteTool,
            json!({"file_path":"forbidden-parent-fallback", "content":"no"}),
            &ctx,
        )
        .await;
        assert!(result.is_error);
        assert_eq!(result.metadata.unwrap()["execution_status"], "not_run");
        assert!(!tmp.path().join("forbidden-parent-fallback").exists());
        let remove = ExitWorktreeTool
            .execute(json!({"action":"remove","discard_changes":true}), &ctx)
            .await;
        assert!(
            remove.is_error,
            "uncertain ownership cannot authorize removal"
        );
        let keep = ExitWorktreeTool
            .execute(json!({"action":"keep"}), &ctx)
            .await;
        assert!(!keep.is_error);
        assert!(!record_path(&key).exists());
        reset_session().await;
    }

    #[tokio::test]
    async fn failed_creation_never_claims_ownership_of_preexisting_branch() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let ctx = allow_all_context(tmp.path().into());
        run_git(tmp.path(), &["branch", "existing"]).await.unwrap();
        let entered = EnterWorktreeTool
            .execute(json!({"branch":"existing"}), &ctx)
            .await;
        assert!(entered.is_error);
        let key = session_key(&ctx);
        let saved = load_session(&key).unwrap().unwrap();
        assert_eq!(saved.phase, WorktreePhase::Unknown);
        assert!(saved.branch.is_none());
        let removed = ExitWorktreeTool
            .execute(json!({"action":"remove","discard_changes":true}), &ctx)
            .await;
        assert!(removed.is_error);
        assert!(run_git(
            tmp.path(),
            &["rev-parse", "--verify", "refs/heads/existing"]
        )
        .await
        .is_ok());
        let kept = ExitWorktreeTool
            .execute(json!({"action":"keep"}), &ctx)
            .await;
        assert!(!kept.is_error);
        reset_session().await;
    }

    #[test]
    fn persistent_scope_identity_is_versioned_and_stable() {
        let key = SessionKey {
            workspace: PathBuf::from("/workspace/project"),
            session: "stable-session".into(),
        };
        assert_eq!(
            record_path(&key).file_name().unwrap(),
            "78368848571ef5f46d10b9a495ad06d377c20bd49722f7d6378579aede04f8c5.json"
        );
    }

    #[tokio::test]
    async fn removal_refuses_a_reused_path_registered_to_another_branch() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().to_path_buf());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered = EnterWorktreeTool
            .execute(json!({"branch": "owned-branch"}), &ctx)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        let worktree = tmp.path().join(".worktrees/owned-branch");

        run_git(
            tmp.path(),
            &["worktree", "remove", "--force", worktree.to_str().unwrap()],
        )
        .await
        .unwrap();
        run_git(tmp.path(), &["branch", "foreign-branch"])
            .await
            .unwrap();
        run_git(
            tmp.path(),
            &[
                "worktree",
                "add",
                worktree.to_str().unwrap(),
                "foreign-branch",
            ],
        )
        .await
        .unwrap();
        let sentinel = worktree.join("foreign-sentinel");
        std::fs::write(&sentinel, b"foreign worktree must survive").unwrap();

        let result = ExitWorktreeTool
            .execute(json!({"action": "remove", "discard_changes": true}), &ctx)
            .await;
        assert!(result.is_error, "foreign registration must block removal");
        assert_eq!(
            std::fs::read(&sentinel).unwrap(),
            b"foreign worktree must survive"
        );
        assert!(run_git(tmp.path(), &["worktree", "list", "--porcelain"])
            .await
            .unwrap()
            .contains("branch refs/heads/foreign-branch"));

        run_git(
            tmp.path(),
            &["worktree", "remove", "--force", worktree.to_str().unwrap()],
        )
        .await
        .unwrap();
        run_git(tmp.path(), &["branch", "-D", "foreign-branch"])
            .await
            .unwrap();
        reset_session().await;
    }

    #[tokio::test]
    async fn removal_refuses_same_path_same_branch_replacement() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().to_path_buf());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered = EnterWorktreeTool
            .execute(json!({"branch": "same-branch"}), &ctx)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        let worktree = tmp.path().join(".worktrees/same-branch");

        run_git(
            tmp.path(),
            &["worktree", "remove", "--force", worktree.to_str().unwrap()],
        )
        .await
        .unwrap();
        run_git(
            tmp.path(),
            &["worktree", "add", worktree.to_str().unwrap(), "same-branch"],
        )
        .await
        .unwrap();
        let sentinel = worktree.join("replacement-sentinel");
        std::fs::write(&sentinel, b"replacement must survive").unwrap();

        let result = ExitWorktreeTool
            .execute(json!({"action": "remove", "discard_changes": true}), &ctx)
            .await;
        assert!(
            result.is_error,
            "same-path same-branch replacement must block removal"
        );
        assert_eq!(
            std::fs::read(&sentinel).unwrap(),
            b"replacement must survive"
        );

        run_git(
            tmp.path(),
            &["worktree", "remove", "--force", worktree.to_str().unwrap()],
        )
        .await
        .unwrap();
        reset_session().await;
    }

    #[tokio::test]
    async fn removal_allows_legitimate_commit_advancement() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().to_path_buf());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered = EnterWorktreeTool
            .execute(json!({"branch": "advancing-branch"}), &ctx)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        let worktree = tmp.path().join(".worktrees/advancing-branch");
        std::fs::write(worktree.join("advanced.txt"), b"committed").unwrap();
        run_git(&worktree, &["add", "advanced.txt"]).await.unwrap();
        run_git(&worktree, &["commit", "-m", "advance worktree branch"])
            .await
            .unwrap();

        let result = ExitWorktreeTool
            .execute(json!({"action": "remove", "discard_changes": true}), &ctx)
            .await;
        assert!(
            !result.is_error,
            "legitimate commit advancement must remain removable: {}",
            result.content
        );
        assert!(!worktree.exists());
        reset_session().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn removal_refuses_symlink_replacement_and_keep_preserves_record() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().to_path_buf());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered = EnterWorktreeTool
            .execute(json!({"branch": "symlink-owned"}), &ctx)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        let worktree = tmp.path().join(".worktrees/symlink-owned");
        let target = tmp.path().join("foreign-target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("sentinel"), b"keep me").unwrap();
        run_git(
            tmp.path(),
            &["worktree", "remove", "--force", worktree.to_str().unwrap()],
        )
        .await
        .unwrap();
        std::os::unix::fs::symlink(&target, &worktree).unwrap();

        let result = ExitWorktreeTool
            .execute(json!({"action": "remove", "discard_changes": true}), &ctx)
            .await;
        assert!(result.is_error, "symlink replacement must block removal");
        assert!(worktree.is_symlink());
        assert_eq!(std::fs::read(target.join("sentinel")).unwrap(), b"keep me");

        let kept = ExitWorktreeTool
            .execute(json!({"action": "keep"}), &ctx)
            .await;
        assert!(
            !kept.is_error,
            "keep must preserve uncertain state: {}",
            kept.content
        );
        assert!(worktree.is_symlink());
        assert_eq!(std::fs::read(target.join("sentinel")).unwrap(), b"keep me");
        std::fs::remove_file(&worktree).unwrap();
        reset_session().await;
    }

    #[tokio::test]
    async fn legacy_active_record_can_keep_but_cannot_remove() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().to_path_buf());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let entered = EnterWorktreeTool
            .execute(json!({"branch": "legacy-record"}), &ctx)
            .await;
        assert!(!entered.is_error, "{}", entered.content);
        let key = session_key(&ctx);
        let mut legacy = load_session(&key).unwrap().unwrap();
        legacy.identity = None;
        persist_session(&key, &legacy).unwrap();
        WORKTREE_SESSIONS
            .write()
            .insert(key.clone(), legacy.clone());

        let removed = ExitWorktreeTool
            .execute(json!({"action": "remove", "discard_changes": true}), &ctx)
            .await;
        assert!(removed.is_error, "legacy identity must block removal");
        assert!(legacy.worktree_path.exists());
        let kept = ExitWorktreeTool
            .execute(json!({"action": "keep"}), &ctx)
            .await;
        assert!(
            !kept.is_error,
            "keep should preserve legacy state: {}",
            kept.content
        );
        assert!(legacy.worktree_path.exists());
        reset_session().await;
    }

    #[tokio::test]
    async fn exit_clears_only_worktree_shell_and_session_cleanup_matches_scoped_keys() {
        let _guard = TEST_LOCK.lock().await;
        reset_session().await;
        let tmp = tempfile::tempdir().unwrap();
        init_git_repo(tmp.path()).await;
        let mut ctx = allow_all_context(tmp.path().into());
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let parent = crate::context_shell_state(&ctx);
        parent
            .lock()
            .env_vars
            .insert("PARENT".into(), "keep".into());
        let entered = EnterWorktreeTool
            .execute(json!({"branch":"shell-cleanup"}), &ctx)
            .await;
        assert!(!entered.is_error);
        let effective = execution_context(&ctx).unwrap();
        let key = crate::shell_scope(&effective);
        let old = crate::context_shell_state(&effective);
        old.lock()
            .env_vars
            .insert("WORKTREE".into(), "clear".into());
        let exited = ExitWorktreeTool
            .execute(json!({"action":"keep"}), &ctx)
            .await;
        assert!(!exited.is_error);
        assert!(!crate::SHELL_STATE_REGISTRY.contains_key(&key));
        assert_eq!(
            crate::context_shell_state(&ctx)
                .lock()
                .env_vars
                .get("PARENT")
                .unwrap(),
            "keep"
        );
        crate::clear_session_shell_state(&ctx.session_id);
        assert!(crate::context_shell_state(&ctx).lock().env_vars.is_empty());
        crate::clear_session_shell_state(&ctx.session_id);
        reset_session().await;
    }
}
