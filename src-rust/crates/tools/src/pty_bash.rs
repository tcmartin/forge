// PTY-backed Bash tool: wraps every command in a real pseudo-terminal so that
// programs that query isatty() (npm, cargo, git, pytest, …) behave correctly.
//
// Shell state (cwd + env) is persisted across calls through the same sentinel
// mechanism as the original BashTool, so `cd` and `export` work as expected.
//
// Platform notes
// ──────────────
//  Unix  → portable_pty (native openpty)
//  Windows → falls back to the existing cmd.exe approach; ConPTY is available
//             in portable_pty but adds complexity for minimal gain on Windows.

use crate::{context_shell_state, PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use claurst_core::bash_classifier::{classify_bash_command, BashRiskLevel};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
#[cfg(windows)]
use std::process::Stdio;
use std::time::Duration;
#[cfg(windows)]
use tokio::process::Command;
use tracing::debug;

// Unix-only imports used by the shell-state helpers and PTY execution path.
#[cfg(unix)]
use crate::ShellState;
#[cfg(unix)]
use regex::Regex;
#[cfg(unix)]
use std::collections::HashMap;

/// Sentinel appended to the shell wrapper script (Unix only).
#[cfg(unix)]
const SHELL_STATE_SENTINEL: &str = "__CC_SHELL_STATE__";

pub struct PtyBashTool;

#[derive(Debug, Deserialize)]
struct BashInput {
    command: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default)]
    run_in_background: bool,
    #[serde(default)]
    cwd: Option<String>,
}

fn default_timeout() -> u64 {
    120_000
}

/// Attach typed process outcome fields without replacing retained evidence or
/// metadata already produced by the tool.
fn with_command_metadata(
    mut result: ToolResult,
    command_status: &str,
    exit_code: Option<i32>,
    partial: bool,
) -> ToolResult {
    let mut metadata = match result.metadata.take() {
        Some(Value::Object(metadata)) => metadata,
        Some(other) => Map::from_iter([("tool_metadata".into(), other)]),
        None => Map::new(),
    };
    metadata.insert(
        "command_status".into(),
        Value::String(command_status.into()),
    );
    metadata.insert(
        "exit_code".into(),
        exit_code.map_or(Value::Null, Value::from),
    );
    metadata.insert("partial".into(), Value::Bool(partial));
    result.metadata = Some(Value::Object(metadata));
    result
}

fn not_run_result(message: impl Into<String>) -> ToolResult {
    with_command_metadata(
        ToolResult::error(message).with_execution_status(crate::ExecutionStatus::NotRun),
        "not_run",
        None,
        false,
    )
}

// ---------------------------------------------------------------------------
// Shell state helpers — Unix only (used by the PTY wrapper script)
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn parse_shell_state_block(lines: &[String]) -> Option<(PathBuf, HashMap<String, String>)> {
    let mut iter = lines.iter();
    let cwd_line = iter.next()?;
    let cwd = PathBuf::from(cwd_line.trim());

    let mut env_vars = HashMap::new();
    for line in iter {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].to_string();
            let val = line[eq + 1..].to_string();
            if !key.starts_with('_')
                && ![
                    "SHLVL",
                    "BASH_LINENO",
                    "BASH_SOURCE",
                    "FUNCNAME",
                    "PIPESTATUS",
                    "OLDPWD",
                ]
                .contains(&key.as_str())
            {
                env_vars.insert(key, val);
            }
        }
    }

    Some((cwd, env_vars))
}

#[cfg(unix)]
fn extract_exports_from_command(command: &str) -> HashMap<String, String> {
    let re =
        Regex::new(r#"(?m)^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"]*)"|'([^']*)'|(\S*))"#)
            .unwrap();
    let mut map = HashMap::new();
    for cap in re.captures_iter(command) {
        let key = cap[1].to_string();
        let val = cap
            .get(2)
            .or_else(|| cap.get(3))
            .or_else(|| cap.get(4))
            .map(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        map.insert(key, val);
    }
    map
}

// SECURITY (#211): restored env vars are NEVER emitted into this script.
// The script becomes an argv element (`bash -c "<script>"`), which is readable
// by any local user via `ps auxww` / `/proc/<pid>/cmdline`. Instead the restored
// vars are handed to the child through its ENVIRONMENT via `apply_restored_env`
// (portable_pty `CommandBuilder::env`), so secret values never touch a command line.
#[cfg(unix)]
fn build_wrapper_script(command: &str, state: &ShellState, base_cwd: &PathBuf) -> String {
    let effective_cwd = state.cwd.as_ref().unwrap_or(base_cwd);
    let cwd_escaped: String = effective_cwd.to_string_lossy().replace('\'', "'\\''");

    format!(
        r#"set -e
cd '{cwd}'
set +e
{user_cmd}
__CC_EXIT_CODE=$?
echo '{sentinel}'
pwd
env | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' || true
exit $__CC_EXIT_CODE
"#,
        cwd = cwd_escaped,
        user_cmd = command,
        sentinel = SHELL_STATE_SENTINEL,
    )
}

/// Plumb the persisted shell env vars into the child process's environment
/// (NOT its argv). Each var is set with `CommandBuilder::env`, so the child
/// inherits it exactly as `export` would have — but the value never appears on
/// any command line. This is the sole path through which restored values reach
/// the child (#211).
#[cfg(unix)]
fn apply_restored_env(cmd: &mut portable_pty::CommandBuilder, env_vars: &HashMap<String, String>) {
    for (k, v) in env_vars {
        cmd.env(k, v);
    }
}

// ---------------------------------------------------------------------------
// Background execution (no PTY needed for detached commands)
// ---------------------------------------------------------------------------

async fn run_in_background(
    command: String,
    cwd: PathBuf,
    timeout_ms: u64,
    ctx: &ToolContext,
) -> ToolResult {
    let task_id = match crate::background_processes::spawn(command.clone(), cwd, timeout_ms, ctx) {
        Ok(task_id) => task_id,
        Err(error) => return not_run_result(error),
    };
    let mut result = ToolResult::success(format!(
        "Command started in background.\nTask ID: {}\nCommand: {}",
        task_id, command
    ));
    result = with_command_metadata(result, "running", None, true);
    if let Some(Value::Object(metadata)) = result.metadata.as_mut() {
        metadata.insert("background".into(), Value::Bool(true));
        metadata.insert("task_id".into(), Value::String(task_id));
    }
    result
}

// ---------------------------------------------------------------------------
// ANSI stripping (Unix only — PTY output only happens on Unix)
// ---------------------------------------------------------------------------

/// Remove ANSI/VT escape sequences from PTY output, producing clean text.
#[cfg(unix)]
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.peek() {
                Some('[') => {
                    chars.next(); // consume '['
                                  // CSI: consume parameter + intermediate bytes, stop at final byte
                    for c in &mut chars {
                        if c.is_ascii_alphabetic() || c == '@' {
                            break;
                        }
                    }
                }
                Some(']') => {
                    // OSC: consume until ST (ESC \) or BEL
                    chars.next(); // consume ']'
                    let mut prev = '\0';
                    for c in &mut chars {
                        if c == '\x07' {
                            break; // BEL terminates OSC
                        }
                        if prev == '\x1b' && c == '\\' {
                            break; // ST = ESC \ terminates OSC
                        }
                        prev = c;
                    }
                }
                Some('(') | Some(')') | Some('*') | Some('+') => {
                    chars.next(); // consume designator introducer
                    chars.next(); // consume charset code
                }
                _ => {
                    // Two-character escape (ESC X): skip next char
                    chars.next();
                }
            }
        } else if ch == '\r' {
            // CR without LF: treat as line reset (discard pending partial line)
            // CR+LF is fine: LF will follow and push the newline
        } else {
            result.push(ch);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Unix PTY execution
// ---------------------------------------------------------------------------

/// Outcome of a single foreground PTY run.
#[cfg(unix)]
enum PtyOutcome {
    /// The command finished; carries (raw PTY output, exit code).
    Completed(String, Option<i32>, bool),
    /// The command exceeded its timeout. The child has been KILLED (#220).
    TimedOut(String),
    Cancelled(String),
    /// The PTY could not be set up / the command could not be spawned.
    Failed(String, bool),
}

/// Guard that guarantees the PTY child is killed if the running future is
/// dropped before the command completes — e.g. the turn is cancelled or the
/// task is aborted mid-command. This is the PTY analogue of tokio's
/// `kill_on_drop(true)`: portable_pty's child is NOT killed on drop, so without
/// this a cancelled command would orphan its child (#220).
///
/// The guard is disarmed on normal completion and fired explicitly on timeout.
#[cfg(unix)]
struct PtyKillGuard {
    killer: Box<dyn portable_pty::ChildKiller + Send + Sync>,
    armed: bool,
}

#[cfg(unix)]
impl PtyKillGuard {
    fn new(killer: Box<dyn portable_pty::ChildKiller + Send + Sync>) -> Self {
        Self {
            killer,
            armed: true,
        }
    }

    /// The command finished on its own — there is nothing left to kill.
    fn disarm(&mut self) {
        self.armed = false;
    }

    /// Kill the child now (timeout path) and disarm so `Drop` is a no-op.
    fn kill_now(&mut self) {
        if self.armed {
            let _ = self.killer.kill();
            self.armed = false;
        }
    }
}

#[cfg(unix)]
impl Drop for PtyKillGuard {
    fn drop(&mut self) {
        if self.armed {
            // Future dropped before completion (cancel / abort): don't leak the child.
            let _ = self.killer.kill();
        }
    }
}

#[cfg(unix)]
async fn run_in_pty(
    script: &str,
    working_dir: &str,
    env_vars: &HashMap<String, String>,
    timeout: Duration,
    cancel: &tokio_util::sync::CancellationToken,
) -> PtyOutcome {
    use portable_pty::{native_pty_system, CommandBuilder, PtySize};

    let pty_system = native_pty_system();

    let pair = match pty_system.openpty(PtySize {
        rows: 50,
        cols: 220,
        pixel_width: 0,
        pixel_height: 0,
    }) {
        Ok(p) => p,
        Err(e) => return PtyOutcome::Failed(format!("Failed to open PTY: {}", e), false),
    };

    let mut cmd = CommandBuilder::new("bash");
    cmd.args(["-c", script]);
    cmd.cwd(working_dir);
    // Restored shell vars go through the child's ENVIRONMENT, never its argv (#211).
    apply_restored_env(&mut cmd, env_vars);

    let child = match pair.slave.spawn_command(cmd) {
        Ok(c) => c,
        Err(e) => return PtyOutcome::Failed(format!("Failed to spawn in PTY: {}", e), false),
    };

    // Arm cleanup immediately after spawn, including reader setup failures.
    let mut guard = PtyKillGuard::new(child.clone_killer());

    // Grab the reader *before* dropping slave so the fd stays valid.
    let reader = match pair.master.try_clone_reader() {
        Ok(r) => r,
        Err(e) => return PtyOutcome::Failed(format!("Failed to clone PTY reader: {}", e), true),
    };

    // A killer handle cloned from the child. This is how a timed-out or
    // cancelled command is explicitly KILLED (portable_pty does not kill on
    // drop, so without this the child would orphan) (#220).
    // Drop slave after spawn — once the child's controlling terminal is gone,
    // the master side will see EOF when the child exits.
    drop(pair.slave);
    // Keep master alive (inside the read thread) until reading is done.
    let master = pair.master;

    // Read all PTY output in a blocking thread (portable_pty reader is sync) and
    // reap the child there so we can return its exit code. drive_pty_child stops
    // reading as soon as the DIRECT child exits rather than waiting for pty EOF,
    // which a detached grandchild would hold open forever (#184).
    let mut read_handle = tokio::task::spawn_blocking(move || {
        let master_fd = master.as_raw_fd();
        let result = drive_pty_child(child, reader, master_fd);
        // Keep the master fd alive until reading + reaping is complete.
        drop(master);
        result
    });

    // Kill-on-drop guard: if this future is dropped mid-command (turn cancelled)
    // the guard fires and kills the child instead of leaking it (#220).
    let interrupted = tokio::select! {
        biased;
        result = tokio::time::timeout(timeout, &mut read_handle) => Some(result),
        _ = cancel.cancelled() => None,
    };
    let was_cancelled = interrupted.is_none();
    match interrupted {
        Some(Ok(Ok((output, exit_code, partial)))) => {
            guard.disarm();
            PtyOutcome::Completed(output, exit_code, partial)
        }
        Some(Ok(Err(e))) => {
            guard.kill_now();
            PtyOutcome::Failed(format!("PTY read thread panicked: {}", e), true)
        }
        Some(Err(_)) | None => {
            let cancelled = was_cancelled;
            // Interrupted: explicitly KILL the child so it can't linger as an
            // orphan (#220). The read thread then observes the exit and finishes.
            guard.kill_now();
            let partial = match tokio::time::timeout(Duration::from_secs(2), &mut read_handle).await
            {
                Ok(Ok((output, _, _))) => output,
                _ => String::new(),
            };
            if cancelled {
                PtyOutcome::Cancelled(partial)
            } else {
                PtyOutcome::TimedOut(partial)
            }
        }
    }
}

/// Poll interval used to wake the PTY read loop so it can notice that the direct
/// child has exited even while a detached grandchild still holds the pty open.
#[cfg(unix)]
const PTY_POLL_INTERVAL_MS: i32 = 20;

/// Wait up to `timeout_ms` for `fd` to become readable. Returns `true` when the
/// caller should attempt a read — data ready, EOF/hangup, or a poll error we'd
/// rather surface through `read` — and `false` on a clean timeout with nothing
/// pending.
#[cfg(unix)]
fn poll_readable(fd: std::os::unix::io::RawFd, timeout_ms: i32) -> bool {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        let rc = unsafe { libc::poll(&mut pfd as *mut libc::pollfd, 1, timeout_ms) };
        if rc < 0 {
            // Retry on EINTR; on any other error let the subsequent `read` surface it.
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return true;
        }
        // rc == 0 → clean timeout, nothing ready. rc > 0 → POLLIN/POLLHUP/POLLERR.
        return rc > 0;
    }
}

/// Drive a spawned PTY child to completion: read its output while it runs, reap
/// it, and return `(output, exit_code)`.
///
/// Crucially — and mirroring bash.rs's `drive_child` — the loop stops as soon as
/// the DIRECT child exits rather than waiting for pty EOF. A grandchild that
/// detached into its own session (`setsid`) inherits the pty slave and would
/// otherwise hold it open indefinitely, hanging the tool for the grandchild's
/// full lifetime (#184). `poll_readable` lets the blocking read wake up
/// periodically so we can re-check `try_wait()` without spinning.
#[cfg(unix)]
fn drive_pty_child(
    mut child: Box<dyn portable_pty::Child + Send + Sync>,
    mut reader: Box<dyn std::io::Read + Send>,
    master_fd: Option<std::os::unix::io::RawFd>,
) -> (String, Option<i32>, bool) {
    use std::io::Read;

    let mut output = Vec::new();
    let mut buf = [0u8; 4096];
    const MAX_BYTES: usize = 32 * 1024 * 1024;
    let mut partial = false;

    // Set once the direct child has exited. We then drain any already-buffered
    // pty output and stop — we never block waiting for EOF that a detached
    // grandchild would hold open forever (#184).
    let mut child_exited = false;

    loop {
        // Wait briefly for data. When we can poll, this lets the loop wake up to
        // re-check the child's liveness even while the pty is held open elsewhere.
        let data_ready = match master_fd {
            Some(fd) => poll_readable(fd, PTY_POLL_INTERVAL_MS),
            None => true, // can't poll — fall back to a plain blocking read
        };

        if data_ready {
            match reader.read(&mut buf) {
                Ok(0) => break, // EOF: every pty writer (incl. the child) is gone
                Ok(n) => {
                    let retained = n.min(MAX_BYTES.saturating_sub(output.len()));
                    output.extend_from_slice(&buf[..retained]);
                    partial |= retained < n;
                    // Keep draining after the capture limit. Stopping reads can
                    // deadlock a still-running child on a full PTY buffer.
                    continue; // keep draining while bytes remain
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => break,
            }
        }

        // Nothing was ready this cycle.
        if child_exited {
            // Direct child already exited and the pty is now idle → stop instead
            // of hanging on a detached grandchild still holding it open (#184).
            break;
        }
        match child.try_wait() {
            Ok(Some(_)) => {
                // Direct child is done. Loop once more to drain any final buffered
                // bytes; the `child_exited` guard above then breaks us out.
                child_exited = true;
            }
            Ok(None) => {}   // still running
            Err(_) => break, // can't observe the child — bail rather than spin
        }
    }

    let exit_code = child.wait().ok().map(|status| status.exit_code() as i32);
    // Decode once; a UTF-8 character may span arbitrary reader chunks.
    (
        String::from_utf8_lossy(&output).into_owned(),
        exit_code,
        partial,
    )
}

// ---------------------------------------------------------------------------
// Windows fallback (cmd.exe, no PTY)
// ---------------------------------------------------------------------------

#[cfg(windows)]
async fn run_windows_fallback(
    command: &str,
    effective_cwd: &PathBuf,
    timeout_dur: Duration,
    timeout_ms: u64,
    ctx: &ToolContext,
) -> ToolResult {
    let child = match Command::new("cmd")
        .arg("/C")
        .arg(command)
        .current_dir(effective_cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return not_run_result(format!("Failed to spawn command: {}", e)),
    };

    let captured =
        match crate::command_capture::capture_with_cancel(child, timeout_dur, &ctx.cancel_token)
            .await
        {
            Ok(output) => output,
            Err(e) => {
                return with_command_metadata(
                    ToolResult::error(format!("Failed to capture command output: {e}")),
                    "unknown",
                    None,
                    true,
                )
            }
        };
    let exit_code = captured.status.and_then(|status| status.code());
    let mut output = String::from_utf8_lossy(&captured.stdout).into_owned();
    if !captured.stderr.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str("STDERR:\n");
        output.push_str(&String::from_utf8_lossy(&captured.stderr));
    }
    if output.is_empty() {
        output = "(no output)".to_string();
    }
    if captured.cancelled || captured.error.is_some() {
        let reason = captured.error.as_deref().unwrap_or("command cancelled");
        let result = crate::evidence::retain_output(
            ctx,
            format!("Command capture incomplete: {reason}\n{output}"),
            true,
            true,
        )
        .with_execution_status(crate::ExecutionStatus::Unknown);
        return with_command_metadata(result, "cancelled", None, true);
    }
    if captured.timed_out {
        let result = crate::evidence::retain_output(
            ctx,
            format!("Command timed out after {timeout_ms}ms; capture incomplete\n{output}"),
            true,
            true,
        )
        .with_execution_status(crate::ExecutionStatus::Unknown);
        return with_command_metadata(result, "timed_out", None, true);
    }
    let status = if exit_code.is_some() {
        "completed"
    } else {
        "unknown"
    };
    let result = truncate_output(output, exit_code, captured.partial, ctx);
    with_command_metadata(result, status, exit_code, captured.partial)
}

// ---------------------------------------------------------------------------
// Shared output truncation helper
// ---------------------------------------------------------------------------

fn truncate_output(
    output: String,
    exit_code: Option<i32>,
    partial: bool,
    ctx: &ToolContext,
) -> ToolResult {
    let output = if partial {
        format!("[Capture incomplete: stream capture limit reached]\n{output}")
    } else {
        output
    };
    let is_error = exit_code != Some(0);
    let output = match exit_code {
        Some(code) if code != 0 => format!("Command exited with code {code}\n{output}"),
        Some(_) => output,
        None => format!("Command exit status unavailable\n{output}"),
    };
    crate::evidence::retain_output(ctx, output, is_error, partial)
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Tool for PtyBashTool {
    // Gates itself: calls `ctx.check_permission*` in `execute()` (#210).
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        claurst_core::constants::TOOL_NAME_BASH
    }

    fn description(&self) -> &str {
        "Executes a given bash command in a real terminal (PTY) and returns its output. \
         The working directory persists between commands. Supports interactive programs, \
         colored output (stripped for readability), and terminal-aware tools like npm, \
         cargo, git, and pytest. Use for running shell commands, scripts, git operations, \
         and system tasks."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn native_bash_execution_witness(
        &self,
        tool_id: &str,
        result: &ToolResult,
        emitted_result: &str,
    ) -> Option<crate::NativeBashExecutionWitness> {
        crate::NativeBashExecutionWitness::from_trusted_result(tool_id, result, emitted_result)
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "description": {
                    "type": "string",
                    "description": "Clear, concise description of what this command does"
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in milliseconds (max 600000, default 120000)"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Set to true to run command in the background"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for this call. Relative paths resolve against the workspace. Does not change the regular shell's working directory."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: BashInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return not_run_result(format!("Invalid input: {}", e)),
        };
        let cwd_override = params.cwd.as_deref().map(|path| ctx.resolve_path(path));
        if cwd_override.as_ref().is_some_and(|path| !path.is_dir()) {
            return not_run_result("The requested command working directory does not exist");
        }
        if let Err(error) = crate::deletion_intent::authorize_bash_command(&params.command) {
            return not_run_result(error);
        }

        // Permission check
        let reason = params
            .description
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("This will execute a shell command.")
            .to_string();

        if let Err(e) = ctx.check_permission_for_path(
            self.name(),
            &reason,
            std::path::PathBuf::from(&params.command),
            false,
        ) {
            return not_run_result(e.to_string());
        }

        // Security classifier — block Critical-risk commands unconditionally.
        if classify_bash_command(&params.command) == BashRiskLevel::Critical {
            return not_run_result(format!(
                "Command blocked: classified as Critical risk by the bash security classifier.\n\
                 Refusing to execute: {}",
                params.command
            ));
        }

        let timeout_ms = params.timeout.min(600_000);
        let timeout_dur = Duration::from_millis(timeout_ms);
        let shell_state_arc = context_shell_state(ctx);

        // ── Background path ──────────────────────────────────────────────────
        if params.run_in_background {
            let cwd = {
                let state = shell_state_arc.lock();
                cwd_override
                    .clone()
                    .or_else(|| state.cwd.clone())
                    .unwrap_or_else(|| ctx.working_dir.clone())
            };
            return run_in_background(params.command, cwd, timeout_ms, ctx).await;
        }

        debug!(
            command_bytes = params.command.len(),
            "Executing bash command via PTY"
        );

        // ── Windows path (no PTY — use cmd.exe fallback) ─────────────────────
        #[cfg(windows)]
        {
            let effective_cwd = {
                let state = shell_state_arc.lock();
                cwd_override
                    .clone()
                    .or_else(|| state.cwd.clone())
                    .unwrap_or_else(|| ctx.working_dir.clone())
            };
            return run_windows_fallback(
                &params.command,
                &effective_cwd,
                timeout_dur,
                timeout_ms,
                ctx,
            )
            .await;
        }

        // ── Unix PTY path ────────────────────────────────────────────────────
        #[cfg(unix)]
        {
            // Build the wrapper script that restores cwd + captures shell state.
            // Restored env vars are cloned out and later injected into the child's
            // environment (never its argv) — see `apply_restored_env` (#211).
            let (script, restored_env, working_dir_str) = {
                let state = shell_state_arc.lock();
                let mut call_state = ShellState::new();
                call_state.cwd = cwd_override.clone().or_else(|| state.cwd.clone());
                let script = build_wrapper_script(&params.command, &call_state, &ctx.working_dir);
                let restored_env = state.env_vars.clone();
                let wd = ctx.working_dir.to_string_lossy().into_owned();
                (script, restored_env, wd)
            };

            // run_in_pty owns the timeout so it can KILL the child when it fires
            // (a bare outer timeout would just drop the future and orphan it) (#220).
            let outcome = run_in_pty(
                &script,
                &working_dir_str,
                &restored_env,
                timeout_dur,
                &ctx.cancel_token,
            )
            .await;

            match outcome {
                PtyOutcome::Completed(raw_output, exit_code, partial) => {
                    // Strip ANSI escape codes from PTY output
                    let cleaned = strip_ansi(&raw_output);

                    // Split into user-visible lines and state block
                    let all_lines: Vec<String> = cleaned.lines().map(|l| l.to_string()).collect();

                    let sentinel_pos = all_lines
                        .iter()
                        .rposition(|l| l.trim() == SHELL_STATE_SENTINEL);

                    let (user_lines, state_lines) = match sentinel_pos {
                        Some(pos) => (&all_lines[..pos], &all_lines[pos + 1..]),
                        None => (all_lines.as_slice(), &[][..]),
                    };

                    // Update persistent shell state
                    if !state_lines.is_empty() {
                        if let Some((new_cwd, env_delta)) = parse_shell_state_block(state_lines) {
                            let mut state = shell_state_arc.lock();
                            if cwd_override.is_none() {
                                state.cwd = Some(new_cwd);
                            }
                            for (k, v) in env_delta {
                                state.env_vars.insert(k, v);
                            }
                        }
                    }

                    // Fast-path export capture
                    {
                        let exports = extract_exports_from_command(&params.command);
                        if !exports.is_empty() {
                            let mut state = shell_state_arc.lock();
                            for (k, v) in exports {
                                state.env_vars.insert(k, v);
                            }
                        }
                    }

                    let mut output = user_lines.join("\n");
                    if output.is_empty() {
                        output = "(no output)".to_string();
                    }

                    let status = if exit_code.is_some() {
                        "completed"
                    } else {
                        "unknown"
                    };
                    let result = truncate_output(output, exit_code, partial, ctx);
                    with_command_metadata(result, status, exit_code, partial)
                }
                PtyOutcome::Failed(e, spawned) => with_command_metadata(
                    ToolResult::error(format!("PTY execution failed: {}", e)),
                    if spawned { "unknown" } else { "not_run" },
                    None,
                    false,
                ),
                PtyOutcome::Cancelled(raw_output) => {
                    let cleaned = strip_ansi(&raw_output);
                    let output = cleaned.split(SHELL_STATE_SENTINEL).next().unwrap_or("");
                    let result = crate::evidence::retain_output(
                        ctx,
                        format!("Command cancelled; capture incomplete\n{output}"),
                        true,
                        true,
                    )
                    .with_execution_status(crate::ExecutionStatus::Unknown);
                    with_command_metadata(result, "cancelled", None, true)
                }
                PtyOutcome::TimedOut(raw_output) => {
                    let cleaned = strip_ansi(&raw_output);
                    let output = cleaned.split(SHELL_STATE_SENTINEL).next().unwrap_or("");
                    let result = crate::evidence::retain_output(
                        ctx,
                        format!(
                            "Command timed out after {timeout_ms}ms; capture incomplete\n{output}"
                        ),
                        true,
                        true,
                    )
                    .with_execution_status(crate::ExecutionStatus::Unknown);
                    with_command_metadata(result, "timed_out", None, true)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn native_witness_is_bounded_and_hashes_emitted_utf8() {
        let cases = [
            ("completed", Some(0), false, true),
            ("completed", Some(7), false, true),
            ("completed", None, false, true),
            ("completed", Some(0), true, true),
            ("cancelled", None, true, true),
            ("timed_out", None, true, true),
            ("not_run", None, false, true),
            ("unknown", None, false, true),
        ];
        for (status, exit_code, partial, expected) in cases {
            let result = ToolResult::success("native display").with_metadata(json!({
                "command_status": status,
                "exit_code": exit_code,
                "partial": partial,
                "private_command": "must not cross the event boundary",
            }));
            assert_eq!(
                PtyBashTool
                    .native_bash_execution_witness("call-1", &result, "native display")
                    .is_some(),
                expected
            );
        }

        let result = ToolResult::success("native display").with_metadata(json!({
            "command_status": "completed",
            "exit_code": 0,
            "partial": false,
        }));
        let witness = PtyBashTool
            .native_bash_execution_witness("call-1", &result, "native display")
            .expect("valid native metadata should produce a witness");
        assert_eq!(witness.tool_id, "call-1");
        assert_eq!(
            witness.displayed_result_sha256,
            claurst_core::crypto_utils::sha256_hex(b"native display")
        );
        assert!(PtyBashTool
            .native_bash_execution_witness("call-1", &result, "forged display")
            .is_some_and(|witness| witness.displayed_result_sha256
                == claurst_core::crypto_utils::sha256_hex(b"forged display")));

        let malformed = [
            json!({"command_status":"running","exit_code":null,"partial":true}),
            json!({"command_status":"completed","exit_code":1.5,"partial":false}),
            json!({"command_status":"completed","exit_code":0,"partial":"false"}),
            json!("forged metadata"),
        ];
        for metadata in malformed {
            let result = ToolResult::success("native display").with_metadata(metadata);
            assert!(PtyBashTool
                .native_bash_execution_witness("call-1", &result, "native display")
                .is_none());
        }
        assert!(PtyBashTool
            .native_bash_execution_witness("", &result, "native display")
            .is_none());
        assert!(PtyBashTool
            .native_bash_execution_witness(&"x".repeat(257), &result, "native display")
            .is_none());
    }
    use std::ffi::OsStr;

    /// #211: restored env values (secrets) must NOT be baked into the wrapper
    /// script, because that script becomes an argv element of `bash -c`, which is
    /// visible to any local user via `ps auxww` / `/proc/<pid>/cmdline`.
    #[test]
    fn wrapper_script_never_embeds_restored_env_values() {
        let mut state = ShellState::new();
        state
            .env_vars
            .insert("SECRET".to_string(), "topsecret".to_string());
        state.env_vars.insert(
            "AWS_SECRET_ACCESS_KEY".to_string(),
            "aws-argv-leak".to_string(),
        );

        let base = PathBuf::from("/tmp");
        let script = build_wrapper_script("echo hi", &state, &base);

        // The secret VALUES must never appear in the script string / argv.
        assert!(
            !script.contains("topsecret"),
            "secret value leaked into wrapper script (argv):\n{script}"
        );
        assert!(
            !script.contains("aws-argv-leak"),
            "AWS secret leaked into wrapper script (argv):\n{script}"
        );
        // And there must be no re-exported `export KEY=` lines for restored vars.
        assert!(
            !script.contains("export SECRET="),
            "restored var was re-exported into the script (argv):\n{script}"
        );
        assert!(
            !script.contains("export AWS_SECRET_ACCESS_KEY="),
            "restored var was re-exported into the script (argv):\n{script}"
        );
    }

    /// #211: the value IS delivered to the child — but through its environment
    /// (`CommandBuilder::env`), which is not part of argv. This exercises the
    /// env-plumbing function directly.
    #[test]
    fn restored_env_reaches_child_via_env_map_not_argv() {
        use portable_pty::CommandBuilder;

        let mut env_vars = HashMap::new();
        env_vars.insert("SECRET".to_string(), "topsecret".to_string());

        let mut cmd = CommandBuilder::new("bash");
        cmd.args(["-c", "true"]);
        apply_restored_env(&mut cmd, &env_vars);

        // Value is present in the child's environment...
        assert_eq!(cmd.get_env("SECRET"), Some(OsStr::new("topsecret")));
        // ...and NOT in argv (only `bash -c true`).
        for arg in cmd.get_argv() {
            assert!(
                !arg.to_string_lossy().contains("topsecret"),
                "secret value leaked into CommandBuilder argv: {arg:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Execution tests (#220 / #184) — exercise the live PTY path end-to-end.
    // -----------------------------------------------------------------------

    /// Permission handler that allows everything — for exercising `execute`.
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

    struct DenyAllHandler;

    impl claurst_core::permissions::PermissionHandler for DenyAllHandler {
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

    fn allow_all_context() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: std::sync::Arc::new(AllowAllHandler),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "pty-bash-test".to_string(),
            file_history: std::sync::Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: claurst_core::config::Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            trusted_checker_catalog: Default::default(),
            catalog_view: None,
        }
    }

    /// #220: a command that exceeds its timeout must have its child KILLED, not
    /// merely dropped. portable_pty does not kill the child on drop, so before
    /// the fix a timed-out command left an orphaned process running to its
    /// natural end. We time out a long `sleep` and then assert the process is
    /// gone.
    #[cfg(not(windows))]
    #[tokio::test]
    async fn timeout_kills_the_child() {
        let tool = PtyBashTool;
        let ctx = allow_all_context();

        // A distinctive duration doubles as a searchable marker for the process.
        let marker = "sleep 31337";
        let input = json!({
            "command": marker,
            "timeout": 500u64, // ms — far shorter than the sleep
        });

        let started = std::time::Instant::now();
        let result = tool.execute(input, &ctx).await;
        let elapsed = started.elapsed();

        assert!(
            result.is_error,
            "a timed-out command should be an error, got: {}",
            result.content
        );
        assert!(
            result.content.contains("timed out"),
            "expected a timeout message, got: {}",
            result.content
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "execute should return shortly after the 500ms timeout, took {:?}",
            elapsed
        );

        // Give the kill a moment to propagate, then assert the sleep is gone.
        tokio::time::sleep(Duration::from_millis(500)).await;
        if let Ok(out) = std::process::Command::new("pgrep")
            .arg("-f")
            .arg(marker)
            .output()
        {
            let pids = String::from_utf8_lossy(&out.stdout);
            let lingering: Vec<&str> = pids.split_whitespace().collect();
            assert!(
                lingering.is_empty(),
                "timed-out child `{marker}` was not killed; still running (pids: {lingering:?})",
            );
        }
    }

    /// Regression test for #184: a tool command that spawns a child which
    /// detaches into its own session (`setsid`) inherits the pty slave. The
    /// read loop must stop once the *direct* child exits instead of blocking on
    /// pty EOF held open by the detached grandchild — otherwise the tool (and the
    /// agent turn) hangs for the grandchild's full lifetime.
    #[cfg(not(windows))]
    #[tokio::test]
    async fn foreground_does_not_hang_on_detached_grandchild() {
        let tool = PtyBashTool;
        let ctx = allow_all_context();

        // `setsid sleep 30` runs in a brand-new session but inherits the pty;
        // `&` lets the wrapper shell return immediately without waiting.
        let input = json!({
            "command": "setsid sleep 30 & echo spawned",
            "timeout": 120000u64,
        });

        let started = std::time::Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(10), tool.execute(input, &ctx)).await;
        let elapsed = started.elapsed();

        let result = result
            .expect("execute hung waiting on the detached grandchild's pty (regression of #184)");
        assert!(
            !result.is_error,
            "command should have succeeded, got: {}",
            result.content
        );
        assert!(
            result.content.contains("spawned"),
            "expected the parent command's output, got: {}",
            result.content
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "execute should return promptly after the direct child exits, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn large_unicode_output_retains_missing_middle_and_failure_tail() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context();
        ctx.working_dir = workspace.path().into();
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let command = "awk 'BEGIN { for (i=0; i<20000; i++) { if(i==10000) print \"distinctive-middle-fact\"; print \"🦀abcdefgh\" } print \"tail-failure\" }'; exit 7";
        let result = PtyBashTool.execute(json!({"command":command}), &ctx).await;
        assert!(result.is_error);
        assert!(result.content.len() < 34_000);
        assert!(result.content.contains("tail-failure"));
        assert!(!result.content.contains("distinctive-middle-fact"));
        assert!(!result.content.contains('\u{fffd}'));
        let metadata = result.metadata.as_ref().unwrap();
        assert_eq!(metadata["command_status"], "completed");
        assert_eq!(metadata["exit_code"], 7);
        assert_eq!(metadata["partial"], false);
        let id = metadata["evidence"]["id"].as_str().unwrap();
        let (reference, original) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert!(!reference.partial);
        assert!(original.starts_with("Command exited with code 7"));
        assert!(original.contains("distinctive-middle-fact"));
        assert!(original.contains("tail-failure"));
        assert!(!original.contains('\u{fffd}'));
        assert!(!original.contains(SHELL_STATE_SENTINEL));
    }

    #[tokio::test]
    async fn successful_foreground_run_reports_typed_completion() {
        let ctx = allow_all_context();
        let result = PtyBashTool
            .execute(json!({"command": "printf 'ok\\n'"}), &ctx)
            .await;

        assert!(!result.is_error);
        let metadata = result.metadata.as_ref().unwrap();
        assert_eq!(metadata["command_status"], "completed");
        assert_eq!(metadata["exit_code"], 0);
        assert_eq!(metadata["partial"], false);
        assert!(metadata.get("evidence").is_some());
    }

    #[tokio::test]
    async fn background_run_reports_running_without_completion() {
        let ctx = allow_all_context();
        let result = PtyBashTool
            .execute(
                json!({"command": "sleep 1", "run_in_background": true}),
                &ctx,
            )
            .await;

        assert!(!result.is_error);
        let metadata = result.metadata.as_ref().unwrap();
        assert_eq!(metadata["command_status"], "running");
        assert_eq!(metadata["exit_code"], Value::Null);
        assert_eq!(metadata["partial"], true);
        assert_eq!(metadata["background"], true);
        let task_id = metadata["task_id"].as_str().unwrap();
        crate::background_processes::stop(task_id, &ctx).unwrap();
    }

    #[tokio::test]
    async fn invalid_foreground_input_is_typed_not_run() {
        let ctx = allow_all_context();
        let result = PtyBashTool.execute(json!({"timeout": "bad"}), &ctx).await;

        assert!(result.is_error);
        let metadata = result.metadata.as_ref().unwrap();
        assert_eq!(metadata["command_status"], "not_run");
        assert_eq!(metadata["exit_code"], Value::Null);
        assert_eq!(metadata["partial"], false);
    }
    #[tokio::test]
    async fn timeout_retains_partial_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context();
        ctx.working_dir = workspace.path().into();
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let result = PtyBashTool
            .execute(
                json!({
                    "command": "printf 'partial-before-timeout🦀\\n'; sleep 10",
                    "timeout": 150u64,
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
        let id = result.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap();
        let (reference, original) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert!(reference.partial);
        assert_eq!(
            result.metadata.as_ref().unwrap()["execution_status"],
            "unknown"
        );
        assert_eq!(
            result.metadata.as_ref().unwrap()["command_status"],
            "timed_out"
        );
        assert_eq!(result.metadata.as_ref().unwrap()["exit_code"], Value::Null);
        assert!(original.contains("partial-before-timeout🦀"));
        assert!(!original.contains(SHELL_STATE_SENTINEL));
    }
    #[tokio::test]
    async fn cooperative_cancel_retains_partial_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context();
        ctx.working_dir = workspace.path().into();
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let ready = workspace.path().join("cancel-ready");
        let trigger = ctx.cancel_token.clone();
        let cancellation = async move {
            tokio::time::timeout(Duration::from_secs(5), async {
                while !ready.exists() {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            trigger.cancel();
        };
        let execution = PtyBashTool.execute(
            json!({
                "command":"printf 'captured-before-cancel🦀\\n'; touch cancel-ready; sleep 10",
            }),
            &ctx,
        );
        let (result, ()) = tokio::join!(execution, cancellation);
        assert!(result.is_error);
        assert_eq!(
            result.metadata.as_ref().unwrap()["execution_status"],
            "unknown"
        );
        assert_eq!(
            result.metadata.as_ref().unwrap()["command_status"],
            "cancelled"
        );
        assert_eq!(result.metadata.as_ref().unwrap()["exit_code"], Value::Null);
        let id = result.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap();
        let (reference, text) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert!(reference.partial);
        assert!(text.contains("captured-before-cancel🦀"));
    }

    #[tokio::test]
    async fn neutral_filename_mention_does_not_authorize_deletion() {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("reportfixture.txt");
        std::fs::write(&target, "keep").unwrap();
        let mut ctx = allow_all_context();
        ctx.working_dir = workspace.path().into();
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let authorization = crate::deletion_intent::DeletionAuthorization::from_user_text(
            "Inspect reportfixture.txt and tell me its length.",
        );

        let result = crate::deletion_intent::with_deletion_authorization(
            authorization,
            PtyBashTool.execute(json!({"command": "rm -- reportfixture.txt"}), &ctx),
        )
        .await;

        assert!(result.is_error);
        assert_eq!(
            result.execution_status(),
            Some(crate::ExecutionStatus::NotRun)
        );
        assert!(target.is_file(), "neutral mention must preserve the target");
    }

    #[tokio::test]
    async fn explicit_target_bound_intent_preserves_allowed_deletion() {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("authorizedfixture.txt");
        std::fs::write(&target, "delete").unwrap();
        let mut ctx = allow_all_context();
        ctx.working_dir = workspace.path().into();
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        let authorization = crate::deletion_intent::DeletionAuthorization::from_user_text(
            "Please delete authorizedfixture.txt.",
        );

        let result = crate::deletion_intent::with_deletion_authorization(
            authorization,
            PtyBashTool.execute(json!({"command": "rm -- authorizedfixture.txt"}), &ctx),
        )
        .await;

        assert!(!result.is_error, "{}", result.content);
        assert!(!target.exists(), "authorized deletion should reach Bash");
    }

    #[tokio::test]
    async fn explicit_intent_does_not_bypass_permission_denial() {
        let workspace = tempfile::tempdir().unwrap();
        let target = workspace.path().join("deniedfixture.txt");
        std::fs::write(&target, "keep").unwrap();
        let mut ctx = allow_all_context();
        ctx.working_dir = workspace.path().into();
        ctx.session_id = uuid::Uuid::new_v4().to_string();
        ctx.permission_handler = std::sync::Arc::new(DenyAllHandler);
        let authorization = crate::deletion_intent::DeletionAuthorization::from_user_text(
            "Please delete deniedfixture.txt.",
        );

        let result = crate::deletion_intent::with_deletion_authorization(
            authorization,
            PtyBashTool.execute(json!({"command": "rm -- deniedfixture.txt"}), &ctx),
        )
        .await;

        assert!(result.is_error);
        assert_eq!(
            result.execution_status(),
            Some(crate::ExecutionStatus::NotRun)
        );
        assert!(
            target.is_file(),
            "permission denial must preserve the target"
        );
    }
}
