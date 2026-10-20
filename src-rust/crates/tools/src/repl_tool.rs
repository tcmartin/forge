// REPLTool: executes code in a persistent, scoped interpreter session.
//
// Each session owns its child process, process group, pipes, cancellation
// token, and execution identity. Calls use an unpredictable per-call frame so
// user output cannot forge completion by guessing a fixed sentinel.

use crate::command_capture::ProcessGroupGuard;
use crate::evidence::retain_output;
use crate::mutation::{register_pending, scope_for_workspace, PendingMutationTicket};
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use claurst_core::bash_classifier::{classify_bash_command, BashRiskLevel};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout, Instant};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

const MAX_REPL_SESSIONS: usize = 64;
const MAX_STREAM_BYTES: usize = 256 * 1024;
const REPL_CALL_DEADLINE: Duration = Duration::from_secs(30);
const FRAME_PREFIX: &str = "__FORGE_REPL_DONE_";
const ASYNC_PENDING_PREFIX: &str = "__FORGE_REPL_ASYNC_PENDING__";

// Node's interactive REPL evaluates each stdin line in its own lexical
// wrapper. A tiny JSON-line driver gives every invocation the same vm context
// while keeping request framing independent from user code.
const NODE_REPL_SERVER: &str = r#"
const fs = require('fs');
const vm = require('vm');
const Console = require('console').Console;
const nativeSetTimeout = setTimeout;
const nativeClearTimeout = clearTimeout;
const nativeSetInterval = setInterval;
const nativeClearInterval = clearInterval;
const activeTimers = new Set();
const trackedSetTimeout = (fn, delay, ...args) => {
  let handle;
  handle = nativeSetTimeout(() => {
    activeTimers.delete(handle);
    fn(...args);
  }, delay);
  activeTimers.add(handle);
  return handle;
};
const trackedClearTimeout = handle => {
  activeTimers.delete(handle);
  nativeClearTimeout(handle);
};
const trackedSetInterval = (fn, delay, ...args) => {
  const handle = nativeSetInterval(fn, delay, ...args);
  activeTimers.add(handle);
  return handle;
};
const trackedClearInterval = handle => {
  activeTimers.delete(handle);
  nativeClearInterval(handle);
};
const context = vm.createContext({process, require, console: new Console(process.stdout, process.stderr), Buffer, setTimeout: trackedSetTimeout, clearTimeout: trackedClearTimeout, setInterval: trackedSetInterval, clearInterval: trackedClearInterval, URL, URLSearchParams, TextEncoder, TextDecoder});
context.globalThis = context;
context.global = context;
let input = '';
let queue = Promise.resolve();
const end = stream => new Promise(resolve => stream.end(resolve));
async function run(request) {
  const oldOut = process.stdout;
  const oldErr = process.stderr;
  let out;
  let err;
  let pendingAsync = false;
  try {
    out = fs.createWriteStream(request.out, {flags: 'w'});
    err = fs.createWriteStream(request.err, {flags: 'w'});
    process.stdout = out;
    process.stderr = err;
    context.console = new Console(out, err);
    try {
      const result = vm.runInContext(request.code, context);
      // Promise.resolve assimilates promises created by the vm context even
      // when their constructor is from a different realm.
      await Promise.resolve(result);
      const deadline = Date.now() + 350;
      while (activeTimers.size && Date.now() < deadline) {
        await new Promise(resolve => nativeSetTimeout(resolve, 10));
      }
      pendingAsync = activeTimers.size !== 0;
    } catch (error) {
      process.stderr.write(String(error));
    }
    await end(out);
    await end(err);
  } finally {
    process.stdout = oldOut;
    process.stderr = oldErr;
  }
  fs.writeSync(3, (pendingAsync ? "__FORGE_REPL_ASYNC_PENDING__" : "") + request.marker + '\n');
}
function dispatch(line) {
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    process.stderr.write(String(error));
    return;
  }
  queue = queue.then(() => run(request));
}
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
  input += chunk;
  let newline;
  while ((newline = input.indexOf('\n')) >= 0) {
    const line = input.slice(0, newline);
    input = input.slice(newline + 1);
    if (line) dispatch(line);
  }
});
"#;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReplKey {
    workspace: PathBuf,
    session: String,
    execution_root: PathBuf,
    language: String,
}

struct ReplSession {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    stderr: ChildStderr,
    group: ProcessGroupGuard,
    cancel: CancellationToken,
    pending: PendingMutationTicket,
    retiring: bool,
}

impl ReplSession {
    fn healthy(&mut self) -> bool {
        if self.retiring {
            return false;
        }
        #[cfg(unix)]
        {
            self.group.child_running().unwrap_or(false)
        }
        #[cfg(windows)]
        {
            matches!(self.child.try_wait(), Ok(None))
        }
    }
}

static REPL_SESSIONS: Lazy<Arc<DashMap<ReplKey, Arc<Mutex<ReplSession>>>>> =
    Lazy::new(|| Arc::new(DashMap::new()));
static REPL_CANCELS: Lazy<Arc<DashMap<ReplKey, (Weak<Mutex<ReplSession>>, CancellationToken)>>> =
    Lazy::new(|| Arc::new(DashMap::new()));
static REPL_REGISTRATION: Lazy<parking_lot::Mutex<()>> = Lazy::new(|| parking_lot::Mutex::new(()));

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn key_for(ctx: &ToolContext, language: &str) -> ReplKey {
    ReplKey {
        workspace: canonical(&crate::worktree::workspace_root(ctx)),
        session: ctx.session_id.clone(),
        execution_root: canonical(&ctx.working_dir),
        language: language.to_string(),
    }
}

fn interpreter_for(language: &str) -> Option<(&'static str, Vec<&'static str>)> {
    match language {
        "python" | "python3" => Some(("python3", vec!["-u", "-i"])),
        // Keep fd 3 attached to the original control pipe. The driver uses a
        // persistent vm context rather than Node's line-oriented REPL.
        "javascript" | "node" => Some((
            "sh",
            vec![
                "-c",
                "exec 3>&1; exec node -e \"$1\"",
                "forge-node",
                NODE_REPL_SERVER,
            ],
        )),
        "bash" | "sh" | "" => Some(("bash", vec!["--norc", "--noprofile"])),
        _ => None,
    }
}

fn frame_marker() -> String {
    format!("{FRAME_PREFIX}{}__", Uuid::new_v4().simple())
}

#[cfg(unix)]
struct CallChannels {
    _directory: tempfile::TempDir,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    stdout_keeper: std::fs::File,
    stderr_keeper: std::fs::File,
}

#[cfg(unix)]
impl CallChannels {
    fn new() -> Result<Self, String> {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::OpenOptionsExt;

        let directory =
            tempfile::tempdir().map_err(|error| format!("REPL channel directory: {error}"))?;
        let stdout_path = directory.path().join("stdout");
        let stderr_path = directory.path().join("stderr");
        for path in [&stdout_path, &stderr_path] {
            let bytes = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| "REPL channel path contains NUL".to_string())?;
            // SAFETY: the path is a validated NUL-free OS path and the mode is
            // private to this invocation's temporary directory.
            if unsafe { libc::mkfifo(bytes.as_ptr(), 0o600) } != 0 {
                return Err(format!(
                    "REPL channel FIFO {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
        }
        let stdout_keeper = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&stdout_path)
            .map_err(|error| format!("REPL stdout FIFO: {error}"))?;
        let stderr_keeper = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&stderr_path)
            .map_err(|error| format!("REPL stderr FIFO: {error}"))?;
        Ok(Self {
            _directory: directory,
            stdout_path,
            stderr_path,
            stdout_keeper,
            stderr_keeper,
        })
    }

    fn activate(self) -> Result<ActiveChannels, String> {
        use std::os::unix::fs::OpenOptionsExt;
        let stdout = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&self.stdout_path)
            .map(tokio::fs::File::from_std)
            .map_err(|error| format!("REPL stdout FIFO reader: {error}"))?;
        let stderr = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&self.stderr_path)
            .map(tokio::fs::File::from_std)
            .map_err(|error| format!("REPL stderr FIFO reader: {error}"))?;
        Ok(ActiveChannels {
            _directory: self._directory,
            stdout,
            stderr,
            keepers: Arc::new(std::sync::Mutex::new((
                Some(self.stdout_keeper),
                Some(self.stderr_keeper),
            ))),
            marker_seen: Arc::new(AtomicBool::new(false)),
        })
    }
}

#[cfg(unix)]
struct ActiveChannels {
    _directory: tempfile::TempDir,
    stdout: tokio::fs::File,
    stderr: tokio::fs::File,
    keepers: Arc<std::sync::Mutex<(Option<std::fs::File>, Option<std::fs::File>)>>,
    marker_seen: Arc<AtomicBool>,
}

#[cfg(unix)]
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn wrap_code(
    language: &str,
    code: &str,
    marker: &str,
    #[cfg(unix)] channels: Option<&CallChannels>,
) -> String {
    #[cfg(unix)]
    if let Some(channels) = channels {
        let out =
            serde_json::to_string(&channels.stdout_path.to_string_lossy().to_string()).unwrap();
        let err =
            serde_json::to_string(&channels.stderr_path.to_string_lossy().to_string()).unwrap();
        let frame = serde_json::to_string(marker).unwrap();
        return match language {
            "python" | "python3" => {
                let inner = format!(concat!(
                "import os,sys,threading,time\n",
                "if '_forge_ctrl_out' not in globals(): _forge_ctrl_out=os.dup(1); _forge_ctrl_err=os.dup(2)\n",
                "_forge_threads_before={{id(_forge_thread) for _forge_thread in threading.enumerate()}}\n",
                "_forge_async_pending=False\n",
                "_forge_out=os.open({out}, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)\n",
                "_forge_err=os.open({err}, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)\n",
                "os.dup2(_forge_out,1); os.close(_forge_out); os.dup2(_forge_err,2); os.close(_forge_err)\n",
                "try:\n try:\n  exec({code:?})\n",
                " except BaseException as _forge_error:\n  print(repr(_forge_error), file=sys.stderr)\n",
                " _forge_wait_deadline=time.monotonic()+0.35\n",
                " while True:\n  _forge_live=[_forge_thread for _forge_thread in threading.enumerate() if id(_forge_thread) not in _forge_threads_before and _forge_thread is not threading.current_thread() and _forge_thread.is_alive()]\n  if not _forge_live or time.monotonic() >= _forge_wait_deadline: break\n  for _forge_thread in _forge_live: _forge_thread.join(max(0,min(0.01,_forge_wait_deadline-time.monotonic())))\n",
                " _forge_async_pending=bool(_forge_live)\n",
                "finally:\n sys.stdout.flush(); sys.stderr.flush(); os.dup2(_forge_ctrl_out,1); os.dup2(_forge_ctrl_err,2)\n",
                "os.write(_forge_ctrl_out, ((\"__FORGE_REPL_ASYNC_PENDING__\" if _forge_async_pending else \"\") + {frame} + '\\n').encode())\n"),
                out = out,
                err = err,
                frame = frame,
                code = code,
                );
                format!("exec({inner:?})\n")
            }
            "javascript" | "node" => {
                serde_json::to_string(&json!({
                    "code": code,
                    "out": channels.stdout_path.to_string_lossy(),
                    "err": channels.stderr_path.to_string_lossy(),
                    "marker": marker,
                }))
                .unwrap()
                    + "\n"
            }
            _ => format!(
                "exec 3>&1\nexec 1>{} 2>{}\n{}\nexec 1>&3 2>&3\nprintf '%s\\n' {} >&3\n",
                shell_quote(&channels.stdout_path.to_string_lossy()),
                shell_quote(&channels.stderr_path.to_string_lossy()),
                code,
                shell_quote(marker),
            ),
        };
    }

    match language {
        "python" | "python3" => {
            // exec(<single line>) avoids the interactive parser's requirement
            // for a blank line after a multi-line try/except suite.
            format!("exec({code:?})\nprint({marker:?})\n")
        }
        "javascript" | "node" => {
            format!("eval({code:?}); process.stdout.write({marker:?} + '\\n')\n")
        }
        _ => format!("{code}\nprintf '%s\\n' {marker:?}\n"),
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn append_bounded(output: &mut Vec<u8>, chunk: &[u8], partial: &mut bool) {
    let remaining = MAX_STREAM_BYTES.saturating_sub(output.len());
    let retained = chunk.len().min(remaining);
    output.extend_from_slice(&chunk[..retained]);
    *partial |= retained < chunk.len();
}

struct SessionCapture {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    complete: bool,
    partial: bool,
    timed_out: bool,
    cancelled: bool,
    error: Option<String>,
}

#[cfg(not(unix))]
async fn collect_until_frame(
    session: &mut ReplSession,
    marker: &str,
    cancel: &CancellationToken,
    deadline: Duration,
) -> SessionCapture {
    let marker_bytes = marker.as_bytes();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut stdout_chunk = [0_u8; 8192];
    let mut stderr_chunk = [0_u8; 8192];
    let mut stdout_open = true;
    let mut stderr_open = true;
    let mut partial = false;
    let mut complete = false;
    let mut error = None;
    let mut timed_out = false;
    let mut cancelled = false;
    let sleep_until = Instant::now() + deadline;

    while (stdout_open || stderr_open) && !complete {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                cancelled = true;
                break;
            }
            _ = sleep(sleep_until.saturating_duration_since(Instant::now())) => {
                timed_out = true;
                break;
            }
            result = session.stdout.read(&mut stdout_chunk), if stdout_open => {
                match result {
                    Ok(0) => {
                        stdout_open = false;
                        if find_bytes(&stdout, marker_bytes).is_none() {
                            error = Some("Interpreter exited before its completion frame.".to_string());
                        }
                    }
                    Ok(count) => {
                        append_bounded(&mut stdout, &stdout_chunk[..count], &mut partial);
                        complete = find_bytes(&stdout, marker_bytes).is_some();
                    }
                    Err(error_value) => {
                        stdout_open = false;
                        error = Some(format!("Read error: {error_value}"));
                    }
                }
            }
            result = session.stderr.read(&mut stderr_chunk), if stderr_open => {
                match result {
                    Ok(0) => stderr_open = false,
                    Ok(count) => append_bounded(&mut stderr, &stderr_chunk[..count], &mut partial),
                    Err(error_value) => {
                        stderr_open = false;
                        error = Some(format!("Stderr read error: {error_value}"));
                    }
                }
            }
        }
    }

    // A program can write stderr immediately before stdout emits its frame.
    // Give already-running stderr writers a short bounded grace period so the
    // next invocation cannot inherit output from this one.
    if complete && stderr_open {
        let grace = sleep(Duration::from_millis(50));
        tokio::pin!(grace);
        while stderr_open {
            tokio::select! {
                _ = &mut grace => break,
                result = session.stderr.read(&mut stderr_chunk) => match result {
                    Ok(0) => stderr_open = false,
                    Ok(count) => append_bounded(&mut stderr, &stderr_chunk[..count], &mut partial),
                    Err(error_value) => {
                        stderr_open = false;
                        error = Some(format!("Stderr read error: {error_value}"));
                    }
                }
            }
        }
    }

    if complete {
        let pending_marker = format!("{ASYNC_PENDING_PREFIX}{marker}");
        if find_bytes(&stdout, pending_marker.as_bytes()).is_some() {
            error = Some("Interpreter reported unresolved asynchronous work.".to_string());
        }
        if let Some(marker_start) = find_bytes(&stdout, marker_bytes) {
            stdout.truncate(marker_start);
        }
    }

    partial |= timed_out || cancelled || !complete || error.is_some();
    SessionCapture {
        stdout,
        stderr,
        complete,
        partial,
        timed_out,
        cancelled,
        error,
    }
}

#[cfg(unix)]
fn scan_control(
    retained: &mut Vec<u8>,
    tail: &mut Vec<u8>,
    chunk: &[u8],
    marker: &[u8],
    partial: &mut bool,
) -> bool {
    let mut probe = Vec::with_capacity(tail.len() + chunk.len());
    probe.extend_from_slice(tail);
    probe.extend_from_slice(chunk);
    let found = probe.windows(marker.len()).any(|window| window == marker);
    append_bounded(retained, chunk, partial);
    let keep = marker.len().saturating_sub(1);
    tail.clear();
    tail.extend_from_slice(&probe[probe.len().saturating_sub(keep)..]);
    found
}

#[cfg(unix)]
async fn drain_channel(
    mut reader: tokio::fs::File,
    output: &mut Vec<u8>,
    marker_seen: Arc<AtomicBool>,
) -> std::io::Result<bool> {
    let mut chunk = [0_u8; 8192];
    let mut partial = false;
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) if marker_seen.load(Ordering::Acquire) => return Ok(partial),
            Ok(0) => {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            Ok(count) => append_bounded(output, &chunk[..count], &mut partial),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                tokio::task::yield_now().await;
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(unix)]
async fn collect_until_frame(
    session: &mut ReplSession,
    marker: &str,
    cancel: &CancellationToken,
    deadline: Duration,
    channels: ActiveChannels,
) -> SessionCapture {
    let marker_bytes = marker.as_bytes();
    let mut control = Vec::new();
    let mut stderr = Vec::new();
    let mut control_tail = Vec::new();
    let mut control_chunk = [0_u8; 8192];
    let mut stderr_chunk = [0_u8; 8192];
    let mut control_open = true;
    let mut stderr_open = true;
    let mut partial = false;
    let mut complete = false;
    let mut error = None;
    let mut timed_out = false;
    let mut cancelled = false;
    let deadline_at = Instant::now() + deadline;
    let keepers = channels.keepers.clone();
    let marker_seen = channels.marker_seen.clone();
    let drain_marker_seen = marker_seen.clone();
    let drain = async move {
        let mut call_stdout = Vec::new();
        let mut call_stderr = Vec::new();
        let (stdout_result, stderr_result) = tokio::join!(
            drain_channel(channels.stdout, &mut call_stdout, drain_marker_seen.clone()),
            drain_channel(channels.stderr, &mut call_stderr, drain_marker_seen.clone()),
        );
        (call_stdout, call_stderr, stdout_result, stderr_result)
    };
    tokio::pin!(drain);
    let mut drain_result = None;

    while control_open && !complete {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => { cancelled = true; break; }
            _ = sleep(deadline_at.saturating_duration_since(Instant::now())) => { timed_out = true; break; }
            result = &mut drain, if drain_result.is_none() => { drain_result = Some(result); }
            result = session.stdout.read(&mut control_chunk) => match result {
                Ok(0) => {
                    control_open = false;
                    error = Some("Interpreter exited before its completion frame.".to_string());
                }
                Ok(count) => {
                    complete = scan_control(&mut control, &mut control_tail, &control_chunk[..count], marker_bytes, &mut partial);
                    if complete {
                        marker_seen.store(true, Ordering::Release);
                        if let Ok(mut keepers) = keepers.lock() {
                            keepers.0.take();
                            keepers.1.take();
                        }
                    }
                }
                Err(error_value) => {
                    control_open = false;
                    error = Some(format!("Read error: {error_value}"));
                }
            },
            result = session.stderr.read(&mut stderr_chunk), if stderr_open => match result {
                Ok(0) => stderr_open = false,
                Ok(count) => append_bounded(&mut stderr, &stderr_chunk[..count], &mut partial),
                Err(error_value) => {
                    stderr_open = false;
                    error = Some(format!("Stderr read error: {error_value}"));
                }
            },
        }
    }

    if complete {
        let pending_marker = format!("{ASYNC_PENDING_PREFIX}{marker}");
        if find_bytes(&control, pending_marker.as_bytes()).is_some() {
            error = Some("Interpreter reported unresolved asynchronous work.".to_string());
        }
        if let Some(marker_start) = find_bytes(&control, marker_bytes) {
            control.truncate(marker_start);
        }
        let remaining = deadline_at.saturating_duration_since(Instant::now());
        if drain_result.is_none() {
            drain_result = timeout(remaining, &mut drain).await.ok();
        }
        match drain_result.as_ref() {
            Some((_, _, Ok(left_partial), Ok(right_partial))) => {
                partial |= *left_partial || *right_partial
            }
            Some((_, _, left, right)) => {
                partial = true;
                error = Some(format!(
                    "REPL call channel drain failed: {left:?}; {right:?}"
                ));
            }
            None => timed_out = true,
        }
    } else {
        // Kill before draining on timeout/cancel. A descendant that retained
        // a FIFO writer keeps the drain open, so failure remains Unknown and
        // the session is retired instead of bleeding into the next call.
        if timed_out || cancelled || error.is_some() {
            marker_seen.store(true, Ordering::Release);
            if let Ok(mut keepers) = keepers.lock() {
                keepers.0.take();
                keepers.1.take();
            }
            session.group.kill_now();
            let _ = session.child.start_kill();
            if drain_result.is_none() {
                let _ = timeout(Duration::from_millis(250), &mut drain).await;
            }
        }
    }

    if let Some((call_stdout, call_stderr, _, _)) = drain_result.take() {
        control.extend_from_slice(&call_stdout);
        stderr.extend_from_slice(&call_stderr);
    }
    partial |= timed_out || cancelled || !complete || error.is_some();
    SessionCapture {
        stdout: control,
        stderr,
        complete,
        partial,
        timed_out,
        cancelled,
        error,
    }
}

fn combine_capture(capture: &SessionCapture) -> String {
    let mut text = String::from_utf8_lossy(&capture.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&capture.stderr);
    if !stderr.is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("[stderr]\n");
        text.push_str(&stderr);
    }
    text
}

fn cancellation_for_session(
    key: &ReplKey,
    session: &Arc<Mutex<ReplSession>>,
) -> Option<CancellationToken> {
    let entry = REPL_CANCELS.get(key)?;
    let current = entry.value().0.upgrade()?;
    Arc::ptr_eq(&current, session).then(|| entry.value().1.clone())
}

fn cancel_entry_matches(key: &ReplKey, session: &Arc<Mutex<ReplSession>>) -> bool {
    cancellation_for_session(key, session).is_some()
}

fn remove_cancel_if_same(key: &ReplKey, session: &Arc<Mutex<ReplSession>>) {
    REPL_CANCELS.remove_if(key, |_, (current, _)| {
        current
            .upgrade()
            .is_some_and(|current| Arc::ptr_eq(&current, session))
    });
}

async fn terminate_session(key: &ReplKey, session: &Arc<Mutex<ReplSession>>) -> bool {
    if let Some(cancel) = cancellation_for_session(key, session) {
        cancel.cancel();
    }
    let mut guard = session.lock().await;
    guard.retiring = true;
    guard.cancel.cancel();
    guard.group.kill_now();
    let _ = guard.child.start_kill();
    let wait_result = timeout(Duration::from_secs(2), guard.child.wait()).await;
    let clean = matches!(wait_result, Ok(Ok(_))) && guard.group.confirmed_empty().await;
    let pending_clean = clean && guard.pending.confirm_terminal().is_ok();
    drop(guard);
    if cancel_entry_matches(key, session) {
        remove_cancel_if_same(key, session);
    }
    pending_clean
}

fn remove_if_same(key: &ReplKey, session: &Arc<Mutex<ReplSession>>) -> bool {
    REPL_SESSIONS
        .remove_if(key, |_, current| Arc::ptr_eq(current, session))
        .is_some()
}

async fn terminate_and_remove_if_clean(key: &ReplKey, session: &Arc<Mutex<ReplSession>>) -> bool {
    let clean = terminate_session(key, session).await;
    if clean {
        remove_if_same(key, session);
    }
    clean
}

async fn drain_startup_pipe<R: AsyncRead + Unpin>(mut pipe: R) {
    let mut buffer = [0_u8; 8192];
    while pipe
        .read(&mut buffer)
        .await
        .ok()
        .is_some_and(|count| count > 0)
    {}
}

async fn cleanup_spawned_session(
    mut child: Child,
    mut group: ProcessGroupGuard,
    pending: PendingMutationTicket,
    message: String,
) -> String {
    group.kill_now();
    let _ = child.start_kill();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let drains = async {
        tokio::join!(
            async {
                if let Some(stdout) = stdout {
                    drain_startup_pipe(stdout).await;
                }
            },
            async {
                if let Some(stderr) = stderr {
                    drain_startup_pipe(stderr).await;
                }
            },
        );
    };
    let wait = timeout(Duration::from_secs(2), child.wait());
    let (wait_result, _) = tokio::join!(wait, drains);
    let clean = matches!(wait_result, Ok(Ok(_))) && group.confirmed_empty().await;
    if clean {
        if let Err(error) = pending.confirm_terminal() {
            return format!("{message}; cleanup remains unknown: {error}");
        }
        message
    } else {
        format!("{message}; cleanup remains unknown")
    }
}

fn confirm_unspawned_pending(pending: PendingMutationTicket, message: String) -> String {
    match pending.confirm_terminal() {
        Ok(()) => message,
        Err(error) => format!("{message}; pending cleanup remains unknown: {error}"),
    }
}

async fn get_or_spawn_session(key: &ReplKey) -> Result<Arc<Mutex<ReplSession>>, String> {
    if let Some(entry) = REPL_SESSIONS.get(key) {
        let session = entry.value().clone();
        drop(entry);
        let healthy = session.lock().await.healthy();
        if healthy {
            return Ok(session);
        }
        if !terminate_and_remove_if_clean(key, &session).await {
            return Err("REPL session cleanup remains unknown; refusing replacement".to_string());
        }
    }

    let stale = {
        let _registration = REPL_REGISTRATION.lock();
        // Recheck after acquiring the registration lock. A concurrent creator
        // may have inserted a live session while the first health check
        // awaited its async mutex. Never overwrite that session.
        if let Some(entry) = REPL_SESSIONS.get(key) {
            let session = entry.value().clone();
            drop(entry);
            let health = match session.try_lock() {
                Ok(mut guard) => Some(guard.healthy()),
                Err(_) => None,
            };
            match health {
                Some(true) | None => return Ok(session),
                Some(false) => Some(session),
            }
        } else {
            None
        }
    };
    if let Some(session) = stale {
        if !terminate_and_remove_if_clean(key, &session).await {
            return Err("REPL session cleanup remains unknown; refusing replacement".to_string());
        }
    }

    // Recheck the map immediately before spawning. The first lookup and
    // health check can race with another creator that inserts a session while
    // this task is waiting on the async mutex. Never overwrite that owner;
    // wait for it outside the registration lock and retire it only when its
    // own cleanup is confirmed.
    loop {
        let existing = {
            let _registration = REPL_REGISTRATION.lock();
            REPL_SESSIONS.get(key).map(|entry| entry.value().clone())
        };
        if let Some(session) = existing {
            if session.lock().await.healthy() {
                return Ok(session);
            }
            if !terminate_and_remove_if_clean(key, &session).await {
                return Err(
                    "REPL session cleanup remains unknown; refusing replacement".to_string()
                );
            }
            continue;
        }

        let _registration = REPL_REGISTRATION.lock();
        // A creator may have inserted the key between the unlocked check and
        // this lock acquisition. Recheck while holding the lock before any
        // spawn or insert.
        if REPL_SESSIONS.contains_key(key) {
            drop(_registration);
            continue;
        }
        if REPL_SESSIONS.len() >= MAX_REPL_SESSIONS {
            return Err(format!(
                "REPL session capacity reached ({MAX_REPL_SESSIONS})"
            ));
        }

        let (program, args) = interpreter_for(&key.language)
            .ok_or_else(|| format!("Unsupported language: {}", key.language))?;
        let pending = register_pending(
            scope_for_workspace(key.workspace.clone(), key.session.clone()),
            format!("REPL {} interpreter", key.language),
        )?;
        let mut command = Command::new(program);
        command
            .args(args)
            .current_dir(&key.execution_root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return Err(confirm_unspawned_pending(
                    pending,
                    format!("Failed to spawn '{program}': {error}"),
                ));
            }
        };
        let group = ProcessGroupGuard::new(child.id());
        let stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                drop(_registration);
                return Err(cleanup_spawned_session(
                    child,
                    group,
                    pending,
                    "Interpreter stdin unavailable".to_string(),
                )
                .await);
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                drop(_registration);
                return Err(cleanup_spawned_session(
                    child,
                    group,
                    pending,
                    "Interpreter stdout unavailable".to_string(),
                )
                .await);
            }
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                drop(_registration);
                return Err(cleanup_spawned_session(
                    child,
                    group,
                    pending,
                    "Interpreter stderr unavailable".to_string(),
                )
                .await);
            }
        };
        let cancel = CancellationToken::new();
        let session = Arc::new(Mutex::new(ReplSession {
            group,
            child,
            stdin,
            stdout,
            stderr,
            cancel: cancel.clone(),
            pending,
            retiring: false,
        }));
        REPL_SESSIONS.insert(key.clone(), session.clone());
        REPL_CANCELS.insert(key.clone(), (Arc::downgrade(&session), cancel));
        return Ok(session);
    }
}

struct InvocationGuard {
    key: ReplKey,
    session: Arc<Mutex<ReplSession>>,
    armed: bool,
}

impl InvocationGuard {
    fn new(key: ReplKey, session: Arc<Mutex<ReplSession>>) -> Self {
        Self {
            key,
            session,
            // A queued invocation is not an owner. Arm cleanup only after it
            // has acquired the session mutex and been admitted exclusively.
            armed: false,
        }
    }

    fn arm(&mut self) {
        self.armed = true;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InvocationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let key = self.key.clone();
        let session = self.session.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = terminate_and_remove_if_clean(&key, &session).await;
            });
        }
    }
}

fn incomplete_capture(
    message: impl Into<String>,
    timed_out: bool,
    cancelled: bool,
) -> SessionCapture {
    SessionCapture {
        stdout: Vec::new(),
        stderr: Vec::new(),
        complete: false,
        partial: true,
        timed_out,
        cancelled,
        error: Some(message.into()),
    }
}

async fn run_in_session(
    key: &ReplKey,
    session: &Arc<Mutex<ReplSession>>,
    language: &str,
    code: &str,
    caller_cancel: &CancellationToken,
    deadline: Duration,
) -> SessionCapture {
    let marker = frame_marker();
    let mut invocation = InvocationGuard::new(key.clone(), session.clone());
    let deadline_at = Instant::now() + deadline;
    let session_cancel = cancellation_for_session(key, session).unwrap_or_default();
    let mut guard = tokio::select! {
        _ = caller_cancel.cancelled() => {
            return incomplete_capture("REPL cancelled before invocation admission", false, true);
        }
        _ = session_cancel.cancelled() => {
            return incomplete_capture("REPL session was shut down before invocation admission", false, true);
        }
        _ = sleep(deadline_at.saturating_duration_since(Instant::now())) => {
            return incomplete_capture("REPL invocation admission timed out", true, false);
        }
        guard = session.lock() => guard,
    };
    // There is no await between acquiring the mutex and arming cleanup. A
    // cancellation/drop while queued therefore cannot retire this owner.
    invocation.arm();
    if !guard.healthy() {
        return incomplete_capture("Interpreter exited unexpectedly.", false, false);
    }

    #[cfg(unix)]
    let channels = match CallChannels::new() {
        Ok(channels) => channels,
        Err(error) => return incomplete_capture(error, false, false),
    };
    #[cfg(unix)]
    let wrapped = wrap_code(language, code, &marker, Some(&channels));
    #[cfg(not(unix))]
    let wrapped = wrap_code(language, code, &marker);
    #[cfg(unix)]
    let channels = match channels.activate() {
        Ok(channels) => channels,
        Err(error) => return incomplete_capture(error, false, false),
    };

    let write_result = tokio::select! {
        _ = caller_cancel.cancelled() => return incomplete_capture("REPL cancelled while writing invocation", false, true),
        _ = session_cancel.cancelled() => return incomplete_capture("REPL session shut down while writing invocation", false, true),
        _ = sleep(deadline_at.saturating_duration_since(Instant::now())) => return incomplete_capture("REPL invocation write timed out", true, false),
        result = guard.stdin.write_all(wrapped.as_bytes()) => result,
    };
    if let Err(error) = write_result {
        guard.group.kill_now();
        return incomplete_capture(
            format!("Write to interpreter stdin failed: {error}"),
            false,
            false,
        );
    }
    let flush_result = tokio::select! {
        _ = caller_cancel.cancelled() => return incomplete_capture("REPL cancelled while flushing invocation", false, true),
        _ = session_cancel.cancelled() => return incomplete_capture("REPL session shut down while flushing invocation", false, true),
        _ = sleep(deadline_at.saturating_duration_since(Instant::now())) => return incomplete_capture("REPL invocation flush timed out", true, false),
        result = guard.stdin.flush() => result,
    };
    if let Err(error) = flush_result {
        guard.group.kill_now();
        return incomplete_capture(
            format!("Flush interpreter stdin failed: {error}"),
            false,
            false,
        );
    }
    let cancellation = CancellationToken::new();
    let caller_watcher = cancellation.clone();
    let session_watcher = cancellation.clone();
    let caller_token = caller_cancel.clone();
    let session_token = guard.cancel.clone();
    let caller_task = tokio::spawn(async move {
        caller_token.cancelled().await;
        caller_watcher.cancel();
    });
    let session_task = tokio::spawn(async move {
        session_token.cancelled().await;
        session_watcher.cancel();
    });
    let remaining = deadline_at.saturating_duration_since(Instant::now());
    #[cfg(unix)]
    let capture =
        collect_until_frame(&mut guard, &marker, &cancellation, remaining, channels).await;
    #[cfg(not(unix))]
    let capture = collect_until_frame(&mut guard, &marker, &cancellation, remaining).await;
    caller_task.abort();
    session_task.abort();
    if capture.timed_out || capture.cancelled || !capture.complete || capture.error.is_some() {
        guard.group.kill_now();
        let _ = guard.child.start_kill();
        let _ = timeout(Duration::from_secs(2), guard.child.wait()).await;
        drop(guard);
        let _ = terminate_and_remove_if_clean(key, session).await;
    }
    invocation.disarm();
    capture
}

fn result_with_status(ctx: &ToolContext, capture: SessionCapture) -> ToolResult {
    let mut text = combine_capture(&capture);
    if capture.timed_out {
        text = format!("REPL timed out after {REPL_CALL_DEADLINE:?}; capture incomplete\n{text}");
    } else if capture.cancelled {
        text = format!("REPL cancelled; capture incomplete\n{text}");
    } else if let Some(error) = &capture.error {
        text = format!("REPL error: {error}\n{text}");
    } else if !capture.complete {
        text = format!("REPL completion frame was not observed; capture incomplete\n{text}");
    }
    let failed =
        capture.timed_out || capture.cancelled || capture.error.is_some() || !capture.complete;
    let partial = capture.partial || failed;
    let mut result = retain_output(ctx, text, failed, partial);
    if failed || partial {
        result = result.with_execution_status(crate::ExecutionStatus::Unknown);
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplCleanup {
    ConfirmedClean,
    CleanupUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReplShutdown {
    pub sessions: usize,
    pub cleanup: ReplCleanup,
}

/// Terminate REPL interpreters owned by this exact execution root/session.
pub async fn shutdown_for_context(ctx: &ToolContext) -> ReplShutdown {
    let workspace = canonical(&crate::worktree::workspace_root(ctx));
    let execution_root = canonical(&ctx.working_dir);
    let owned: Vec<(ReplKey, Arc<Mutex<ReplSession>>)> = REPL_SESSIONS
        .iter()
        .filter(|entry| {
            entry.key().session == ctx.session_id
                && entry.key().workspace == workspace
                && entry.key().execution_root == execution_root
        })
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();
    let mut removed = 0;
    let mut clean = true;
    for (key, session) in owned {
        let terminated = timeout(
            Duration::from_secs(2),
            terminate_and_remove_if_clean(&key, &session),
        )
        .await
        .unwrap_or(false);
        clean &= terminated;
        removed += 1;
    }
    ReplShutdown {
        sessions: removed,
        cleanup: if clean {
            ReplCleanup::ConfirmedClean
        } else {
            ReplCleanup::CleanupUnknown
        },
    }
}

/// Terminate every interpreter owned by this workspace/session across its
/// execution roots. The workspace is part of the ownership key: a session ID
/// alone is never sufficient to select interpreters for teardown.
pub async fn shutdown_for_workspace(ctx: &ToolContext) -> ReplShutdown {
    let workspace = canonical(&crate::worktree::workspace_root(ctx));
    let owned: Vec<(ReplKey, Arc<Mutex<ReplSession>>)> = REPL_SESSIONS
        .iter()
        .filter(|entry| entry.key().session == ctx.session_id && entry.key().workspace == workspace)
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();
    let mut removed = 0;
    let mut clean = true;
    for (key, session) in owned {
        let terminated = timeout(
            Duration::from_secs(2),
            terminate_and_remove_if_clean(&key, &session),
        )
        .await
        .unwrap_or(false);
        clean &= terminated;
        removed += 1;
    }
    ReplShutdown {
        sessions: removed,
        cleanup: if clean {
            ReplCleanup::ConfirmedClean
        } else {
            ReplCleanup::CleanupUnknown
        },
    }
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

pub struct ReplTool;

#[derive(Debug, Deserialize)]
struct ReplInput {
    code: String,
    #[serde(default)]
    language: Option<String>,
}

#[async_trait]
impl Tool for ReplTool {
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "REPL"
    }

    fn description(&self) -> &str {
        "Execute code in a persistent interpreter session. Variables, imports, and state persist between calls. Supports bash, python, and javascript (node)."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code to execute in the persistent interpreter"},
                "language": {"type": "string", "enum": ["bash", "python", "javascript"], "description": "Interpreter language; defaults to bash"}
            },
            "required": ["code"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ReplInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(error) => return ToolResult::error(format!("Invalid input: {error}")),
        };
        let language = params.language.as_deref().unwrap_or("bash").to_lowercase();
        let preview = params
            .code
            .chars()
            .take(80)
            .collect::<String>()
            .replace('\n', " ");
        let reason = format!("REPL ({language}): {preview}");
        if let Err(error) = ctx.check_permission(self.name(), &reason, false) {
            return ToolResult::error(error.to_string());
        }
        if matches!(language.as_str(), "bash" | "sh" | "")
            && classify_bash_command(&params.code) == BashRiskLevel::Critical
        {
            return ToolResult::error(format!(
                "Command blocked: classified as Critical risk by the bash security classifier.\nRefusing to execute REPL code: {preview}"
            ));
        }
        let key = key_for(ctx, &language);
        debug!(session = %ctx.session_id, language = %language, cwd = ?key.execution_root, "ReplTool execute");
        let session = match get_or_spawn_session(&key).await {
            Ok(session) => session,
            Err(error) => {
                return ToolResult::error(format!("Failed to start REPL session: {error}"))
            }
        };
        let capture = run_in_session(
            &key,
            &session,
            &language,
            &params.code,
            &ctx.cancel_token,
            REPL_CALL_DEADLINE,
        )
        .await;
        let result = result_with_status(ctx, capture);
        if result.is_error {
            let _ = terminate_and_remove_if_clean(&key, &session).await;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct AllowHandler;
    impl claurst_core::permissions::PermissionHandler for AllowHandler {
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

    struct DenyHandler;
    impl claurst_core::permissions::PermissionHandler for DenyHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Ask {
                reason: "denied".into(),
            }
        }
        fn request_permission(
            &self,
            request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            self.check_permission(request)
        }
    }

    fn ctx_with(
        handler: Arc<dyn claurst_core::permissions::PermissionHandler>,
        session_id: &str,
        working_dir: PathBuf,
    ) -> ToolContext {
        ToolContext {
            working_dir,
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: session_id.to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
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

    #[test]
    fn frame_is_unpredictable_per_call() {
        assert_ne!(frame_marker(), frame_marker());
        assert!(frame_marker().starts_with(FRAME_PREFIX));
    }

    #[test]
    fn python_wrap_is_single_line_exec() {
        #[cfg(unix)]
        let wrapped = wrap_code("python", "try:\n    value = 1", "marker", None);
        #[cfg(not(unix))]
        let wrapped = wrap_code("python", "try:\n    value = 1", "marker");
        assert!(wrapped.starts_with("exec("));
        assert!(wrapped.contains("marker"));
    }

    #[tokio::test]
    async fn denied_permission_does_not_spawn() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(DenyHandler),
            "repl-denied",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let result = ReplTool
            .execute(json!({"language": "bash", "code": "echo no"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(!REPL_SESSIONS.contains_key(&key));
    }

    #[tokio::test]
    async fn critical_bash_does_not_spawn() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-critical",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let result = ReplTool
            .execute(json!({"language": "bash", "code": "rm -rf /"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("Critical"));
        assert!(!REPL_SESSIONS.contains_key(&key));
    }

    #[tokio::test]
    async fn invalid_interpreter_does_not_register_or_spawn() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-invalid-language",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "ruby");
        let result = ReplTool
            .execute(json!({"language": "ruby", "code": "puts 1"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(!REPL_SESSIONS.contains_key(&key));
        assert!(!crate::mutation::pending_for_scope(&ctx));
    }

    #[tokio::test]
    async fn invalid_cwd_clears_unspawned_pending_ticket() {
        let workspace = tempfile::tempdir().unwrap();
        let missing = workspace.path().join("missing-working-directory");
        let ctx = ctx_with(Arc::new(AllowHandler), "repl-invalid-cwd", missing.clone());
        let key = key_for(&ctx, "bash");
        let result = ReplTool
            .execute(json!({"language": "bash", "code": "printf never"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(!REPL_SESSIONS.contains_key(&key));
        assert!(!crate::mutation::pending_for_scope(&ctx));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn partial_startup_is_reaped_and_clears_pending() {
        let workspace = tempfile::tempdir().unwrap();
        let scope = scope_for_workspace(workspace.path().to_path_buf(), "repl-partial-startup");
        let pending = register_pending(scope, "REPL partial startup").unwrap();
        let mut command = Command::new("sh");
        command
            .args(["-c", "printf startup"])
            .current_dir(workspace.path())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);
        let child = command.spawn().unwrap();
        let group = ProcessGroupGuard::new(child.id());
        let message = cleanup_spawned_session(
            child,
            group,
            pending,
            "Interpreter stdin unavailable".to_string(),
        )
        .await;
        assert!(!message.contains("cleanup remains unknown"), "{message}");
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-partial-startup",
            workspace.path().into(),
        );
        assert!(!crate::mutation::pending_for_scope(&ctx));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn durable_pending_confirmation_failure_stays_unknown() {
        let workspace = tempfile::tempdir().unwrap();
        let scope = scope_for_workspace(workspace.path().to_path_buf(), "repl-confirm-failure");
        let pending = register_pending(scope, "REPL durable confirmation failure").unwrap();
        let record = claurst_core::config::Settings::config_dir()
            .join("mutation-pending")
            .join(format!("{}.json", pending.id()));
        std::fs::remove_file(record).unwrap();

        let mut command = Command::new("sh");
        command
            .args(["-c", "sleep 10"])
            .current_dir(workspace.path())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);
        let child = command.spawn().unwrap();
        let group = ProcessGroupGuard::new(child.id());
        let message = cleanup_spawned_session(
            child,
            group,
            pending,
            "Interpreter startup failed".to_string(),
        )
        .await;
        assert!(message.contains("cleanup remains unknown"), "{message}");
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-confirm-failure",
            workspace.path().into(),
        );
        assert!(crate::mutation::pending_for_scope(&ctx));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_persists_and_uses_explicit_cwd() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(Arc::new(AllowHandler), "repl-bash", workspace.path().into());
        let first = ReplTool
            .execute(
                json!({"language": "bash", "code": "shared=41; printf FIRST"}),
                &ctx,
            )
            .await;
        assert!(!first.is_error && first.content.contains("FIRST"));
        let second = ReplTool
            .execute(
                json!({"language": "bash", "code": "printf '%s %s' \"$shared\" \"$PWD\""}),
                &ctx,
            )
            .await;
        assert!(!second.is_error && second.content.contains("41"));
        assert!(second
            .content
            .contains(&*workspace.path().to_string_lossy()));
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn python_persists_and_uses_explicit_cwd() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-python",
            workspace.path().into(),
        );
        let first = ReplTool
            .execute(
                json!({"language": "python", "code": "shared_value = 41\nprint('PY_FIRST')"}),
                &ctx,
            )
            .await;
        assert!(
            !first.is_error && first.content.contains("PY_FIRST"),
            "{}",
            first.content
        );
        let second = ReplTool
            .execute(
                json!({"language": "python", "code": "print('PY_SECOND', shared_value)"}),
                &ctx,
            )
            .await;
        assert!(
            !second.is_error && second.content.contains("PY_SECOND 41"),
            "{}",
            second.content
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn python_finite_thread_output_stays_in_its_invocation() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-python-thread",
            workspace.path().into(),
        );
        let first = ReplTool
            .execute(
                json!({
                    "language": "python",
                    "code": "import threading,time\nthreading.Thread(target=lambda: (time.sleep(0.2), print('LATE_PY_PRIOR_CALL'))).start()\nprint('PY_FIRST')"
                }),
                &ctx,
            )
            .await;
        assert!(
            !first.is_error && first.content.contains("PY_FIRST"),
            "{}",
            first.content
        );
        assert!(
            first.content.contains("LATE_PY_PRIOR_CALL"),
            "{}",
            first.content
        );
        let second = ReplTool
            .execute(
                json!({"language": "python", "code": "print('PY_SECOND')"}),
                &ctx,
            )
            .await;
        assert!(
            !second.is_error && second.content.contains("PY_SECOND"),
            "{}",
            second.content
        );
        assert!(
            !second.content.contains("LATE_PY_PRIOR_CALL"),
            "{}",
            second.content
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn node_async_promises_and_timers_are_attributed_to_their_call() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-node-async",
            workspace.path().into(),
        );
        let promise = ReplTool
            .execute(
                json!({
                    "language": "javascript",
                    "code": "(async () => { await new Promise(r => setTimeout(r, 100)); console.log('NODE_ASYNC') })()"
                }),
                &ctx,
            )
            .await;
        assert!(
            !promise.is_error && promise.content.contains("NODE_ASYNC"),
            "{}",
            promise.content
        );

        let timer = ReplTool
            .execute(
                json!({
                    "language": "javascript",
                    "code": "setTimeout(() => console.log('NODE_TIMER_FIRST'), 200); console.log('NODE_TIMER_READY')"
                }),
                &ctx,
            )
            .await;
        assert!(
            !timer.is_error && timer.content.contains("NODE_TIMER_READY"),
            "{}",
            timer.content
        );
        assert!(
            timer.content.contains("NODE_TIMER_FIRST"),
            "{}",
            timer.content
        );
        let second = ReplTool
            .execute(
                json!({"language": "javascript", "code": "console.log('NODE_TIMER_SECOND')"}),
                &ctx,
            )
            .await;
        assert!(
            !second.is_error && second.content.contains("NODE_TIMER_SECOND"),
            "{}",
            second.content
        );
        assert!(
            !second.content.contains("NODE_TIMER_FIRST"),
            "{}",
            second.content
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unresolved_async_work_is_partial_and_retires_the_session() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-async-pending",
            workspace.path().into(),
        );
        let result = ReplTool
            .execute(
                json!({
                    "language": "javascript",
                    "code": "setTimeout(() => console.log('NEVER_REPORTED'), 5000); console.log('PENDING_READY')"
                }),
                &ctx,
            )
            .await;
        assert!(
            result.is_error,
            "unresolved async work must be reported as incomplete"
        );
        assert!(
            result.content.contains("unresolved asynchronous work"),
            "{}",
            result.content
        );
        assert!(result
            .metadata
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|m| m.get("evidence"))
            .is_some());
        let fresh = ReplTool
            .execute(
                json!({"language": "javascript", "code": "console.log('FRESH_AFTER_PENDING')"}),
                &ctx,
            )
            .await;
        assert!(
            !fresh.is_error && fresh.content.contains("FRESH_AFTER_PENDING"),
            "{}",
            fresh.content
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn same_session_different_workspaces_do_not_share_state() {
        let first_workspace = tempfile::tempdir().unwrap();
        let second_workspace = tempfile::tempdir().unwrap();
        let first = ctx_with(
            Arc::new(AllowHandler),
            "repl-scoped",
            first_workspace.path().into(),
        );
        let second = ctx_with(
            Arc::new(AllowHandler),
            "repl-scoped",
            second_workspace.path().into(),
        );
        let first_result = ReplTool
            .execute(
                json!({"language": "bash", "code": "shared=41; printf first"}),
                &first,
            )
            .await;
        assert!(!first_result.is_error && first_result.content.contains("first"));
        let second_result = ReplTool
            .execute(
                json!({"language": "bash", "code": "printf '%s' \"${shared:-missing}\""}),
                &second,
            )
            .await;
        assert!(!second_result.is_error && second_result.content.contains("missing"));
        let first_shutdown = shutdown_for_workspace(&first).await;
        assert_eq!(first_shutdown.sessions, 1);
        assert_eq!(first_shutdown.cleanup, ReplCleanup::ConfirmedClean);
        shutdown_for_context(&second).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn concurrent_registry_admission_reuses_one_session() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-admission",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let (first, second) = tokio::join!(get_or_spawn_session(&key), get_or_spawn_session(&key));
        let first = first.unwrap();
        let second = second.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            REPL_SESSIONS
                .iter()
                .filter(|entry| entry.key() == &key)
                .count(),
            1
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stale_cleanup_cannot_cancel_or_remove_replacement() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-replacement",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let old = get_or_spawn_session(&key).await.unwrap();
        assert!(remove_if_same(&key, &old));
        let replacement = get_or_spawn_session(&key).await.unwrap();
        assert!(!Arc::ptr_eq(&old, &replacement));
        let _ = terminate_session(&key, &old).await;
        assert!(REPL_SESSIONS
            .get(&key)
            .is_some_and(|entry| Arc::ptr_eq(entry.value(), &replacement)));
        let result = ReplTool
            .execute(
                json!({"language": "bash", "code": "printf replacement"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error && result.content.contains("replacement"));
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn node_lexical_bindings_persist_between_calls() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-node-lexical",
            workspace.path().into(),
        );
        let first = ReplTool
            .execute(
                json!({"language": "javascript", "code": "let lexical=41; const stable=1; process.stdout.write('LEXICAL_READY')"}),
                &ctx,
            )
            .await;
        assert!(
            !first.is_error && first.content.contains("LEXICAL_READY"),
            "{}",
            first.content
        );
        let second = ReplTool
            .execute(
                json!({"language": "javascript", "code": "process.stdout.write(String(lexical + stable))"}),
                &ctx,
            )
            .await;
        assert!(
            !second.is_error && second.content.contains("42"),
            "{}",
            second.content
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn finite_output_cap_keeps_frame_detection_and_marks_partial() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(Arc::new(AllowHandler), "repl-cap", workspace.path().into());
        let result = ReplTool
            .execute(
                json!({"language": "bash", "code": "printf '%300000s' x"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        let metadata = result.metadata.as_ref().and_then(Value::as_object).unwrap();
        assert_eq!(
            metadata
                .get("evidence")
                .and_then(Value::as_object)
                .and_then(|e| e.get("partial"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(metadata.get("execution_status").is_some());
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn delayed_child_output_stays_in_its_invocation_channel() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-attribution",
            workspace.path().into(),
        );
        let first = ReplTool
            .execute(
                json!({"language": "bash", "code": "(sleep 0.2; printf LATE_OUT; printf LATE_ERR >&2) & printf READY"}),
                &ctx,
            )
            .await;
        assert!(!first.is_error && first.content.contains("READY"));
        assert!(first.content.contains("LATE_OUT"));
        assert!(first.content.contains("LATE_ERR"));
        let second = ReplTool
            .execute(json!({"language": "bash", "code": "printf SECOND"}), &ctx)
            .await;
        assert!(!second.is_error && second.content.contains("SECOND"));
        assert!(!second.content.contains("LATE"));
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn old_fixed_sentinel_does_not_complete_a_call() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-frame",
            workspace.path().into(),
        );
        let result = ReplTool
            .execute(
                json!({"language": "bash", "code": "printf '__REPL_DONE_7f3a9b__ safe'"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("__REPL_DONE_7f3a9b__ safe"));
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stderr_flood_is_bounded_and_drained() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-stderr-flood",
            workspace.path().into(),
        );
        let result = ReplTool
            .execute(
                json!({"language": "bash", "code": "printf '%400000s' e >&2; printf done"}),
                &ctx,
            )
            .await;
        assert!(result.content.contains("done"), "{}", result.content);
        let metadata = result.metadata.as_ref().and_then(Value::as_object).unwrap();
        assert!(metadata.get("evidence").is_some(), "{}", result.content);
        assert!(
            metadata.get("execution_status").is_some(),
            "{}",
            result.content
        );
        assert_eq!(
            metadata
                .get("evidence")
                .and_then(Value::as_object)
                .and_then(|evidence| evidence.get("partial"))
                .and_then(Value::as_bool),
            Some(true),
            "stderr flood should retain bounded partial evidence"
        );
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn direct_deadline_returns_partial_capture() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-deadline",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let session = get_or_spawn_session(&key).await.unwrap();
        let capture = run_in_session(
            &key,
            &session,
            "bash",
            "printf before; while :; do printf x; done",
            &ctx.cancel_token,
            Duration::from_millis(100),
        )
        .await;
        assert!(capture.timed_out && capture.partial && !capture.complete);
        remove_if_same(&key, &session);
        terminate_session(&key, &session).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn uncertain_shutdown_retains_owner_for_retry() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-shutdown-retry",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let session = get_or_spawn_session(&key).await.unwrap();
        let release = Arc::new(tokio::sync::Notify::new());
        let release_task = release.clone();
        let lock_task = tokio::spawn(async move {
            let _guard = session.lock().await;
            release_task.notified().await;
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        let first = shutdown_for_context(&ctx).await;
        assert_eq!(first.sessions, 1);
        assert_eq!(first.cleanup, ReplCleanup::CleanupUnknown);
        assert!(REPL_SESSIONS.contains_key(&key));
        release.notify_one();
        lock_task.await.unwrap();
        let second = shutdown_for_context(&ctx).await;
        assert_eq!(second.sessions, 1);
        assert_eq!(second.cleanup, ReplCleanup::ConfirmedClean);
        assert!(!REPL_SESSIONS.contains_key(&key));
        assert!(!crate::mutation::pending_for_scope(&ctx));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_kills_owned_descendant() {
        let workspace = tempfile::tempdir().unwrap();
        let marker = workspace.path().join("late-child-marker");
        let invocation_ready = workspace.path().join("invocation-ready");
        let child_started = workspace.path().join("child-started");
        let ctx = Arc::new(ctx_with(
            Arc::new(AllowHandler),
            "repl-descendant",
            workspace.path().into(),
        ));
        let key = key_for(&ctx, "bash");
        let code = format!(
            ": > {}; (: > {}; sleep 1; : > {}) >/dev/null 2>&1 & printf ready",
            shell_quote(&invocation_ready.to_string_lossy()),
            shell_quote(&child_started.to_string_lossy()),
            shell_quote(&marker.to_string_lossy()),
        );
        let task_ctx = ctx.clone();
        let mut task = tokio::spawn(async move {
            ReplTool
                .execute(json!({"language": "bash", "code": code}), &task_ctx)
                .await
        });

        let ready = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if REPL_SESSIONS.contains_key(&key)
                    && invocation_ready.exists()
                    && child_started.exists()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        if ready.is_err() {
            task.abort();
            let _ = task.await;
            let _ = shutdown_for_context(&ctx).await;
            panic!("REPL invocation and owned descendant did not become ready");
        }

        let shutdown = shutdown_for_context(&ctx).await;
        assert_eq!(shutdown.sessions, 1);

        let owner_after_first = REPL_SESSIONS.get(&key).map(|entry| entry.value().clone());
        let pending_after_first = crate::mutation::pending_for_scope(&ctx);
        if shutdown.cleanup == ReplCleanup::CleanupUnknown {
            let owner = owner_after_first.expect("unknown cleanup must retain the REPL owner");
            assert!(
                pending_after_first,
                "unknown cleanup must retain the pending barrier"
            );
            let result = match tokio::time::timeout(Duration::from_secs(5), &mut task).await {
                Ok(result) => result.unwrap(),
                Err(_) => {
                    task.abort();
                    let _ = task.await;
                    panic!("REPL invocation did not join after bounded shutdown");
                }
            };
            assert!(result.is_error || result.content.contains("ready"));

            let retry = shutdown_for_context(&ctx).await;
            if retry.cleanup == ReplCleanup::ConfirmedClean {
                assert!(!REPL_SESSIONS.contains_key(&key));
                assert!(!crate::mutation::pending_for_scope(&ctx));
            } else {
                let retained = REPL_SESSIONS
                    .get(&key)
                    .map(|entry| entry.value().clone())
                    .expect("repeated unknown cleanup must retain the REPL owner");
                assert!(Arc::ptr_eq(&owner, &retained));
                assert!(crate::mutation::pending_for_scope(&ctx));
            }
        } else {
            assert_eq!(shutdown.cleanup, ReplCleanup::ConfirmedClean);
            assert!(
                owner_after_first.is_none(),
                "confirmed cleanup must remove the REPL owner"
            );
            assert!(
                !pending_after_first,
                "confirmed cleanup must clear the pending barrier"
            );
            let result = match tokio::time::timeout(Duration::from_secs(5), &mut task).await {
                Ok(result) => result.unwrap(),
                Err(_) => {
                    task.abort();
                    let _ = task.await;
                    panic!("REPL invocation did not join after confirmed shutdown");
                }
            };
            assert!(result.is_error || result.content.contains("ready"));
        }

        // Observe beyond the descendant's scheduled write. An absent marker
        // at one instant does not prove the writer was stopped.
        let observation_end = tokio::time::Instant::now() + Duration::from_millis(1200);
        loop {
            assert!(!marker.exists(), "owned descendant wrote after shutdown");
            if tokio::time::Instant::now() >= observation_end {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dead_interpreter_is_restarted() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-restart",
            workspace.path().into(),
        );
        let dead = ReplTool
            .execute(json!({"language": "bash", "code": "exit 7"}), &ctx)
            .await;
        assert!(dead.is_error && dead.content.contains("completion frame"));
        let fresh = ReplTool
            .execute(json!({"language": "bash", "code": "printf fresh"}), &ctx)
            .await;
        assert!(!fresh.is_error && fresh.content.contains("fresh"));
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_retains_partial_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-cancel",
            workspace.path().into(),
        );
        let trigger = ctx.cancel_token.clone();
        let execution = ReplTool.execute(
            json!({"language": "bash", "code": "printf before; sleep 10"}),
            &ctx,
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
        let result = execution.await;
        assert!(result.is_error);
        assert!(result
            .metadata
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|m| m.get("evidence"))
            .is_some());
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropped_invocation_retires_owned_session() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = Arc::new(ctx_with(
            Arc::new(AllowHandler),
            "repl-drop",
            workspace.path().into(),
        ));
        let key = key_for(&ctx, "bash");
        let task_ctx = ctx.clone();
        let task = tokio::spawn(async move {
            ReplTool
                .execute(
                    json!({"language": "bash", "code": "printf ready > repl-drop-ready; sleep 10"}),
                    &task_ctx,
                )
                .await
        });

        let ready_deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let ready = std::fs::read(workspace.path().join("repl-drop-ready"))
                .ok()
                .is_some_and(|bytes| bytes == b"ready");
            if ready && REPL_SESSIONS.contains_key(&key) && crate::mutation::pending_for_scope(&ctx)
            {
                break;
            }
            assert!(
                Instant::now() < ready_deadline,
                "REPL invocation never reached the owned execution point"
            );
            tokio::task::yield_now().await;
        }

        task.abort();
        let _ = task.await;

        let cleanup_deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let owner_present = REPL_SESSIONS.contains_key(&key);
            let pending = crate::mutation::pending_for_scope(&ctx);
            if !owner_present && !pending {
                break;
            }
            assert!(
                Instant::now() < cleanup_deadline,
                "dropped REPL owner cleanup remained pending: owner={owner_present}, pending={pending}"
            );
            tokio::task::yield_now().await;
        }

        let shutdown = shutdown_for_workspace(&ctx).await;
        assert_eq!(shutdown.sessions, 0);
        assert_eq!(shutdown.cleanup, ReplCleanup::ConfirmedClean);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn queued_cancel_does_not_retire_admitted_session() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-queued-cancel",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let session = get_or_spawn_session(&key).await.unwrap();
        let first = run_in_session(
            &key,
            &session,
            "bash",
            "shared=41; printf READY",
            &ctx.cancel_token,
            REPL_CALL_DEADLINE,
        )
        .await;
        assert!(first.complete && first.stdout == b"READY");

        let held = session.lock().await;
        let queued_cancel = CancellationToken::new();
        let queued_key = key.clone();
        let queued_session = session.clone();
        let queued_token = queued_cancel.clone();
        let queued = tokio::spawn(async move {
            run_in_session(
                &queued_key,
                &queued_session,
                "bash",
                "printf SHOULD_NOT_RUN",
                &queued_token,
                REPL_CALL_DEADLINE,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        queued_cancel.cancel();
        let capture = queued.await.unwrap();
        assert!(capture.cancelled && capture.partial && !capture.complete);
        drop(held);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let owner = REPL_SESSIONS
            .get(&key)
            .map(|entry| entry.value().clone())
            .expect("queued cancellation must not remove the admitted owner");
        assert!(Arc::ptr_eq(&owner, &session));
        assert!(crate::mutation::pending_for_scope(&ctx));
        let next = run_in_session(
            &key,
            &session,
            "bash",
            "printf %s \"$shared\"",
            &ctx.cancel_token,
            REPL_CALL_DEADLINE,
        )
        .await;
        assert!(next.complete && next.stdout == b"41");
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn queued_timeout_does_not_retire_admitted_session() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-queued-timeout",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let session = get_or_spawn_session(&key).await.unwrap();
        let held = session.lock().await;
        let queued_key = key.clone();
        let queued_session = session.clone();
        let queued_token = CancellationToken::new();
        let queued = tokio::spawn(async move {
            run_in_session(
                &queued_key,
                &queued_session,
                "bash",
                "printf SHOULD_NOT_RUN",
                &queued_token,
                Duration::from_millis(20),
            )
            .await
        });
        let capture = queued.await.unwrap();
        assert!(capture.timed_out && capture.partial && !capture.complete);
        drop(held);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let owner = REPL_SESSIONS
            .get(&key)
            .map(|entry| entry.value().clone())
            .expect("queued timeout must not remove the admitted owner");
        assert!(Arc::ptr_eq(&owner, &session));
        assert!(crate::mutation::pending_for_scope(&ctx));
        shutdown_for_context(&ctx).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropped_queued_invocation_preserves_session_state() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = ctx_with(
            Arc::new(AllowHandler),
            "repl-queued-drop",
            workspace.path().into(),
        );
        let key = key_for(&ctx, "bash");
        let session = get_or_spawn_session(&key).await.unwrap();
        let first = run_in_session(
            &key,
            &session,
            "bash",
            "shared=41; printf READY",
            &ctx.cancel_token,
            REPL_CALL_DEADLINE,
        )
        .await;
        assert!(first.complete && first.stdout == b"READY");

        // Hold the admitted owner mutex. The queued future must be polled and
        // observed Pending before it is dropped; no scheduler sleep or abort
        // timing can make this test vacuous.
        let held = session.lock().await;
        let queued_key = key.clone();
        let queued_session = session.clone();
        let queued_token = CancellationToken::new();
        let mut queued = Box::pin(run_in_session(
            &queued_key,
            &queued_session,
            "bash",
            "printf SHOULD_NOT_RUN",
            &queued_token,
            REPL_CALL_DEADLINE,
        ));
        let polled = tokio::time::timeout(
            Duration::from_secs(1),
            std::future::poll_fn(|cx| {
                let state = std::future::Future::poll(queued.as_mut(), cx);
                assert!(
                    state.is_pending(),
                    "queued invocation must block on the admitted owner mutex"
                );
                std::task::Poll::Ready(())
            }),
        )
        .await;
        assert!(
            polled.is_ok(),
            "queued invocation was not polled while mutex was held"
        );
        drop(queued);
        drop(held);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let owner = REPL_SESSIONS
            .get(&key)
            .map(|entry| entry.value().clone())
            .expect("dropped queued invocation must not remove the admitted owner");
        assert!(Arc::ptr_eq(&owner, &session));
        assert!(crate::mutation::pending_for_scope(&ctx));
        let next = run_in_session(
            &key,
            &session,
            "bash",
            "printf %s \"$shared\"",
            &ctx.cancel_token,
            REPL_CALL_DEADLINE,
        )
        .await;
        assert!(next.complete && next.stdout == b"41");
        shutdown_for_context(&ctx).await;
    }
}
