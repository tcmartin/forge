//! Bounded pipe capture shared by non-PTY command tools.

use serde::Serialize;
use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

const MAX_STREAM_BYTES: usize = 16 * 1024 * 1024;

/// Cleanup guard for the process group owned by one command invocation.
///
/// Unix launches create a fresh process group whose id is the direct child's
/// pid.  This makes timeout, cancellation, and dropped futures kill
/// descendants too. Windows currently only has Tokio's direct-child
/// `kill_on_drop` guarantee; Job Object cleanup is deliberately left to a
/// future platform-specific implementation.
pub(crate) struct ProcessGroupGuard {
    #[cfg(unix)]
    pid: Option<u32>,
    armed: bool,
    ownership_lost: bool,
}

impl ProcessGroupGuard {
    #[cfg(unix)]
    pub(crate) fn new(pid: Option<u32>) -> Self {
        Self {
            pid,
            armed: true,
            ownership_lost: false,
        }
    }

    #[cfg(windows)]
    pub(crate) fn new(_: Option<u32>) -> Self {
        Self {
            armed: true,
            ownership_lost: false,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    pub(crate) fn kill_now(&mut self) {
        if !self.armed {
            return;
        }
        #[cfg(unix)]
        if let Some(pid) = self.pid.filter(|pid| *pid > 0 && *pid <= i32::MAX as u32) {
            // Keep the unreaped direct child as the group identity anchor.
            // If ownership has already been lost, never signal a reused PGID.
            if owned_child_state(pid).is_err() {
                self.ownership_lost = true;
                self.armed = false;
                return;
            }
            let _ = nix::sys::signal::killpg(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
        self.armed = false;
    }

    pub(crate) fn child_running(&mut self) -> io::Result<bool> {
        #[cfg(unix)]
        if let Some(pid) = self.pid {
            return match owned_child_state(pid) {
                Ok(exited) => Ok(!exited),
                Err(error) => {
                    self.ownership_lost = true;
                    Err(error)
                }
            };
        }
        self.ownership_lost = true;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "owned child identity unavailable",
        ))
    }

    pub(crate) async fn confirmed_empty(&self) -> bool {
        if self.ownership_lost {
            return false;
        }
        #[cfg(unix)]
        if let Some(pid) = self.pid.filter(|pid| *pid > 0 && *pid <= i32::MAX as u32) {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
            loop {
                match nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid as i32), None) {
                    Err(nix::errno::Errno::ESRCH) => return true,
                    Ok(()) if tokio::time::Instant::now() < deadline => {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    _ => return false,
                }
            }
        }
        false
    }
}

/// The child remains unreaped until its group has been signalled. Its PID
/// therefore cannot be reused between observing exit and group cleanup.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd"
))]
fn owned_child_state(pid: u32) -> io::Result<bool> {
    if pid == 0 || pid > i32::MAX as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid owned child id",
        ));
    }
    // SAFETY: waitid receives initialized, writable siginfo storage and a
    // positive PID. WNOWAIT observes only our child without reaping it.
    let (result, info) = unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        let result = libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        );
        (result, info)
    };
    if result == 0 {
        Ok(info.si_signo != 0)
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd"
)))]
fn owned_child_state(_: u32) -> io::Result<bool> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "nonreaping child ownership unavailable",
    ))
}

async fn observe_owned_exit(pid: Option<u32>) -> io::Result<()> {
    let pid =
        pid.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "owned child identity missing"))?;
    loop {
        match owned_child_state(pid) {
            Ok(true) => return Ok(()),
            Ok(false) => tokio::time::sleep(Duration::from_millis(10)).await,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProcessCleanup {
    ConfirmedClean,
    CleanupUnknown,
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        self.kill_now();
    }
}

/// Spawn a direct argv command with bounded-pipe capture and owned lifecycle.
/// The argv is passed verbatim; no shell interpolation occurs here.
pub(crate) fn spawn_owned_argv(
    argv: &[String],
    cwd: &Path,
) -> io::Result<(Child, ProcessGroupGuard)> {
    let Some(program) = argv.first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "formatter command is empty",
        ));
    };
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let child = command.spawn()?;
    let group = ProcessGroupGuard::new(child.id());
    Ok((child, group))
}

pub(crate) struct CapturedCommand {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub status: Option<ExitStatus>,
    pub partial: bool,
    pub timed_out: bool,
    pub cancelled: bool,
    pub error: Option<String>,
    pub cleanup: ProcessCleanup,
}

async fn drain<R: AsyncRead + Unpin>(pipe: Option<R>, output: &mut Vec<u8>) -> io::Result<bool> {
    let Some(mut pipe) = pipe else {
        return Ok(false);
    };
    let mut buffer = [0; 8192];
    let mut partial = false;
    loop {
        let count = pipe.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        let retained = count.min(MAX_STREAM_BYTES.saturating_sub(output.len()));
        output.extend_from_slice(&buffer[..retained]);
        partial |= retained < count;
    }
    Ok(partial)
}

/// Callers spawn with kill_on_drop(true). Both pipes drain concurrently, and
/// buffers live outside the deadline future so timeout retains partial output.
#[cfg(test)]
pub(crate) async fn capture(child: Child, timeout: Duration) -> io::Result<CapturedCommand> {
    capture_with_cancel(child, timeout, &CancellationToken::new()).await
}

/// Keep the stream futures alive during interruption cleanup. This retains both
/// already-read text and buffered pipe text available after the child is killed.
pub(crate) async fn capture_with_cancel(
    child: Child,
    timeout: Duration,
    cancel: &CancellationToken,
) -> io::Result<CapturedCommand> {
    capture_with_cancel_inner(child, timeout, cancel, None).await
}

/// Capture while killing the owned process group immediately on interruption,
/// before waiting for inherited pipe handles to drain.
pub(crate) async fn capture_owned_with_cancel(
    child: Child,
    timeout: Duration,
    cancel: &CancellationToken,
    group: &mut ProcessGroupGuard,
) -> io::Result<CapturedCommand> {
    capture_with_cancel_inner(child, timeout, cancel, Some(group)).await
}

struct CaptureProcess<'a> {
    child: Child,
    group: Option<&'a mut ProcessGroupGuard>,
}

impl Drop for CaptureProcess<'_> {
    fn drop(&mut self) {
        // Signal the group while the Child is still owned and unreaped. A
        // dropped capture future must not drop/reap the anchor first.
        if let Some(group) = self.group.as_mut() {
            group.kill_now();
        }
    }
}

async fn capture_with_cancel_inner(
    child: Child,
    timeout: Duration,
    cancel: &CancellationToken,
    group: Option<&mut ProcessGroupGuard>,
) -> io::Result<CapturedCommand> {
    let mut process = CaptureProcess { child, group };
    let child = &mut process.child;
    let group = &mut process.group;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let mut stdout_bytes = Vec::new();
    let mut stderr_bytes = Vec::new();
    let mut status = None;
    let mut streams_result = None;
    let mut timed_out = false;
    let mut cancelled = false;
    let mut capture_failed = false;
    let mut errors = Vec::new();
    let owned = group.is_some();
    let observe_owned = owned
        && cfg!(any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos",
            target_os = "ios",
            target_os = "freebsd"
        ));
    let child_id = child.id();
    {
        let streams = async {
            tokio::join!(
                drain(stdout, &mut stdout_bytes),
                drain(stderr, &mut stderr_bytes)
            )
        };
        tokio::pin!(streams);
        let deadline = tokio::time::sleep(timeout);
        tokio::pin!(deadline);
        while streams_result.is_none() || status.is_none() {
            tokio::select! {
                biased;
                result = &mut streams, if streams_result.is_none() => {
                    capture_failed = result.0.is_err() || result.1.is_err();
                    streams_result = Some(result);
                    if capture_failed { break; }
                },
                result = observe_owned_exit(child_id), if observe_owned => {
                    if let Err(error) = result {
                        errors.push(format!("owned wait: {error}"));
                        capture_failed = true;
                    }
                    break;
                },
                result = child.wait(), if !observe_owned && status.is_none() => match result {
                    Ok(completed) => status = Some(completed),
                    Err(error) => { errors.push(format!("wait: {error}")); capture_failed = true; break; }
                },
                _ = cancel.cancelled() => { cancelled = true; break; }
                _ = &mut deadline => { timed_out = true; break; }
            }
        }
        if owned || timed_out || cancelled || capture_failed {
            if let Some(group) = group.as_mut() {
                group.kill_now();
            }
            // An exited direct child may leave an inherited pipe open; killing
            // can then be redundant. Keep the partial evidence in either case.
            let _ = child.start_kill();
            let cleanup_wait = async {
                if status.is_none() {
                    tokio::time::timeout(Duration::from_secs(2), child.wait())
                        .await
                        .ok()
                        .and_then(Result::ok)
                } else {
                    status
                }
            };
            let cleanup_read = async {
                if streams_result.is_none() {
                    tokio::time::timeout(Duration::from_secs(2), &mut streams)
                        .await
                        .ok()
                } else {
                    streams_result
                }
            };
            (status, streams_result) = tokio::join!(cleanup_wait, cleanup_read);
        }
    }
    let mut partial = capture_failed;
    match streams_result.as_ref() {
        Some((stdout, stderr)) => {
            for (name, result) in [("stdout", stdout), ("stderr", stderr)] {
                match result {
                    Ok(limited) => partial |= *limited,
                    Err(error) => {
                        partial = true;
                        errors.push(format!("{name}: {error}"));
                    }
                }
            }
        }
        None => partial = true,
    }
    let cleanup = if let Some(group) = group.as_ref() {
        let pipes_closed = matches!(streams_result, Some((Ok(_), Ok(_))));
        let uncertainty = if status.is_none() {
            Some("direct-child wait incomplete")
        } else if !pipes_closed {
            Some("pipe closure unconfirmed")
        } else if group.ownership_lost {
            Some("owned child identity lost")
        } else if !group.confirmed_empty().await {
            Some("owned process group remains present")
        } else {
            None
        };
        if let Some(reason) = uncertainty {
            partial = true;
            errors.push(format!(
                "Owned process cleanup could not be confirmed: {reason}"
            ));
            ProcessCleanup::CleanupUnknown
        } else {
            ProcessCleanup::ConfirmedClean
        }
    } else {
        ProcessCleanup::CleanupUnknown
    };
    Ok(CapturedCommand {
        stdout: stdout_bytes,
        stderr: stderr_bytes,
        status,
        partial: partial || timed_out || cancelled,
        timed_out,
        cancelled,
        cleanup,
        error: if errors.is_empty() {
            None
        } else {
            Some(errors.join("; "))
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;
    use tokio::process::Command;

    #[cfg(unix)]
    #[tokio::test]
    async fn owned_completion_requires_confirmed_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let argv = vec!["sh".into(), "-c".into(), "printf complete".into()];
        let (child, mut group) = spawn_owned_argv(&argv, dir.path()).unwrap();
        let output = capture_owned_with_cancel(
            child,
            Duration::from_secs(3),
            &CancellationToken::new(),
            &mut group,
        )
        .await
        .unwrap();
        assert_eq!(
            output.cleanup,
            ProcessCleanup::ConfirmedClean,
            "error={:?}, status={:?}, partial={}",
            output.error,
            output.status,
            output.partial
        );
        assert_eq!(output.stdout, b"complete");
        assert!(output.status.unwrap().success());
        assert!(!output.partial);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn owned_exit_cleans_redirected_descendant_before_return() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("escaped-write");
        let gate = dir.path().join("allow-write");
        let argv = vec![
            "sh".into(),
            "-c".into(),
            format!(
                "(while [ ! -e '{}' ]; do sleep 0.01; done; touch '{}') >/dev/null 2>&1 & printf direct-done; exit 0",
                gate.display(), marker.display()
            ),
        ];
        let (child, mut group) = spawn_owned_argv(&argv, dir.path()).unwrap();
        let output = capture_owned_with_cancel(
            child,
            Duration::from_secs(3),
            &CancellationToken::new(),
            &mut group,
        )
        .await
        .unwrap();
        assert!(output.status.unwrap().success());
        if output.cleanup == ProcessCleanup::CleanupUnknown {
            assert!(output.partial);
            assert!(output.error.as_deref().is_some_and(
                |error| error.contains("Owned process cleanup could not be confirmed")
            ));
        }
        assert!(!marker.exists());
        std::fs::write(&gate, b"release").unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !marker.exists(),
            "redirected descendant survived finite command completion"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unowned_capture_never_claims_group_cleanup() {
        let child = Command::new("sh")
            .args(["-c", "exit 0"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let output = capture(child, Duration::from_secs(3)).await.unwrap();
        assert_eq!(output.cleanup, ProcessCleanup::CleanupUnknown);
        assert!(
            !output.partial,
            "capture completeness is separate from group ownership"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn lost_child_identity_cannot_be_certified_clean() {
        let dir = tempfile::tempdir().unwrap();
        let argv = vec!["sh".into(), "-c".into(), "exit 0".into()];
        let (mut child, mut group) = spawn_owned_argv(&argv, dir.path()).unwrap();
        child.wait().await.unwrap();
        assert!(group.child_running().is_err());
        group.kill_now();
        assert!(!group.confirmed_empty().await);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropped_owned_capture_kills_before_reaping_child() {
        let dir = tempfile::tempdir().unwrap();
        let ready = dir.path().join("ready");
        let marker = dir.path().join("late-write");
        let gate = dir.path().join("allow-write");
        let argv = vec![
            "sh".into(),
            "-c".into(),
            format!(
                "(touch '{}'; while [ ! -e '{}' ]; do sleep 0.01; done; touch '{}') & wait",
                ready.display(),
                gate.display(),
                marker.display()
            ),
        ];
        let cwd = dir.path().to_path_buf();
        let task = tokio::spawn(async move {
            let (child, mut group) = spawn_owned_argv(&argv, &cwd).unwrap();
            capture_owned_with_cancel(
                child,
                Duration::from_secs(10),
                &CancellationToken::new(),
                &mut group,
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(3), async {
            while !ready.exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            !marker.exists(),
            "write gate must remain closed before cancellation"
        );
        task.abort();
        let _ = task.await;
        std::fs::write(&gate, b"release").unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !marker.exists(),
            "dropped capture left a writing descendant"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn concurrent_pipes_preserve_large_stdout_and_stderr() {
        let child = Command::new("sh").args(["-c", "awk 'BEGIN { for(i=0;i<20000;i++) { print \"out🦀\"; print \"err🦀\" > \"/dev/stderr\" }}'"])
            .stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null()).kill_on_drop(true).spawn().unwrap();
        let output = capture(child, Duration::from_secs(5)).await.unwrap();
        assert!(!output.timed_out);
        assert!(!output.partial);
        assert!(output.status.unwrap().success());
        assert_eq!(
            String::from_utf8(output.stdout)
                .unwrap()
                .matches("out🦀")
                .count(),
            20000
        );
        assert_eq!(
            String::from_utf8(output.stderr)
                .unwrap()
                .matches("err🦀")
                .count(),
            20000
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_retains_both_partial_streams() {
        let child = Command::new("sh")
            .args([
                "-c",
                "printf before-timeout; printf stderr-before-timeout >&2; exec sleep 10",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let output = capture(child, Duration::from_millis(100)).await.unwrap();
        assert!(output.timed_out && output.partial);
        assert_eq!(output.stdout, b"before-timeout");
        assert_eq!(output.stderr, b"stderr-before-timeout");
        assert!(output.status.is_some());
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_keeps_available_output() {
        let workspace = tempfile::tempdir().unwrap();
        let ready = workspace.path().join("ready");
        let child = Command::new("sh")
            .args(["-c", "printf stdout-before-cancel; printf stderr-before-cancel >&2; touch ready; exec sleep 10"])
            .current_dir(workspace.path())
            .stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null())
            .kill_on_drop(true).spawn().unwrap();
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
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
        let (output, ()) = tokio::join!(
            capture_with_cancel(child, Duration::from_secs(10), &cancel),
            cancellation
        );
        let output = output.unwrap();
        assert!(output.cancelled && output.partial && !output.timed_out);
        assert_eq!(output.stdout, b"stdout-before-cancel");
        assert_eq!(output.stderr, b"stderr-before-cancel");
        assert!(output.status.is_some());
    }
}
