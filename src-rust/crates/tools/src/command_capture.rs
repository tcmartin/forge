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

#[cfg(test)]
macro_rules! process_group_diag {
    ($($arg:tt)*) => {
        eprintln!("[owned-process-group] {}", format_args!($($arg)*));
    };
}
#[cfg(not(test))]
macro_rules! process_group_diag {
    ($($arg:tt)*) => {{
        let _ = ();
    }};
}

const MAX_STREAM_BYTES: usize = 16 * 1024 * 1024;

/// Cleanup guard for the process group owned by one command invocation.
///
/// Unix launches create a fresh process group whose id is the direct child's
/// pid. Windows launches suspend the child, assign it to a kill-on-close Job
/// Object, verify membership and the initial thread identity, then resume it.
pub(crate) struct ProcessGroupGuard {
    #[cfg(unix)]
    pid: Option<u32>,
    #[cfg(windows)]
    job: Option<claurst_core::windows_job_object::JobObjectGuard>,
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

    #[cfg(unix)]
    pub(crate) fn arm(child: &Child) -> io::Result<Self> {
        Ok(Self::new(child.id()))
    }

    #[cfg(all(not(unix), not(windows)))]
    pub(crate) fn arm(child: &Child) -> io::Result<Self> {
        let _ = child;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "owned process identity unavailable on this platform",
        ))
    }

    pub(crate) fn disarm(&mut self) {
        #[cfg(windows)]
        {
            if let Some(job) = self.job.as_mut() {
                if job.disarm() {
                    self.armed = false;
                }
            }
            return;
        }
        #[cfg(unix)]
        {
            self.armed = false;
        }
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
        #[cfg(windows)]
        if let Some(job) = self.job.as_mut() {
            job.kill_now();
            self.ownership_lost |= job.ownership_lost();
        } else {
            self.ownership_lost = true;
        }
        #[cfg(unix)]
        {
            self.armed = false;
        }
    }

    pub(crate) fn child_running(&mut self) -> io::Result<bool> {
        #[cfg(windows)]
        {
            let Some(job) = self.job.as_mut() else {
                self.ownership_lost = true;
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "owned child identity unavailable",
                ));
            };
            return match job.child_running() {
                Ok(running) => Ok(running),
                Err(error) => {
                    self.ownership_lost = true;
                    Err(error)
                }
            };
        }
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
        #[cfg(not(windows))]
        {
            self.ownership_lost = true;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "owned child identity unavailable",
            ))
        }
    }

    pub(crate) async fn confirmed_empty(&self) -> bool {
        if self.ownership_lost {
            process_group_diag!("ownership_lost=true");
            return false;
        }
        #[cfg(windows)]
        if let Some(job) = self.job.as_ref() {
            return job.confirmed_empty().await;
        }
        #[cfg(unix)]
        if let Some(pid) = self.pid.filter(|pid| *pid > 0 && *pid <= i32::MAX as u32) {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
            #[cfg(test)]
            let mut probes = 0_u32;
            loop {
                #[cfg(test)]
                {
                    probes += 1;
                }
                match nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid as i32), None) {
                    Err(nix::errno::Errno::ESRCH) => {
                        process_group_diag!("pid={} probes={} group_absent=true", pid, probes);
                        return true;
                    }
                    Ok(()) if tokio::time::Instant::now() < deadline => {
                        #[cfg(test)]
                        if probes == 1 {
                            process_group_diag!("pid={} probes={} group_absent=false", pid, probes);
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    Ok(()) => {
                        process_group_diag!(
                            "pid={} probes={} group_absent=false timeout=true",
                            pid,
                            probes
                        );
                        return false;
                    }
                    Err(error) => {
                        process_group_diag!(
                            "pid={} probes={} group_absent=false probe_error={:?}",
                            pid,
                            probes,
                            error
                        );
                        #[cfg(not(test))]
                        let _ = &error;
                        return false;
                    }
                }
            }
        }
        process_group_diag!("identity_missing=true");
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

#[cfg(windows)]
pub(crate) use claurst_core::windows_job_object::SpawnOwnedError;

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
    #[cfg(windows)]
    {
        let _ = (argv, cwd);
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "direct spawn is unavailable on Windows without the checked owned launch sequence",
        ));
    }
    #[cfg(not(windows))]
    {
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
        let group = ProcessGroupGuard::arm(&child)?;
        return Ok((child, group));
    }
    #[allow(unreachable_code)]
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "owned direct spawn unavailable on this platform",
    ))
}

#[cfg(windows)]
pub(crate) fn spawn_owned_argv_checked(
    argv: &[String],
    cwd: &Path,
) -> Result<(Child, ProcessGroupGuard), SpawnOwnedError> {
    spawn_owned_argv_checked_with_stdio(argv, cwd, Stdio::null(), Stdio::piped(), Stdio::piped())
}

#[cfg(windows)]
pub(crate) fn spawn_owned_argv_checked_with_stdio(
    argv: &[String],
    cwd: &Path,
    stdin: Stdio,
    stdout: Stdio,
    stderr: Stdio,
) -> Result<(Child, ProcessGroupGuard), SpawnOwnedError> {
    let Some(program) = argv.first() else {
        return Err(SpawnOwnedError::Spawn(io::Error::new(
            io::ErrorKind::InvalidInput,
            "owned command is empty",
        )));
    };
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .current_dir(cwd)
        .stdout(stdout)
        .stderr(stderr)
        .stdin(stdin)
        .kill_on_drop(true);
    spawn_owned_command_checked(command)
}

/// Spawn one authorized raw shell command through cmd.exe without applying
/// CRT argv quoting to the command text. cmd.exe parses /C text itself and
/// does not follow CommandLineToArgvW rules; raw_arg preserves embedded quotes
/// and shell metacharacters while the outer pair supplies cmd /S /C framing.
#[cfg(windows)]
pub(crate) fn spawn_owned_cmd_shell_checked(
    command_text: &str,
    cwd: &Path,
) -> Result<(Child, ProcessGroupGuard), SpawnOwnedError> {
    let mut command = claurst_core::windows_job_object::cmd_shell_command(command_text);
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    spawn_owned_command_checked(command)
}

#[cfg(windows)]
pub(crate) fn spawn_owned_command_checked(
    command: Command,
) -> Result<(Child, ProcessGroupGuard), SpawnOwnedError> {
    match claurst_core::windows_job_object::spawn_owned_command_checked(command) {
        Ok((child, job)) => Ok((
            child,
            ProcessGroupGuard {
                job: Some(job),
                armed: true,
                ownership_lost: false,
            },
        )),
        Err(SpawnOwnedError::Spawn(error)) => Err(SpawnOwnedError::Spawn(error)),
        Err(SpawnOwnedError::Arm { child, error }) => Err(SpawnOwnedError::Arm { child, error }),
    }
}

#[cfg(windows)]
pub(crate) async fn cleanup_unarmed_child(mut child: Child) -> io::Result<()> {
    let _ = child.start_kill();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let mut stdout_bytes = Vec::new();
    let mut stderr_bytes = Vec::new();
    let wait = async {
        tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .ok()
            .and_then(Result::ok)
    };
    let streams = async {
        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(
                drain(stdout, &mut stdout_bytes),
                drain(stderr, &mut stderr_bytes),
            )
        })
        .await
        .ok()
    };
    let (wait_result, streams_result) = tokio::join!(wait, streams);
    let wait_complete = wait_result.is_some();
    let pipes_closed = matches!(streams_result, Some((Ok(_), Ok(_))));
    Err(io::Error::new(
        io::ErrorKind::Other,
        format!(
            "job arm failed; bounded direct-child cleanup completed (wait={wait_complete}, pipes={pipes_closed}) but process-tree cleanup remains unknown"
        ),
    ))
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

    #[cfg(windows)]
    fn powershell_argv(script: &str) -> Vec<String> {
        vec![
            "powershell".into(),
            "-NoProfile".into(),
            "-NonInteractive".into(),
            "-Command".into(),
            script.into(),
        ]
    }

    // These ignored fixtures are invoked by the owned-process-tree tests. The
    // exact filter prevents a broad or zero-match test invocation from being
    // mistaken for the intended parent/descendant witness.
    #[cfg(windows)]
    const WINDOWS_TREE_MARKER: &str = "windows-job-object-pids.txt";

    #[cfg(windows)]
    const WINDOWS_TREE_STARTED_MARKER: &str = "windows-job-object-descendant-started.txt";

    #[cfg(windows)]
    fn windows_tree_fixture_argv(test_name: &str) -> Vec<String> {
        vec![
            std::env::current_exe()
                .expect("test executable")
                .to_string_lossy()
                .into_owned(),
            test_name.into(),
            "--exact".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ]
    }

    #[cfg(windows)]
    fn write_windows_marker(path: &Path, contents: &str) -> io::Result<()> {
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, contents)?;
        std::fs::rename(temporary, path)
    }

    #[cfg(windows)]
    fn wait_for_windows_fixture_started(marker: &Path, descendant_pid: u32) -> io::Result<()> {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            if let Ok(contents) = std::fs::read_to_string(marker)
                && contents.trim().parse::<u32>().ok() == Some(descendant_pid)
            {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "exact descendant fixture did not publish its startup witness",
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(windows)]
    fn checked_spawn(argv: &[String], cwd: &Path) -> (Child, ProcessGroupGuard) {
        spawn_owned_argv_checked(argv, cwd).unwrap_or_else(|error| panic!("{}", error.message()))
    }

    #[cfg(windows)]
    fn windows_process_exists(pid: u32) -> io::Result<bool> {
        let query = format!(
            "if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"
        );
        std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &query])
            .status()
            .map(|status| status.success())
    }

    #[cfg(windows)]
    async fn wait_for_windows_processes_gone(pids: [u32; 2]) -> io::Result<()> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            let parent_exists = windows_process_exists(pids[0])?;
            let descendant_exists = windows_process_exists(pids[1])?;
            if !parent_exists && !descendant_exists {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "owned parent or descendant remained after bounded witness polling",
                ));
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    #[cfg(windows)]
    async fn wait_for_windows_pid_witness(marker: &Path, parent_pid: u32) -> io::Result<[u32; 2]> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            if let Ok(contents) = std::fs::read_to_string(marker) {
                let mut fields = contents.trim().split(',');
                let observed_parent = fields.next().and_then(|value| value.trim().parse().ok());
                let descendant = fields.next().and_then(|value| value.trim().parse().ok());
                if fields.next().is_none()
                    && observed_parent == Some(parent_pid)
                    && descendant.is_some_and(|pid| pid != 0)
                {
                    return Ok([parent_pid, descendant.unwrap()]);
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "owned parent and descendant startup witness was not observed",
                ));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[cfg(windows)]
    async fn capture_windows(
        argv: &[String],
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> CapturedCommand {
        let dir = tempfile::tempdir().unwrap();
        let (child, mut group) = checked_spawn(argv, dir.path());
        capture_owned_with_cancel(child, timeout, cancel, &mut group)
            .await
            .unwrap()
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "launched as a child by the owned-process-tree tests"]
    fn windows_process_tree_parent_fixture() {
        let marker = Path::new(WINDOWS_TREE_MARKER);
        let started = Path::new(WINDOWS_TREE_STARTED_MARKER);
        let mut descendant =
            std::process::Command::new(std::env::current_exe().expect("test executable"))
                .args([
                    "windows_process_tree_descendant_fixture",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                ])
                .spawn()
                .expect("spawn native descendant fixture");
        if let Err(error) = wait_for_windows_fixture_started(started, descendant.id()) {
            let _ = descendant.kill();
            let _ = descendant.wait();
            panic!("exact descendant fixture startup witness: {error}");
        }
        write_windows_marker(
            marker,
            &format!("{},{}", std::process::id(), descendant.id()),
        )
        .expect("publish process-tree witness");
        let _ = descendant.wait();
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "launched as a child by the owned-process-tree tests"]
    fn windows_process_tree_descendant_fixture() {
        write_windows_marker(
            Path::new(WINDOWS_TREE_STARTED_MARKER),
            &std::process::id().to_string(),
        )
        .expect("publish descendant startup witness");
        std::thread::sleep(Duration::from_secs(30));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_direct_completion_proves_empty_job() {
        let output = capture_windows(
            &powershell_argv("exit 0"),
            Duration::from_secs(5),
            &CancellationToken::new(),
        )
        .await;
        assert_eq!(output.cleanup, ProcessCleanup::ConfirmedClean);
        assert!(output.status.unwrap().success());
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_descendant_cancel_proves_job_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join(WINDOWS_TREE_MARKER);
        let argv = windows_tree_fixture_argv("windows_process_tree_parent_fixture");
        let (child, mut group) = checked_spawn(&argv, dir.path());
        let parent_pid = child.id().expect("owned parent pid");
        let cancel = CancellationToken::new();
        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                capture_owned_with_cancel(child, Duration::from_secs(30), &cancel, &mut group)
                    .await
                    .unwrap()
            }
        });
        let pids = wait_for_windows_pid_witness(&marker, parent_pid)
            .await
            .expect("parent and descendant startup witness");
        cancel.cancel();
        let output = task.await.unwrap();
        assert!(output.cancelled);
        assert_eq!(output.cleanup, ProcessCleanup::ConfirmedClean);
        wait_for_windows_processes_gone(pids)
            .await
            .expect("owned parent and descendant are gone");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_timeout_proves_job_cleanup() {
        let output = capture_windows(
            &powershell_argv("Start-Sleep 30"),
            Duration::from_millis(150),
            &CancellationToken::new(),
        )
        .await;
        assert!(output.timed_out);
        assert_eq!(output.cleanup, ProcessCleanup::ConfirmedClean);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_dropped_capture_proves_parent_and_descendant_gone() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join(WINDOWS_TREE_MARKER);
        let argv = windows_tree_fixture_argv("windows_process_tree_parent_fixture");
        let cwd = dir.path().to_path_buf();
        let (child, mut group) = checked_spawn(&argv, &cwd);
        let parent_pid = child.id().expect("owned parent pid");
        let task = tokio::spawn(async move {
            capture_owned_with_cancel(
                child,
                Duration::from_secs(30),
                &CancellationToken::new(),
                &mut group,
            )
            .await
        });
        let pids = wait_for_windows_pid_witness(&marker, parent_pid)
            .await
            .expect("parent and descendant pid witness");
        task.abort();
        let _ = task.await;
        tokio::time::timeout(
            Duration::from_secs(3),
            wait_for_windows_processes_gone(pids),
        )
        .await
        .expect("owned parent and descendant are gone")
        .expect("process witness query");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_foreign_process_survives_owner_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let (mut foreign, mut foreign_group) =
            checked_spawn(&powershell_argv("Start-Sleep 30"), dir.path());
        let marker = dir.path().join(WINDOWS_TREE_MARKER);
        let argv = windows_tree_fixture_argv("windows_process_tree_parent_fixture");
        let (child, mut group) = checked_spawn(&argv, dir.path());
        let parent_pid = child.id().expect("owned parent pid");
        let cancel = CancellationToken::new();
        let task = tokio::spawn({
            let cancel = cancel.clone();
            async move {
                capture_owned_with_cancel(child, Duration::from_secs(30), &cancel, &mut group)
                    .await
                    .unwrap()
            }
        });
        let pids = wait_for_windows_pid_witness(&marker, parent_pid)
            .await
            .expect("owned parent and descendant started");
        assert!(foreign.try_wait().unwrap().is_none());
        cancel.cancel();
        let result = task.await.unwrap();
        assert!(result.cancelled);
        assert_eq!(result.cleanup, ProcessCleanup::ConfirmedClean);
        wait_for_windows_processes_gone(pids)
            .await
            .expect("owned parent and descendant are gone");
        assert!(foreign.try_wait().unwrap().is_none());
        foreign_group.kill_now();
        foreign.wait().await.expect("join foreign fixture");
        assert!(foreign_group.confirmed_empty().await);
        foreign_group.disarm();
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_reaped_child_cannot_be_used_as_pid_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (mut child, mut group) = checked_spawn(&powershell_argv("exit 0"), dir.path());
        let status = child.wait().await.unwrap();
        assert!(status.success());
        assert!(!group.child_running().unwrap());
        group.kill_now();
        assert!(group.confirmed_empty().await);
    }

    #[cfg(windows)]
    #[test]
    fn windows_spawn_failure_does_not_create_owned_task() {
        let dir = tempfile::tempdir().unwrap();
        let argv = vec!["definitely-not-a-real-program-for-r55".into()];
        assert!(spawn_owned_argv_checked(&argv, dir.path()).is_err());
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_cmd_shell_preserves_quoted_path_and_metacharacters() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("owned marker & exact.txt");
        let command = format!("echo owned > \"{}\"", marker.display());
        let (child, mut group) = spawn_owned_cmd_shell_checked(&command, dir.path())
            .unwrap_or_else(|error| panic!("owned cmd shell: {}", error.message()));
        let output = capture_owned_with_cancel(
            child,
            Duration::from_secs(5),
            &CancellationToken::new(),
            &mut group,
        )
        .await
        .expect("bounded cmd capture");
        assert!(output.status.expect("cmd status").success());
        assert!(
            marker.is_file(),
            "quoted metacharacter path was not written"
        );
        assert!(group.confirmed_empty().await);
    }

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
