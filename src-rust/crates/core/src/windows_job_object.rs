//! Shared Windows process-tree ownership primitive.
//!
//! The checked factory is the only launch entry point. It creates the child
//! suspended, establishes Job Object ownership and identity, then resumes it.
//! Callers never receive an armed guard for an already-running child.

#![cfg(windows)]

use std::io;
use std::mem::size_of;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::process::{Child, Command};

type Bool = i32;
type Dword = u32;
type Long = i32;

const TRUE: Bool = 1;
const CREATE_SNAPSHOT_THREAD: Dword = 0x0000_0004;
const CREATE_SUSPENDED: Dword = 0x0000_0004;
const INVALID_HANDLE_VALUE: RawHandle = -1isize as RawHandle;
const ERROR_NO_MORE_FILES: i32 = 18;
const STILL_ACTIVE: Dword = 259;
const DUPLICATE_SAME_ACCESS: Dword = 0x0000_0002;
const THREAD_SUSPEND_RESUME: Dword = 0x0002;
const THREAD_QUERY_LIMITED_INFORMATION: Dword = 0x0800;
const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: Dword = 9;
const JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION: Dword = 1;
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Dword = 0x0000_2000;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FileTime {
    low_date_time: Dword,
    high_date_time: Dword,
}

impl FileTime {
    fn as_u64(self) -> u64 {
        (u64::from(self.high_date_time) << 32) | u64::from(self.low_date_time)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JobObjectBasicLimitInformation {
    per_process_user_time_limit: i64,
    per_job_user_time_limit: i64,
    limit_flags: Dword,
    minimum_working_set_size: usize,
    maximum_working_set_size: usize,
    active_process_limit: Dword,
    affinity: usize,
    priority_class: Dword,
    scheduling_class: Dword,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IoCounters {
    read_operation_count: u64,
    write_operation_count: u64,
    other_operation_count: u64,
    read_transfer_count: u64,
    write_transfer_count: u64,
    other_transfer_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JobObjectExtendedLimitInformation {
    basic_limit_information: JobObjectBasicLimitInformation,
    io_info: IoCounters,
    process_memory_limit: usize,
    job_memory_limit: usize,
    peak_process_memory_used: usize,
    peak_job_memory_used: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JobObjectBasicAccountingInformation {
    total_user_time: i64,
    total_kernel_time: i64,
    this_period_total_user_time: i64,
    this_period_total_kernel_time: i64,
    total_page_faults: Dword,
    total_processes: Dword,
    active_processes: Dword,
    total_terminated_processes: Dword,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ThreadEntry32 {
    dw_size: Dword,
    cnt_usage: Dword,
    th32_thread_id: Dword,
    th32_owner_process_id: Dword,
    tp_base_pri: Long,
    tp_delta_pri: Long,
    dw_flags: Dword,
}

#[link(name = "kernel32")]
extern "system" {
    fn AssignProcessToJobObject(job: RawHandle, process: RawHandle) -> Bool;
    fn CreateJobObjectW(attributes: *const std::ffi::c_void, name: *const u16) -> RawHandle;
    fn CreateToolhelp32Snapshot(flags: Dword, process_id: Dword) -> RawHandle;
    fn DuplicateHandle(
        source_process: RawHandle,
        source_handle: RawHandle,
        target_process: RawHandle,
        target_handle: *mut RawHandle,
        desired_access: Dword,
        inherit_handle: Bool,
        options: Dword,
    ) -> Bool;
    fn GetCurrentProcess() -> RawHandle;
    fn GetExitCodeProcess(process: RawHandle, exit_code: *mut Dword) -> Bool;
    fn GetLastError() -> Dword;
    fn GetProcessId(process: RawHandle) -> Dword;
    fn GetProcessIdOfThread(thread: RawHandle) -> Dword;
    fn GetProcessTimes(
        process: RawHandle,
        creation_time: *mut FileTime,
        exit_time: *mut FileTime,
        kernel_time: *mut FileTime,
        user_time: *mut FileTime,
    ) -> Bool;
    fn IsProcessInJob(process: RawHandle, job: RawHandle, result: *mut Bool) -> Bool;
    fn OpenThread(desired_access: Dword, inherit_handle: Bool, thread_id: Dword) -> RawHandle;
    fn QueryInformationJobObject(
        job: RawHandle,
        info_class: Dword,
        info: *mut u8,
        info_len: Dword,
        return_len: *mut Dword,
    ) -> Bool;
    fn ResumeThread(thread: RawHandle) -> Dword;
    fn SetInformationJobObject(
        job: RawHandle,
        info_class: Dword,
        info: *const u8,
        info_len: Dword,
    ) -> Bool;
    fn TerminateJobObject(job: RawHandle, exit_code: Dword) -> Bool;
    fn Thread32First(snapshot: RawHandle, entry: *mut ThreadEntry32) -> Bool;
    fn Thread32Next(snapshot: RawHandle, entry: *mut ThreadEntry32) -> Bool;
}

fn last_error(operation: &str) -> io::Error {
    // SAFETY: GetLastError reads thread-local error state and takes no pointers.
    let code = unsafe { GetLastError() };
    io::Error::new(
        io::ErrorKind::Other,
        format!("{operation} failed with Win32 error {code}"),
    )
}

fn bool_result(operation: &str, result: Bool) -> io::Result<()> {
    if result == TRUE {
        Ok(())
    } else {
        Err(last_error(operation))
    }
}

fn owned_handle(handle: RawHandle, operation: &str) -> io::Result<OwnedHandle> {
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        return Err(last_error(operation));
    }
    // SAFETY: the caller has checked the Win32 handle sentinel and transfers
    // this live handle's sole close responsibility to OwnedHandle.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
}

fn duplicate_process_handle(raw: RawHandle) -> io::Result<OwnedHandle> {
    let mut duplicate = std::ptr::null_mut();
    bool_result(
        "DuplicateHandle",
        // SAFETY: raw is borrowed from the still-live Tokio Child; the target
        // is this process and DuplicateHandle initializes `duplicate`.
        unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                raw,
                GetCurrentProcess(),
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        },
    )?;
    owned_handle(duplicate, "DuplicateHandle returned no handle")
}

fn process_creation_time(process: RawHandle) -> io::Result<u64> {
    let mut creation = FileTime::default();
    let mut exit = FileTime::default();
    let mut kernel = FileTime::default();
    let mut user = FileTime::default();
    bool_result(
        "GetProcessTimes",
        // SAFETY: all FILETIME outputs are initialized writable storage.
        unsafe { GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user) },
    )?;
    Ok(creation.as_u64())
}

fn sole_initial_thread(process_id: Dword) -> io::Result<OwnedHandle> {
    let snapshot = owned_handle(
        // SAFETY: the call has no borrowed Rust pointers and requests a thread
        // snapshot for the retained process identity.
        unsafe { CreateToolhelp32Snapshot(CREATE_SNAPSHOT_THREAD, 0) },
        "CreateToolhelp32Snapshot",
    )?;
    let mut entry = ThreadEntry32 {
        dw_size: size_of::<ThreadEntry32>() as Dword,
        ..ThreadEntry32::default()
    };
    let mut candidate = None;
    // SAFETY: snapshot is owned and entry is initialized writable THREADENTRY32 storage.
    let first = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) };
    if first != TRUE {
        // SAFETY: GetLastError reads thread-local error state and takes no pointers.
        let error = unsafe { GetLastError() } as i32;
        return Err(io::Error::new(
            io::ErrorKind::Other,
            if error == ERROR_NO_MORE_FILES {
                "Thread32First found no initial thread"
            } else {
                "Thread32First failed"
            },
        ));
    }
    loop {
        if entry.th32_owner_process_id == process_id {
            if candidate.replace(entry.th32_thread_id).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "suspended child has multiple initial-thread candidates",
                ));
            }
        }
        // SAFETY: snapshot is owned and entry is initialized writable THREADENTRY32 storage.
        let next = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) };
        if next == TRUE {
            continue;
        }
        // SAFETY: GetLastError reads thread-local error state and takes no pointers.
        let error = unsafe { GetLastError() } as i32;
        if error != ERROR_NO_MORE_FILES {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Thread32Next failed with Win32 error {error}"),
            ));
        }
        break;
    }
    let thread_id = candidate.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "suspended child has no initial thread candidate",
        )
    })?;
    owned_handle(
        // SAFETY: the snapshot identified this thread id and OpenThread returns
        // an owned kernel handle or the documented NULL failure sentinel.
        unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_QUERY_LIMITED_INFORMATION,
                0,
                thread_id,
            )
        },
        "OpenThread",
    )
}

pub struct JobObjectGuard {
    job: Option<OwnedHandle>,
    process: OwnedHandle,
    direct_pid: Dword,
    creation_time: u64,
    kill_requested: bool,
    ownership_lost: AtomicBool,
    empty_verified: AtomicBool,
}

#[cfg(test)]
std::thread_local! {
    // Arming is synchronous before the first await. Scope a fault to its test
    // thread rather than allowing another parallel test to consume it.
    static FAIL_BEFORE_RESUME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn fail_next_arm_before_resume() {
    FAIL_BEFORE_RESUME.with(|failure| failure.set(true));
}

pub enum SpawnOwnedError {
    Spawn(io::Error),
    Arm { child: Child, error: io::Error },
}

impl SpawnOwnedError {
    pub fn message(&self) -> String {
        match self {
            Self::Spawn(error) => format!("owned child spawn failed: {error}"),
            Self::Arm { error, .. } => format!("owned job arm failed: {error}"),
        }
    }
}

/// Spawn a Windows child through the complete suspended -> owned -> resumed
/// sequence. The returned child is still owned by the caller when arming
/// fails, allowing bounded cleanup and an explicit Unknown result.
pub fn spawn_owned_command_checked(
    mut command: Command,
) -> Result<(Child, JobObjectGuard), SpawnOwnedError> {
    command.creation_flags(CREATE_SUSPENDED);
    let child = command.spawn().map_err(SpawnOwnedError::Spawn)?;
    match JobObjectGuard::arm(&child) {
        Ok(job) => Ok((child, job)),
        Err(error) => Err(SpawnOwnedError::Arm { child, error }),
    }
}

impl JobObjectGuard {
    /// Arm ownership while the Tokio child is still suspended.  Every failure
    /// leaves the child suspended or kill-on-drop owned by Tokio; the job's
    /// close-on-drop rule handles the assigned-process case.
    fn arm(child: &Child) -> io::Result<Self> {
        let direct_pid = child
            .id()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "child exited before arm"))?;
        let raw = child
            .raw_handle()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "child handle unavailable"))?;
        let process = duplicate_process_handle(raw)?;
        // SAFETY: the process handle is owned and retained through this identity query.
        let retained_pid = unsafe { GetProcessId(process.as_raw_handle()) };
        if retained_pid == 0 || retained_pid != direct_pid {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "retained process handle identity does not match Tokio child",
            ));
        }
        let creation_time = process_creation_time(process.as_raw_handle())?;
        let job = owned_handle(
            // SAFETY: NULL attributes/name creates a fresh unnamed job.
            unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) },
            "CreateJobObjectW",
        )?;
        let mut limits = JobObjectExtendedLimitInformation::default();
        limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        bool_result(
            "SetInformationJobObject",
            // SAFETY: limits has the exact repr(C) ABI layout and remains live
            // for the duration of this synchronous call.
            unsafe {
                SetInformationJobObject(
                    job.as_raw_handle(),
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    (&limits as *const JobObjectExtendedLimitInformation).cast(),
                    size_of::<JobObjectExtendedLimitInformation>() as Dword,
                )
            },
        )?;
        bool_result(
            "AssignProcessToJobObject",
            // SAFETY: both handles are owned and valid until this call returns.
            unsafe { AssignProcessToJobObject(job.as_raw_handle(), process.as_raw_handle()) },
        )?;
        let mut in_job = 0;
        bool_result(
            "IsProcessInJob",
            // SAFETY: in_job is writable storage for the BOOL result.
            unsafe { IsProcessInJob(process.as_raw_handle(), job.as_raw_handle(), &mut in_job) },
        )?;
        if in_job != TRUE {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "AssignProcessToJobObject did not establish membership",
            ));
        }
        let thread = sole_initial_thread(retained_pid)?;
        // SAFETY: thread is a live owned handle retained through this identity query.
        let thread_pid = unsafe { GetProcessIdOfThread(thread.as_raw_handle()) };
        if thread_pid == 0 || thread_pid != retained_pid {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "initial thread identity does not match retained child handle",
            ));
        }
        #[cfg(test)]
        if FAIL_BEFORE_RESUME.with(|failure| failure.replace(false)) {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "test-only pre-resume arm failure",
            ));
        }
        // SAFETY: thread is the retained initial thread whose process identity was verified.
        let previous_suspend_count = unsafe { ResumeThread(thread.as_raw_handle()) };
        if previous_suspend_count != 1 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("ResumeThread returned unexpected count {previous_suspend_count}"),
            ));
        }
        Ok(Self {
            job: Some(job),
            process,
            direct_pid: retained_pid,
            creation_time,
            kill_requested: false,
            ownership_lost: AtomicBool::new(false),
            empty_verified: AtomicBool::new(false),
        })
    }

    pub fn ownership_lost(&self) -> bool {
        self.ownership_lost.load(Ordering::Acquire)
    }

    pub fn kill_now(&mut self) {
        if self.kill_requested {
            return;
        }
        let Some(job) = self.job.as_ref() else {
            self.ownership_lost.store(true, Ordering::Release);
            return;
        };
        // SAFETY: job is a live owned handle retained for the termination call.
        let result = unsafe { TerminateJobObject(job.as_raw_handle(), 1) };
        if result != TRUE {
            self.ownership_lost.store(true, Ordering::Release);
        } else {
            self.kill_requested = true;
        }
    }

    pub fn child_running(&mut self) -> io::Result<bool> {
        let process = self.process.as_raw_handle();
        // SAFETY: the process handle is owned and retained through this identity query.
        let pid = unsafe { GetProcessId(process) };
        let creation_time = process_creation_time(process)?;
        if pid == 0 || pid != self.direct_pid || creation_time != self.creation_time {
            self.ownership_lost.store(true, Ordering::Release);
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "retained child identity changed",
            ));
        }
        let mut exit_code = 0;
        bool_result(
            "GetExitCodeProcess",
            // SAFETY: exit_code is writable storage and process is retained.
            unsafe { GetExitCodeProcess(process, &mut exit_code) },
        )?;
        Ok(exit_code == STILL_ACTIVE)
    }

    pub async fn confirmed_empty(&self) -> bool {
        if self.ownership_lost() {
            return false;
        }
        let Some(job) = self.job.as_ref() else {
            return false;
        };
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let mut accounting = JobObjectBasicAccountingInformation::default();
            // SAFETY: job is retained and accounting is initialized writable repr(C) storage.
            let result = unsafe {
                QueryInformationJobObject(
                    job.as_raw_handle(),
                    JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                    (&mut accounting as *mut JobObjectBasicAccountingInformation).cast(),
                    size_of::<JobObjectBasicAccountingInformation>() as Dword,
                    std::ptr::null_mut(),
                )
            };
            if result != TRUE {
                return false;
            }
            if accounting.active_processes == 0 {
                self.empty_verified.store(true, Ordering::Release);
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Kill any remaining owned process, prove the Job Object is empty, then
    /// release the job authority. Failure keeps the guard armed for Drop.
    pub async fn finish(&mut self) -> io::Result<()> {
        if self.ownership_lost() {
            return Err(io::Error::other("owned Job Object identity was lost"));
        }
        if !self.confirmed_empty().await {
            self.kill_now();
            if !self.confirmed_empty().await {
                return Err(io::Error::other(
                    "owned Job Object did not reach a confirmed empty state",
                ));
            }
        }
        if !self.disarm() {
            return Err(io::Error::other("owned Job Object could not be disarmed"));
        }
        Ok(())
    }

    pub fn disarm(&mut self) -> bool {
        if !self.empty_verified.load(Ordering::Acquire) {
            return false;
        }
        // Closing an already-empty job is safe and releases the ownership
        // authority.  A non-empty job is never disarmed.
        self.job.take();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;

    #[tokio::test]
    async fn pre_resume_arm_failure_returns_owned_child_without_resuming_it() {
        fail_next_arm_before_resume();
        let mut command = Command::new("cmd");
        command
            .args(["/C", "exit", "0"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let result = spawn_owned_command_checked(command);
        let Err(SpawnOwnedError::Arm { mut child, error }) = result else {
            panic!("pre-resume fault must be returned as an arm error");
        };
        assert!(error.to_string().contains("pre-resume"));
        child
            .kill()
            .await
            .expect("the returned suspended child remains cleanup-owned");
        child
            .wait()
            .await
            .expect("cleanup must reap the returned child");
    }
}
