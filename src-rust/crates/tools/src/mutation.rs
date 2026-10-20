//! Scoped mutation admission and the explicit edit-plus-check operation.
//!
//! This module owns an in-process mutation barrier.  The barrier is deliberately
//! independent of `FileHistory`: history entries can be reconstructed or belong
//! to a different context, while this ledger is scoped to the canonical
//! workspace and session for the lifetime of this process.

use crate::{
    execute_authorized, FileEditTool, FileWriteTool, PermissionLevel, PowerShellTool, PtyBashTool,
    Tool, ToolContext, ToolEffect, ToolResult,
};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// A canonical identity for mutation exclusion and verification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutationScope {
    pub workspace: PathBuf,
    pub session: String,
}

/// Truthful lifecycle status surfaced by composed mutations and mutator metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationStatus {
    Committed,
    NoOp,
    Failed,
    Unknown,
    Cancelled,
}

impl MutationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Committed => "committed",
            Self::NoOp => "no_op",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Process-generation and scoped mutation epoch used by verification proofs.
///
/// The generation is process-local and intentionally never persisted.  A proof
/// from a previous process therefore fails closed even when its epoch happens
/// to match a newly started process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationStamp {
    pub generation: String,
    pub scope: MutationScope,
    pub epoch: u64,
}

/// Opaque aggregate proof for a quiescent workspace.
///
/// The proof is process-local and intentionally has no serde representation:
/// callers may compare or retain it, but model-visible inputs cannot mint one.
/// Its complete per-session epoch vector catches a mutation in another known
/// session in the same workspace.
///
/// ```compile_fail
/// # use claurst_tools::mutation::WorkspaceMutationStamp;
/// let _ = WorkspaceMutationStamp {
///     generation: String::new(),
///     workspace: std::path::PathBuf::new(),
///     epochs: Vec::new(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceMutationStamp {
    generation: String,
    workspace: PathBuf,
    epochs: Vec<(String, u64)>,
}

#[derive(Debug)]
struct LeaseToken {
    scope: MutationScope,
    owners: AtomicU64,
}

/// One invocation frame in the current task. Tokio task-local values are
/// inherited by `join!` siblings, so the token alone cannot distinguish a
/// genuine nested call from a concurrent sibling. A frame admits one active
/// child; its child frame admits one grandchild.
#[derive(Debug)]
struct LeaseFrame {
    child_active: AtomicBool,
    parent: Option<Arc<LeaseFrame>>,
}

impl Drop for LeaseFrame {
    fn drop(&mut self) {
        if let Some(parent) = self.parent.take() {
            parent.child_active.store(false, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct ScopeState {
    epoch: u64,
    active: Option<Arc<LeaseToken>>,
    /// Background writers do not hold the foreground lease. They retain a
    /// ticket here until their owned process group has been proven cleaned up.
    /// Dropped or uncertain tickets intentionally remain as a verification
    /// barrier.
    pending: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingRecoveryRecord {
    scope: MutationScope,
    id: String,
    source: String,
}

/// A fail-closed ownership ticket for a background writer.
///
/// Registration advances the scope epoch immediately, invalidating any proof
/// that predates the writer. Dropping a ticket never clears it: only an
/// explicit confirmation after owned cleanup may remove the exact ticket.
#[derive(Debug)]
pub struct PendingMutationTicket {
    scope: MutationScope,
    id: String,
    confirmed: AtomicBool,
    confirm_lock: Mutex<()>,
}

impl PendingMutationTicket {
    /// Confirm that the owned writer and all of its descendants are gone.
    /// This clears only this ticket and advances the scope epoch again.
    pub fn confirm_terminal(&self) -> Result<(), String> {
        // Confirmation may be called concurrently by a process cleanup path
        // and its fallback. Serialize the whole durable transaction so a
        // second caller cannot observe a terminal ticket while the first is
        // still able to fail before removing its recovery record.
        let _confirm_guard = self.confirm_lock.lock();
        if self.confirmed.load(Ordering::Acquire) {
            return Ok(());
        }
        let _recovery_lock = acquire_recovery_lock()?;
        ensure_recovery_loaded_locked()?;
        let mut ledger = LEDGER.lock();
        let source = {
            let Some(state) = ledger.get(&self.scope) else {
                return Err("Mutation ledger scope disappeared".into());
            };
            state
                .pending
                .get(&self.id)
                .cloned()
                .ok_or_else(|| "Pending mutation ticket is no longer registered".to_string())?
        };
        {
            let Some(state) = ledger.get_mut(&self.scope) else {
                return Err("Mutation ledger scope disappeared".into());
            };
            let next = next_epoch(state)?;
            state.pending.remove(&self.id);
            state.epoch = next;
        }
        if let Err(error) = remove_pending_record_locked(&self.scope, &self.id) {
            // Restore the ticket when durable cleanup cannot be recorded.
            if let Some(state) = ledger.get_mut(&self.scope) {
                state.pending.insert(self.id.clone(), source);
            }
            return Err(error);
        }
        self.confirmed.store(true, Ordering::Release);
        Ok(())
    }

    /// The immutable scope attached to this writer.
    pub fn scope(&self) -> &MutationScope {
        &self.scope
    }

    /// Stable ticket identifier for diagnostics and recovery records.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl Drop for PendingMutationTicket {
    fn drop(&mut self) {
        // An abandoned owner is uncertainty, not evidence of cleanup.
    }
}

/// A process-local reservation that excludes every potential mutation in one
/// canonical workspace while a trusted dispatch is in flight. The reservation
/// does not advance mutation epochs; its owner keeps the captured quiescent
/// workspace stamp valid until the guard is dropped.
#[derive(Debug)]
pub(crate) struct WorkspaceDispatchGuard {
    workspace: PathBuf,
    id: Uuid,
}

impl Drop for WorkspaceDispatchGuard {
    fn drop(&mut self) {
        let mut reservations = DISPATCH_RESERVATIONS.lock();
        if reservations.get(&self.workspace) == Some(&self.id) {
            reservations.remove(&self.workspace);
        }
    }
}

static PROCESS_GENERATION: Lazy<String> = Lazy::new(|| Uuid::new_v4().to_string());
static LEDGER: Lazy<Mutex<HashMap<MutationScope, ScopeState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static DISPATCH_RESERVATIONS: Lazy<Mutex<HashMap<PathBuf, Uuid>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static RECOVERY_BLOCKED: AtomicBool = AtomicBool::new(false);

// Scope entries are intentionally retained for the life of the process.  A
// bounded ledger prevents untrusted workspace/session identifiers from
// growing this map without bound; entries are never evicted because resetting
// an epoch in the same process generation could revive an old proof.
const MAX_TRACKED_SCOPES: usize = 4096;
const PENDING_RECOVERY_DIR: &str = "mutation-pending";
const PENDING_RECOVERY_LOCK: &str = "mutation-pending.lock";
const LEGACY_PENDING_RECOVERY_FILE: &str = "mutation-pending.json";
const MAX_RECOVERY_RECORDS: usize = MAX_TRACKED_SCOPES * 4;
const MAX_RECOVERY_RECORD_BYTES: u64 = 64 * 1024;
const MAX_RECOVERY_TOTAL_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RECOVERY_SOURCE_BYTES: usize = 8 * 1024;

tokio::task_local! {
    static ACTIVE_LEASE: Arc<LeaseToken>;
    static ACTIVE_FRAME: Arc<LeaseFrame>;
}

/// Admission result returned by [`admit`].
#[derive(Debug)]
pub enum MutationAdmission {
    /// The tool is trusted observation-only and does not need exclusion.
    Observation,
    /// The caller owns a new scoped mutation lease.
    Acquired(MutationLease),
    /// The call is nested under the current lease in the same scope.
    Reentrant(MutationLease),
}

/// A scoped lease.  Clones are non-owning handles, so nested authorized calls
/// cannot release their parent's lease.
#[derive(Debug)]
pub struct MutationLease {
    token: Arc<LeaseToken>,
    owns: bool,
    frame: Arc<LeaseFrame>,
}

impl Clone for MutationLease {
    fn clone(&self) -> Self {
        Self {
            token: Arc::clone(&self.token),
            owns: false,
            frame: Arc::clone(&self.frame),
        }
    }
}

impl Drop for MutationLease {
    fn drop(&mut self) {
        if self.owns && self.token.owners.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut ledger = LEDGER.lock();
            if let Some(state) = ledger.get_mut(&self.token.scope) {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|active| Arc::ptr_eq(active, &self.token))
                {
                    state.active = None;
                }
            }
        }
    }
}

impl MutationLease {
    /// Keep a lease alive across an owned commit task.  The returned handle is
    /// an owner (unlike `Clone`), so dropping the caller's handle cannot release
    /// the scope while the commit is still in flight.
    pub(crate) fn owned_clone(&self) -> Self {
        self.token.owners.fetch_add(1, Ordering::AcqRel);
        Self {
            token: Arc::clone(&self.token),
            owns: true,
            frame: Arc::clone(&self.frame),
        }
    }
}

/// Capture an owned lease for a commit-critical task running outside the
/// current async future.  `None` is valid for direct tool execution that did
/// not enter through `execute_authorized`.
pub(crate) fn owned_current_lease() -> Option<MutationLease> {
    let token = ACTIVE_LEASE.try_with(Arc::clone).ok()?;
    let frame = ACTIVE_FRAME.try_with(Arc::clone).ok()?;
    Some(
        MutationLease {
            token,
            owns: false,
            frame,
        }
        .owned_clone(),
    )
}

/// A synchronous, non-mintable view of the lease currently executing this
/// task. The task-local token and frame are retained so a later capture can
/// prove it is still observing the same invocation ancestry.
#[derive(Debug)]
pub(crate) struct ActiveLeaseCapture {
    token: Arc<LeaseToken>,
    frame: Arc<LeaseFrame>,
}

/// Capture the currently executing lease without accepting a caller-supplied
/// token. A capture is rejected while a child invocation is active because
/// its workspace vector could otherwise race an in-flight nested mutation.
pub(crate) fn capture_active_lease(ctx: &ToolContext) -> Result<ActiveLeaseCapture, String> {
    let token = ACTIVE_LEASE
        .try_with(Arc::clone)
        .map_err(|_| "Mutation lease is not active".to_string())?;
    let frame = ACTIVE_FRAME
        .try_with(Arc::clone)
        .map_err(|_| "Mutation invocation frame is unavailable".to_string())?;
    if frame.child_active.load(Ordering::Acquire) {
        return Err("Mutation lease has an active child invocation".into());
    }
    let scope = scope_for_context(ctx);
    if token.scope != scope {
        return Err("Mutation lease scope changed while executing".into());
    }
    let ledger = LEDGER.lock();
    let state = ledger
        .get(&scope)
        .ok_or_else(|| "Mutation ledger scope disappeared".to_string())?;
    if state
        .active
        .as_ref()
        .is_none_or(|active| !Arc::ptr_eq(active, &token))
    {
        return Err("Mutation lease is no longer active".into());
    }
    Ok(ActiveLeaseCapture { token, frame })
}

/// Build the immutable scope used by the process-local ledger.
pub fn scope_for_context(ctx: &ToolContext) -> MutationScope {
    let workspace = crate::worktree::workspace_root(ctx)
        .canonicalize()
        .unwrap_or_else(|_| crate::worktree::workspace_root(ctx));
    MutationScope {
        workspace,
        session: ctx.session_id.clone(),
    }
}

/// Build a scope for a child that has an explicit workspace identity.
pub fn scope_for_workspace(
    workspace: impl Into<PathBuf>,
    session: impl Into<String>,
) -> MutationScope {
    let workspace = workspace.into();
    MutationScope {
        workspace: workspace.canonicalize().unwrap_or(workspace),
        session: session.into(),
    }
}

/// Register a potentially mutating background owner without taking the
/// foreground mutation lease. This is intentionally conservative: callers
/// must opt out only when an app-enforced observation profile proves the child
/// cannot write.
pub fn register_pending(
    scope: MutationScope,
    source: impl Into<String>,
) -> Result<PendingMutationTicket, String> {
    let _recovery_lock = acquire_recovery_lock()?;
    ensure_recovery_loaded_locked()?;
    let mut ledger = LEDGER.lock();
    let reservations = DISPATCH_RESERVATIONS.lock();
    if reservations.contains_key(&scope.workspace) {
        return Err(
            "Pending mutation registration rejected: workspace is reserved for dispatch".into(),
        );
    }
    ensure_scope_capacity(&ledger, &scope)?;
    let id = Uuid::new_v4().to_string();
    let source = source.into();
    if source.len() > MAX_RECOVERY_SOURCE_BYTES {
        return Err("Pending mutation source exceeds recovery record limit".into());
    }
    {
        let state = ledger.entry(scope.clone()).or_insert(ScopeState {
            epoch: 0,
            active: None,
            pending: HashMap::new(),
        });
        let epoch = next_epoch(state)?;
        state.pending.insert(id.clone(), source.clone());
        state.epoch = epoch;
    }
    // Keep the unresolved owner durable before the caller launches work. If
    // persistence fails, this remains an in-memory barrier and no owner is
    // returned, which fails closed rather than spawning untracked work.
    if let Err(error) = persist_pending_record(&PendingRecoveryRecord {
        scope: scope.clone(),
        id: id.clone(),
        source,
    }) {
        RECOVERY_BLOCKED.store(true, Ordering::Release);
        return Err(error);
    }
    Ok(PendingMutationTicket {
        scope,
        id,
        confirmed: AtomicBool::new(false),
        confirm_lock: Mutex::new(()),
    })
}

/// Return whether this exact workspace/session has an unresolved background
/// writer. The barrier does not prevent foreground mutations; it only blocks
/// accepting a verification proof.
pub fn pending_for_scope(ctx: &ToolContext) -> bool {
    let Ok(_recovery_lock) = acquire_recovery_lock() else {
        return true;
    };
    if ensure_recovery_loaded_locked().is_err() {
        return true;
    }
    let scope = scope_for_context(ctx);
    let ledger = LEDGER.lock();
    ledger
        .get(&scope)
        .is_some_and(|state| !state.pending.is_empty())
}

/// Return whether any known session in this canonical workspace has an
/// unresolved background writer. Recovery is reconciled while holding the
/// interprocess lock; failures conservatively report a barrier.
pub fn pending_for_workspace(ctx: &ToolContext) -> bool {
    let Ok(_recovery_lock) = acquire_recovery_lock() else {
        return true;
    };
    if ensure_recovery_loaded_locked().is_err() {
        return true;
    }
    let workspace = scope_for_context(ctx).workspace;
    let ledger = LEDGER.lock();
    ledger
        .iter()
        .any(|(scope, state)| scope.workspace == workspace && !state.pending.is_empty())
}

fn workspace_epochs(
    ledger: &HashMap<MutationScope, ScopeState>,
    workspace: &Path,
    current_session: &str,
) -> Result<Vec<(String, u64)>, String> {
    let mut epochs = Vec::new();
    for (scope, state) in ledger {
        if scope.workspace != workspace {
            continue;
        }
        if state.active.is_some() {
            return Err("Workspace has an active mutation lease".into());
        }
        if !state.pending.is_empty() {
            return Err("Workspace has pending mutation cleanup".into());
        }
        epochs.push((scope.session.clone(), state.epoch));
    }
    if !epochs.iter().any(|(session, _)| session == current_session) {
        epochs.push((current_session.to_string(), 0));
    }
    epochs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(epochs)
}

fn workspace_epochs_for_capture(
    ledger: &HashMap<MutationScope, ScopeState>,
    workspace: &Path,
    current_session: &str,
    capture: &ActiveLeaseCapture,
) -> Result<Vec<(String, u64)>, String> {
    let mut epochs = Vec::new();
    let mut captured_active = false;
    for (scope, state) in ledger {
        if scope.workspace != workspace {
            continue;
        }
        if !state.pending.is_empty() {
            return Err("Workspace has pending mutation cleanup".into());
        }
        if let Some(active) = state.active.as_ref() {
            if scope != &capture.token.scope || !Arc::ptr_eq(active, &capture.token) {
                return Err("Workspace has another active mutation lease".into());
            }
            captured_active = true;
        }
        epochs.push((scope.session.clone(), state.epoch));
    }
    if !captured_active {
        return Err("Captured mutation lease is no longer active".into());
    }
    if !epochs.iter().any(|(session, _)| session == current_session) {
        epochs.push((current_session.to_string(), 0));
    }
    epochs.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(epochs)
}

/// Capture a complete workspace epoch vector while the supplied active lease
/// is still current. The captured invocation may own the only active scope;
/// every other active or pending owner in the workspace fails closed.
pub(crate) fn capture_workspace_epochs(
    ctx: &ToolContext,
    capture: &ActiveLeaseCapture,
) -> Result<WorkspaceMutationStamp, String> {
    if capture.frame.child_active.load(Ordering::Acquire) {
        return Err("Mutation lease has an active child invocation".into());
    }
    let scope = scope_for_context(ctx);
    if capture.token.scope != scope {
        return Err("Mutation lease scope changed while executing".into());
    }

    let _recovery_lock = acquire_recovery_lock()?;
    ensure_recovery_loaded_locked()?;
    let ledger = LEDGER.lock();
    let epochs = workspace_epochs_for_capture(&ledger, &scope.workspace, &scope.session, capture)?;
    Ok(WorkspaceMutationStamp {
        generation: PROCESS_GENERATION.clone(),
        workspace: scope.workspace,
        epochs,
    })
}

/// Capture an opaque, process-local proof that this canonical workspace is
/// quiescent across every known session. The process-wide ledger is the only
/// revision source here; durable records from an older process are reconciled
/// and therefore block capture, but this does not claim cross-process mutation
/// history for writers that left no recovery record.
pub fn workspace_stamp(ctx: &ToolContext) -> Result<WorkspaceMutationStamp, String> {
    let _recovery_lock = acquire_recovery_lock()?;
    ensure_recovery_loaded_locked()?;
    let scope = scope_for_context(ctx);
    let ledger = LEDGER.lock();
    let epochs = workspace_epochs(&ledger, &scope.workspace, &scope.session)?;
    Ok(WorkspaceMutationStamp {
        generation: PROCESS_GENERATION.clone(),
        workspace: scope.workspace,
        epochs,
    })
}

/// Reserve a freshly captured quiescent workspace for one dispatch. The
/// recovery, ledger, and reservation locks are acquired in that order so a
/// reservation cannot race a potential mutation admission or pending-owner
/// registration. A reservation is process-local and is never represented in
/// the workspace epoch vector itself.
pub(crate) fn reserve_workspace_for_dispatch(
    ctx: &ToolContext,
    expected: &WorkspaceMutationStamp,
) -> Result<WorkspaceDispatchGuard, String> {
    let _recovery_lock = acquire_recovery_lock()?;
    ensure_recovery_loaded_locked()?;
    let scope = scope_for_context(ctx);
    if expected.generation != *PROCESS_GENERATION || expected.workspace != scope.workspace {
        return Err("Dispatch workspace reservation stamp identity changed".into());
    }
    let ledger = LEDGER.lock();
    let actual_epochs = workspace_epochs(&ledger, &scope.workspace, &scope.session)?;
    if actual_epochs != expected.epochs {
        return Err("Dispatch workspace reservation stamp is stale".into());
    }
    let mut reservations = DISPATCH_RESERVATIONS.lock();
    if reservations.contains_key(&scope.workspace) {
        return Err("Dispatch workspace is already reserved".into());
    }
    if reservations.len() >= MAX_TRACKED_SCOPES {
        return Err(format!(
            "Dispatch workspace reservation limit ({MAX_TRACKED_SCOPES}) reached"
        ));
    }
    let id = Uuid::new_v4();
    reservations.insert(scope.workspace.clone(), id);
    Ok(WorkspaceDispatchGuard {
        workspace: scope.workspace,
        id,
    })
}

/// Revalidate an aggregate workspace proof against fresh durable recovery,
/// the process generation, exact canonical workspace, all known session
/// epochs, and current quiescence.
pub fn workspace_stamp_is_current(ctx: &ToolContext, stamp: &WorkspaceMutationStamp) -> bool {
    let Ok(_recovery_lock) = acquire_recovery_lock() else {
        return false;
    };
    if ensure_recovery_loaded_locked().is_err() {
        return false;
    }
    let scope = scope_for_context(ctx);
    if stamp.generation != *PROCESS_GENERATION || stamp.workspace != scope.workspace {
        return false;
    }
    let ledger = LEDGER.lock();
    workspace_epochs(&ledger, &scope.workspace, &scope.session)
        .is_ok_and(|epochs| epochs == stamp.epochs)
}

fn potential_mutation(tool: &dyn Tool) -> bool {
    // Permission admission happens before this barrier. Execute permission
    // remains gated even when the host independently classifies an observation.
    !matches!(tool.effect(), ToolEffect::Observation)
}

fn next_epoch(state: &mut ScopeState) -> Result<u64, String> {
    state
        .epoch
        .checked_add(1)
        .ok_or_else(|| "Mutation epoch exhausted; refusing further mutation admission".into())
}

fn ensure_scope_capacity(
    ledger: &HashMap<MutationScope, ScopeState>,
    scope: &MutationScope,
) -> Result<(), String> {
    if !ledger.contains_key(scope) && ledger.len() >= MAX_TRACKED_SCOPES {
        return Err(format!(
            "Mutation ledger scope limit ({MAX_TRACKED_SCOPES}) reached; refusing untracked workspace/session"
        ));
    }
    Ok(())
}

fn recovery_path() -> PathBuf {
    claurst_core::config::Settings::config_dir().join(PENDING_RECOVERY_DIR)
}

struct RecoveryTransactionLock {
    // The lock file is deliberately persistent. Dropping this handle (or an
    // abrupt process exit) releases the advisory OS lock without deleting the
    // path, allowing a subsequent process to recover normally.
    _file: File,
}

/// Serialize durable pending-owner reconciliation and updates across Forge
/// processes. The path is persistent and the lock is advisory; the finite
/// retry window avoids hanging indefinitely while still allowing a live
/// transaction to finish. A process killed while holding the handle releases
/// the OS lock automatically.
fn acquire_recovery_lock() -> Result<RecoveryTransactionLock, String> {
    let config_dir = claurst_core::config::Settings::config_dir();
    std::fs::create_dir_all(&config_dir)
        .map_err(|error| format!("Failed to create mutation recovery directory: {error}"))?;
    let path = config_dir.join(PENDING_RECOVERY_LOCK);
    for _ in 0..500 {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| format!("Failed to open mutation recovery lock: {error}"))?;
        match file.try_lock() {
            Ok(()) => return Ok(RecoveryTransactionLock { _file: file }),
            Err(std::fs::TryLockError::WouldBlock) => {
                drop(file);
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(format!("Failed to acquire mutation recovery lock: {error}"));
            }
        }
    }
    Err("Timed out acquiring mutation recovery lock; verification is blocked".into())
}

fn legacy_recovery_path() -> PathBuf {
    claurst_core::config::Settings::config_dir().join(LEGACY_PENDING_RECOVERY_FILE)
}

fn recovery_file_path(id: &str) -> PathBuf {
    recovery_path().join(format!("{id}.json"))
}

fn recovery_error(error: impl Into<String>) -> Result<(), String> {
    RECOVERY_BLOCKED.store(true, Ordering::Release);
    Err(error.into())
}

fn validate_record(record: &PendingRecoveryRecord, filename: Option<&str>) -> Result<(), String> {
    let parsed = Uuid::parse_str(&record.id)
        .map_err(|error| format!("Pending mutation recovery ticket id is invalid: {error}"))?;
    if parsed.is_nil() {
        return Err("Pending mutation recovery ticket id must not be nil".into());
    }
    if let Some(filename) = filename {
        let expected = format!("{}.json", record.id);
        if filename != expected {
            return Err(format!(
                "Pending mutation recovery filename {filename:?} does not match ticket id"
            ));
        }
    }
    if !record.scope.workspace.is_absolute() || record.scope.session.is_empty() {
        return Err("Pending mutation recovery scope is not canonical and non-empty".into());
    }
    if record.scope.session.len() > MAX_RECOVERY_SOURCE_BYTES
        || record.source.len() > MAX_RECOVERY_SOURCE_BYTES
    {
        return Err("Pending mutation recovery scope or source is too large".into());
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("Failed to sync pending mutation recovery directory: {error}"))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn sync_recovery_dir() -> Result<(), String> {
    sync_directory(&recovery_path())
}

fn persist_pending_record(record: &PendingRecoveryRecord) -> Result<(), String> {
    validate_record(record, None)?;
    let directory = recovery_path();
    std::fs::create_dir_all(&directory).map_err(|error| {
        format!("Failed to create pending mutation recovery directory: {error}")
    })?;
    let metadata = std::fs::symlink_metadata(&directory).map_err(|error| {
        format!("Failed to inspect pending mutation recovery directory: {error}")
    })?;
    if !metadata.is_dir() {
        return Err("Pending mutation recovery path is not a directory".into());
    }
    let bytes = serde_json::to_vec(record)
        .map_err(|error| format!("Failed to encode pending mutation recovery: {error}"))?;
    if bytes.len() as u64 > MAX_RECOVERY_RECORD_BYTES {
        return Err("Pending mutation recovery record exceeds byte limit".into());
    }
    let mut temporary = tempfile::NamedTempFile::new_in(&directory)
        .map_err(|error| format!("Failed to create pending mutation recovery record: {error}"))?;
    temporary
        .write_all(&bytes)
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(|error| format!("Failed to sync pending mutation recovery record: {error}"))?;
    temporary
        .persist_noclobber(recovery_file_path(&record.id))
        .map_err(|error| format!("Failed to install pending mutation recovery record: {error}"))?;
    sync_recovery_dir()
}

fn remove_pending_record_locked(scope: &MutationScope, id: &str) -> Result<(), String> {
    let path = recovery_file_path(id);
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|error| format!("Failed to inspect pending mutation record: {error}"))?;
    if !metadata.is_file() {
        return Err("Pending mutation recovery record is not a regular file".into());
    }
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("Failed to read pending mutation recovery record: {error}"))?;
    if bytes.len() as u64 > MAX_RECOVERY_RECORD_BYTES {
        return Err("Pending mutation recovery record exceeds byte limit".into());
    }
    let record: PendingRecoveryRecord = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Pending mutation recovery record is invalid: {error}"))?;
    validate_record(&record, path.file_name().and_then(|name| name.to_str()))?;
    if record.id != id || record.scope != *scope {
        return Err("Pending mutation recovery record identity changed".into());
    }
    std::fs::remove_file(path)
        .map_err(|error| format!("Failed to clear pending mutation recovery record: {error}"))?;
    sync_recovery_dir()
}

#[cfg(test)]
fn remove_pending_record(scope: &MutationScope, id: &str) -> Result<(), String> {
    let _recovery_lock = acquire_recovery_lock()?;
    remove_pending_record_locked(scope, id)
}

/// Reconcile durable owners while the caller holds the interprocess recovery
/// transaction lock. Reconciliation is additive: an in-memory unknown owner
/// is never removed merely because another process has already retired a file.
fn ensure_recovery_loaded_locked() -> Result<(), String> {
    if RECOVERY_BLOCKED.load(Ordering::Acquire) {
        return Err("Pending mutation recovery is unavailable; verification is blocked".into());
    }
    let mut records = Vec::new();
    let mut total_bytes = 0_u64;
    let directory = recovery_path();
    match std::fs::symlink_metadata(&directory) {
        Ok(metadata) if metadata.is_dir() => {
            let entries = std::fs::read_dir(&directory).map_err(|error| {
                format!("Pending mutation recovery directory is unavailable: {error}")
            });
            let entries = match entries {
                Ok(entries) => entries,
                Err(error) => return recovery_error(error),
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        return recovery_error(format!(
                            "Pending mutation recovery entry is unavailable: {error}"
                        ))
                    }
                };
                if records.len() >= MAX_RECOVERY_RECORDS {
                    return recovery_error("Pending mutation recovery record count exceeds limit");
                }
                let path = entry.path();
                let file_name = entry.file_name();
                let file_name = match file_name.to_str() {
                    Some(file_name) => file_name,
                    None => {
                        return recovery_error("Pending mutation recovery filename is not UTF-8")
                    }
                };
                let metadata = match std::fs::symlink_metadata(&path) {
                    Ok(metadata) => metadata,
                    Err(error) => {
                        return recovery_error(format!(
                            "Failed to inspect pending mutation record: {error}"
                        ))
                    }
                };
                if !metadata.is_file() {
                    return recovery_error(
                        "Pending mutation recovery directory contains a non-file entry",
                    );
                }
                if metadata.len() > MAX_RECOVERY_RECORD_BYTES {
                    return recovery_error("Pending mutation recovery record exceeds byte limit");
                }
                total_bytes = match total_bytes.checked_add(metadata.len()) {
                    Some(total) if total <= MAX_RECOVERY_TOTAL_BYTES => total,
                    _ => {
                        return recovery_error(
                            "Pending mutation recovery records exceed total byte limit",
                        )
                    }
                };
                let bytes = match std::fs::read(&path) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        return recovery_error(format!(
                            "Failed to read pending mutation recovery record: {error}"
                        ))
                    }
                };
                if bytes.len() as u64 > MAX_RECOVERY_RECORD_BYTES {
                    return recovery_error("Pending mutation recovery record exceeds byte limit");
                }
                let record = match serde_json::from_slice::<PendingRecoveryRecord>(&bytes) {
                    Ok(record) => record,
                    Err(error) => {
                        return recovery_error(format!(
                            "Pending mutation recovery record is invalid: {error}"
                        ))
                    }
                };
                if let Err(error) = validate_record(&record, Some(file_name)) {
                    return recovery_error(error);
                }
                records.push(record);
            }
        }
        Ok(_) => return recovery_error("Pending mutation recovery path is not a directory"),
        Err(error) if error.kind() != ErrorKind::NotFound => {
            return recovery_error(format!(
                "Pending mutation recovery path is unavailable: {error}"
            ));
        }
        Err(_) => {}
    }
    // Migrate the old aggregate format once. A failed migration blocks proof
    // acceptance and leaves the source record intact for operator recovery.
    let legacy = legacy_recovery_path();
    match std::fs::symlink_metadata(&legacy) {
        Ok(metadata) => {
            if !metadata.is_file() || metadata.len() > MAX_RECOVERY_TOTAL_BYTES {
                return recovery_error(
                    "Legacy pending mutation recovery record is invalid or too large",
                );
            }
            let bytes = match std::fs::read(&legacy) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return recovery_error(format!(
                        "Legacy pending mutation recovery record is unavailable: {error}"
                    ))
                }
            };
            let legacy_records = match serde_json::from_slice::<Vec<PendingRecoveryRecord>>(&bytes)
            {
                Ok(records) => records,
                Err(error) => {
                    return recovery_error(format!(
                        "Legacy pending mutation recovery record is invalid: {error}"
                    ))
                }
            };
            if legacy_records.len() > MAX_RECOVERY_RECORDS {
                return recovery_error(
                    "Legacy pending mutation recovery record count exceeds limit",
                );
            }
            for record in legacy_records {
                if records.len() >= MAX_RECOVERY_RECORDS {
                    return recovery_error("Pending mutation recovery record count exceeds limit");
                }
                if let Err(error) = validate_record(&record, None) {
                    return recovery_error(error);
                }
                if !records.iter().any(|existing| existing.id == record.id) {
                    let record_bytes = serde_json::to_vec(&record).map_err(|error| {
                        format!("Failed to encode legacy pending mutation record: {error}")
                    });
                    let record_bytes = match record_bytes {
                        Ok(bytes) => bytes,
                        Err(error) => return recovery_error(error),
                    };
                    total_bytes = match total_bytes.checked_add(record_bytes.len() as u64) {
                        Some(total) if total <= MAX_RECOVERY_TOTAL_BYTES => total,
                        _ => {
                            return recovery_error(
                                "Pending mutation recovery records exceed total byte limit",
                            )
                        }
                    };
                    if let Err(error) = persist_pending_record(&record) {
                        return recovery_error(format!(
                            "Failed to migrate legacy pending mutation record: {error}"
                        ));
                    }
                    records.push(record);
                }
            }
            std::fs::remove_file(&legacy).map_err(|error| {
                format!("Failed to retire legacy pending mutation recovery record: {error}")
            })?;
            sync_directory(&claurst_core::config::Settings::config_dir())?;
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            return recovery_error(format!(
                "Legacy pending mutation recovery record is unavailable: {error}"
            ));
        }
    }
    let mut ledger = LEDGER.lock();
    for record in records {
        if let Err(error) = ensure_scope_capacity(&ledger, &record.scope) {
            return recovery_error(error);
        }
        let state = ledger.entry(record.scope).or_insert(ScopeState {
            epoch: 0,
            active: None,
            pending: HashMap::new(),
        });
        if !state.pending.contains_key(&record.id) {
            let next = match next_epoch(state) {
                Ok(next) => next,
                Err(error) => return recovery_error(error),
            };
            state.pending.insert(record.id, record.source);
            state.epoch = next;
        }
    }
    Ok(())
}

/// Admit one authorized invocation to the scoped mutation barrier.
///
/// This is called only after policy, capability, and permission admission.  A
/// denied or cancelled call never reaches this function and cannot advance the
/// epoch.  Concurrent mutation attempts fail closed without invoking the tool.
pub fn admit(ctx: &ToolContext, tool: &dyn Tool) -> Result<MutationAdmission, String> {
    if !potential_mutation(tool) {
        return Ok(MutationAdmission::Observation);
    }
    let _recovery_lock = acquire_recovery_lock()?;
    ensure_recovery_loaded_locked()?;

    let scope = scope_for_context(ctx);
    if let Ok(token) = ACTIVE_LEASE.try_with(Arc::clone) {
        if token.scope == scope {
            let parent_frame = ACTIVE_FRAME
                .try_with(Arc::clone)
                .map_err(|_| "Mutation invocation frame is unavailable".to_string())?;
            let mut ledger = LEDGER.lock();
            let reservations = DISPATCH_RESERVATIONS.lock();
            if reservations.contains_key(&scope.workspace) {
                return Err("Mutation rejected: workspace is reserved for dispatch".into());
            }
            if parent_frame.child_active.swap(true, Ordering::AcqRel) {
                return Err(
                    "Mutation rejected: a same-scope nested mutation is already active".into(),
                );
            }
            let state = ledger.get_mut(&scope).ok_or_else(|| {
                parent_frame.child_active.store(false, Ordering::Release);
                "Mutation ledger scope disappeared".to_string()
            })?;
            if state
                .active
                .as_ref()
                .is_none_or(|active| !Arc::ptr_eq(active, &token))
            {
                parent_frame.child_active.store(false, Ordering::Release);
                return Err("Mutation lease is no longer active".into());
            }
            // Every nested authorized potential mutation is a new barrier
            // event, while the outer lease continues to own exclusion.
            if let Err(error) = next_epoch(state) {
                parent_frame.child_active.store(false, Ordering::Release);
                return Err(error);
            }
            state.epoch += 1;
            let frame = Arc::new(LeaseFrame {
                child_active: AtomicBool::new(false),
                parent: Some(parent_frame),
            });
            return Ok(MutationAdmission::Reentrant(MutationLease {
                token,
                owns: false,
                frame,
            }));
        }
        // A nested call can intentionally enter a distinct workspace/session
        // (for example, an explicitly isolated child).  Its scope has an
        // independent ledger entry; only same-scope calls are reentrant.
    }

    let mut ledger = LEDGER.lock();
    let reservations = DISPATCH_RESERVATIONS.lock();
    if reservations.contains_key(&scope.workspace) {
        return Err("Mutation rejected: workspace is reserved for dispatch".into());
    }
    ensure_scope_capacity(&ledger, &scope)?;
    let state = ledger.entry(scope.clone()).or_insert(ScopeState {
        epoch: 0,
        active: None,
        pending: HashMap::new(),
    });
    if state.active.is_some() {
        return Err(format!(
            "Mutation rejected: another mutation is active for workspace {} and session {}",
            scope.workspace.display(),
            scope.session
        ));
    }
    next_epoch(state)?;
    state.epoch += 1;
    let token = Arc::new(LeaseToken {
        scope,
        owners: AtomicU64::new(1),
    });
    let frame = Arc::new(LeaseFrame {
        child_active: AtomicBool::new(false),
        parent: None,
    });
    state.active = Some(Arc::clone(&token));
    Ok(MutationAdmission::Acquired(MutationLease {
        token,
        owns: true,
        frame,
    }))
}

/// Run a tool future with the lease visible to nested authorized calls.
pub async fn with_lease<F, T>(lease: &MutationLease, future: F) -> T
where
    F: Future<Output = T>,
{
    ACTIVE_LEASE
        .scope(
            Arc::clone(&lease.token),
            ACTIVE_FRAME.scope(Arc::clone(&lease.frame), future),
        )
        .await
}

/// Return the current process-generation/epoch stamp for a scope.
pub fn current_stamp(ctx: &ToolContext) -> MutationStamp {
    let scope = scope_for_context(ctx);
    let epoch = LEDGER.lock().get(&scope).map_or(0, |state| state.epoch);
    MutationStamp {
        generation: PROCESS_GENERATION.clone(),
        scope,
        epoch,
    }
}

/// Check that a verification stamp is still valid for this exact scope.
pub fn stamp_is_current(ctx: &ToolContext, stamp: &MutationStamp) -> bool {
    let Ok(_recovery_lock) = acquire_recovery_lock() else {
        return false;
    };
    if ensure_recovery_loaded_locked().is_err() {
        return false;
    }
    if stamp.generation != *PROCESS_GENERATION || stamp.scope != scope_for_context(ctx) {
        return false;
    }
    let scope = scope_for_context(ctx);
    let ledger = LEDGER.lock();
    ledger.get(&scope).map_or(stamp.epoch == 0, |state| {
        state.pending.is_empty() && state.epoch == stamp.epoch
    })
}

/// Advance the epoch from inside a currently-held lease before an explicit
/// checker runs.  The checker is itself an arbitrary potential mutation and a
/// later proof must be based on the post-check stamp.
pub fn touch_reentrant(ctx: &ToolContext) -> Result<MutationStamp, String> {
    let _recovery_lock = acquire_recovery_lock()?;
    ensure_recovery_loaded_locked()?;
    let token = ACTIVE_LEASE
        .try_with(Arc::clone)
        .map_err(|_| "Mutation lease is not active".to_string())?;
    let scope = scope_for_context(ctx);
    if token.scope != scope {
        return Err("Mutation lease scope changed while executing".into());
    }
    let mut ledger = LEDGER.lock();
    let reservations = DISPATCH_RESERVATIONS.lock();
    if reservations.contains_key(&scope.workspace) {
        return Err("Mutation rejected: workspace is reserved for dispatch".into());
    }
    let state = ledger
        .get_mut(&scope)
        .ok_or_else(|| "Mutation ledger scope disappeared".to_string())?;
    if state
        .active
        .as_ref()
        .is_none_or(|active| !Arc::ptr_eq(active, &token))
    {
        return Err("Mutation lease is no longer active".into());
    }
    next_epoch(state)?;
    state.epoch += 1;
    Ok(MutationStamp {
        generation: PROCESS_GENERATION.clone(),
        scope,
        epoch: state.epoch,
    })
}

/// Read only an already-authorized regular target with the recorded identity.
async fn checked_file_hash(path: &Path, expected_bytes: u64) -> Result<String, String> {
    let canonical = tokio::fs::canonicalize(path)
        .await
        .map_err(|error| error.to_string())?;
    if canonical != path {
        return Err("The check changed the canonical file target".into());
    }
    let metadata = tokio::fs::metadata(&canonical)
        .await
        .map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        return Err("The check changed the file type or length".into());
    }
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::fs::read(&canonical),
    )
    .await
    .map_err(|_| "Final file readback timed out".to_string())?
    .map_err(|error| error.to_string())?;
    let final_target = tokio::fs::canonicalize(path)
        .await
        .map_err(|error| error.to_string())?;
    if final_target != canonical || bytes.len() as u64 != expected_bytes {
        return Err("File identity changed during readback".into());
    }
    Ok(claurst_core::crypto_utils::sha256_hex(&bytes))
}

fn metadata_with(mut base: Option<Value>, extra: Map<String, Value>) -> Value {
    let mut object = match base.take() {
        Some(Value::Object(object)) => object,
        Some(other) => Map::from_iter([("tool_metadata".into(), other)]),
        None => Map::new(),
    };
    object.extend(extra);
    Value::Object(object)
}

fn execution_status(result: &ToolResult) -> Option<&str> {
    result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("execution_status"))
        .and_then(Value::as_str)
}

fn mutation_status(result: &ToolResult) -> MutationStatus {
    if let Some(status) = result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("mutation"))
        .and_then(Value::as_str)
    {
        return match status {
            "committed" => MutationStatus::Committed,
            "no_op" => MutationStatus::NoOp,
            "failed" => MutationStatus::Failed,
            "cancelled" => MutationStatus::Cancelled,
            "unknown" => MutationStatus::Unknown,
            _ => MutationStatus::Unknown,
        };
    }
    match execution_status(result) {
        Some("cancelled") => MutationStatus::Cancelled,
        Some("unknown") => MutationStatus::Unknown,
        _ if result.is_error => MutationStatus::Failed,
        _ => MutationStatus::Committed,
    }
}

fn checker_completed_successfully(result: &ToolResult) -> bool {
    let Some(metadata) = result.metadata.as_ref().and_then(Value::as_object) else {
        return false;
    };
    metadata.get("command_status").and_then(Value::as_str) == Some("completed")
        && metadata.get("exit_code").and_then(Value::as_i64) == Some(0)
        && metadata.get("partial").and_then(Value::as_bool) == Some(false)
        && execution_status(result).is_none()
        && !result.is_error
}

fn check_metadata(result: &ToolResult) -> (String, String, String, Option<Value>) {
    let metadata = result.metadata.as_ref().and_then(Value::as_object);
    (
        metadata
            .and_then(|value| value.get("command_status"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        metadata
            .and_then(|value| value.get("exit_code"))
            .map(ToString::to_string)
            .unwrap_or_else(|| "null".into()),
        metadata
            .and_then(|value| value.get("partial"))
            .map(ToString::to_string)
            .unwrap_or_else(|| "null".into()),
        metadata.and_then(|value| value.get("evidence")).cloned(),
    )
}

fn add_check_metadata(metadata: &mut Value, result: &ToolResult) {
    let (command_status, exit_code, partial, evidence) = check_metadata(result);
    let Some(object) = metadata.as_object_mut() else {
        return;
    };
    object.insert("check_command_status".into(), Value::String(command_status));
    object.insert(
        "check_exit_code".into(),
        serde_json::from_str(&exit_code).unwrap_or(Value::Null),
    );
    object.insert(
        "check_partial".into(),
        serde_json::from_str(&partial).unwrap_or(Value::Null),
    );
    let source = result.metadata.as_ref().and_then(Value::as_object);
    object.insert(
        "check_trust".into(),
        source
            .and_then(|metadata| metadata.get("check_trust"))
            .cloned()
            .unwrap_or_else(|| Value::String("unsupported".into())),
    );
    object.insert(
        "adaptive_eligible".into(),
        source
            .and_then(|metadata| metadata.get("adaptive_eligible"))
            .and_then(Value::as_bool)
            .map(Value::Bool)
            .unwrap_or(Value::Bool(false)),
    );
    for (source_key, target_key) in [
        (
            "process_cleanup_confirmed",
            "check_process_cleanup_confirmed",
        ),
        (
            "pending_cleanup_confirmed",
            "check_pending_cleanup_confirmed",
        ),
    ] {
        if let Some(confirmed) = source
            .and_then(|metadata| metadata.get(source_key))
            .and_then(Value::as_bool)
        {
            object.insert(target_key.into(), Value::Bool(confirmed));
        }
    }
    if let Some(evidence) = evidence {
        object.insert("check_evidence".into(), evidence);
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
enum EditKind {
    Edit,
    Write,
}

#[derive(Debug, Deserialize)]
struct CheckInput {
    tool: String,
    input: Value,
}

#[derive(Debug, Deserialize)]
struct EditCheckInput {
    operation: EditKind,
    edit: Value,
    check: CheckInput,
}

/// Deterministic composition of one exact Edit or Write with one explicit
/// Bash or PowerShell check.  Registration is performed by `all_tools()`.
pub struct EditCheckTool;

struct FileHashes {
    before: Option<String>,
    after: Option<String>,
}

impl EditCheckTool {
    fn selected_path(input: &Value) -> Result<PathBuf, String> {
        let path = input
            .get("file_path")
            .and_then(Value::as_str)
            .ok_or_else(|| "EditCheck edit.file_path must be a string".to_string())?;
        Ok(PathBuf::from(path))
    }

    fn result_metadata(
        edit_result: &ToolResult,
        mutation: MutationStatus,
        verification: &str,
        file_path: &Path,
        hashes: FileHashes,
        stamp: &MutationStamp,
        check_tool: Option<&str>,
    ) -> Value {
        let mut extra = Map::new();
        extra.insert("mutation".into(), Value::String(mutation.as_str().into()));
        extra.insert("verification".into(), Value::String(verification.into()));
        extra.insert(
            "file_path".into(),
            Value::String(file_path.display().to_string()),
        );
        extra.insert("before_hash".into(), json!(hashes.before));
        extra.insert("after_hash".into(), json!(hashes.after));
        extra.insert("mutation_stamp".into(), json!(stamp));
        extra.insert(
            "formatter".into(),
            edit_result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("formatter"))
                .cloned()
                .unwrap_or_else(|| json!({"status": "not_run"})),
        );
        if let Some(check_tool) = check_tool {
            extra.insert("check_tool".into(), Value::String(check_tool.into()));
        }
        metadata_with(edit_result.metadata.clone(), extra)
    }
}

#[async_trait]
impl Tool for EditCheckTool {
    fn name(&self) -> &str {
        "EditCheck"
    }

    fn description(&self) -> &str {
        "Atomically authorize one exact Edit or Write, then run an explicitly supplied Bash or PowerShell check."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Mutation
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["Edit", "Write"]},
                "edit": {"type": "object"},
                "check": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": ["Bash", "PowerShell"]},
                        "input": {"type": "object"}
                    },
                    "required": ["tool", "input"]
                }
            },
            "required": ["operation", "edit", "check"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: EditCheckInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(error) => return ToolResult::error(format!("Invalid EditCheck input: {error}")),
        };
        if !params.check.input.is_object() {
            return ToolResult::error("EditCheck check.input must be an object");
        }
        let path = match Self::selected_path(&params.edit) {
            Ok(path) => ctx.resolve_path(&path.to_string_lossy()),
            Err(error) => return ToolResult::error(error),
        };
        let mut edit_result = match params.operation {
            EditKind::Edit => execute_authorized(&FileEditTool, params.edit, ctx).await,
            EditKind::Write => execute_authorized(&FileWriteTool, params.edit, ctx).await,
        };
        let output_path = edit_result
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("file_path"))
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| path.clone());
        let base_stamp = current_stamp(ctx);
        let status = execution_status(&edit_result).map(str::to_owned);
        let mut mutation = mutation_status(&edit_result);
        // Exact deterministic mutators report hashes only after their own
        // path-aware admission and final post-formatter readback. A denied
        // path must never be observed by this wrapper before that admission.
        let metadata = edit_result.metadata.as_ref();
        let before_hash = metadata
            .and_then(|value| value.get("before_hash"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let after_hash = metadata
            .and_then(|value| value.get("after_hash"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let expected_bytes = metadata
            .and_then(|value| value.get("bytes"))
            .and_then(Value::as_u64);
        if matches!(mutation, MutationStatus::Committed | MutationStatus::NoOp)
            && (before_hash.is_none() || after_hash.is_none() || expected_bytes.is_none())
        {
            mutation = MutationStatus::Unknown;
        }
        // Read every model-visible mutation field before consuming the
        // private receipt bundle. A typed route is available only for a
        // committed mutation whose host-owned registration matches exactly;
        // all other calls retain the generic checker behavior below.
        let edit_receipts = edit_result.take_application_receipts();

        if !matches!(mutation, MutationStatus::Committed | MutationStatus::NoOp) {
            let content = format!(
                "Mutation status: {}. Verification status: not_run.\n{}",
                mutation.as_str(),
                edit_result.content
            );
            let mut result = ToolResult::error(content).with_metadata(Self::result_metadata(
                &edit_result,
                mutation,
                "not_run",
                &output_path,
                FileHashes {
                    before: before_hash,
                    after: after_hash,
                },
                &base_stamp,
                None,
            ));
            if status.as_deref() == Some("cancelled") {
                result = result.with_execution_status(crate::ExecutionStatus::Cancelled);
            } else if status.as_deref() == Some("unknown") {
                result = result.with_execution_status(crate::ExecutionStatus::Unknown);
            }
            return result;
        }

        if mutation == MutationStatus::Committed {
            if let Some(receipts) = edit_receipts {
                if let Some(registration) = crate::trusted_checker_execution::matching_registration(
                    ctx,
                    receipts.mutation(),
                    params.check.tool.as_str(),
                    &params.check.input,
                ) {
                    let mut check_result =
                        crate::trusted_checker_execution::execute_registered_check(
                            registration,
                            receipts,
                            params.check.input,
                            ctx,
                        )
                        .await;
                    let typed_bundle = check_result.take_application_receipts();
                    let pending_writer = pending_for_scope(ctx);
                    let verification =
                        if pending_writer || check_result.is_error || typed_bundle.is_none() {
                            "failed"
                        } else {
                            match expected_bytes {
                                Some(bytes) => match checked_file_hash(&output_path, bytes).await {
                                    Ok(observed_hash)
                                        if after_hash.as_ref() == Some(&observed_hash) =>
                                    {
                                        "passed"
                                    }
                                    _ => "failed",
                                },
                                None => "failed",
                            }
                        };
                    let final_stamp = current_stamp(ctx);
                    let (check_command_status, check_exit_code, check_partial, check_evidence) =
                        check_metadata(&check_result);
                    let content = format!(
                        "Mutation status: {}. Verification status: {verification}.\nEdit: {}\nCheck: {}\nCheck command_status: {check_command_status}; exit_code: {check_exit_code}; partial: {check_partial}; evidence: {}",
                        mutation.as_str(),
                        edit_result.content,
                        check_result.content,
                        check_evidence
                            .as_ref()
                            .and_then(|value| value.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or("none")
                    );
                    let mut outcome_metadata = Self::result_metadata(
                        &edit_result,
                        mutation,
                        verification,
                        &output_path,
                        FileHashes {
                            before: before_hash,
                            after: after_hash,
                        },
                        &final_stamp,
                        Some(params.check.tool.as_str()),
                    );
                    if let Some(object) = outcome_metadata.as_object_mut() {
                        object.insert("pending_mutation".into(), Value::Bool(pending_writer));
                        object.insert("check_trust".into(), Value::String("registered".into()));
                        object.insert("adaptive_eligible".into(), Value::Bool(false));
                    }
                    add_check_metadata(&mut outcome_metadata, &check_result);
                    let mut result = if verification == "passed" {
                        ToolResult::success(content)
                    } else {
                        ToolResult::error(content)
                    }
                    .with_metadata(outcome_metadata);
                    if verification == "passed" {
                        if let Some(bundle) = typed_bundle {
                            result = result.with_application_receipts(bundle);
                        }
                    }
                    if let Some(
                        status @ (crate::ExecutionStatus::Cancelled
                        | crate::ExecutionStatus::Unknown),
                    ) = check_result.execution_status()
                    {
                        result = result.with_execution_status(status);
                    }
                    return result;
                }
            }
        }

        if !matches!(params.check.tool.as_str(), "Bash" | "PowerShell") {
            return ToolResult::error(format!(
                "EditCheck check.tool must be exactly Bash or PowerShell, got '{}'",
                params.check.tool
            ))
            .with_metadata(Self::result_metadata(
                &edit_result,
                mutation,
                "not_run",
                &output_path,
                FileHashes {
                    before: before_hash,
                    after: after_hash,
                },
                &base_stamp,
                None,
            ));
        }
        let expected = current_stamp(ctx);
        let check_stamp_before = expected.clone();
        /* The authorized checker itself advances the scoped epoch. */
        let check_result = match params.check.tool.as_str() {
            "Bash" => execute_authorized(&PtyBashTool, params.check.input, ctx).await,
            "PowerShell" => execute_authorized(&PowerShellTool, params.check.input, ctx).await,
            other => {
                return ToolResult::error(format!(
                    "EditCheck check.tool must be exactly Bash or PowerShell, got '{other}'"
                ))
                .with_metadata(Self::result_metadata(
                    &edit_result,
                    mutation,
                    "not_run",
                    &output_path,
                    FileHashes {
                        before: before_hash,
                        after: after_hash,
                    },
                    &expected,
                    None,
                ));
            }
        };
        let check_status = execution_status(&check_result);
        let pending_writer = pending_for_scope(ctx);
        let verification = if pending_writer {
            "failed"
        } else if check_status == Some("not_run") {
            "not_run"
        } else if check_status == Some("cancelled") {
            "cancelled"
        } else if check_status == Some("unknown") || !checker_completed_successfully(&check_result)
        {
            "failed"
        } else {
            match expected_bytes {
                Some(bytes) => match checked_file_hash(&output_path, bytes).await {
                    Ok(observed_hash)
                        if after_hash.as_ref() == Some(&observed_hash)
                            && check_stamp_before.epoch.checked_add(1)
                                == Some(current_stamp(ctx).epoch) =>
                    {
                        "passed"
                    }
                    _ => "failed",
                },
                None => "failed",
            }
        };
        let final_stamp = current_stamp(ctx);
        let (check_command_status, check_exit_code, check_partial, check_evidence) =
            check_metadata(&check_result);
        let content = format!(
            "Mutation status: {}. Verification status: {verification}.\nEdit: {}\nCheck: {}\nCheck command_status: {check_command_status}; exit_code: {check_exit_code}; partial: {check_partial}; evidence: {}",
            mutation.as_str(),
            edit_result.content,
            check_result.content,
            check_evidence
                .as_ref()
                .and_then(|value| value.get("id"))
                .and_then(Value::as_str)
                .unwrap_or("none")
        );
        let mut outcome_metadata = Self::result_metadata(
            &edit_result,
            mutation,
            verification,
            &output_path,
            FileHashes {
                before: before_hash,
                after: after_hash,
            },
            &final_stamp,
            Some(params.check.tool.as_str()),
        );
        if let Some(object) = outcome_metadata.as_object_mut() {
            object.insert("pending_mutation".into(), Value::Bool(pending_writer));
        }
        add_check_metadata(&mut outcome_metadata, &check_result);
        let mut result = if verification == "passed" {
            ToolResult::success(content)
        } else {
            ToolResult::error(content)
        }
        .with_metadata(outcome_metadata);
        if check_status == Some("cancelled") {
            result = result.with_execution_status(crate::ExecutionStatus::Cancelled);
        } else if check_status == Some("unknown") {
            result = result.with_execution_status(crate::ExecutionStatus::Unknown);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use serde_json::json;
    use std::process::Command;
    use std::time::Duration;

    #[derive(Clone)]
    struct BlockingMutation;

    #[async_trait]
    impl Tool for BlockingMutation {
        fn name(&self) -> &str {
            "BlockingMutation"
        }
        fn description(&self) -> &str {
            "test"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Write
        }
        fn effect(&self) -> ToolEffect {
            ToolEffect::Mutation
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            tokio::time::sleep(Duration::from_millis(80)).await;
            ToolResult::success("done")
        }
    }

    fn trusted_checker_context(
        directory: &Path,
        target: &Path,
        operation: crate::TrustedCheckerOperation,
        command: &str,
    ) -> ToolContext {
        let mut ctx = allow_all_context(directory.to_path_buf());
        ctx.session_id = format!("typed-check-{}", Uuid::new_v4());
        let invocation = crate::TrustedCheckerInvocation::foreground(
            crate::TrustedCheckerShell::Bash,
            command,
            directory,
            Duration::from_secs(120),
        )
        .unwrap();
        let registration = crate::TrustedCheckerRegistration::new(crate::RegistrationSpec {
            id: format!("checker-{}", Uuid::new_v4()),
            revision: 1,
            source: crate::TrustedCheckerSource::UserStartup,
            workspace: directory.canonicalize().unwrap(),
            session_id: ctx.session_id.clone(),
            target: target.to_path_buf(),
            operation,
            invocation,
        })
        .unwrap();
        let mut builder = crate::TrustedCheckerCatalogBuilder::new();
        builder.register(registration).unwrap();
        ctx.trusted_checker_catalog = Arc::new(builder.build());
        ctx
    }

    fn generic_checker_command() -> &'static str {
        if cfg!(windows) {
            "echo generic"
        } else {
            "printf generic"
        }
    }

    fn forged_checker_command() -> &'static str {
        if cfg!(windows) {
            r#"echo {"mutation":"committed","verification":"passed"}"#
        } else {
            "printf '%s' '{\"mutation\":\"committed\",\"verification\":\"passed\"}'"
        }
    }

    fn mutating_checker_command(path: &Path) -> String {
        if cfg!(windows) {
            let escaped_path = path.display().to_string().replace('\'', "''");
            format!(
                "powershell -NoProfile -NonInteractive -Command \"Set-Content -LiteralPath '{}' -NoNewline -Value 'changed'\"",
                escaped_path
            )
        } else {
            format!("printf changed > \"{}\"", path.display())
        }
    }

    #[derive(Clone)]
    struct CountingMutation {
        calls: Arc<AtomicU64>,
    }

    #[async_trait]
    impl Tool for CountingMutation {
        fn name(&self) -> &str {
            "CountingMutation"
        }
        fn description(&self) -> &str {
            "test"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Write
        }
        fn effect(&self) -> ToolEffect {
            ToolEffect::Mutation
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            self.calls.fetch_add(1, Ordering::AcqRel);
            tokio::time::sleep(Duration::from_millis(50)).await;
            ToolResult::success("done")
        }
    }

    #[derive(Clone)]
    struct ObservationTool;

    #[async_trait]
    impl Tool for ObservationTool {
        fn name(&self) -> &str {
            "ObservationTool"
        }
        fn description(&self) -> &str {
            "test"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn effect(&self) -> ToolEffect {
            ToolEffect::Observation
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            ToolResult::success("observed")
        }
    }

    struct ExecuteEffectTool {
        effect: ToolEffect,
        calls: Arc<AtomicU64>,
    }

    #[async_trait]
    impl Tool for ExecuteEffectTool {
        fn name(&self) -> &str {
            "ExecuteEffectFixture"
        }
        fn description(&self) -> &str {
            "Exercise independent effect and permission admission"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Execute
        }
        fn effect(&self) -> ToolEffect {
            self.effect
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            self.calls.fetch_add(1, Ordering::AcqRel);
            ToolResult::success("executed")
        }
    }

    struct DenyExecute;

    impl claurst_core::permissions::PermissionHandler for DenyExecute {
        fn check_permission(
            &self,
            request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            assert!(!request.is_read_only);
            claurst_core::permissions::PermissionDecision::Deny
        }
        fn request_permission(
            &self,
            request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            self.check_permission(request)
        }
    }

    #[tokio::test]
    async fn execute_observation_preserves_permission_gate_without_mutation_bookkeeping() {
        let directory = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(directory.path().to_path_buf());
        ctx.session_id = format!("execute-observation-{}", Uuid::new_v4());
        let scope = scope_for_context(&ctx);
        let calls = Arc::new(AtomicU64::new(0));
        let tool = ExecuteEffectTool {
            effect: ToolEffect::Observation,
            calls: Arc::clone(&calls),
        };
        assert!(!LEDGER.lock().contains_key(&scope));
        let allowed = execute_authorized(&tool, json!({}), &ctx).await;
        assert!(!allowed.is_error, "{}", allowed.content);
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert!(!LEDGER.lock().contains_key(&scope));
        assert_eq!(current_stamp(&ctx).epoch, 0);

        ctx.permission_handler = Arc::new(DenyExecute);
        let denied = execute_authorized(&tool, json!({}), &ctx).await;
        assert!(denied.is_error);
        assert_eq!(
            denied.metadata.as_ref().unwrap()["execution_status"],
            "not_run"
        );
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert!(!LEDGER.lock().contains_key(&scope));
        assert_eq!(current_stamp(&ctx).epoch, 0);
    }

    #[tokio::test]
    async fn execute_unknown_and_mutation_keep_lease_exclusion_and_epochs() {
        for effect in [ToolEffect::Unknown, ToolEffect::Mutation] {
            let directory = tempfile::tempdir().unwrap();
            let mut ctx = allow_all_context(directory.path().to_path_buf());
            ctx.session_id = format!("execute-barrier-{}", Uuid::new_v4());
            let scope = scope_for_context(&ctx);
            let calls = Arc::new(AtomicU64::new(0));
            let tool = ExecuteEffectTool {
                effect,
                calls: Arc::clone(&calls),
            };
            let lease = admit(&ctx, &tool).unwrap();
            assert!(matches!(lease, MutationAdmission::Acquired(_)));
            assert_eq!(current_stamp(&ctx).epoch, 1);
            let blocked = execute_authorized(&tool, json!({}), &ctx).await;
            assert!(blocked.is_error);
            assert_eq!(
                blocked.metadata.as_ref().unwrap()["execution_status"],
                "not_run"
            );
            assert_eq!(calls.load(Ordering::Acquire), 0);
            assert_eq!(current_stamp(&ctx).epoch, 1);
            drop(lease);
            let allowed = execute_authorized(&tool, json!({}), &ctx).await;
            assert!(!allowed.is_error, "{}", allowed.content);
            assert_eq!(calls.load(Ordering::Acquire), 1);
            assert_eq!(current_stamp(&ctx).epoch, 2);
            let ledger = LEDGER.lock();
            assert!(ledger.get(&scope).unwrap().active.is_none());
            drop(ledger);
            LEDGER.lock().remove(&scope);
        }
    }

    #[test]
    fn dispatch_reservation_excludes_mutations_across_sessions_and_releases() {
        let directory = tempfile::tempdir().unwrap();
        let mut first = allow_all_context(directory.path().to_path_buf());
        first.session_id = format!("dispatch-owner-{}", Uuid::new_v4());
        let mut foreign = first.clone();
        foreign.session_id = format!("dispatch-foreign-{}", Uuid::new_v4());
        let other_directory = tempfile::tempdir().unwrap();
        let other = allow_all_context(other_directory.path().to_path_buf());
        let calls = Arc::new(AtomicU64::new(0));
        let mutation = CountingMutation {
            calls: Arc::clone(&calls),
        };
        let expected = workspace_stamp(&first).unwrap();
        let guard = reserve_workspace_for_dispatch(&first, &expected).unwrap();

        assert!(admit(&first, &mutation).is_err());
        assert!(admit(&foreign, &mutation).is_err());
        assert!(register_pending(scope_for_context(&foreign), "reserved writer").is_err());
        assert_eq!(calls.load(Ordering::Acquire), 0);
        assert_eq!(current_stamp(&first).epoch, 0);
        assert!(matches!(
            admit(&first, &ObservationTool),
            Ok(MutationAdmission::Observation)
        ));

        let other_lease = admit(&other, &mutation).unwrap();
        drop(other_lease);
        drop(guard);

        let retry = admit(&first, &mutation).unwrap();
        drop(retry);
        assert_eq!(current_stamp(&first).epoch, 1);
        LEDGER.lock().remove(&scope_for_context(&first));
        LEDGER.lock().remove(&scope_for_context(&foreign));
        LEDGER.lock().remove(&scope_for_context(&other));
    }

    #[test]
    fn dispatch_reservation_rejects_stale_active_and_pending_stamps() {
        let directory = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(directory.path().to_path_buf());
        ctx.session_id = format!("dispatch-stale-{}", Uuid::new_v4());
        let stale = workspace_stamp(&ctx).unwrap();
        let lease = admit(&ctx, &BlockingMutation).unwrap();
        assert!(reserve_workspace_for_dispatch(&ctx, &stale).is_err());
        drop(lease);

        let expected = workspace_stamp(&ctx).unwrap();
        let ticket = register_pending(scope_for_context(&ctx), "pending writer").unwrap();
        assert!(reserve_workspace_for_dispatch(&ctx, &expected).is_err());
        ticket.confirm_terminal().unwrap();

        let guard = reserve_workspace_for_dispatch(&ctx, &workspace_stamp(&ctx).unwrap()).unwrap();
        drop(guard);
        LEDGER.lock().remove(&scope_for_context(&ctx));
    }

    struct NestedOtherScope {
        ctx: ToolContext,
    }

    #[async_trait]
    impl Tool for NestedOtherScope {
        fn name(&self) -> &str {
            "NestedOtherScope"
        }
        fn description(&self) -> &str {
            "test"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Write
        }
        fn effect(&self) -> ToolEffect {
            ToolEffect::Mutation
        }
        fn input_schema(&self) -> Value {
            json!({"type":"object"})
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            execute_authorized(&BlockingMutation, json!({}), &self.ctx).await
        }
    }

    #[test]
    fn workspace_stamp_requires_foreign_session_quiescence_and_reconciles_pending() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = allow_all_context(dir.path().to_path_buf());
        first.session_id = format!("workspace-stamp-first-{}", Uuid::new_v4());
        let mut foreign = first.clone();
        foreign.session_id = format!("workspace-stamp-foreign-{}", Uuid::new_v4());

        let active = match admit(&foreign, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected foreign session lease"),
        };
        assert!(workspace_stamp(&first).is_err());
        drop(active);

        let stamp = workspace_stamp(&first).unwrap();
        let ticket = register_pending(scope_for_context(&foreign), "foreign writer").unwrap();
        assert!(pending_for_workspace(&first));
        assert!(workspace_stamp(&first).is_err());
        assert!(!workspace_stamp_is_current(&first, &stamp));
        let abandoned_id = ticket.id().to_string();
        drop(ticket);

        // A durable owner remains visible after its in-memory scope is
        // discarded, proving fresh reconciliation blocks aggregate capture.
        let recovered = register_pending(scope_for_context(&foreign), "recovered writer").unwrap();
        let foreign_scope = scope_for_context(&foreign);
        LEDGER.lock().remove(&foreign_scope);
        assert!(pending_for_workspace(&first));
        assert!(workspace_stamp(&first).is_err());
        recovered.confirm_terminal().unwrap();
        remove_pending_record(&foreign_scope, &abandoned_id).unwrap();
        LEDGER.lock().remove(&foreign_scope);
        let clean_stamp = workspace_stamp(&first).unwrap();
        assert!(workspace_stamp_is_current(&first, &clean_stamp));

        LEDGER.lock().remove(&scope_for_context(&first));
    }

    #[tokio::test]
    async fn workspace_stamp_tracks_foreign_epoch_and_ignores_other_workspace() {
        let first_dir = tempfile::tempdir().unwrap();
        let other_dir = tempfile::tempdir().unwrap();
        let mut first = allow_all_context(first_dir.path().to_path_buf());
        first.session_id = format!("workspace-stamp-owner-{}", Uuid::new_v4());
        let mut foreign = first.clone();
        foreign.session_id = format!("workspace-stamp-later-{}", Uuid::new_v4());
        let mut other = allow_all_context(other_dir.path().to_path_buf());
        other.session_id = format!("workspace-stamp-other-{}", Uuid::new_v4());

        let stamp = workspace_stamp(&first).unwrap();
        let other_lease = match admit(&other, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected other workspace lease"),
        };
        drop(other_lease);
        assert!(workspace_stamp_is_current(&first, &stamp));

        let same_content = first_dir.path().join("same-content.txt");
        std::fs::write(&same_content, "unchanged").unwrap();
        let result = execute_authorized(
            &FileWriteTool,
            json!({"file_path": same_content, "content": "unchanged"}),
            &foreign,
        )
        .await;
        assert!(
            !result.is_error,
            "same-content write failed: {}",
            result.content
        );
        assert_eq!(std::fs::read_to_string(&same_content).unwrap(), "unchanged");
        assert!(!workspace_stamp_is_current(&first, &stamp));

        let mut stale_generation = workspace_stamp(&first).unwrap();
        stale_generation.generation = "different-process".into();
        assert!(!workspace_stamp_is_current(&first, &stale_generation));

        LEDGER.lock().remove(&scope_for_context(&first));
        LEDGER.lock().remove(&scope_for_context(&foreign));
        LEDGER.lock().remove(&scope_for_context(&other));
    }

    #[test]
    fn active_lease_capture_requires_current_task_local_lease() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        assert!(capture_active_lease(&ctx).is_err());
    }

    #[tokio::test]
    async fn active_capture_rejects_foreign_active_or_pending_owner() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = allow_all_context(dir.path().to_path_buf());
        first.session_id = format!("capture-first-{}", Uuid::new_v4());
        let mut foreign = first.clone();
        foreign.session_id = format!("capture-foreign-{}", Uuid::new_v4());
        let outer = match admit(&first, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected outer lease"),
        };
        with_lease(&outer, async {
            let capture = capture_active_lease(&first).unwrap();
            let foreign_lease = match admit(&foreign, &BlockingMutation).unwrap() {
                MutationAdmission::Acquired(lease) => lease,
                _ => panic!("expected foreign lease"),
            };
            assert!(capture_workspace_epochs(&first, &capture).is_err());
            drop(foreign_lease);

            let ticket = register_pending(scope_for_context(&foreign), "foreign writer").unwrap();
            assert!(capture_workspace_epochs(&first, &capture).is_err());
            ticket.confirm_terminal().unwrap();
        })
        .await;
        drop(outer);
        LEDGER.lock().remove(&scope_for_context(&first));
        LEDGER.lock().remove(&scope_for_context(&foreign));
    }

    #[tokio::test]
    async fn nested_capture_accepts_the_same_token_without_a_child() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let outer = match admit(&ctx, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected outer lease"),
        };
        with_lease(&outer, async {
            let outer_capture = capture_active_lease(&ctx).unwrap();
            assert!(capture_workspace_epochs(&ctx, &outer_capture).is_ok());
            let nested = match admit(&ctx, &BlockingMutation).unwrap() {
                MutationAdmission::Reentrant(lease) => lease,
                _ => panic!("expected nested lease"),
            };
            assert!(capture_active_lease(&ctx).is_err());
            with_lease(&nested, async {
                let nested_capture = capture_active_lease(&ctx).unwrap();
                assert!(capture_workspace_epochs(&ctx, &nested_capture).is_ok());
            })
            .await;
            drop(nested);
        })
        .await;
        drop(outer);
        LEDGER.lock().remove(&scope_for_context(&ctx));
    }

    #[tokio::test]
    async fn captured_workspace_epochs_can_move_to_blocking_task() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let outer = match admit(&ctx, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected outer lease"),
        };
        let captured = with_lease(&outer, async { capture_active_lease(&ctx).unwrap() }).await;
        let blocking_ctx = ctx.clone();
        let stamp =
            tokio::task::spawn_blocking(move || capture_workspace_epochs(&blocking_ctx, &captured))
                .await
                .unwrap()
                .unwrap();
        assert_eq!(stamp.workspace, scope_for_context(&ctx).workspace);
        drop(outer);
        LEDGER.lock().remove(&scope_for_context(&ctx));
    }

    #[tokio::test]
    async fn captured_workspace_vector_rejects_intervening_foreign_same_byte_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("same-byte.txt");
        std::fs::write(&path, "unchanged").unwrap();
        let mut first = allow_all_context(dir.path().to_path_buf());
        first.session_id = format!("capture-vector-first-{}", Uuid::new_v4());
        let mut foreign = first.clone();
        foreign.session_id = format!("capture-vector-foreign-{}", Uuid::new_v4());
        let outer = match admit(&first, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected outer lease"),
        };
        let stamp = with_lease(&outer, async {
            let capture = capture_active_lease(&first).unwrap();
            capture_workspace_epochs(&first, &capture).unwrap()
        })
        .await;
        drop(outer);

        let result = execute_authorized(
            &FileWriteTool,
            json!({"file_path": path, "content": "unchanged"}),
            &foreign,
        )
        .await;
        assert!(
            !result.is_error,
            "same-byte foreign write failed: {}",
            result.content
        );
        assert!(!workspace_stamp_is_current(&first, &stamp));
        LEDGER.lock().remove(&scope_for_context(&first));
        LEDGER.lock().remove(&scope_for_context(&foreign));
    }

    #[tokio::test]
    async fn scopes_share_epoch_but_distinct_workspace_does_not() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let mut first = allow_all_context(a.path().to_path_buf());
        first.session_id = "same".into();
        let second = first.clone();
        let mut other = allow_all_context(b.path().to_path_buf());
        other.session_id = "same".into();
        let before = current_stamp(&first);
        let lease = match admit(&first, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected lease"),
        };
        assert_eq!(scope_for_context(&first), scope_for_context(&second));
        assert!(!stamp_is_current(&first, &before));
        assert!(!stamp_is_current(&other, &before));
        assert!(stamp_is_current(&other, &current_stamp(&other)));
        drop(lease);
    }

    #[test]
    fn stale_generation_and_exhausted_epoch_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let mut stale = current_stamp(&ctx);
        stale.generation = "different-process".into();
        assert!(!stamp_is_current(&ctx, &stale));

        let scope = scope_for_context(&ctx);
        {
            let mut ledger = LEDGER.lock();
            ledger.insert(
                scope,
                ScopeState {
                    epoch: u64::MAX,
                    active: None,
                    pending: HashMap::new(),
                },
            );
        }
        assert!(admit(&ctx, &BlockingMutation).is_err());
        LEDGER.lock().remove(&scope_for_context(&ctx));
    }

    #[test]
    fn scope_capacity_rejects_only_untracked_entries() {
        let mut ledger = HashMap::with_capacity(MAX_TRACKED_SCOPES);
        for index in 0..MAX_TRACKED_SCOPES {
            ledger.insert(
                MutationScope {
                    workspace: PathBuf::from(format!("/mutation-capacity-{index}")),
                    session: index.to_string(),
                },
                ScopeState {
                    epoch: 1,
                    active: None,
                    pending: HashMap::new(),
                },
            );
        }
        let existing = ledger.keys().next().unwrap().clone();
        assert!(ensure_scope_capacity(&ledger, &existing).is_ok());
        let new_scope = MutationScope {
            workspace: PathBuf::from("/mutation-capacity-new"),
            session: "new".into(),
        };
        assert!(ensure_scope_capacity(&ledger, &new_scope).is_err());
        assert_eq!(ledger.len(), MAX_TRACKED_SCOPES);
    }

    #[test]
    fn pending_ticket_invalidates_and_blocks_only_verification() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.session_id = format!("pending-{}", Uuid::new_v4());
        let before = current_stamp(&ctx);
        let ticket = register_pending(scope_for_context(&ctx), "background shell").unwrap();
        assert!(!stamp_is_current(&ctx, &before));
        assert!(pending_for_scope(&ctx));
        let lease = match admit(&ctx, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            other => panic!("unexpected admission: {other:?}"),
        };
        drop(lease);
        ticket.confirm_terminal().unwrap();
        assert!(!pending_for_scope(&ctx));
    }

    #[test]
    fn pending_owner_has_an_individual_durable_record() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.session_id = format!("pending-record-{}", Uuid::new_v4());
        let scope = scope_for_context(&ctx);
        let ticket = register_pending(scope.clone(), "background shell").unwrap();
        let path = recovery_file_path(ticket.id());
        let metadata = std::fs::symlink_metadata(&path).unwrap();
        assert!(metadata.is_file());
        assert!(metadata.len() <= MAX_RECOVERY_RECORD_BYTES);
        assert!(!legacy_recovery_path().exists());
        let bytes = std::fs::read(&path).unwrap();
        let record: PendingRecoveryRecord = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(record.id, ticket.id());
        assert_eq!(record.scope, scope);
        ticket.confirm_terminal().unwrap();
        assert!(std::fs::symlink_metadata(path).is_err());
    }

    #[test]
    fn concurrent_pending_confirmation_waits_for_durable_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.session_id = format!("pending-confirm-{}", Uuid::new_v4());
        let scope = scope_for_context(&ctx);
        let ticket = Arc::new(register_pending(scope.clone(), "background shell").unwrap());
        let held_lock = acquire_recovery_lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();

        for _ in 0..2 {
            let ticket = Arc::clone(&ticket);
            let tx = tx.clone();
            std::thread::spawn(move || {
                tx.send(ticket.confirm_terminal()).unwrap();
            });
        }
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            rx.try_recv().is_err(),
            "confirmation completed while durable lock was held"
        );
        drop(held_lock);

        let first = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let second = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(first.is_ok() && second.is_ok());
        assert!(!pending_for_scope(&ctx));
        LEDGER.lock().remove(&scope);
    }

    #[test]
    fn recovery_lock_released_after_sigkill() {
        struct ChildGuard(Option<std::process::Child>);

        impl ChildGuard {
            fn new(child: std::process::Child) -> Self {
                Self(Some(child))
            }

            fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
                self.0
                    .as_mut()
                    .expect("child guard was already reaped")
                    .try_wait()
            }

            fn kill_and_wait(&mut self) {
                if let Some(mut child) = self.0.take() {
                    match child.try_wait() {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        }

        impl Drop for ChildGuard {
            fn drop(&mut self) {
                self.kill_and_wait();
            }
        }

        const CHILD: &str = "FORGE_MUTATION_RECOVERY_LOCK_CHILD";
        const COORDINATOR: &str = "FORGE_MUTATION_RECOVERY_LOCK_COORDINATOR";
        const READY: &str = "FORGE_MUTATION_RECOVERY_LOCK_READY";
        if std::env::var_os(CHILD).is_some() {
            let _lock = acquire_recovery_lock().expect("child acquires advisory recovery lock");
            let ready = PathBuf::from(std::env::var(READY).expect("ready path"));
            std::fs::write(ready, b"locked").expect("publish lock ownership");
            std::thread::sleep(Duration::from_secs(60));
            return;
        }

        if std::env::var_os(COORDINATOR).is_some() {
            let root = PathBuf::from(std::env::var("FORGE_MUTATION_RECOVERY_ROOT").unwrap());
            let ready = root.join("ready");
            let mut holder = ChildGuard::new(
                Command::new(std::env::current_exe().expect("test executable"))
                    .arg("recovery_lock_released_after_sigkill")
                    .arg("--nocapture")
                    .env(CHILD, "1")
                    .env(READY, &ready)
                    .spawn()
                    .expect("spawn lock holder"),
            );
            for _ in 0..100 {
                if ready.exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(ready.exists(), "child did not publish lock ownership");
            let workspace = root.join("workspace");
            std::fs::create_dir_all(&workspace).expect("create isolated workspace");
            let ctx = allow_all_context(workspace);
            assert!(
                workspace_stamp(&ctx).is_err(),
                "aggregate capture must fail closed while recovery lock is held"
            );
            holder.kill_and_wait();

            let lock_path =
                claurst_core::config::Settings::config_dir().join(PENDING_RECOVERY_LOCK);
            assert!(
                lock_path.exists(),
                "advisory lock path must remain persistent"
            );
            let _lock = acquire_recovery_lock().expect("OS releases lock after SIGKILL");
            return;
        }

        let scratch = tempfile::tempdir().expect("lock test scratch directory");
        let home = scratch.path().join("forge-home");
        let mut coordinator = ChildGuard::new(
            Command::new(std::env::current_exe().expect("test executable"))
                .arg("recovery_lock_released_after_sigkill")
                .arg("--nocapture")
                .env(COORDINATOR, "1")
                .env("FORGE_MUTATION_RECOVERY_ROOT", scratch.path())
                .env("FORGE_HOME", &home)
                .env("CLAURST_HOME", &home)
                .spawn()
                .expect("spawn isolated lock coordinator"),
        );
        let mut status = None;
        for _ in 0..500 {
            status = coordinator.try_wait().expect("poll lock coordinator");
            if status.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        if status.is_none() {
            coordinator.kill_and_wait();
            panic!("isolated lock coordinator exceeded bounded deadline");
        }
        assert!(
            status.unwrap().success(),
            "isolated lock coordinator failed"
        );
    }

    #[test]
    fn pending_unknown_or_abandoned_owner_remains_a_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.session_id = format!("pending-unknown-{}", Uuid::new_v4());
        let scope = scope_for_context(&ctx);
        let ticket = register_pending(scope.clone(), "unknown process").unwrap();
        let abandoned_id = ticket.id().to_string();
        drop(ticket);
        assert!(pending_for_scope(&ctx));
        let second = register_pending(scope, "second process").unwrap();
        second.confirm_terminal().unwrap();
        assert!(pending_for_scope(&ctx));
        remove_pending_record(&scope_for_context(&ctx), &abandoned_id).unwrap();
        LEDGER.lock().remove(&scope_for_context(&ctx));
    }

    #[tokio::test]
    async fn pending_writer_rejects_verification_acceptance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target.txt");
        std::fs::write(&path, "before").unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.session_id = format!("pending-check-{}", Uuid::new_v4());
        let ticket = register_pending(scope_for_context(&ctx), "background shell").unwrap();
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": "exit 0"}}
            }),
            &ctx,
        )
        .await;
        assert!(result.is_error);
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "failed");
        assert_eq!(result.metadata.as_ref().unwrap()["pending_mutation"], true);
        ticket.confirm_terminal().unwrap();
        LEDGER.lock().remove(&scope_for_context(&ctx));
    }

    #[tokio::test]
    async fn concurrent_mutation_fails_closed_without_running_tool() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.session_id = Uuid::new_v4().to_string();
        let first = tokio::spawn({
            let ctx = ctx.clone();
            async move { execute_authorized(&BlockingMutation, json!({}), &ctx).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let second = execute_authorized(&BlockingMutation, json!({}), &ctx).await;
        assert!(second.is_error);
        assert_eq!(second.metadata.unwrap()["execution_status"], "not_run");
        let first = first.await.unwrap();
        assert!(!first.is_error);
    }

    #[tokio::test]
    async fn nested_same_scope_is_reentrant() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let outer = match admit(&ctx, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected lease"),
        };
        with_lease(&outer, async {
            match admit(&ctx, &BlockingMutation).unwrap() {
                MutationAdmission::Reentrant(_) => {}
                _ => panic!("expected reentrant admission"),
            }
        })
        .await;
    }

    #[tokio::test]
    async fn same_task_join_siblings_fail_closed_before_second_tool_runs() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let calls = Arc::new(AtomicU64::new(0));
        let tool = CountingMutation {
            calls: Arc::clone(&calls),
        };
        let outer = match admit(&ctx, &tool).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected lease"),
        };
        let (first, second) = with_lease(&outer, async {
            tokio::join!(
                execute_authorized(&tool, json!({}), &ctx),
                execute_authorized(&tool, json!({}), &ctx),
            )
        })
        .await;
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert_ne!(first.is_error, second.is_error);
        let rejected = if first.is_error { first } else { second };
        assert_eq!(
            rejected.metadata.as_ref().unwrap()["execution_status"],
            "not_run"
        );
    }

    #[tokio::test]
    async fn same_task_join_distinct_scopes_can_progress() {
        let outer_dir = tempfile::tempdir().unwrap();
        let inner_dir = tempfile::tempdir().unwrap();
        let outer_ctx = allow_all_context(outer_dir.path().to_path_buf());
        let inner_ctx = allow_all_context(inner_dir.path().to_path_buf());
        let calls = Arc::new(AtomicU64::new(0));
        let tool = CountingMutation {
            calls: Arc::clone(&calls),
        };
        let outer = match admit(&outer_ctx, &tool).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected lease"),
        };
        let (outer_result, inner_result) = with_lease(&outer, async {
            tokio::join!(
                execute_authorized(&tool, json!({}), &outer_ctx),
                execute_authorized(&tool, json!({}), &inner_ctx),
            )
        })
        .await;
        assert!(!outer_result.is_error, "{}", outer_result.content);
        assert!(!inner_result.is_error, "{}", inner_result.content);
        assert_eq!(calls.load(Ordering::Acquire), 2);
    }

    #[tokio::test]
    async fn nested_distinct_scope_can_acquire_independent_lease() {
        let outer_dir = tempfile::tempdir().unwrap();
        let inner_dir = tempfile::tempdir().unwrap();
        let outer = allow_all_context(outer_dir.path().to_path_buf());
        let inner = allow_all_context(inner_dir.path().to_path_buf());
        let result = execute_authorized(&NestedOtherScope { ctx: inner }, json!({}), &outer).await;
        assert!(!result.is_error, "{}", result.content);
    }

    #[tokio::test]
    async fn owned_commit_handle_keeps_exclusion_after_caller_drop() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let outer = match admit(&ctx, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected lease"),
        };
        let held = with_lease(&outer, async { owned_current_lease().unwrap() }).await;
        drop(outer);
        assert!(admit(&ctx, &BlockingMutation).is_err());
        drop(held);
        assert!(matches!(
            admit(&ctx, &BlockingMutation),
            Ok(MutationAdmission::Acquired(_))
        ));
    }

    #[tokio::test]
    async fn nested_owned_commit_keeps_parent_frame_guard() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let outer = match admit(&ctx, &BlockingMutation).unwrap() {
            MutationAdmission::Acquired(lease) => lease,
            _ => panic!("expected lease"),
        };
        with_lease(&outer, async {
            let nested = match admit(&ctx, &BlockingMutation).unwrap() {
                MutationAdmission::Reentrant(lease) => lease,
                _ => panic!("expected nested lease"),
            };
            let owned_commit = nested.owned_clone();
            drop(nested);
            assert!(admit(&ctx, &BlockingMutation).is_err());
            drop(owned_commit);
            assert!(matches!(
                admit(&ctx, &BlockingMutation),
                Ok(MutationAdmission::Reentrant(_))
            ));
        })
        .await;
    }

    #[tokio::test]
    async fn edit_check_keeps_committed_mutation_when_checker_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target.txt");
        std::fs::write(&path, "before").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": "exit 7"}}
            }),
            &ctx,
        )
        .await;
        assert!(result.is_error);
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "committed");
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "failed");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn trusted_edit_check_attaches_only_a_host_receipt_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("typed.txt");
        std::fs::write(&path, "before").unwrap();
        let ctx = trusted_checker_context(
            dir.path(),
            &path,
            crate::TrustedCheckerOperation::Write,
            "true",
        );
        let mut result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": "true"}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            result.metadata.as_ref().unwrap()["check_trust"],
            "registered"
        );
        assert!(result.take_application_receipts().is_some());
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn unavailable_windows_bash_registration_fails_closed_without_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("windows-fail-closed.txt");
        std::fs::write(&path, "before").unwrap();
        let ctx = trusted_checker_context(
            dir.path(),
            &path,
            crate::TrustedCheckerOperation::Write,
            "echo checker",
        );
        // The registration API has no cmd.exe shell variant. The Bash
        // registration therefore cannot acquire a trusted receipt on Windows;
        // the ordinary cmd fallback must still complete without a receipt.
        let mut result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": "echo checker"}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            result.metadata.as_ref().unwrap()["check_trust"],
            "unsupported"
        );
        assert!(result.take_application_receipts().is_none());
    }

    #[tokio::test]
    async fn mismatched_trusted_registration_uses_generic_checker_without_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mismatch.txt");
        std::fs::write(&path, "before").unwrap();
        let ctx = trusted_checker_context(
            dir.path(),
            &path,
            crate::TrustedCheckerOperation::Write,
            "true",
        );
        let mut result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": generic_checker_command()}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            result.metadata.as_ref().unwrap()["check_trust"],
            "unsupported"
        );
        assert!(result.take_application_receipts().is_none());
    }

    #[tokio::test]
    async fn forged_checker_json_never_creates_a_typed_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("forged.json");
        std::fs::write(&path, "before").unwrap();
        let ctx = trusted_checker_context(
            dir.path(),
            &path,
            crate::TrustedCheckerOperation::Write,
            "true",
        );
        let mut result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": forged_checker_command()}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            result.metadata.as_ref().unwrap()["check_trust"],
            "unsupported"
        );
        assert!(result.take_application_receipts().is_none());
    }

    #[tokio::test]
    async fn failed_edit_skips_checker_and_reports_not_run() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target.txt");
        let marker = dir.path().join("checker-ran");
        std::fs::write(&path, "before").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Edit",
                "edit": {"file_path": path, "old_string": "missing", "new_string": "after"},
                "check": {"tool": "Bash", "input": {"command": format!("touch {}", marker.display())}}
            }),
            &ctx,
        )
        .await;
        assert!(result.is_error);
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "failed");
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "not_run");
        assert!(!marker.exists());
    }

    #[tokio::test]
    async fn same_byte_write_is_no_op_but_checker_still_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target.txt");
        std::fs::write(&path, "same").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let mut result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "same"},
                "check": {"tool": "Bash", "input": {"command": "exit 0"}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "no_op");
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "passed");
        assert!(result.take_application_receipts().is_none());
    }

    #[tokio::test]
    async fn creating_a_new_empty_file_is_committed_not_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new-empty.txt");
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": ""},
                "check": {"tool": "Bash", "input": {"command": "exit 0"}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "committed");
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "passed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn denied_path_is_not_read_before_nested_permission() {
        use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
        struct DenyPaths;
        impl PermissionHandler for DenyPaths {
            fn check_permission(&self, request: &PermissionRequest) -> PermissionDecision {
                if request.path.is_some() {
                    PermissionDecision::Deny
                } else {
                    PermissionDecision::Allow
                }
            }
            fn request_permission(&self, request: &PermissionRequest) -> PermissionDecision {
                self.check_permission(request)
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("private-fifo");
        assert!(std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .unwrap()
            .success());
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.permission_handler = Arc::new(DenyPaths);
        let result = tokio::time::timeout(
            Duration::from_millis(250),
            execute_authorized(
                &EditCheckTool,
                json!({
                    "operation": "Write", "edit": {"file_path": path, "content": "never"},
                    "check": {"tool": "Bash", "input": {"command": "exit 0"}}
                }),
                &ctx,
            ),
        )
        .await;
        if result.is_err() {
            // Release the old implementation's unauthorized blocking FIFO
            // reader before the test runtime shuts down.
            let release_path = path.clone();
            std::thread::spawn(move || std::fs::write(release_path, b"private bytes").unwrap())
                .join()
                .unwrap();
        }
        assert!(
            result.is_ok(),
            "denied target was read before path permission"
        );
        let result = result.unwrap();
        assert!(result.is_error);
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "not_run");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn checker_deletion_without_user_authority_is_not_run() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write", "edit": {"file_path": path, "content": ""},
                "check": {"tool": "Bash", "input": {"command": "rm empty.txt"}}
            }),
            &ctx,
        )
        .await;
        assert!(result.is_error, "missing empty target was falsely verified");
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "committed");
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "not_run");
        assert!(path.is_file(), "unauthorized checker deleted the target");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn checker_retargeting_same_bytes_cannot_pass() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target.txt");
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = execute_authorized(&EditCheckTool, json!({
            "operation": "Write", "edit": {"file_path": path, "content": "target"},
            "check": {"tool": "Bash", "input": {"command": "mv target.txt moved.txt; ln -s moved.txt target.txt"}}
        }), &ctx).await;
        assert!(
            result.is_error,
            "different canonical target was falsely verified"
        );
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "failed");
    }

    #[tokio::test]
    async fn checker_mutating_target_cannot_claim_verification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target.txt");
        std::fs::write(&path, "before").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": path, "content": "after"},
                "check": {"tool": "Bash", "input": {"command": mutating_checker_command(&path)}}
            }),
            &ctx,
        )
        .await;
        assert!(result.is_error);
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "committed");
        assert_eq!(result.metadata.as_ref().unwrap()["verification"], "failed");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "changed");
    }
}
