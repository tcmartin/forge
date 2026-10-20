//! Private, process-local authority for trusted mutation/checker receipts.
//!
//! Receipt values intentionally have no serde representation.  They are
//! created by the already-authorized tool path and are consumed by the host
//! verifier; model-visible JSON cannot create or replace one.

use crate::command_capture::{CapturedCommand, ProcessCleanup};
use crate::mutation::{self, MutationScope, MutationStamp, WorkspaceMutationStamp};
use crate::trusted_checker::{TrustedCheckerOperation, TrustedCheckerRegistration};
use crate::ToolContext;
use claurst_core::crypto_utils::sha256_hex;
use claurst_core::dispatch::{
    DispatchAttempt, DispatchAuthorization, DispatchAuthorizer, DispatchDecision, DispatchDenied,
    DispatchPermit, DispatchReservation, DispatchRoute,
};
use parking_lot::Mutex;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::future::Future;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const MAX_SNAPSHOT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_OPERATION_ID_BYTES: usize = 256;
const SNAPSHOT_DEADLINE: Duration = Duration::from_secs(5);

struct ReceiptState {
    finalized: AtomicBool,
    // All proof and permit clones retain this exact state. A new permit can
    // never manufacture another verified logical continuation.
    dispatch_use: Mutex<Option<DispatchAttempt>>,
    dispatch_in_flight: AtomicBool,
}

/// Errors from receipt construction and proof validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    Cancelled,
    MissingLease,
    UnsupportedPlatform,
    InvalidInput,
    TargetUnavailable,
    TargetChanged,
    SizeLimit,
    Timeout,
    Io,
    ScopeMismatch,
    StampMismatch,
    CheckerNotRegistered,
    CheckerIncomplete,
    MissingChecker,
    AlreadyConsumed,
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Cancelled => "verification cancelled",
            Self::MissingLease => "an active mutation lease is required",
            Self::UnsupportedPlatform => "verification receipts are unsupported on this platform",
            Self::InvalidInput => "invalid receipt input",
            Self::TargetUnavailable => "the target file is unavailable",
            Self::TargetChanged => "the target changed during verification",
            Self::SizeLimit => "the target exceeds the verification size limit",
            Self::Timeout => "verification exceeded its deadline",
            Self::Io => "verification I/O failed",
            Self::ScopeMismatch => "receipt scope does not match the active context",
            Self::StampMismatch => "mutation epoch changed during verification",
            Self::CheckerNotRegistered => "checker registration is not in the immutable catalog",
            Self::CheckerIncomplete => "checker completion evidence is incomplete",
            Self::MissingChecker => "a completed checker receipt is required",
            Self::AlreadyConsumed => "verification proof was already consumed",
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for ReceiptError {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TargetSnapshot {
    parent: PathBuf,
    basename: Vec<u8>,
    dev: u64,
    ino: u64,
    len: u64,
    hash: String,
}

/// A host-created record of one committed mutation.
///
/// The fields are private by design.  Callers can only obtain one after the
/// active mutation lease and the exact on-disk target have been checked.
///
/// ```compile_fail
/// # use claurst_tools::MutationReceipt;
/// # use claurst_tools::TrustedCheckerOperation;
/// # let _ = (MutationReceipt {}, TrustedCheckerOperation::Write);
/// ```
#[derive(Clone)]
pub struct MutationReceipt {
    scope: MutationScope,
    path: PathBuf,
    operation: TrustedCheckerOperation,
    operation_id: String,
    expected_hash: String,
    expected_len: u64,
    snapshot: TargetSnapshot,
    stamp: MutationStamp,
    state: Arc<ReceiptState>,
}

impl fmt::Debug for MutationReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MutationReceipt")
            .field("operation", &self.operation)
            .field("operation_id", &"<redacted>")
            .field("expected_hash", &"<redacted>")
            .field("expected_len", &self.expected_len)
            .field("scope", &"<redacted>")
            .field("path", &"<redacted>")
            .finish()
    }
}

impl MutationReceipt {
    /// Capture a committed regular file under the currently active mutation.
    pub(crate) async fn committed(
        ctx: &ToolContext,
        path: impl Into<PathBuf>,
        operation: TrustedCheckerOperation,
        operation_id: impl Into<String>,
        expected_hash: impl Into<String>,
        expected_len: u64,
    ) -> Result<Self, ReceiptError> {
        #[cfg(windows)]
        return Err(ReceiptError::UnsupportedPlatform);

        #[cfg(not(windows))]
        {
            mutation::capture_active_lease(ctx).map_err(|_| ReceiptError::MissingLease)?;
            let operation_id = operation_id.into();
            if operation_id.is_empty() || operation_id.len() > MAX_OPERATION_ID_BYTES {
                return Err(ReceiptError::InvalidInput);
            }
            let expected_hash = expected_hash.into();
            if expected_hash.is_empty() || expected_hash.len() > 128 {
                return Err(ReceiptError::InvalidInput);
            }
            let path = path.into();
            let path = if path.is_absolute() {
                path
            } else {
                ctx.working_dir.join(path)
            };
            let scope = mutation::scope_for_context(ctx);
            let before_stamp = mutation::current_stamp(ctx);
            if before_stamp.scope != scope {
                return Err(ReceiptError::ScopeMismatch);
            }
            let before = snapshot_with_deadline(&path, ctx).await?;
            if before.hash != expected_hash || before.len != expected_len {
                return Err(ReceiptError::TargetChanged);
            }
            let canonical_path = before
                .parent
                .join(path.file_name().ok_or(ReceiptError::InvalidInput)?);
            let after = snapshot_with_deadline(&path, ctx).await?;
            let after_stamp = mutation::current_stamp(ctx);
            if before != after || before_stamp != after_stamp {
                return Err(ReceiptError::TargetChanged);
            }
            Ok(Self {
                scope,
                path: canonical_path,
                operation,
                operation_id,
                expected_hash,
                expected_len,
                snapshot: after,
                stamp: after_stamp,
                state: Arc::new(ReceiptState {
                    finalized: AtomicBool::new(false),
                    dispatch_use: Mutex::new(None),
                    dispatch_in_flight: AtomicBool::new(false),
                }),
            })
        }
    }

    pub(crate) fn scope(&self) -> &MutationScope {
        &self.scope
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn operation(&self) -> TrustedCheckerOperation {
        self.operation
    }

    pub(crate) fn stamp(&self) -> &MutationStamp {
        &self.stamp
    }
}

/// A host-created record of a successfully completed pinned checker.
#[derive(Clone)]
pub struct CheckerReceipt {
    registration: TrustedCheckerRegistration,
    mutation_stamp: MutationStamp,
    start_stamp: MutationStamp,
    final_stamp: MutationStamp,
    workspace_stamp: WorkspaceMutationStamp,
    target: TargetSnapshot,
}

impl fmt::Debug for CheckerReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckerReceipt")
            .field("registration", &"<redacted>")
            .field("mutation_stamp", &"<redacted>")
            .field("start_stamp", &"<redacted>")
            .field("final_stamp", &"<redacted>")
            .finish()
    }
}

impl CheckerReceipt {
    /// Build a checker receipt after its durable pending ticket has been
    /// cleared.  The explicit start stamp prevents an unrelated epoch from
    /// being mistaken for the checker admission transition.
    pub(crate) async fn completed(
        ctx: &ToolContext,
        registration: &TrustedCheckerRegistration,
        mutation: &MutationReceipt,
        start_stamp: &MutationStamp,
        captured: &CapturedCommand,
    ) -> Result<Self, ReceiptError> {
        if captured.status.is_none_or(|status| !status.success())
            || captured.partial
            || captured.timed_out
            || captured.cancelled
            || captured.error.is_some()
            || captured.cleanup != ProcessCleanup::ConfirmedClean
        {
            return Err(ReceiptError::CheckerIncomplete);
        }
        if ctx.cancel_token.is_cancelled() {
            return Err(ReceiptError::Cancelled);
        }
        let scope = mutation::scope_for_context(ctx);
        if mutation.scope != scope
            || registration.workspace() != scope.workspace
            || registration.session_id() != scope.session
            || registration.target() != mutation.path
            || registration.operation() != mutation.operation
        {
            return Err(ReceiptError::ScopeMismatch);
        }
        if ctx.trusted_checker_catalog.lookup(registration).is_none() {
            return Err(ReceiptError::CheckerNotRegistered);
        }
        let Some(expected_start_epoch) = mutation.stamp.epoch.checked_add(1) else {
            return Err(ReceiptError::StampMismatch);
        };
        let Some(expected_final_epoch) = start_stamp.epoch.checked_add(2) else {
            return Err(ReceiptError::StampMismatch);
        };
        if start_stamp.generation != mutation.stamp.generation
            || start_stamp.scope != scope
            || start_stamp.epoch != expected_start_epoch
        {
            return Err(ReceiptError::StampMismatch);
        }
        let final_stamp = mutation::current_stamp(ctx);
        if final_stamp.generation != start_stamp.generation
            || final_stamp.scope != scope
            || final_stamp.epoch != expected_final_epoch
        {
            return Err(ReceiptError::StampMismatch);
        }
        let active = mutation::capture_active_lease(ctx).map_err(|_| ReceiptError::MissingLease)?;
        let workspace_stamp = capture_workspace_stamp(ctx, active).await?;
        Ok(Self {
            registration: registration.clone(),
            mutation_stamp: mutation.stamp.clone(),
            start_stamp: start_stamp.clone(),
            final_stamp,
            workspace_stamp,
            target: mutation.snapshot.clone(),
        })
    }

    pub(crate) fn workspace_stamp(&self) -> &WorkspaceMutationStamp {
        &self.workspace_stamp
    }
}

/// The private bundle passed between the mutator, checker runner and verifier.
#[derive(Clone)]
pub struct ReceiptBundle {
    mutation: MutationReceipt,
    checker: Option<CheckerReceipt>,
    state: Arc<ReceiptState>,
}

impl fmt::Debug for ReceiptBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReceiptBundle")
            .field("mutation", &self.mutation)
            .field("checker", &self.checker.as_ref().map(|_| "<present>"))
            .finish()
    }
}

impl ReceiptBundle {
    pub(crate) fn new(mutation: MutationReceipt) -> Self {
        let state = Arc::clone(&mutation.state);
        Self {
            mutation,
            checker: None,
            state,
        }
    }

    pub(crate) fn attach_checker(mut self, checker: CheckerReceipt) -> Self {
        self.checker = Some(checker);
        self
    }

    pub(crate) fn mutation(&self) -> &MutationReceipt {
        &self.mutation
    }
}

/// A finalized, revalidatable proof of a pinned mutation and checker.
///
/// Receipt finalization and actual-dispatch consumption have separate shared
/// states. Finalization alone never authorizes an inference transport.
#[derive(Clone)]
pub struct TrustedVerificationProof {
    bundle: ReceiptBundle,
}

impl fmt::Debug for TrustedVerificationProof {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedVerificationProof")
            .field("bundle", &"<redacted>")
            .finish()
    }
}

impl TrustedVerificationProof {
    /// Revalidate the immutable registration, process-local epochs, and target
    /// identity before a future dispatch uses this proof.
    pub async fn revalidate(&self, ctx: &ToolContext) -> Result<(), ReceiptError> {
        validate_bundle(ctx, &self.bundle).await
    }

    /// Bind a host-selected route to the shared one-shot proof authority.
    /// Construction is pure: actual leaf dispatch must call `authorize` after
    /// local wire validation and inference admission, immediately before send.
    pub fn dispatch_permit(&self, ctx: &ToolContext, route: DispatchRoute) -> DispatchPermit {
        DispatchPermit::new(Arc::new(ProofDispatchAuthorizer {
            proof: self.clone(),
            context: ctx.clone(),
            route,
        }))
    }
}

struct ProofDispatchAuthorizer {
    proof: TrustedVerificationProof,
    context: ToolContext,
    route: DispatchRoute,
}

struct ProofInFlightGuard(Arc<ReceiptState>);

impl Drop for ProofInFlightGuard {
    fn drop(&mut self) {
        self.0.dispatch_in_flight.store(false, Ordering::Release);
    }
}

struct ProofDispatchReservation {
    // Release workspace admission before making a serialized retry available.
    _workspace: mutation::WorkspaceDispatchGuard,
    _in_flight: ProofInFlightGuard,
}

impl DispatchReservation for ProofDispatchReservation {}

async fn reserve_dispatch_workspace(
    ctx: &ToolContext,
    stamp: &WorkspaceMutationStamp,
) -> Result<mutation::WorkspaceDispatchGuard, DispatchDenied> {
    let context = ctx.clone();
    let stamp = stamp.clone();
    let mut task = tokio::task::spawn_blocking(move || {
        mutation::reserve_workspace_for_dispatch(&context, &stamp)
    });
    tokio::select! {
        result = &mut task => result
            .map_err(|_| DispatchDenied::ProofInvalidated)?
            .map_err(|_| DispatchDenied::ProofInvalidated),
        _ = ctx.cancel_token.cancelled() => {
            task.abort();
            Err(DispatchDenied::Cancelled)
        }
        _ = tokio::time::sleep(SNAPSHOT_DEADLINE) => {
            task.abort();
            Err(DispatchDenied::Timeout)
        }
    }
}

impl DispatchAuthorizer for ProofDispatchAuthorizer {
    fn authorize<'a>(
        &'a self,
        attempt: &'a DispatchAttempt,
    ) -> Pin<Box<dyn Future<Output = Result<DispatchAuthorization, DispatchDenied>> + Send + 'a>>
    {
        Box::pin(async move {
            if attempt.route() != &self.route {
                return Err(DispatchDenied::RouteMismatch);
            }
            if self
                .proof
                .bundle
                .state
                .dispatch_use
                .lock()
                .as_ref()
                .is_some_and(|previous| previous != attempt)
            {
                return Err(DispatchDenied::AlreadyConsumed);
            }
            self.proof
                .bundle
                .state
                .dispatch_in_flight
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| DispatchDenied::InFlight)?;
            let in_flight = ProofInFlightGuard(Arc::clone(&self.proof.bundle.state));
            let checker = self
                .proof
                .bundle
                .checker
                .as_ref()
                .ok_or(DispatchDenied::ProofInvalidated)?;
            // Reserve under the mutation ledger lock before asynchronous file
            // validation. Admitted writers cannot slip into its final await
            // boundary or the subsequent transport send.
            let workspace =
                reserve_dispatch_workspace(&self.context, &checker.workspace_stamp).await?;
            self.proof
                .revalidate(&self.context)
                .await
                .map_err(|error| match error {
                    ReceiptError::Cancelled => DispatchDenied::Cancelled,
                    ReceiptError::Timeout => DispatchDenied::Timeout,
                    _ => DispatchDenied::ProofInvalidated,
                })?;
            if self.context.cancel_token.is_cancelled() {
                return Err(DispatchDenied::Cancelled);
            }
            let decision = {
                let mut used = self.proof.bundle.state.dispatch_use.lock();
                match used.as_ref() {
                    Some(previous) if previous == attempt => DispatchDecision::AuthorizedRetry,
                    Some(_) => return Err(DispatchDenied::AlreadyConsumed),
                    None => {
                        *used = Some(attempt.clone());
                        DispatchDecision::AuthorizedFresh
                    }
                }
            };
            Ok(DispatchAuthorization::new(
                decision,
                Box::new(ProofDispatchReservation {
                    _workspace: workspace,
                    _in_flight: in_flight,
                }),
            ))
        })
    }
}

/// Finalize one receipt bundle into the single consumable verification proof.
pub async fn finalize_proof(
    ctx: &ToolContext,
    bundle: &ReceiptBundle,
) -> Result<TrustedVerificationProof, ReceiptError> {
    validate_bundle(ctx, bundle).await?;
    if bundle
        .state
        .finalized
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(ReceiptError::AlreadyConsumed);
    }
    Ok(TrustedVerificationProof {
        bundle: bundle.clone(),
    })
}

async fn validate_bundle(ctx: &ToolContext, bundle: &ReceiptBundle) -> Result<(), ReceiptError> {
    if ctx.cancel_token.is_cancelled() {
        return Err(ReceiptError::Cancelled);
    }
    let checker = bundle
        .checker
        .as_ref()
        .ok_or(ReceiptError::MissingChecker)?;
    let mutation = &bundle.mutation;
    if mutation.operation_id.is_empty()
        || mutation.snapshot.hash != mutation.expected_hash
        || mutation.snapshot.len != mutation.expected_len
    {
        return Err(ReceiptError::TargetChanged);
    }
    let scope = mutation::scope_for_context(ctx);
    if mutation.scope != scope
        || checker.registration.workspace() != scope.workspace
        || checker.registration.session_id() != scope.session
        || checker.registration.target() != mutation.path
        || checker.registration.operation() != mutation.operation
    {
        return Err(ReceiptError::ScopeMismatch);
    }
    if ctx
        .trusted_checker_catalog
        .lookup(&checker.registration)
        .is_none()
    {
        return Err(ReceiptError::CheckerNotRegistered);
    }
    let Some(expected_start_epoch) = mutation.stamp.epoch.checked_add(1) else {
        return Err(ReceiptError::StampMismatch);
    };
    let Some(expected_final_epoch) = checker.start_stamp.epoch.checked_add(2) else {
        return Err(ReceiptError::StampMismatch);
    };
    if checker.mutation_stamp != mutation.stamp
        || checker.start_stamp.generation != mutation.stamp.generation
        || checker.start_stamp.scope != scope
        || checker.start_stamp.epoch != expected_start_epoch
        || checker.final_stamp.generation != checker.start_stamp.generation
        || checker.final_stamp.scope != scope
        || checker.final_stamp.epoch != expected_final_epoch
    {
        return Err(ReceiptError::StampMismatch);
    }
    workspace_stamp_is_current_bounded(ctx, checker.workspace_stamp()).await?;
    let current = mutation::current_stamp(ctx);
    if current != checker.final_stamp {
        return Err(ReceiptError::StampMismatch);
    }
    let first = snapshot_with_deadline(mutation.path(), ctx).await?;
    if first != mutation.snapshot || first != checker.target {
        return Err(ReceiptError::TargetChanged);
    }
    workspace_stamp_is_current_bounded(ctx, checker.workspace_stamp()).await?;
    let second = snapshot_with_deadline(mutation.path(), ctx).await?;
    if second != first || second != mutation.snapshot {
        return Err(ReceiptError::TargetChanged);
    }
    Ok(())
}

async fn snapshot_with_deadline(
    path: &Path,
    ctx: &ToolContext,
) -> Result<TargetSnapshot, ReceiptError> {
    let path = path.to_path_buf();
    let mut task = tokio::task::spawn_blocking(move || snapshot_sync(&path));
    tokio::select! {
        result = &mut task => result.map_err(|_| ReceiptError::Io)?,
        _ = ctx.cancel_token.cancelled() => {
            task.abort();
            Err(ReceiptError::Cancelled)
        }
        _ = tokio::time::sleep(SNAPSHOT_DEADLINE) => {
            task.abort();
            Err(ReceiptError::Timeout)
        }
    }
}

async fn capture_workspace_stamp(
    ctx: &ToolContext,
    active: crate::mutation::ActiveLeaseCapture,
) -> Result<WorkspaceMutationStamp, ReceiptError> {
    let owned_context = ctx.clone();
    let mut task = tokio::task::spawn_blocking(move || {
        mutation::capture_workspace_epochs(&owned_context, &active)
    });
    tokio::select! {
        result = &mut task => match result {
            Ok(Ok(stamp)) => Ok(stamp),
            Ok(Err(_)) | Err(_) => Err(ReceiptError::StampMismatch),
        },
        _ = ctx.cancel_token.cancelled() => {
            task.abort();
            Err(ReceiptError::Cancelled)
        }
        _ = tokio::time::sleep(SNAPSHOT_DEADLINE) => {
            task.abort();
            Err(ReceiptError::Timeout)
        }
    }
}

async fn workspace_stamp_is_current_bounded(
    ctx: &ToolContext,
    stamp: &WorkspaceMutationStamp,
) -> Result<(), ReceiptError> {
    let owned_context = ctx.clone();
    let stamp = stamp.clone();
    let mut task = tokio::task::spawn_blocking(move || {
        mutation::workspace_stamp_is_current(&owned_context, &stamp)
    });
    tokio::select! {
        result = &mut task => match result {
            Ok(true) => Ok(()),
            Ok(false) | Err(_) => Err(ReceiptError::StampMismatch),
        },
        _ = ctx.cancel_token.cancelled() => {
            task.abort();
            Err(ReceiptError::Cancelled)
        }
        _ = tokio::time::sleep(SNAPSHOT_DEADLINE) => {
            task.abort();
            Err(ReceiptError::Timeout)
        }
    }
}

fn snapshot_sync(path: &Path) -> Result<TargetSnapshot, ReceiptError> {
    #[cfg(windows)]
    {
        let _ = path;
        return Err(ReceiptError::UnsupportedPlatform);
    }
    #[cfg(not(windows))]
    {
        let basename = path
            .file_name()
            .ok_or(ReceiptError::InvalidInput)?
            .as_encoded_bytes()
            .to_vec();
        let parent = path
            .parent()
            .ok_or(ReceiptError::InvalidInput)?
            .canonicalize()
            .map_err(|_| ReceiptError::TargetUnavailable)?;
        let target = parent.join(path.file_name().ok_or(ReceiptError::InvalidInput)?);
        let canonical_parent = path
            .parent()
            .ok_or(ReceiptError::InvalidInput)?
            .canonicalize()
            .map_err(|_| ReceiptError::TargetUnavailable)?;
        if canonical_parent != parent {
            return Err(ReceiptError::TargetChanged);
        }
        let link_meta =
            fs::symlink_metadata(&target).map_err(|_| ReceiptError::TargetUnavailable)?;
        if link_meta.file_type().is_symlink() || !link_meta.file_type().is_file() {
            return Err(ReceiptError::TargetUnavailable);
        }
        let (dev, ino) = native_identity(&link_meta)?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let mut file = options
            .open(&target)
            .map_err(|_| ReceiptError::TargetUnavailable)?;
        let opened_metadata = file
            .metadata()
            .map_err(|_| ReceiptError::TargetUnavailable)?;
        if !opened_metadata.file_type().is_file()
            || native_identity(&opened_metadata)? != (dev, ino)
            || opened_metadata.len() > MAX_SNAPSHOT_BYTES
        {
            return Err(ReceiptError::TargetChanged);
        }
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            let count = file.read(&mut buffer).map_err(|_| ReceiptError::Io)?;
            if count == 0 {
                break;
            }
            if bytes.len().saturating_add(count) as u64 > MAX_SNAPSHOT_BYTES {
                return Err(ReceiptError::SizeLimit);
            }
            bytes.extend_from_slice(&buffer[..count]);
        }
        let final_metadata = file
            .metadata()
            .map_err(|_| ReceiptError::TargetUnavailable)?;
        let final_link_meta =
            fs::symlink_metadata(&target).map_err(|_| ReceiptError::TargetUnavailable)?;
        if final_link_meta.file_type().is_symlink()
            || !final_link_meta.file_type().is_file()
            || native_identity(&final_metadata)? != native_identity(&final_link_meta)?
            || native_identity(&final_metadata)? != (dev, ino)
            || final_metadata.len() != opened_metadata.len()
            || bytes.len() as u64 != final_metadata.len()
        {
            return Err(ReceiptError::TargetChanged);
        }
        Ok(TargetSnapshot {
            parent,
            basename,
            dev,
            ino,
            len: bytes.len() as u64,
            hash: sha256_hex(&bytes),
        })
    }
}

#[cfg(unix)]
fn native_identity(metadata: &fs::Metadata) -> Result<(u64, u64), ReceiptError> {
    use std::os::unix::fs::MetadataExt;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn native_identity(_: &fs::Metadata) -> Result<(u64, u64), ReceiptError> {
    Err(ReceiptError::UnsupportedPlatform)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use claurst_core::dispatch::{DispatchRouteSpec, LogicalRequestId};

    #[cfg(unix)]
    fn test_route() -> DispatchRoute {
        DispatchRoute::new(DispatchRouteSpec {
            provider_id: "fixture".into(),
            provider_instance_id: "instance-canary".into(),
            account_identity: "account-canary".into(),
            model_id: "model-fixture".into(),
            endpoint_identity: "endpoint-fixture".into(),
            effort: "medium".into(),
            policy_revision: 1,
        })
        .unwrap()
    }

    #[cfg(unix)]
    async fn decision(
        permit: &DispatchPermit,
        attempt: &DispatchAttempt,
    ) -> Result<DispatchDecision, DispatchDenied> {
        // Model a completed/dropped send before starting a serialized retry.
        permit
            .authorize(attempt)
            .await
            .map(|authorization| authorization.decision())
    }

    #[cfg(unix)]
    async fn verified_proof(root: &Path) -> (ToolContext, TrustedVerificationProof) {
        use crate::{
            execute_authorized, EditCheckTool, RegistrationSpec, TrustedCheckerCatalogBuilder,
            TrustedCheckerInvocation, TrustedCheckerShell, TrustedCheckerSource,
        };
        use serde_json::json;
        let mut ctx = crate::test_support::allow_all_context(root.to_path_buf());
        let command = "test \"$(cat target)\" = after; printf x >> checker-runs";
        let registration = TrustedCheckerRegistration::new(RegistrationSpec {
            id: "dispatch-fixture".into(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.to_path_buf(),
            session_id: ctx.session_id.clone(),
            target: root.join("target"),
            operation: TrustedCheckerOperation::Write,
            invocation: TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                command,
                root,
                Duration::from_secs(5),
            )
            .unwrap(),
        })
        .unwrap();
        let mut catalog = TrustedCheckerCatalogBuilder::new();
        catalog.register(registration).unwrap();
        ctx.trusted_checker_catalog = Arc::new(catalog.build());
        let mut result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write", "edit": {"file_path": "target", "content": "after"},
                "check": {"tool": "Bash", "input": {"command": command, "timeout": 5000}},
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        let receipts = result.take_application_receipts().unwrap();
        let proof = finalize_proof(&ctx, &receipts).await.unwrap();
        (ctx, proof)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dispatch_consumes_once_and_exact_retry_reuses_shared_authority() {
        let root = tempfile::tempdir().unwrap();
        let (ctx, proof) = verified_proof(root.path()).await;
        let route = test_route();
        let id = LogicalRequestId::fresh();
        let body = sha256_hex(b"exact final prepared body");
        let attempt = DispatchAttempt::new(route.clone(), id.clone(), body.clone()).unwrap();
        let permit = proof.dispatch_permit(&ctx, route.clone());
        assert!(
            proof.bundle.state.dispatch_use.lock().is_none(),
            "construction must be pure"
        );
        assert_eq!(
            decision(&permit, &attempt).await,
            Ok(DispatchDecision::AuthorizedFresh)
        );
        assert_eq!(
            decision(&permit.clone(), &attempt).await,
            Ok(DispatchDecision::AuthorizedRetry)
        );
        let rebuilt = proof.clone().dispatch_permit(&ctx, route.clone());
        assert_eq!(
            decision(&rebuilt, &attempt).await,
            Ok(DispatchDecision::AuthorizedRetry)
        );
        let another_id =
            DispatchAttempt::new(route.clone(), LogicalRequestId::fresh(), body).unwrap();
        assert_eq!(
            decision(&rebuilt, &another_id).await,
            Err(DispatchDenied::AlreadyConsumed)
        );
        let another_body =
            DispatchAttempt::new(route, id, sha256_hex(b"different prepared body")).unwrap();
        assert_eq!(
            decision(&rebuilt, &another_body).await,
            Err(DispatchDenied::AlreadyConsumed)
        );
        assert_eq!(
            std::fs::read(root.path().join("checker-runs")).unwrap(),
            b"x"
        );
        assert_eq!(
            ctx.file_history.lock().len(),
            1,
            "authorization must never replay edits"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dispatch_route_changes_deny_and_new_permit_cannot_reset_consumption() {
        let root = tempfile::tempdir().unwrap();
        let (ctx, proof) = verified_proof(root.path()).await;
        let original = test_route();
        let id = LogicalRequestId::fresh();
        let body = sha256_hex(b"final prepared body");
        let permit = proof.dispatch_permit(&ctx, original.clone());
        let attempt = DispatchAttempt::new(original, id.clone(), body.clone()).unwrap();
        assert_eq!(
            decision(&permit, &attempt).await,
            Ok(DispatchDecision::AuthorizedFresh)
        );
        for field in 0..7 {
            let mut spec = DispatchRouteSpec {
                provider_id: "fixture".into(),
                provider_instance_id: "instance-canary".into(),
                account_identity: "account-canary".into(),
                model_id: "model-fixture".into(),
                endpoint_identity: "endpoint-fixture".into(),
                effort: "medium".into(),
                policy_revision: 1,
            };
            match field {
                0 => spec.provider_id = "other-provider".into(),
                1 => spec.provider_instance_id = "other-instance".into(),
                2 => spec.account_identity = "other-account".into(),
                3 => spec.model_id = "other-model".into(),
                4 => spec.endpoint_identity = "other-endpoint".into(),
                5 => spec.effort = "low".into(),
                _ => spec.policy_revision = 2,
            }
            let changed = DispatchRoute::new(spec).unwrap();
            let attempted =
                DispatchAttempt::new(changed.clone(), id.clone(), body.clone()).unwrap();
            assert_eq!(
                decision(&permit, &attempted).await,
                Err(DispatchDenied::RouteMismatch)
            );
            assert_eq!(
                decision(&proof.dispatch_permit(&ctx, changed), &attempted).await,
                Err(DispatchDenied::AlreadyConsumed)
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn exact_retry_revalidates_workspace_and_target_after_first_dispatch() {
        let root = tempfile::tempdir().unwrap();
        let (ctx, proof) = verified_proof(root.path()).await;
        let route = test_route();
        let attempt = DispatchAttempt::new(
            route.clone(),
            LogicalRequestId::fresh(),
            sha256_hex(b"body"),
        )
        .unwrap();
        let permit = proof.dispatch_permit(&ctx, route);
        assert_eq!(
            decision(&permit, &attempt).await,
            Ok(DispatchDecision::AuthorizedFresh)
        );
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, b"after").unwrap();
        std::fs::rename(&replacement, root.path().join("target")).unwrap();
        assert_eq!(
            decision(&permit, &attempt).await,
            Err(DispatchDenied::ProofInvalidated)
        );

        let root = tempfile::tempdir().unwrap();
        let (ctx, proof) = verified_proof(root.path()).await;
        let route = test_route();
        let attempt = DispatchAttempt::new(
            route.clone(),
            LogicalRequestId::fresh(),
            sha256_hex(b"body"),
        )
        .unwrap();
        let permit = proof.dispatch_permit(&ctx, route);
        assert_eq!(
            decision(&permit, &attempt).await,
            Ok(DispatchDecision::AuthorizedFresh)
        );
        let ticket = mutation::register_pending(
            mutation::scope_for_workspace(root.path(), "foreign"),
            "fixture",
        )
        .unwrap();
        assert_eq!(
            decision(&permit, &attempt).await,
            Err(DispatchDenied::ProofInvalidated)
        );
        ticket.confirm_terminal().unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_and_wrong_scope_do_not_consume_dispatch_authority() {
        let root = tempfile::tempdir().unwrap();
        let (ctx, proof) = verified_proof(root.path()).await;
        let route = test_route();
        let attempt = DispatchAttempt::new(
            route.clone(),
            LogicalRequestId::fresh(),
            sha256_hex(b"body"),
        )
        .unwrap();
        let mut foreign = ctx.clone();
        foreign.session_id = "foreign".into();
        assert_eq!(
            decision(&proof.dispatch_permit(&foreign, route.clone()), &attempt).await,
            Err(DispatchDenied::ProofInvalidated)
        );
        assert!(proof.bundle.state.dispatch_use.lock().is_none());
        ctx.cancel_token.cancel();
        assert_eq!(
            decision(&proof.dispatch_permit(&ctx, route), &attempt).await,
            Err(DispatchDenied::Cancelled)
        );
        assert!(proof.bundle.state.dispatch_use.lock().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn held_authorization_blocks_writers_and_concurrent_retry_until_send_drop() {
        use crate::{execute_authorized, FileReadTool, FileWriteTool};
        use serde_json::json;
        let root = tempfile::tempdir().unwrap();
        let (ctx, proof) = verified_proof(root.path()).await;
        let route = test_route();
        let attempt = DispatchAttempt::new(
            route.clone(),
            LogicalRequestId::fresh(),
            sha256_hex(b"body"),
        )
        .unwrap();
        let permit = proof.dispatch_permit(&ctx, route);
        let authorization = permit.authorize(&attempt).await.unwrap();
        assert_eq!(authorization.decision(), DispatchDecision::AuthorizedFresh);
        assert_eq!(
            permit.clone().authorize(&attempt).await.unwrap_err(),
            DispatchDenied::InFlight
        );
        let stamp = mutation::current_stamp(&ctx);
        let denied = execute_authorized(
            &FileWriteTool,
            json!({"file_path": "target", "content": "changed"}),
            &ctx,
        )
        .await;
        assert!(
            denied.is_error,
            "writer must not enter after proof validation"
        );
        assert_eq!(
            mutation::current_stamp(&ctx),
            stamp,
            "denial must not advance epoch"
        );
        let mut foreign = ctx.clone();
        foreign.session_id = "foreign-dispatch-writer".into();
        let denied = execute_authorized(
            &FileWriteTool,
            json!({"file_path": "other", "content": "changed"}),
            &foreign,
        )
        .await;
        assert!(denied.is_error);
        assert!(!root.path().join("other").exists());
        let read = execute_authorized(&FileReadTool, json!({"file_path": "target"}), &ctx).await;
        assert!(!read.is_error, "{}", read.content);
        assert!(read.content.contains("after"));
        // An unsuccessful transport also drops its authorization before retry.
        drop(authorization);
        let retry = permit.authorize(&attempt).await.unwrap();
        assert_eq!(retry.decision(), DispatchDecision::AuthorizedRetry);
        drop(retry);
        let result = execute_authorized(
            &FileWriteTool,
            json!({"file_path": "target", "content": "changed"}),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            permit.authorize(&attempt).await.unwrap_err(),
            DispatchDenied::ProofInvalidated
        );
    }

    #[test]
    fn receipt_debug_is_redacted() {
        let debug = format!("{:?}", ReceiptError::TargetChanged);
        assert!(!debug.contains("/"));
    }

    #[cfg(unix)]
    #[test]
    fn target_snapshot_uses_native_identity_and_exact_bytes() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("target");
        std::fs::write(&path, b"bytes").expect("write");
        let first = snapshot_sync(&path).expect("snapshot");
        let replacement = directory.path().join("replacement");
        std::fs::write(&replacement, b"bytes").expect("replacement");
        std::fs::rename(&replacement, &path).expect("atomic replace");
        let second = snapshot_sync(&path).expect("snapshot");
        assert_eq!(first.len, second.len);
        assert_eq!(first.hash, second.hash);
        assert_ne!((first.dev, first.ino), (0, 0));
        assert_ne!((first.dev, first.ino), (second.dev, second.ino));
    }

    #[cfg(unix)]
    #[test]
    fn final_symlink_is_not_a_verified_target() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().expect("tempdir");
        let source = directory.path().join("source");
        let link = directory.path().join("target");
        std::fs::write(&source, b"bytes").expect("write");
        symlink(&source, &link).expect("symlink");
        assert_eq!(snapshot_sync(&link), Err(ReceiptError::TargetUnavailable));
    }

    #[test]
    fn cloned_bundles_share_one_consumption_authority() {
        let directory = tempfile::tempdir().expect("tempdir");
        let snapshot = TargetSnapshot {
            parent: directory.path().to_path_buf(),
            basename: b"target".to_vec(),
            dev: 1,
            ino: 1,
            len: 0,
            hash: sha256_hex(&[]),
        };
        let mutation = MutationReceipt {
            scope: MutationScope {
                workspace: directory.path().to_path_buf(),
                session: "session".to_string(),
            },
            path: directory.path().join("target"),
            operation: TrustedCheckerOperation::Write,
            operation_id: "CANARY_OPERATION_ID".to_string(),
            expected_hash: "CANARY_HASH".to_string(),
            expected_len: 0,
            snapshot,
            stamp: MutationStamp {
                generation: "generation".to_string(),
                scope: MutationScope {
                    workspace: directory.path().to_path_buf(),
                    session: "session".to_string(),
                },
                epoch: 1,
            },
            state: Arc::new(ReceiptState {
                finalized: AtomicBool::new(false),
                dispatch_use: Mutex::new(None),
                dispatch_in_flight: AtomicBool::new(false),
            }),
        };
        let debug = format!("{:?}", mutation);
        assert!(!debug.contains("CANARY_OPERATION_ID"));
        assert!(!debug.contains("CANARY_HASH"));
        let first = ReceiptBundle::new(mutation);
        let second = first.clone();
        assert!(first
            .state
            .finalized
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok());
        assert!(second
            .state
            .finalized
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err());
    }
}
