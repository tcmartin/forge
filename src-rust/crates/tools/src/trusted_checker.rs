//! Host-owned registrations for narrowly scoped trusted checkers.
//!
//! This module deliberately contains no model-facing or serde representation.
//! A host constructs a catalog once, then shares its `Arc` through
//! [`crate::ToolContext`].  The catalog has no mutation or replacement API.

use claurst_core::crypto_utils::sha256_hex;
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

/// Maximum number of registrations in one immutable catalog.
pub const MAX_TRUSTED_CHECKERS: usize = 64;
/// Maximum byte length of an opaque registration identifier.
pub const MAX_TRUSTED_CHECKER_ID_BYTES: usize = 128;
/// Maximum byte length of a session identifier.
pub const MAX_TRUSTED_CHECKER_SESSION_BYTES: usize = 256;
/// Maximum byte length of a checker command.
pub const MAX_TRUSTED_CHECKER_COMMAND_BYTES: usize = 16 * 1024;
/// Maximum byte length of a canonical path stored in a registration.
pub const MAX_TRUSTED_CHECKER_PATH_BYTES: usize = 4096;
/// Maximum effective checker timeout.
pub const MAX_TRUSTED_CHECKER_TIMEOUT: Duration = Duration::from_secs(300);

/// The host-controlled source of a pinned registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustedCheckerSource {
    /// Explicitly supplied by a user at startup.
    UserStartup,
    /// Explicitly supplied by the embedding application.
    Application,
}

/// The only file mutation operations a checker may be associated with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustedCheckerOperation {
    Write,
    Edit,
}

/// Supported foreground checker interpreters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustedCheckerShell {
    Bash,
    PowerShell,
}

/// A finite foreground shell invocation supplied by the host.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TrustedCheckerInvocation {
    shell: TrustedCheckerShell,
    command: String,
    cwd: PathBuf,
    timeout: Duration,
    foreground: bool,
    // Captured when the host constructs its immutable registration. Never
    // source values from the model-managed persistent shell environment.
    environment: BTreeMap<OsString, OsString>,
    program: Option<PathBuf>,
}

impl fmt::Debug for TrustedCheckerInvocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedCheckerInvocation")
            .field("shell", &self.shell)
            .field("command", &"<redacted>")
            .field("cwd", &"<redacted>")
            .field("timeout", &self.timeout)
            .field("foreground", &self.foreground)
            .field("environment", &"<host-pinned>")
            .field("program", &"<host-pinned>")
            .finish()
    }
}

impl TrustedCheckerInvocation {
    /// Construct a foreground invocation. The working directory must be an
    /// absolute existing directory and the timeout must be finite and bounded.
    /// A minimal environment is pinned from the constructing host. Shell
    /// startup hooks and session exports are excluded.
    pub fn foreground(
        shell: TrustedCheckerShell,
        command: impl Into<String>,
        cwd: impl Into<PathBuf>,
        timeout: Duration,
    ) -> Result<Self, TrustedCheckerError> {
        let command = command.into();
        let cwd = canonical_directory(&cwd.into(), "checker cwd")?;
        validate_command(&command)?;
        validate_timeout(timeout)?;
        let mut environment = BTreeMap::new();
        for name in ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"] {
            if let Some(value) = std::env::var_os(name) {
                if value.as_encoded_bytes().len() > 16 * 1024 {
                    return Err(TrustedCheckerError::EnvironmentTooLong);
                }
                environment.insert(OsString::from(name), value);
            }
        }
        environment
            .entry(OsString::from("PATH"))
            .or_insert_with(|| OsString::from("/usr/bin:/bin"));
        let program = match shell {
            TrustedCheckerShell::Bash if cfg!(unix) => Path::new("/bin/bash").canonicalize().ok(),
            TrustedCheckerShell::PowerShell if cfg!(unix) => environment
                .get(OsStr::new("PATH"))
                .and_then(|path| {
                    std::env::split_paths(path)
                        .filter(|directory| directory.is_absolute())
                        .map(|directory| directory.join("pwsh"))
                        .find(|candidate| candidate.is_file())
                })
                .and_then(|path| path.canonicalize().ok()),
            _ => None,
        };
        if let Some(program) = &program {
            validate_path_length(program, "checker program")?;
        }
        Ok(Self {
            shell,
            command,
            cwd,
            timeout,
            foreground: true,
            environment,
            program,
        })
    }

    /// The interpreter selected by the host.
    pub fn shell(&self) -> TrustedCheckerShell {
        self.shell
    }

    /// The exact command string selected by the host.
    pub fn command(&self) -> &str {
        &self.command
    }

    /// The canonical effective working directory.
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// The finite effective timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Always true for a valid invocation. This is exposed for typed dispatch
    /// checks while construction intentionally offers no background variant.
    pub fn is_foreground(&self) -> bool {
        self.foreground
    }

    /// Whether the constructing host resolved a supported interpreter.
    /// Execution still validates process creation and owned completion.
    pub fn is_available(&self) -> bool {
        self.program.is_some()
    }

    pub(crate) fn environment(&self) -> &BTreeMap<OsString, OsString> {
        &self.environment
    }

    pub(crate) fn program(&self) -> Option<&Path> {
        self.program.as_deref()
    }
}

/// Raw host input used to create one validated registration.
///
/// This is intentionally not a serde or model-facing type. Callers must pass
/// it through [`TrustedCheckerRegistration::new`] before it can enter a
/// catalog.
#[derive(Clone)]
pub struct RegistrationSpec {
    pub id: String,
    pub revision: u64,
    pub source: TrustedCheckerSource,
    pub workspace: PathBuf,
    pub session_id: String,
    pub target: PathBuf,
    pub operation: TrustedCheckerOperation,
    pub invocation: TrustedCheckerInvocation,
}

impl fmt::Debug for RegistrationSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegistrationSpec")
            .field("id", &"<redacted>")
            .field("revision", &self.revision)
            .field("source", &self.source)
            .field("workspace", &"<redacted>")
            .field("session_id", &"<redacted>")
            .field("target", &"<redacted>")
            .field("operation", &self.operation)
            .field("invocation", &self.invocation)
            .finish()
    }
}

/// A host-owned, validated trusted checker registration.
///
/// Fields remain private so a caller cannot construct an unvalidated
/// descriptor or alter one after it has been inserted into a catalog.
#[derive(Clone, PartialEq, Eq)]
pub struct TrustedCheckerRegistration {
    id: String,
    revision: u64,
    source: TrustedCheckerSource,
    workspace: PathBuf,
    session_id: String,
    target: PathBuf,
    operation: TrustedCheckerOperation,
    invocation: TrustedCheckerInvocation,
    digest: String,
}

impl fmt::Debug for TrustedCheckerRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrustedCheckerRegistration")
            .field("id", &"<redacted>")
            .field("revision", &self.revision)
            .field("source", &self.source)
            .field("workspace", &"<redacted>")
            .field("session_id", &"<redacted>")
            .field("target", &"<redacted>")
            .field("operation", &self.operation)
            .field("invocation", &self.invocation)
            .field("digest", &self.digest)
            .finish()
    }
}

impl TrustedCheckerRegistration {
    /// Validate and canonicalize one host registration.
    pub fn new(spec: RegistrationSpec) -> Result<Self, TrustedCheckerError> {
        let RegistrationSpec {
            id,
            revision,
            source,
            workspace,
            session_id,
            target,
            operation,
            invocation,
        } = spec;
        validate_id(&id)?;
        if revision == 0 {
            return Err(TrustedCheckerError::InvalidRevision);
        }
        let workspace = canonical_directory(&workspace, "checker workspace")?;
        validate_session(&session_id)?;
        let target = canonical_target(&target)?;
        if !invocation.is_foreground() {
            return Err(TrustedCheckerError::BackgroundInvocation);
        }

        let mut registration = Self {
            id,
            revision,
            source,
            workspace,
            session_id,
            target,
            operation,
            invocation,
            digest: String::new(),
        };
        registration.digest = stable_digest(&registration);
        Ok(registration)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn source(&self) -> TrustedCheckerSource {
        self.source
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Canonical parent plus the exact target basename. The target itself may
    /// not exist yet because a write can be verified before its first creation.
    pub fn target(&self) -> &Path {
        &self.target
    }

    pub fn operation(&self) -> TrustedCheckerOperation {
        self.operation
    }

    pub fn invocation(&self) -> &TrustedCheckerInvocation {
        &self.invocation
    }

    /// Stable SHA-256 digest of the validated descriptor fields.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// Immutable catalog construction errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustedCheckerError {
    EmptyId,
    IdTooLong,
    InvalidRevision,
    EmptySession,
    SessionTooLong,
    EmptyCommand,
    CommandTooLong,
    InvalidTimeout,
    TimeoutTooLong,
    EnvironmentTooLong,
    InvalidPath { field: &'static str, reason: String },
    BackgroundInvocation,
    CapacityExceeded,
    DuplicateRegistration { id: String, revision: u64 },
}

impl fmt::Display for TrustedCheckerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => formatter.write_str("trusted checker id is empty"),
            Self::IdTooLong => formatter.write_str("trusted checker id is too long"),
            Self::InvalidRevision => {
                formatter.write_str("trusted checker revision must be positive")
            }
            Self::EmptySession => formatter.write_str("trusted checker session is empty"),
            Self::SessionTooLong => formatter.write_str("trusted checker session is too long"),
            Self::EmptyCommand => formatter.write_str("trusted checker command is empty"),
            Self::CommandTooLong => formatter.write_str("trusted checker command is too long"),
            Self::InvalidTimeout => formatter.write_str("trusted checker timeout must be positive"),
            Self::TimeoutTooLong => {
                formatter.write_str("trusted checker timeout exceeds the host bound")
            }
            Self::EnvironmentTooLong => {
                formatter.write_str("trusted checker environment exceeds the host bound")
            }
            Self::InvalidPath { field, reason } => write!(formatter, "invalid {field}: {reason}"),
            Self::BackgroundInvocation => {
                formatter.write_str("trusted checker invocation must be foreground")
            }
            Self::CapacityExceeded => {
                formatter.write_str("trusted checker catalog capacity exceeded")
            }
            Self::DuplicateRegistration { id, revision } => {
                write!(
                    formatter,
                    "trusted checker registration already exists: {id}@{revision}"
                )
            }
        }
    }
}

impl std::error::Error for TrustedCheckerError {}

/// Host-only builder for an immutable trusted checker catalog.
#[derive(Debug, Default)]
pub struct TrustedCheckerCatalogBuilder {
    registrations: Vec<TrustedCheckerRegistration>,
}

impl TrustedCheckerCatalogBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one validated host registration. Duplicate id/revision pairs are
    /// rejected and the catalog remains unchanged on failure.
    pub fn register(
        &mut self,
        registration: TrustedCheckerRegistration,
    ) -> Result<(), TrustedCheckerError> {
        if self.registrations.iter().any(|existing| {
            existing.id == registration.id && existing.revision == registration.revision
        }) {
            return Err(TrustedCheckerError::DuplicateRegistration {
                id: registration.id.clone(),
                revision: registration.revision,
            });
        }
        if self.registrations.len() >= MAX_TRUSTED_CHECKERS {
            return Err(TrustedCheckerError::CapacityExceeded);
        }
        self.registrations.push(registration);
        Ok(())
    }

    /// Freeze the builder into a read-only catalog.
    pub fn build(self) -> TrustedCheckerCatalog {
        TrustedCheckerCatalog {
            registrations: self.registrations,
        }
    }
}

/// Immutable, read-only host registration catalog.
#[derive(Debug, Clone, Default)]
pub struct TrustedCheckerCatalog {
    registrations: Vec<TrustedCheckerRegistration>,
}

impl TrustedCheckerCatalog {
    /// Return an empty catalog without enabling any checker implicitly.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }

    pub fn registrations(&self) -> impl Iterator<Item = &TrustedCheckerRegistration> {
        self.registrations.iter()
    }

    /// Exact field-for-field lookup. This has no mutation, proof, or receipt
    /// side effects and returns only a registration already owned by the
    /// immutable catalog.
    pub fn lookup(
        &self,
        request: &TrustedCheckerRegistration,
    ) -> Option<&TrustedCheckerRegistration> {
        self.registrations
            .iter()
            .find(|registration| *registration == request)
    }
}

fn validate_id(id: &str) -> Result<(), TrustedCheckerError> {
    if id.is_empty() {
        return Err(TrustedCheckerError::EmptyId);
    }
    if id.len() > MAX_TRUSTED_CHECKER_ID_BYTES {
        return Err(TrustedCheckerError::IdTooLong);
    }
    Ok(())
}

fn validate_session(session: &str) -> Result<(), TrustedCheckerError> {
    if session.is_empty() {
        return Err(TrustedCheckerError::EmptySession);
    }
    if session.len() > MAX_TRUSTED_CHECKER_SESSION_BYTES {
        return Err(TrustedCheckerError::SessionTooLong);
    }
    Ok(())
}

fn validate_command(command: &str) -> Result<(), TrustedCheckerError> {
    if command.is_empty() {
        return Err(TrustedCheckerError::EmptyCommand);
    }
    if command.len() > MAX_TRUSTED_CHECKER_COMMAND_BYTES {
        return Err(TrustedCheckerError::CommandTooLong);
    }
    Ok(())
}

fn validate_timeout(timeout: Duration) -> Result<(), TrustedCheckerError> {
    if timeout.is_zero() {
        return Err(TrustedCheckerError::InvalidTimeout);
    }
    if timeout > MAX_TRUSTED_CHECKER_TIMEOUT {
        return Err(TrustedCheckerError::TimeoutTooLong);
    }
    Ok(())
}

fn canonical_directory(path: &Path, field: &'static str) -> Result<PathBuf, TrustedCheckerError> {
    if !path.is_absolute() {
        return Err(TrustedCheckerError::InvalidPath {
            field,
            reason: "path must be absolute".to_string(),
        });
    }
    let metadata = std::fs::metadata(path).map_err(|error| TrustedCheckerError::InvalidPath {
        field,
        reason: error.to_string(),
    })?;
    if !metadata.is_dir() {
        return Err(TrustedCheckerError::InvalidPath {
            field,
            reason: "path must name a directory".to_string(),
        });
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| TrustedCheckerError::InvalidPath {
            field,
            reason: error.to_string(),
        })?;
    validate_path_length(&canonical, field)?;
    Ok(canonical)
}

fn canonical_target(path: &Path) -> Result<PathBuf, TrustedCheckerError> {
    if !path.is_absolute() {
        return Err(TrustedCheckerError::InvalidPath {
            field: "checker target",
            reason: "path must be absolute".to_string(),
        });
    }
    if matches!(
        path.components().next_back(),
        Some(Component::CurDir | Component::ParentDir)
    ) {
        return Err(TrustedCheckerError::InvalidPath {
            field: "checker target",
            reason: "target must end in an explicit basename".to_string(),
        });
    }
    let parent = path
        .parent()
        .ok_or_else(|| TrustedCheckerError::InvalidPath {
            field: "checker target",
            reason: "target has no parent".to_string(),
        })?;
    let basename = path
        .file_name()
        .ok_or_else(|| TrustedCheckerError::InvalidPath {
            field: "checker target",
            reason: "target has no basename".to_string(),
        })?;
    if basename == OsStr::new(".") || basename == OsStr::new("..") {
        return Err(TrustedCheckerError::InvalidPath {
            field: "checker target",
            reason: "target basename must name a file".to_string(),
        });
    }
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            return Err(TrustedCheckerError::InvalidPath {
                field: "checker target",
                reason: "target must not be a directory".to_string(),
            });
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(TrustedCheckerError::InvalidPath {
                field: "checker target",
                reason: error.to_string(),
            });
        }
    }
    let canonical_parent = canonical_directory(parent, "checker target parent")?;
    let target = canonical_parent.join(basename);
    validate_path_length(&target, "checker target")?;
    Ok(target)
}

fn validate_path_length(path: &Path, field: &'static str) -> Result<(), TrustedCheckerError> {
    if path.as_os_str().as_encoded_bytes().len() > MAX_TRUSTED_CHECKER_PATH_BYTES {
        return Err(TrustedCheckerError::InvalidPath {
            field,
            reason: "path is too long".to_string(),
        });
    }
    Ok(())
}

fn append_field(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn stable_digest(registration: &TrustedCheckerRegistration) -> String {
    let mut bytes = b"forge/trusted-checker/v1\0".to_vec();
    append_field(&mut bytes, registration.id.as_bytes());
    append_field(&mut bytes, &registration.revision.to_le_bytes());
    append_field(
        &mut bytes,
        match registration.source {
            TrustedCheckerSource::UserStartup => b"user_startup",
            TrustedCheckerSource::Application => b"application",
        },
    );
    append_field(
        &mut bytes,
        registration.workspace.as_os_str().as_encoded_bytes(),
    );
    append_field(&mut bytes, registration.session_id.as_bytes());
    append_field(
        &mut bytes,
        registration.target.as_os_str().as_encoded_bytes(),
    );
    append_field(
        &mut bytes,
        match registration.operation {
            TrustedCheckerOperation::Write => b"write",
            TrustedCheckerOperation::Edit => b"edit",
        },
    );
    append_field(
        &mut bytes,
        match registration.invocation.shell {
            TrustedCheckerShell::Bash => b"bash",
            TrustedCheckerShell::PowerShell => b"powershell",
        },
    );
    append_field(&mut bytes, registration.invocation.command.as_bytes());
    append_field(
        &mut bytes,
        registration.invocation.cwd.as_os_str().as_encoded_bytes(),
    );
    append_field(
        &mut bytes,
        &registration.invocation.timeout.as_nanos().to_le_bytes(),
    );
    append_field(&mut bytes, &[u8::from(registration.invocation.foreground)]);
    append_field(
        &mut bytes,
        registration
            .invocation
            .program
            .as_deref()
            .map_or(&[][..], |path| path.as_os_str().as_encoded_bytes()),
    );
    for (name, value) in &registration.invocation.environment {
        append_field(&mut bytes, name.as_encoded_bytes());
        append_field(&mut bytes, value.as_encoded_bytes());
    }
    sha256_hex(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn spec(root: &Path, target: &Path, command: &str) -> RegistrationSpec {
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            command,
            root,
            Duration::from_secs(5),
        )
        .expect("valid invocation");
        RegistrationSpec {
            id: "checker".to_string(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.to_path_buf(),
            session_id: "session-a".to_string(),
            target: target.to_path_buf(),
            operation: TrustedCheckerOperation::Write,
            invocation,
        }
    }

    fn registration(root: &Path, target: &Path, command: &str) -> TrustedCheckerRegistration {
        TrustedCheckerRegistration::new(spec(root, target, command)).expect("valid registration")
    }

    #[test]
    fn exact_lookup_matches_all_bound_fields() {
        let root = tempdir().expect("temp root");
        let target = root.path().join("result.txt");
        let base = spec(root.path(), &target, "printf '%s' '$HOME'");
        let first = TrustedCheckerRegistration::new(base.clone()).expect("valid registration");
        let mut builder = TrustedCheckerCatalogBuilder::new();
        builder.register(first.clone()).expect("register");
        let catalog = builder.build();
        assert_eq!(catalog.lookup(&first), Some(&first));
        assert!(TrustedCheckerCatalog::empty().lookup(&first).is_none());

        let alternate_root = tempdir().expect("alternate root");
        let mut candidates = Vec::new();
        let mut candidate = base.clone();
        candidate.id = "other-checker".to_string();
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.revision = 2;
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.source = TrustedCheckerSource::UserStartup;
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.workspace = alternate_root.path().to_path_buf();
        candidate.invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            "printf '%s' '$HOME'",
            root.path(),
            Duration::from_secs(5),
        )
        .expect("valid invocation");
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.session_id = "session-b".to_string();
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.target = root.path().join("other.txt");
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.operation = TrustedCheckerOperation::Edit;
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::PowerShell,
            "printf '%s' '$HOME'",
            root.path(),
            Duration::from_secs(5),
        )
        .expect("valid invocation");
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            "printf '%s' '$HOME'",
            alternate_root.path(),
            Duration::from_secs(5),
        )
        .expect("valid invocation");
        candidates.push(candidate);
        let mut candidate = base.clone();
        candidate.invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            "printf '%s' '$HOME'",
            root.path(),
            Duration::from_secs(6),
        )
        .expect("valid invocation");
        candidates.push(candidate);
        let mut candidate = base;
        candidate.invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            "printf '%s' '$PATH'",
            root.path(),
            Duration::from_secs(5),
        )
        .expect("valid invocation");
        candidates.push(candidate);

        for candidate in candidates {
            let registration = TrustedCheckerRegistration::new(candidate)
                .expect("candidate should remain structurally valid");
            assert!(catalog.lookup(&registration).is_none());
        }
    }

    #[test]
    fn canonicalizes_workspace_cwd_and_symlink_parent_for_missing_target() {
        let root = tempdir().expect("temp root");
        let real_parent = root.path().join("real");
        std::fs::create_dir(&real_parent).expect("real parent");
        let link_parent = root.path().join("link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real_parent, &link_parent).expect("parent symlink");
        #[cfg(not(unix))]
        let link_parent = real_parent.clone();

        let target = link_parent.join("new.txt");
        let entry = registration(root.path(), &target, "true");
        assert_eq!(
            entry.workspace(),
            root.path().canonicalize().expect("canonical root")
        );
        assert_eq!(
            entry.invocation().cwd(),
            root.path().canonicalize().expect("canonical root")
        );
        assert_eq!(
            entry.target(),
            &real_parent
                .canonicalize()
                .expect("canonical parent")
                .join("new.txt")
        );
        assert!(!entry.target().exists());
    }

    #[test]
    fn rejects_directory_and_dot_targets() {
        let root = tempdir().expect("temp root");
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            "true",
            root.path(),
            Duration::from_secs(1),
        )
        .expect("valid invocation");
        let directory = root.path().join("directory");
        std::fs::create_dir(&directory).expect("target directory");
        let directory_error = TrustedCheckerRegistration::new(RegistrationSpec {
            id: "checker".to_string(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.path().to_path_buf(),
            session_id: "session".to_string(),
            target: directory,
            operation: TrustedCheckerOperation::Write,
            invocation: invocation.clone(),
        })
        .unwrap_err();
        assert!(matches!(
            directory_error,
            TrustedCheckerError::InvalidPath { .. }
        ));

        let dot_error = TrustedCheckerRegistration::new(RegistrationSpec {
            id: "checker".to_string(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.path().to_path_buf(),
            session_id: "session".to_string(),
            target: root.path().join("."),
            operation: TrustedCheckerOperation::Write,
            invocation,
        })
        .unwrap_err();
        assert!(matches!(dot_error, TrustedCheckerError::InvalidPath { .. }));
    }

    #[test]
    fn debug_output_redacts_commands_and_paths() {
        let root = tempdir().expect("temp root");
        let target = root.path().join("CANARY_TARGET_SECRET");
        let command = "CANARY_COMMAND_SECRET";
        let session = "CANARY_SESSION_SECRET";
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            command,
            root.path(),
            Duration::from_secs(1),
        )
        .expect("valid invocation");
        let spec = RegistrationSpec {
            id: "CANARY_ID_SECRET".to_string(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.path().to_path_buf(),
            session_id: session.to_string(),
            target: target.clone(),
            operation: TrustedCheckerOperation::Write,
            invocation: invocation.clone(),
        };
        let registration = TrustedCheckerRegistration::new(spec.clone()).expect("registration");
        for output in [
            format!("{registration:?}"),
            format!("{invocation:?}"),
            format!("{spec:?}"),
        ] {
            assert!(!output.contains(command));
            assert!(!output.contains(session));
            assert!(!output.contains("CANARY_ID_SECRET"));
            assert!(!output.contains("CANARY_TARGET_SECRET"));
            assert!(!output.contains(&root.path().display().to_string()));
        }
    }

    #[cfg(unix)]
    #[test]
    fn digest_and_bounds_preserve_non_utf8_unix_path_bytes() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        fn in_memory_registration(path: PathBuf) -> TrustedCheckerRegistration {
            let invocation = TrustedCheckerInvocation {
                shell: TrustedCheckerShell::Bash,
                command: "true".to_string(),
                cwd: path.clone(),
                timeout: Duration::from_secs(1),
                foreground: true,
                environment: BTreeMap::new(),
                program: None,
            };
            let mut registration = TrustedCheckerRegistration {
                id: "checker".to_string(),
                revision: 1,
                source: TrustedCheckerSource::Application,
                workspace: path.clone(),
                session_id: "session".to_string(),
                target: path,
                operation: TrustedCheckerOperation::Write,
                invocation,
                digest: String::new(),
            };
            registration.digest = stable_digest(&registration);
            registration
        }

        let first_path = PathBuf::from(OsString::from_vec(b"/prefix/\x80".to_vec()));
        let second_path = PathBuf::from(OsString::from_vec(b"/prefix/\x81".to_vec()));
        assert_eq!(first_path.to_string_lossy(), second_path.to_string_lossy());
        assert_ne!(
            first_path.as_os_str().as_encoded_bytes(),
            second_path.as_os_str().as_encoded_bytes()
        );
        assert!(validate_path_length(&first_path, "test").is_ok());
        assert!(validate_path_length(&second_path, "test").is_ok());
        let first = in_memory_registration(first_path);
        let second = in_memory_registration(second_path);
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn rejects_empty_revision_relative_paths_and_unbounded_values() {
        let root = tempdir().expect("temp root");
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::PowerShell,
            "Write-Output ok",
            root.path(),
            Duration::from_secs(1),
        )
        .expect("valid invocation");
        assert_eq!(
            TrustedCheckerRegistration::new(RegistrationSpec {
                id: String::new(),
                revision: 1,
                source: TrustedCheckerSource::UserStartup,
                workspace: root.path().to_path_buf(),
                session_id: "session".to_string(),
                target: root.path().join("out"),
                operation: TrustedCheckerOperation::Edit,
                invocation: invocation.clone(),
            })
            .unwrap_err(),
            TrustedCheckerError::EmptyId
        );
        assert_eq!(
            TrustedCheckerRegistration::new(RegistrationSpec {
                id: "checker".to_string(),
                revision: 0,
                source: TrustedCheckerSource::UserStartup,
                workspace: root.path().to_path_buf(),
                session_id: "session".to_string(),
                target: root.path().join("out"),
                operation: TrustedCheckerOperation::Edit,
                invocation: invocation.clone(),
            })
            .unwrap_err(),
            TrustedCheckerError::InvalidRevision
        );
        assert!(matches!(
            TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                "true",
                "relative",
                Duration::from_secs(1),
            ),
            Err(TrustedCheckerError::InvalidPath { .. })
        ));
        assert_eq!(
            TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                "true",
                root.path(),
                Duration::ZERO,
            )
            .unwrap_err(),
            TrustedCheckerError::InvalidTimeout
        );
        assert_eq!(
            TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                "true",
                root.path(),
                MAX_TRUSTED_CHECKER_TIMEOUT + Duration::from_nanos(1),
            )
            .unwrap_err(),
            TrustedCheckerError::TimeoutTooLong
        );
    }

    #[test]
    fn duplicate_and_capacity_are_rejected_without_mutating_frozen_catalog() {
        let root = tempdir().expect("temp root");
        let mut builder = TrustedCheckerCatalogBuilder::new();
        let first = registration(root.path(), &root.path().join("a"), "true");
        builder.register(first.clone()).expect("first");
        assert!(matches!(
            builder.register(first),
            Err(TrustedCheckerError::DuplicateRegistration { .. })
        ));

        for index in 1..MAX_TRUSTED_CHECKERS {
            let invocation = TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                format!("printf {index}"),
                root.path(),
                Duration::from_secs(1),
            )
            .expect("valid invocation");
            let entry = TrustedCheckerRegistration::new(RegistrationSpec {
                id: format!("checker-{index}"),
                revision: 1,
                source: TrustedCheckerSource::Application,
                workspace: root.path().to_path_buf(),
                session_id: "session".to_string(),
                target: root.path().join(format!("target-{index}")),
                operation: TrustedCheckerOperation::Write,
                invocation,
            })
            .expect("valid registration");
            builder.register(entry).expect("within cap");
        }
        let overflow = TrustedCheckerRegistration::new(RegistrationSpec {
            id: "overflow".to_string(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.path().to_path_buf(),
            session_id: "session".to_string(),
            target: root.path().join("overflow"),
            operation: TrustedCheckerOperation::Write,
            invocation: TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                "true",
                root.path(),
                Duration::from_secs(1),
            )
            .expect("valid invocation"),
        })
        .expect("valid overflow registration");
        assert_eq!(
            builder.register(overflow).unwrap_err(),
            TrustedCheckerError::CapacityExceeded
        );
        assert_eq!(builder.build().len(), MAX_TRUSTED_CHECKERS);
    }

    #[test]
    fn digest_is_stable_and_changes_when_a_bound_field_changes() {
        let root = tempdir().expect("temp root");
        let first = registration(root.path(), &root.path().join("out"), "true");
        let same = registration(root.path(), &root.path().join("out"), "true");
        assert_eq!(first.digest(), same.digest());
        let changed = registration(root.path(), &root.path().join("out"), "false");
        assert_ne!(first.digest(), changed.digest());
        assert_eq!(first.digest().len(), 64);
    }

    #[test]
    fn environment_is_minimal_redacted_and_bound_to_registration() {
        let root = tempdir().expect("temp root");
        let first = registration(root.path(), &root.path().join("out"), "true");
        assert!(first
            .invocation
            .environment
            .contains_key(OsStr::new("PATH")));
        assert!(first.invocation.environment.keys().all(|name| {
            ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
                .iter()
                .any(|allowed| name == OsStr::new(allowed))
        }));
        let mut changed = first.clone();
        changed.invocation.environment.insert(
            OsString::from("HOME"),
            OsString::from("CANARY_PINNED_HOME_SECRET"),
        );
        changed.digest = stable_digest(&changed);
        assert_ne!(first.digest(), changed.digest());
        assert!(!format!("{changed:?}").contains("CANARY_PINNED_HOME_SECRET"));
    }
}
