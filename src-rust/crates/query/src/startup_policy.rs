//! Bounded, immutable host input for verified continuations.
//!
//! The document is read once at startup. Only validated non-serde values leave
//! this module; later session binding uses the cached descriptors and never
//! rereads a file that a running tool could have changed.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use claurst_api::dispatch_policy::{DispatchModelSpec, DispatchPolicy, EffortWireBinding};
use claurst_api::LlmProvider;
use claurst_core::dispatch::DispatchDenied;
use claurst_core::effort::EffortLevel;
use claurst_tools::{
    RegistrationSpec, TrustedCheckerCatalog, TrustedCheckerCatalogBuilder,
    TrustedCheckerInvocation, TrustedCheckerOperation, TrustedCheckerRegistration,
    TrustedCheckerShell, TrustedCheckerSource,
};
use serde::Deserialize;

use crate::effort_policy::{EffortPolicy, VerifiedEffortPolicy};

pub const MAX_STARTUP_POLICY_BYTES: u64 = 128 * 1024;
const MAX_MODELS: usize = 64;
const MAX_CHECKERS: usize = 64;
const MAX_PATH_BYTES: usize = 4096;
const VALIDATION_SESSION: &str = "host-startup-unbound";

/// Host-observed source used to resolve the current provider configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderSelectionSource {
    Configured,
    Runtime,
    Registry,
}

/// Categorical errors deliberately omit document values and filesystem paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StartupPolicyError {
    #[error("verified policy file is unavailable")]
    FileUnavailable,
    #[error("verified policy must be a regular file")]
    NotRegular,
    #[error("verified policy exceeds 128 KiB")]
    TooLarge,
    #[error("verified policy document is malformed")]
    MalformedDocument,
    #[error("verified policy schema version is unsupported")]
    UnsupportedVersion,
    #[error("verified policy effort or model registration is invalid")]
    InvalidPolicy,
    #[error("verified policy checker registration is invalid")]
    InvalidChecker,
    #[error("verified policy workspace is unavailable")]
    InvalidWorkspace,
    #[error("verified policy belongs to a different workspace")]
    WorkspaceMismatch,
    #[error("verified policy workspace identity is unsupported on this platform")]
    UnsupportedPlatform,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyDocument {
    schema_version: u32,
    revision: u64,
    #[serde(default = "default_provider_id")]
    provider: String,
    base: String,
    continuation: String,
    models: Vec<ModelDocument>,
    checkers: Vec<CheckerDocument>,
}

fn default_provider_id() -> String {
    "anthropic".to_string()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelDocument {
    id: String,
    efforts: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckerDocument {
    id: String,
    revision: u64,
    target: PathBuf,
    operation: OperationDocument,
    shell: ShellDocument,
    command: String,
    cwd: PathBuf,
    timeout_ms: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum OperationDocument {
    Write,
    Edit,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum ShellDocument {
    Bash,
    Powershell,
}

/// Validated startup declarations without live session authority.
pub struct LoadedVerifiedPolicy {
    workspace: PathBuf,
    workspace_identity: WorkspaceIdentity,
    effort: VerifiedEffortPolicy,
    provider_id: String,
    dispatch: Arc<DispatchPolicy>,
    templates: Vec<TrustedCheckerRegistration>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct WorkspaceIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl fmt::Debug for LoadedVerifiedPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LoadedVerifiedPolicy(..)")
    }
}

/// Host-bound values for one session in the originally selected workspace.
pub struct BoundVerifiedPolicy {
    effort: VerifiedEffortPolicy,
    provider_id: String,
    dispatch: Arc<DispatchPolicy>,
    catalog: Arc<TrustedCheckerCatalog>,
    bound_provider_cache: Mutex<Option<BoundProviderCacheEntry>>,
}

struct BoundProviderCacheEntry {
    fingerprint: String,
    source: ProviderSelectionSource,
    registry_owner: Option<Arc<dyn LlmProvider>>,
    provider: Arc<dyn LlmProvider>,
}

impl BoundProviderCacheEntry {
    fn matches(
        &self,
        fingerprint: &str,
        source: ProviderSelectionSource,
        registry_owner: Option<&Arc<dyn LlmProvider>>,
    ) -> bool {
        self.fingerprint == fingerprint
            && self.source == source
            && match source {
                ProviderSelectionSource::Registry => match (&self.registry_owner, registry_owner) {
                    (Some(cached), Some(current)) => Arc::ptr_eq(cached, current),
                    _ => false,
                },
                ProviderSelectionSource::Configured | ProviderSelectionSource::Runtime => true,
            }
    }
}

impl fmt::Debug for BoundVerifiedPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BoundVerifiedPolicy(..)")
    }
}

impl BoundVerifiedPolicy {
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn effort_policy(&self) -> EffortPolicy {
        EffortPolicy::Verified(self.effort)
    }

    pub fn dispatch_policy(&self) -> &Arc<DispatchPolicy> {
        &self.dispatch
    }

    pub fn checker_catalog(&self) -> &Arc<TrustedCheckerCatalog> {
        &self.catalog
    }

    pub(crate) fn clear_bound_provider_cache(&self) {
        if let Ok(mut cache) = self.bound_provider_cache.lock() {
            *cache = None;
        }
    }

    /// Bind a selected provider to the host policy, preserving a typed failure
    /// for verified callers. A raw provider remains usable by the historical
    /// fixed path only; callers must not treat it as policy-bound.
    pub(crate) fn try_bind_provider_for_selection(
        &self,
        provider: Arc<dyn LlmProvider>,
        source: ProviderSelectionSource,
        registry_owner: Option<Arc<dyn LlmProvider>>,
    ) -> Result<Arc<dyn LlmProvider>, DispatchDenied> {
        let Some(fingerprint) = provider.host_configuration_fingerprint() else {
            self.clear_bound_provider_cache();
            return provider.bind_dispatch_policy(Arc::clone(&self.dispatch));
        };

        let Ok(mut cache) = self.bound_provider_cache.lock() else {
            return Err(DispatchDenied::UnsupportedRoute);
        };
        if let Some(entry) = cache.as_ref() {
            if entry.matches(&fingerprint, source, registry_owner.as_ref()) {
                return Ok(Arc::clone(&entry.provider));
            }
        }

        let bound = match provider.bind_dispatch_policy(Arc::clone(&self.dispatch)) {
            Ok(bound) => bound,
            Err(error) => {
                *cache = None;
                return Err(error);
            }
        };
        *cache = Some(BoundProviderCacheEntry {
            fingerprint,
            source,
            registry_owner,
            provider: Arc::clone(&bound),
        });
        Ok(bound)
    }

    /// Historical fixed callers retain the raw provider on a binding failure.
    /// Verified callers use `try_bind_provider_for_selection` and must preserve
    /// the error instead of treating this fallback as policy-bound.
    #[cfg(test)]
    pub(crate) fn bind_provider_for_selection(
        &self,
        provider: Arc<dyn LlmProvider>,
        source: ProviderSelectionSource,
        registry_owner: Option<Arc<dyn LlmProvider>>,
    ) -> Arc<dyn LlmProvider> {
        let fallback = Arc::clone(&provider);
        self.try_bind_provider_for_selection(provider, source, registry_owner)
            .unwrap_or(fallback)
    }
}

impl LoadedVerifiedPolicy {
    /// Load an explicitly selected file, bounded independently of its metadata.
    /// Relative document paths are resolved within the host's workspace.
    pub fn load(path: &Path, workspace: &Path) -> Result<Self, StartupPolicyError> {
        let workspace = canonical_workspace(workspace)?;
        let workspace_identity = capture_workspace_identity(&workspace)?;
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            workspace.join(path)
        };
        let metadata = std::fs::metadata(&path).map_err(|_| StartupPolicyError::FileUnavailable)?;
        if !metadata.is_file() {
            return Err(StartupPolicyError::NotRegular);
        }
        if metadata.len() > MAX_STARTUP_POLICY_BYTES {
            return Err(StartupPolicyError::TooLarge);
        }
        let file = File::open(path).map_err(|_| StartupPolicyError::FileUnavailable)?;
        if !file
            .metadata()
            .map_err(|_| StartupPolicyError::FileUnavailable)?
            .is_file()
        {
            return Err(StartupPolicyError::NotRegular);
        }
        let mut bytes = Vec::new();
        file.take(MAX_STARTUP_POLICY_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| StartupPolicyError::FileUnavailable)?;
        if bytes.len() as u64 > MAX_STARTUP_POLICY_BYTES {
            return Err(StartupPolicyError::TooLarge);
        }
        let document: PolicyDocument =
            serde_json::from_slice(&bytes).map_err(|_| StartupPolicyError::MalformedDocument)?;
        let policy = Self::validate(document, workspace, workspace_identity)?;
        if capture_workspace_identity(&policy.workspace)? != policy.workspace_identity {
            return Err(StartupPolicyError::WorkspaceMismatch);
        }
        Ok(policy)
    }

    fn validate(
        document: PolicyDocument,
        workspace: PathBuf,
        workspace_identity: WorkspaceIdentity,
    ) -> Result<Self, StartupPolicyError> {
        if document.schema_version != 1 {
            return Err(StartupPolicyError::UnsupportedVersion);
        }
        if !matches!(document.provider.as_str(), "anthropic" | "openai") {
            return Err(StartupPolicyError::InvalidPolicy);
        }
        let base = canonical_effort(&document.base)?;
        let continuation = canonical_effort(&document.continuation)?;
        let effort = VerifiedEffortPolicy::new(base, continuation)
            .map_err(|_| StartupPolicyError::InvalidPolicy)?;
        if document.models.is_empty() || document.models.len() > MAX_MODELS {
            return Err(StartupPolicyError::InvalidPolicy);
        }
        let mut models = Vec::with_capacity(document.models.len());
        for model in document.models {
            if model.efforts.is_empty() || model.efforts.len() > 8 {
                return Err(StartupPolicyError::InvalidPolicy);
            }
            let levels: Vec<_> = model
                .efforts
                .iter()
                .map(|value| canonical_effort(value))
                .collect::<Result<_, _>>()?;
            if !levels.contains(&base) || !levels.contains(&continuation) {
                return Err(StartupPolicyError::InvalidPolicy);
            }
            let efforts = if document.provider == "openai" {
                levels
                    .into_iter()
                    .map(EffortWireBinding::openai_reasoning)
                    .collect()
            } else {
                levels
                    .into_iter()
                    .map(EffortWireBinding::anthropic_thinking)
                    .collect()
            };
            models.push(DispatchModelSpec {
                model_id: model.id,
                efforts,
            });
        }
        let dispatch = Arc::new(
            DispatchPolicy::new(document.revision, models)
                .map_err(|_| StartupPolicyError::InvalidPolicy)?,
        );
        if document.checkers.len() > MAX_CHECKERS {
            return Err(StartupPolicyError::InvalidChecker);
        }
        let mut builder = TrustedCheckerCatalogBuilder::new();
        for checker in document.checkers {
            let target = relative_workspace_path(&workspace, &checker.target)?;
            validate_target_leaf(&target)?;
            let cwd = relative_workspace_path(&workspace, &checker.cwd)?;
            let cwd = cwd
                .canonicalize()
                .map_err(|_| StartupPolicyError::InvalidChecker)?;
            if !cwd.starts_with(&workspace) {
                return Err(StartupPolicyError::InvalidChecker);
            }
            let invocation = TrustedCheckerInvocation::foreground(
                match checker.shell {
                    ShellDocument::Bash => TrustedCheckerShell::Bash,
                    ShellDocument::Powershell => TrustedCheckerShell::PowerShell,
                },
                checker.command,
                cwd,
                Duration::from_millis(checker.timeout_ms),
            )
            .map_err(|_| StartupPolicyError::InvalidChecker)?;
            if !invocation.is_available() {
                return Err(StartupPolicyError::InvalidChecker);
            }
            // This private placeholder validates immutable descriptors only.
            // It never leaves this object or enters a live ToolContext.
            let template = TrustedCheckerRegistration::new(RegistrationSpec {
                id: checker.id,
                revision: checker.revision,
                source: TrustedCheckerSource::UserStartup,
                workspace: workspace.clone(),
                session_id: VALIDATION_SESSION.into(),
                target,
                operation: match checker.operation {
                    OperationDocument::Write => TrustedCheckerOperation::Write,
                    OperationDocument::Edit => TrustedCheckerOperation::Edit,
                },
                invocation,
            })
            .map_err(|_| StartupPolicyError::InvalidChecker)?;
            if !template.target().starts_with(&workspace) {
                return Err(StartupPolicyError::InvalidChecker);
            }
            builder
                .register(template)
                .map_err(|_| StartupPolicyError::InvalidChecker)?;
        }
        let catalog = builder.build();
        Ok(Self {
            workspace,
            workspace_identity,
            effort,
            provider_id: document.provider,
            dispatch,
            templates: catalog.registrations().cloned().collect(),
        })
    }

    pub fn dispatch_policy(&self) -> &Arc<DispatchPolicy> {
        &self.dispatch
    }

    /// Return the immutable effort policy captured at startup.  Session
    /// binding only adds the session-scoped checker catalog; it never rereads
    /// or reconstructs the policy document.
    pub fn effort_policy(&self) -> EffortPolicy {
        EffortPolicy::Verified(self.effort)
    }

    /// Validate that a resumed session still refers to the exact workspace
    /// whose policy was loaded.  This is intentionally cheaper than building
    /// a checker catalog and is used before each new logical selection.
    pub fn validate_workspace(&self, workspace: &Path) -> Result<(), StartupPolicyError> {
        if canonical_workspace(workspace)? != self.workspace {
            return Err(StartupPolicyError::WorkspaceMismatch);
        }
        if capture_workspace_identity(&self.workspace)? != self.workspace_identity {
            return Err(StartupPolicyError::WorkspaceMismatch);
        }
        Ok(())
    }

    /// Bind cached declarations to a new host session in the same workspace.
    /// The interpreter and environment remain pinned from startup.
    pub fn bind_session(
        &self,
        workspace: &Path,
        session_id: &str,
    ) -> Result<BoundVerifiedPolicy, StartupPolicyError> {
        self.validate_workspace(workspace)?;
        let mut builder = TrustedCheckerCatalogBuilder::new();
        for template in &self.templates {
            validate_target_leaf(template.target())?;
            if template
                .invocation()
                .cwd()
                .canonicalize()
                .map_err(|_| StartupPolicyError::InvalidChecker)?
                != template.invocation().cwd()
            {
                return Err(StartupPolicyError::InvalidChecker);
            }
            let registration = TrustedCheckerRegistration::new(RegistrationSpec {
                id: template.id().to_owned(),
                revision: template.revision(),
                source: TrustedCheckerSource::UserStartup,
                workspace: self.workspace.clone(),
                session_id: session_id.to_owned(),
                target: template.target().to_path_buf(),
                operation: template.operation(),
                invocation: template.invocation().clone(),
            })
            .map_err(|_| StartupPolicyError::InvalidChecker)?;
            if registration.target() != template.target() {
                return Err(StartupPolicyError::InvalidChecker);
            }
            builder
                .register(registration)
                .map_err(|_| StartupPolicyError::InvalidChecker)?;
        }
        // Even an empty catalog must not accept an invalid host session.
        if session_id.is_empty() || session_id.len() > 256 {
            return Err(StartupPolicyError::InvalidChecker);
        }
        if capture_workspace_identity(&self.workspace)? != self.workspace_identity {
            return Err(StartupPolicyError::WorkspaceMismatch);
        }
        Ok(BoundVerifiedPolicy {
            effort: self.effort,
            provider_id: self.provider_id.clone(),
            dispatch: Arc::clone(&self.dispatch),
            catalog: Arc::new(builder.build()),
            bound_provider_cache: Mutex::new(None),
        })
    }
}

fn capture_workspace_identity(path: &Path) -> Result<WorkspaceIdentity, StartupPolicyError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::metadata(path).map_err(|_| StartupPolicyError::InvalidWorkspace)?;
        if !metadata.is_dir() {
            return Err(StartupPolicyError::InvalidWorkspace);
        }
        Ok(WorkspaceIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err(StartupPolicyError::UnsupportedPlatform)
    }
}

fn validate_target_leaf(target: &Path) -> Result<(), StartupPolicyError> {
    match std::fs::symlink_metadata(target) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(StartupPolicyError::InvalidChecker)
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(StartupPolicyError::InvalidChecker),
    }
}

fn canonical_workspace(workspace: &Path) -> Result<PathBuf, StartupPolicyError> {
    if !workspace.is_absolute() {
        return Err(StartupPolicyError::InvalidWorkspace);
    }
    let canonical = workspace
        .canonicalize()
        .map_err(|_| StartupPolicyError::InvalidWorkspace)?;
    if !canonical.is_dir() || canonical.as_os_str().as_encoded_bytes().len() > MAX_PATH_BYTES {
        return Err(StartupPolicyError::InvalidWorkspace);
    }
    Ok(canonical)
}

fn relative_workspace_path(
    workspace: &Path,
    relative: &Path,
) -> Result<PathBuf, StartupPolicyError> {
    if relative.as_os_str().is_empty()
        || relative.as_os_str().as_encoded_bytes().len() > MAX_PATH_BYTES
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err(StartupPolicyError::InvalidChecker);
    }
    Ok(workspace.join(relative))
}

fn canonical_effort(value: &str) -> Result<EffortLevel, StartupPolicyError> {
    EffortLevel::from_str(value)
        .filter(|level| level.as_str() == value)
        .ok_or(StartupPolicyError::InvalidPolicy)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use claurst_api::{
        LlmProvider, OpenAiProvider, ProviderCapabilities, ProviderError, ProviderRequest,
        ProviderResponse, ProviderStatus, ResponseContinuationSupport, StreamEvent,
        SystemPromptStyle,
    };
    use claurst_core::dispatch::{DispatchRoute, DispatchRouteSpec};
    use claurst_core::provider_id::ProviderId;
    use futures::Stream;
    use serde_json::{json, Value};
    use std::pin::Pin;
    use tempfile::TempDir;

    fn fixture() -> (TempDir, PathBuf, Value) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("target"), "before").unwrap();
        let document = json!({
            "schema_version":1, "revision":7,
            "base":"high", "continuation":"medium",
            "models":[{"id":"exact-model", "efforts":["high","medium"]}],
            "checkers":[{
                "id":"target-check", "revision":3, "target":"target",
                "operation":"edit", "shell":"bash", "command":"test -s target",
                "cwd":".", "timeout_ms":1000
            }]
        });
        (dir, root, document)
    }

    fn load(root: &Path, document: &Value) -> Result<LoadedVerifiedPolicy, StartupPolicyError> {
        std::fs::write(
            root.join("policy.json"),
            serde_json::to_vec(document).unwrap(),
        )
        .unwrap();
        LoadedVerifiedPolicy::load(Path::new("policy.json"), root)
    }

    struct RouteButCannotBindProvider {
        id: ProviderId,
        fingerprint: Option<String>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for RouteButCannotBindProvider {
        fn id(&self) -> &ProviderId {
            &self.id
        }

        fn name(&self) -> &str {
            "route-but-cannot-bind-test"
        }

        async fn create_message(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            panic!("binding fixture never dispatches inference")
        }

        async fn create_message_stream(
            &self,
            _request: ProviderRequest,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
            panic!("binding fixture never dispatches inference")
        }

        async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
            Ok(ProviderStatus::Healthy)
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                streaming: true,
                tool_calling: true,
                thinking: true,
                image_input: false,
                pdf_input: false,
                audio_input: false,
                video_input: false,
                caching: false,
                structured_output: false,
                system_prompt_style: SystemPromptStyle::TopLevel,
            }
        }

        fn response_continuation_support(&self) -> ResponseContinuationSupport {
            ResponseContinuationSupport::FullContextFallback
        }

        fn host_configuration_fingerprint(&self) -> Option<String> {
            self.fingerprint.clone()
        }

        fn dispatch_route(
            &self,
            model: &str,
            effort: EffortLevel,
        ) -> Result<DispatchRoute, DispatchDenied> {
            DispatchRoute::new(DispatchRouteSpec {
                provider_id: self.id.to_string(),
                provider_instance_id: "unbound-instance".into(),
                account_identity: "unbound-account".into(),
                model_id: model.into(),
                endpoint_identity: "unbound-endpoint".into(),
                effort: effort.as_str().into(),
                policy_revision: 7,
            })
            .map_err(|_| DispatchDenied::UnsupportedRoute)
        }

        fn bind_dispatch_policy(
            &self,
            _policy: Arc<claurst_api::dispatch_policy::DispatchPolicy>,
        ) -> Result<Arc<dyn LlmProvider>, DispatchDenied> {
            Err(DispatchDenied::UnsupportedRoute)
        }
    }

    fn rejecting_provider(fingerprint: Option<&str>) -> Arc<dyn LlmProvider> {
        Arc::new(RouteButCannotBindProvider {
            id: ProviderId::new("rejecting-bind"),
            fingerprint: fingerprint.map(str::to_string),
        })
    }

    #[test]
    fn failed_binding_is_typed_for_fingerprinted_and_unfingerprinted_routes() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let bound = loaded.bind_session(&root, "binding-failure").unwrap();

        for fingerprint in [Some("host-a"), None] {
            let provider = rejecting_provider(fingerprint);
            assert!(provider
                .dispatch_route("exact-model", EffortLevel::High)
                .is_ok());
            assert_eq!(
                bound
                    .try_bind_provider_for_selection(
                        Arc::clone(&provider),
                        ProviderSelectionSource::Configured,
                        None,
                    )
                    .err(),
                Some(DispatchDenied::UnsupportedRoute)
            );
            // The legacy fixed path still receives the raw provider; only
            // verified route/proof eligibility is denied by the resolver.
            assert!(bound
                .bind_provider_for_selection(provider, ProviderSelectionSource::Configured, None,)
                .dispatch_route("exact-model", EffortLevel::High)
                .is_ok());
        }
    }

    #[test]
    fn poisoned_binding_cache_fails_closed_without_claiming_a_bound_route() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let bound = loaded
            .bind_session(&root, "poisoned-binding-cache")
            .unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = bound.bound_provider_cache.lock().unwrap();
            panic!("poison test cache");
        }));
        assert_eq!(
            bound
                .try_bind_provider_for_selection(
                    rejecting_provider(Some("host-a")),
                    ProviderSelectionSource::Configured,
                    None,
                )
                .err(),
            Some(DispatchDenied::UnsupportedRoute)
        );
    }

    #[test]
    fn cached_policy_rebinds_session_without_rereading_changed_file() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let first = loaded.bind_session(&root, "session-one").unwrap();
        std::fs::write(root.join("policy.json"), b"model-edited invalid policy").unwrap();
        let second = loaded.bind_session(&root, "session-two").unwrap();
        assert!(Arc::ptr_eq(
            first.dispatch_policy(),
            second.dispatch_policy()
        ));
        assert_eq!(
            second.effort_policy().verified_policy().unwrap().base(),
            EffortLevel::High
        );
        let first_registration = first.checker_catalog().registrations().next().unwrap();
        let second_registration = second.checker_catalog().registrations().next().unwrap();
        assert_eq!(first_registration.session_id(), "session-one");
        assert_eq!(second_registration.session_id(), "session-two");
        assert_eq!(
            second_registration.source(),
            TrustedCheckerSource::UserStartup
        );
        assert_eq!(
            first_registration.invocation(),
            second_registration.invocation()
        );
        assert_ne!(first_registration.digest(), second_registration.digest());
        assert_eq!(format!("{loaded:?}"), "LoadedVerifiedPolicy(..)");
        assert_eq!(format!("{second:?}"), "BoundVerifiedPolicy(..)");
    }

    #[test]
    fn different_workspace_and_invalid_sessions_reject() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        assert!(loaded.validate_workspace(&root).is_ok());
        let other = tempfile::tempdir().unwrap();
        assert_eq!(
            loaded.validate_workspace(other.path()).unwrap_err(),
            StartupPolicyError::WorkspaceMismatch
        );
        assert_eq!(
            loaded.bind_session(other.path(), "new").unwrap_err(),
            StartupPolicyError::WorkspaceMismatch
        );
        assert_eq!(
            loaded.bind_session(&root, "").unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
    }

    #[test]
    fn file_bounds_and_regular_file_requirement_are_local() {
        let (_dir, root, _) = fixture();
        assert_eq!(
            LoadedVerifiedPolicy::load(&root, &root).unwrap_err(),
            StartupPolicyError::NotRegular
        );
        let oversized = File::create(root.join("oversized")).unwrap();
        oversized.set_len(MAX_STARTUP_POLICY_BYTES + 1).unwrap();
        assert_eq!(
            LoadedVerifiedPolicy::load(Path::new("oversized"), &root).unwrap_err(),
            StartupPolicyError::TooLarge
        );
        std::fs::write(root.join("invalid"), b"{\"command\":\"secret-canary\"}").unwrap();
        let error = LoadedVerifiedPolicy::load(Path::new("invalid"), &root).unwrap_err();
        assert_eq!(error, StartupPolicyError::MalformedDocument);
        assert!(!error.to_string().contains("secret-canary"));
    }

    #[test]
    fn strict_schema_rejects_unknown_duplicate_and_unsupported_fields() {
        let (_dir, root, mut document) = fixture();
        document["unexpected"] = json!(true);
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::MalformedDocument
        );
        document.as_object_mut().unwrap().remove("unexpected");
        document["schema_version"] = json!(2);
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::UnsupportedVersion
        );
        let text = serde_json::to_string(&document)
            .unwrap()
            .replacen("{", "{\"revision\":8,", 1);
        std::fs::write(root.join("policy.json"), text).unwrap();
        assert_eq!(
            LoadedVerifiedPolicy::load(Path::new("policy.json"), &root).unwrap_err(),
            StartupPolicyError::MalformedDocument
        );
    }

    #[test]
    fn exact_distinct_efforts_and_model_registrations_are_required() {
        let (_dir, root, original) = fixture();
        for efforts in [
            json!(["high"]),
            json!(["high", "normal"]),
            json!(["high", "medium", "medium"]),
            json!(["high", "medium", "max", "ultracode"]),
        ] {
            let mut document = original.clone();
            document["models"][0]["efforts"] = efforts;
            assert_eq!(
                load(&root, &document).unwrap_err(),
                StartupPolicyError::InvalidPolicy
            );
        }
        let mut document = original.clone();
        document["models"]
            .as_array_mut()
            .unwrap()
            .push(original["models"][0].clone());
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidPolicy
        );
        document = original;
        document["continuation"] = json!("high");
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidPolicy
        );
    }

    #[test]
    fn descriptors_reject_escape_duplicates_background_and_unbounded_timeout() {
        let (_dir, root, original) = fixture();
        for target in ["../outside", "/outside", "sub/../../outside"] {
            let mut document = original.clone();
            document["checkers"][0]["target"] = json!(target);
            assert_eq!(
                load(&root, &document).unwrap_err(),
                StartupPolicyError::InvalidChecker
            );
        }
        let mut document = original.clone();
        document["checkers"]
            .as_array_mut()
            .unwrap()
            .push(original["checkers"][0].clone());
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
        document = original.clone();
        document["checkers"][0]["background"] = json!(true);
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::MalformedDocument
        );
        for timeout in [0, 300_001] {
            document = original.clone();
            document["checkers"][0]["timeout_ms"] = json!(timeout);
            assert_eq!(
                load(&root, &document).unwrap_err(),
                StartupPolicyError::InvalidChecker
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_descriptor_parent_cannot_escape_workspace() {
        let (_dir, root, mut document) = fixture();
        let other = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(other.path(), root.join("escape")).unwrap();
        document["checkers"][0]["target"] = json!("escape/new-file");
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
        document["checkers"][0]["target"] = json!("target");
        document["checkers"][0]["cwd"] = json!("escape");
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
    }

    #[cfg(unix)]
    #[test]
    fn target_symlink_is_rejected_at_load_and_session_rebind() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let other = tempfile::tempdir().unwrap();
        let outside = other.path().join("outside");
        std::fs::write(&outside, "outside").unwrap();
        std::fs::remove_file(root.join("target")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("target")).unwrap();
        assert_eq!(
            loaded.bind_session(&root, "later").unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
        assert_eq!(std::fs::read(outside).unwrap(), b"outside");
    }

    #[test]
    fn workspace_replacement_at_same_path_cannot_rebind() {
        let (_dir, root, document) = fixture();
        let workspace = root.join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("target"), "before").unwrap();
        let loaded = load(&workspace, &document).unwrap();
        std::fs::rename(&workspace, root.join("retired-workspace")).unwrap();
        std::fs::create_dir(&workspace).unwrap();
        std::fs::write(workspace.join("target"), "before").unwrap();
        assert_eq!(
            loaded.validate_workspace(&workspace).unwrap_err(),
            StartupPolicyError::WorkspaceMismatch
        );
        assert_eq!(
            loaded.bind_session(&workspace, "later").unwrap_err(),
            StartupPolicyError::WorkspaceMismatch
        );
    }

    #[test]
    fn unavailable_interpreter_rejects_during_loading() {
        let (_dir, root, mut document) = fixture();
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::PowerShell,
            "exit 0",
            &root,
            Duration::from_secs(1),
        )
        .unwrap();
        document["checkers"][0]["shell"] = json!("powershell");
        if invocation.is_available() {
            assert!(load(&root, &document).is_ok());
        } else {
            assert_eq!(
                load(&root, &document).unwrap_err(),
                StartupPolicyError::InvalidChecker
            );
        }
    }

    #[test]
    fn registration_counts_are_bounded() {
        let (_dir, root, mut document) = fixture();
        document["models"] = Value::Array(
            (0..65)
                .map(|index| json!({"id":format!("model-{index}"),"efforts":["high","medium"]}))
                .collect(),
        );
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidPolicy
        );
        let (_, _, original) = fixture();
        document = original;
        let checker = document["checkers"][0].clone();
        document["checkers"] = Value::Array(
            (0..65)
                .map(|index| {
                    let mut item = checker.clone();
                    item["id"] = json!(format!("check-{index}"));
                    item
                })
                .collect(),
        );
        assert_eq!(
            load(&root, &document).unwrap_err(),
            StartupPolicyError::InvalidChecker
        );
    }

    #[test]
    fn openai_same_source_and_configuration_reuses_one_bound_provider() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let bound = loaded.bind_session(&root, "cache-session").unwrap();

        let configured_first: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("same-key".to_string()));
        let configured = bound.bind_provider_for_selection(
            configured_first,
            ProviderSelectionSource::Configured,
            None,
        );
        let configured_second: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("same-key".to_string()));
        let configured_again = bound.bind_provider_for_selection(
            configured_second,
            ProviderSelectionSource::Configured,
            None,
        );
        assert!(Arc::ptr_eq(&configured, &configured_again));

        let runtime_first: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("same-key".to_string()));
        let runtime = bound.bind_provider_for_selection(
            runtime_first,
            ProviderSelectionSource::Runtime,
            None,
        );
        assert!(!Arc::ptr_eq(&configured, &runtime));
        let runtime_second: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("same-key".to_string()));
        let runtime_again = bound.bind_provider_for_selection(
            runtime_second,
            ProviderSelectionSource::Runtime,
            None,
        );
        assert!(Arc::ptr_eq(&runtime, &runtime_again));
    }

    #[test]
    fn provider_changes_rotate_and_failures_clear_the_single_entry() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let bound = loaded.bind_session(&root, "cache-session").unwrap();

        let first: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new("key-a".to_string()));
        let first =
            bound.bind_provider_for_selection(first, ProviderSelectionSource::Configured, None);
        let second: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new("key-b".to_string()));
        let second =
            bound.bind_provider_for_selection(second, ProviderSelectionSource::Configured, None);
        assert!(!Arc::ptr_eq(&first, &second));

        let third: Arc<dyn LlmProvider> = Arc::new(
            OpenAiProvider::new("key-b".to_string())
                .with_base_url("https://alternate.example".to_string()),
        );
        let third =
            bound.bind_provider_for_selection(third, ProviderSelectionSource::Configured, None);
        assert!(!Arc::ptr_eq(&second, &third));

        let back_to_a: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new("key-a".to_string()));
        let back_to_a =
            bound.bind_provider_for_selection(back_to_a, ProviderSelectionSource::Configured, None);
        assert!(!Arc::ptr_eq(&first, &back_to_a));

        let registry_owner: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("registry-key".to_string()));
        let registry_bound = bound.bind_provider_for_selection(
            Arc::clone(&registry_owner),
            ProviderSelectionSource::Registry,
            Some(Arc::clone(&registry_owner)),
        );
        let registry_again = bound.bind_provider_for_selection(
            Arc::new(OpenAiProvider::new("registry-key".to_string())),
            ProviderSelectionSource::Registry,
            Some(Arc::clone(&registry_owner)),
        );
        assert!(Arc::ptr_eq(&registry_bound, &registry_again));
        let replacement_owner: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("registry-key".to_string()));
        let registry_rotated = bound.bind_provider_for_selection(
            Arc::clone(&replacement_owner),
            ProviderSelectionSource::Registry,
            Some(replacement_owner),
        );
        assert!(!Arc::ptr_eq(&registry_bound, &registry_rotated));

        let failed: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::new(String::new()));
        let failed_result =
            bound.bind_provider_for_selection(failed, ProviderSelectionSource::Configured, None);
        let fresh_after_failure: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new("registry-key".to_string()));
        let fresh_after_failure = bound.bind_provider_for_selection(
            fresh_after_failure,
            ProviderSelectionSource::Configured,
            None,
        );
        assert!(!Arc::ptr_eq(&failed_result, &fresh_after_failure));
    }

    #[test]
    fn session_bound_provider_caches_are_not_shared() {
        let (_dir, root, document) = fixture();
        let loaded = load(&root, &document).unwrap();
        let first_session = loaded.bind_session(&root, "session-one").unwrap();
        let second_session = loaded.bind_session(&root, "session-two").unwrap();
        let first = first_session.bind_provider_for_selection(
            Arc::new(OpenAiProvider::new("same-key".to_string())),
            ProviderSelectionSource::Configured,
            None,
        );
        let second = second_session.bind_provider_for_selection(
            Arc::new(OpenAiProvider::new("same-key".to_string())),
            ProviderSelectionSource::Configured,
            None,
        );
        assert!(!Arc::ptr_eq(&first, &second));
    }
}
