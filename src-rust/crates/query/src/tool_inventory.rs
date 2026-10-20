//! Query-owned runtime inventory snapshots.
//!
//! A snapshot is the owning boundary between a host's policy-filtered tool
//! assembly and the query loop.  The query loop may continue to support the
//! historical borrowed tool slice; source-backed runs retain an immutable
//! `Arc` for the lifetime of the current request and replace it only between
//! completed batches.

use async_trait::async_trait;
use claurst_tools::{ChildWriterCapability, Tool, ToolCatalogScope, ToolContext, ToolEffect};
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{workspace_scope_id, HistoryIndex};

/// Monotonic host inventory generation.  A changed generation invalidates all
/// catalog and deferred-schema authority captured from the previous snapshot.
pub type InventoryGeneration = u64;

tokio::task_local! {
    static CURRENT_INVENTORY_SOURCE: Option<Arc<dyn ToolInventorySource>>;
}

/// Run one authorized tool under the source that produced its active
/// inventory.  AgentTool captures this task-local before it spawns a child;
/// ordinary callers remain compatible when no source is installed.
pub(crate) async fn with_inventory_source<F, T>(
    source: Option<Arc<dyn ToolInventorySource>>,
    future: F,
) -> T
where
    F: Future<Output = T>,
{
    CURRENT_INVENTORY_SOURCE.scope(source, future).await
}

pub(crate) fn current_inventory_source() -> Option<Arc<dyn ToolInventorySource>> {
    CURRENT_INVENTORY_SOURCE
        .try_with(|source| source.clone())
        .ok()
        .flatten()
}

/// Effect capability granted to a source-backed child inventory.
///
/// Names are not sufficient authority: a refreshed handle with the same name
/// must still satisfy the capability selected during child planning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChildInventoryCapability {
    /// Only tools which explicitly declare observation are eligible.
    #[default]
    ObservationOnly,
    /// Observation plus the bounded native filesystem writers confined to a
    /// successfully-created child worktree.
    ScopedFilesystemWriters,
}

/// The effective tool profile requested by a query boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolInventoryProfile {
    #[default]
    Interactive,
    Acp,
    /// `None` is used only while a child is being planned.  Once planning
    /// completes, the query stores `Some(names)`, where `Some([])` is an
    /// intentional deny-all result.
    AgentChild {
        allowed_tools: Option<Arc<[String]>>,
        capability: ChildInventoryCapability,
    },
    Cron,
}

impl ToolInventoryProfile {
    /// Whether a host-authorized handle may appear in this effective profile.
    /// A `None` child profile is used only before capability planning and lets
    /// the host return its candidate ceiling.  `Some([])` is deny-all.
    pub fn allows_tool(&self, name: &str) -> bool {
        match self {
            Self::AgentChild { allowed_tools, .. } => allowed_tools
                .as_ref()
                .is_none_or(|allowed| allowed.iter().any(|candidate| candidate == name)),
            Self::Interactive | Self::Acp | Self::Cron => true,
        }
    }

    /// Check both the selected names and the effect capability of a live
    /// handle. This must be applied again when a source refreshes handles.
    pub fn allows_handle(&self, tool: &dyn Tool) -> bool {
        if !self.allows_tool(tool.name()) {
            return false;
        }
        match self {
            Self::AgentChild { capability, .. } => match capability {
                ChildInventoryCapability::ObservationOnly => {
                    tool.effect() == ToolEffect::Observation
                }
                ChildInventoryCapability::ScopedFilesystemWriters => {
                    tool.effect() == ToolEffect::Observation
                        || (tool.effect() == ToolEffect::Mutation
                            && tool.child_writer_capability()
                                == ChildWriterCapability::ScopedFilesystem
                            && matches!(
                                tool.name(),
                                "Edit" | "Write" | "BatchEdit" | "ApplyPatch" | "NotebookEdit"
                            ))
                }
            },
            Self::Interactive | Self::Acp | Self::Cron => true,
        }
    }
}

/// Query-owned request for a fresh, scoped host inventory.
#[derive(Clone)]
pub struct InventoryRequest {
    workspace_root: PathBuf,
    session_id: String,
    profile: ToolInventoryProfile,
    history: Option<HistoryBindingRequest>,
    /// Captured from the live ToolContext.  Source admission requires this
    /// native scope rather than caller-supplied workspace/session labels.
    scope: ToolCatalogScope,
}

impl InventoryRequest {
    pub(crate) fn for_context_with_binding(
        ctx: &ToolContext,
        profile: ToolInventoryProfile,
        history_binding: Option<&HistoryBindingRequest>,
    ) -> Result<Self, InventoryError> {
        let scope = ToolCatalogScope::from_context(ctx)
            .map_err(|error| InventoryError::InvalidScope(error.to_string()))?;
        if history_binding.is_some_and(|binding| !binding.scope().matches_context(ctx)) {
            return Err(InventoryError::InvalidScope(
                "history binding scope no longer matches the query context".to_string(),
            ));
        }
        Self::from_scope(scope, profile, history_binding.cloned())
    }

    pub(crate) fn for_context(
        ctx: &ToolContext,
        profile: ToolInventoryProfile,
        history_index: Option<&Arc<parking_lot::Mutex<HistoryIndex>>>,
    ) -> Result<Self, InventoryError> {
        let scope = ToolCatalogScope::from_context(ctx)
            .map_err(|error| InventoryError::InvalidScope(error.to_string()))?;
        let history = history_index
            .map(|index| HistoryBindingRequest::from_index(Arc::clone(index), scope.clone()))
            .transpose()?;
        Self::from_scope(scope, profile, history)
    }

    fn from_scope(
        scope: ToolCatalogScope,
        profile: ToolInventoryProfile,
        history: Option<HistoryBindingRequest>,
    ) -> Result<Self, InventoryError> {
        let workspace_root = scope.workspace().to_path_buf();
        let session_id = scope.session_id().to_owned();
        Self::validate_history(&workspace_root, &session_id, &scope, history.as_ref())?;
        Ok(Self {
            workspace_root,
            session_id,
            profile,
            history,
            scope,
        })
    }

    fn validate_history(
        workspace_root: &std::path::Path,
        session_id: &str,
        scope: &ToolCatalogScope,
        history: Option<&HistoryBindingRequest>,
    ) -> Result<(), InventoryError> {
        if scope.workspace() != workspace_root || scope.session_id() != session_id {
            return Err(InventoryError::InvalidScope(
                "inventory scope labels do not match captured scope".to_string(),
            ));
        }
        if let Some(binding) = history {
            if binding.session_id != session_id
                || binding.canonical_workspace != workspace_root
                || binding.scope() != scope
            {
                return Err(InventoryError::InvalidScope(
                    "history binding does not match inventory scope".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn workspace_root(&self) -> &std::path::Path {
        &self.workspace_root
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn profile(&self) -> &ToolInventoryProfile {
        &self.profile
    }

    pub fn history(&self) -> Option<&HistoryBindingRequest> {
        self.history.as_ref()
    }

    /// Native workspace/session identity captured from the ToolContext.
    pub fn scope(&self) -> &ToolCatalogScope {
        &self.scope
    }
}

/// A durable history index bound to one workspace and session.
#[derive(Clone)]
pub struct HistoryBindingRequest {
    index: Arc<parking_lot::Mutex<HistoryIndex>>,
    canonical_workspace: PathBuf,
    session_id: String,
    scope: ToolCatalogScope,
}

impl HistoryBindingRequest {
    /// Bind an index to the native scope captured from the current query
    /// context.  The journal labels are checked against the stable workspace
    /// scope before the binding is accepted; callers cannot relabel a parent
    /// index as a child or cron workspace.
    pub(crate) fn from_index(
        index: Arc<parking_lot::Mutex<HistoryIndex>>,
        scope: ToolCatalogScope,
    ) -> Result<Self, InventoryError> {
        let journal_scope = {
            let guard = index.lock();
            guard.ledger().journal().scope().clone()
        };
        if journal_scope.workspace != workspace_scope_id(scope.workspace())
            || journal_scope.session != scope.session_id()
        {
            return Err(InventoryError::InvalidScope(
                "history index belongs to a different workspace or session".to_string(),
            ));
        }
        Ok(Self {
            index,
            canonical_workspace: scope.workspace().to_path_buf(),
            session_id: scope.session_id().to_owned(),
            scope,
        })
    }

    pub fn index(&self) -> &Arc<parking_lot::Mutex<HistoryIndex>> {
        &self.index
    }

    pub fn canonical_workspace(&self) -> &std::path::Path {
        &self.canonical_workspace
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn scope(&self) -> &ToolCatalogScope {
        &self.scope
    }
}

/// Immutable actual tool handles retained for one query boundary.
pub struct ToolInventorySnapshot {
    tools: Arc<Vec<Box<dyn Tool>>>,
    generation: InventoryGeneration,
    scope: ToolCatalogScope,
}

/// Cheap leaf-admission guard for a captured source-backed snapshot.
#[derive(Clone)]
pub(crate) struct InventoryGuard {
    source: Arc<dyn ToolInventorySource>,
    request: InventoryRequest,
    generation: InventoryGeneration,
}

impl InventoryGuard {
    pub(crate) fn new(
        source: Arc<dyn ToolInventorySource>,
        request: InventoryRequest,
        generation: InventoryGeneration,
    ) -> Self {
        Self {
            source,
            request,
            generation,
        }
    }

    pub(crate) fn is_current(&self, ctx: &ToolContext) -> bool {
        self.request.scope().matches_context(ctx)
            && self
                .source
                .current_generation(&self.request, ctx)
                .is_ok_and(|generation| generation == self.generation)
    }

    pub(crate) fn source(&self) -> Arc<dyn ToolInventorySource> {
        Arc::clone(&self.source)
    }
}

impl ToolInventorySnapshot {
    pub fn new(
        tools: Arc<Vec<Box<dyn Tool>>>,
        generation: InventoryGeneration,
        scope: ToolCatalogScope,
    ) -> Result<Arc<Self>, InventoryError> {
        if generation == 0 {
            return Err(InventoryError::InvalidGeneration);
        }
        Ok(Arc::new(Self {
            tools,
            generation,
            scope,
        }))
    }

    pub fn tools(&self) -> &[Box<dyn Tool>] {
        self.tools.as_slice()
    }

    pub fn tools_arc(&self) -> Arc<Vec<Box<dyn Tool>>> {
        Arc::clone(&self.tools)
    }

    pub fn generation(&self) -> InventoryGeneration {
        self.generation
    }

    pub fn scope(&self) -> &ToolCatalogScope {
        &self.scope
    }
}

/// Errors from host inventory assembly.  Messages are categorical and do not
/// expose credentials, routes, or tool input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InventoryError {
    InvalidScope(String),
    InvalidGeneration,
    Unavailable(String),
}

impl fmt::Display for InventoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScope(message) => write!(f, "invalid inventory scope: {message}"),
            Self::InvalidGeneration => f.write_str("invalid inventory generation"),
            Self::Unavailable(message) => write!(f, "inventory unavailable: {message}"),
        }
    }
}

impl std::error::Error for InventoryError {}

/// Host-owned source for fresh, policy-filtered tool snapshots.
// async_trait itself adds #[must_use] to the already-must-use boxed Future.
// Keep the same narrow macro lint exception used by the other async traits.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait ToolInventorySource: Send + Sync {
    /// Return the current immutable snapshot for this effective request.
    async fn snapshot(
        &self,
        request: &InventoryRequest,
        ctx: &ToolContext,
    ) -> Result<Arc<ToolInventorySnapshot>, InventoryError>;

    /// Cheap current-generation check used before reusing a captured snapshot.
    fn current_generation(
        &self,
        request: &InventoryRequest,
        ctx: &ToolContext,
    ) -> Result<InventoryGeneration, InventoryError>;
}

/// The query's compatibility bridge for manual borrowed tools and owning
/// source snapshots.  The owned Arc stays alive while its slice is borrowed.
pub(crate) enum CapturedInventory<'a> {
    Borrowed(&'a [Box<dyn Tool>]),
    Owned(Arc<ToolInventorySnapshot>),
}

impl<'a> CapturedInventory<'a> {
    pub(crate) fn borrowed(tools: &'a [Box<dyn Tool>]) -> Self {
        Self::Borrowed(tools)
    }

    pub(crate) fn owned(snapshot: Arc<ToolInventorySnapshot>) -> Self {
        Self::Owned(snapshot)
    }

    pub(crate) fn tools(&self) -> &[Box<dyn Tool>] {
        match self {
            Self::Borrowed(tools) => tools,
            Self::Owned(snapshot) => snapshot.tools(),
        }
    }

    pub(crate) fn generation(&self) -> Option<InventoryGeneration> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(snapshot) => Some(snapshot.generation()),
        }
    }

    pub(crate) fn scope(&self) -> Option<&ToolCatalogScope> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(snapshot) => Some(snapshot.scope()),
        }
    }

    pub(crate) fn same_handles(&self, other: &Arc<ToolInventorySnapshot>) -> bool {
        match self {
            Self::Borrowed(_) => false,
            Self::Owned(snapshot) => Arc::ptr_eq(&snapshot.tools, &other.tools),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use claurst_core::config::PermissionMode;
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
    use claurst_tools::{ToolEffect, ToolResult};
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicU64, Ordering};
    use tempfile::tempdir;

    struct AllowAll;

    impl PermissionHandler for AllowAll {
        fn check_permission(&self, _: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }

        fn request_permission(&self, _: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    struct InventoryTestTool;

    struct MutationInventoryTestTool;

    #[async_trait]
    impl Tool for InventoryTestTool {
        fn name(&self) -> &str {
            "InventoryTest"
        }

        fn description(&self) -> &str {
            "inventory test tool"
        }

        fn permission_level(&self) -> claurst_tools::PermissionLevel {
            claurst_tools::PermissionLevel::ReadOnly
        }

        fn effect(&self) -> ToolEffect {
            ToolEffect::Observation
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    #[async_trait]
    impl Tool for MutationInventoryTestTool {
        fn name(&self) -> &str {
            "Write"
        }

        fn description(&self) -> &str {
            "mutation inventory test tool"
        }

        fn permission_level(&self) -> claurst_tools::PermissionLevel {
            claurst_tools::PermissionLevel::ReadOnly
        }

        fn effect(&self) -> ToolEffect {
            ToolEffect::Mutation
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("mutated")
        }
    }

    fn context(workspace: PathBuf) -> ToolContext {
        ToolContext {
            working_dir: workspace,
            permission_mode: PermissionMode::BypassPermissions,
            permission_handler: Arc::new(AllowAll),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "inventory-test-session".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: claurst_core::config::Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            trusted_checker_catalog: Arc::new(claurst_tools::TrustedCheckerCatalog::empty()),
            catalog_view: None,
        }
    }

    #[test]
    fn child_capability_rejects_forged_writer_and_accepts_native_handle() {
        let mutation = MutationInventoryTestTool;
        let names: Arc<[String]> = Arc::from(vec!["Write".to_string()].into_boxed_slice());
        let observation_only = ToolInventoryProfile::AgentChild {
            allowed_tools: Some(Arc::clone(&names)),
            capability: ChildInventoryCapability::ObservationOnly,
        };
        let scoped_writers = ToolInventoryProfile::AgentChild {
            allowed_tools: Some(names),
            capability: ChildInventoryCapability::ScopedFilesystemWriters,
        };
        assert!(!observation_only.allows_handle(&mutation));
        assert!(!scoped_writers.allows_handle(&mutation));
        assert!(scoped_writers.allows_handle(&claurst_tools::FileWriteTool));
    }

    struct StaticSource {
        snapshot: Arc<ToolInventorySnapshot>,
        generation: AtomicU64,
    }

    #[async_trait]
    impl ToolInventorySource for StaticSource {
        async fn snapshot(
            &self,
            _: &InventoryRequest,
            _: &ToolContext,
        ) -> Result<Arc<ToolInventorySnapshot>, InventoryError> {
            Ok(Arc::clone(&self.snapshot))
        }

        fn current_generation(
            &self,
            _: &InventoryRequest,
            _: &ToolContext,
        ) -> Result<InventoryGeneration, InventoryError> {
            Ok(self.generation.load(Ordering::Acquire))
        }
    }

    #[tokio::test]
    async fn async_snapshot_reuses_exact_handles_and_generation() {
        let dir = tempdir().expect("temporary workspace");
        let ctx = context(dir.path().to_path_buf());
        let scope = ToolCatalogScope::from_context(&ctx).expect("scope");
        let tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(InventoryTestTool)]);
        let snapshot =
            ToolInventorySnapshot::new(Arc::clone(&tools), 7, scope).expect("valid snapshot");
        let source = StaticSource {
            snapshot: Arc::clone(&snapshot),
            generation: AtomicU64::new(7),
        };
        let request = InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None)
            .expect("request");

        let first = source.snapshot(&request, &ctx).await.expect("snapshot");
        let second = source.snapshot(&request, &ctx).await.expect("snapshot");
        assert!(Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first.tools, &tools));
        assert_eq!(source.current_generation(&request, &ctx).unwrap(), 7);
    }

    #[test]
    fn inventory_guard_rejects_scope_replacement_and_generation_change() {
        let dir = tempdir().expect("temporary workspace");
        let other = tempdir().expect("replacement workspace");
        let ctx = context(dir.path().to_path_buf());
        let scope = ToolCatalogScope::from_context(&ctx).expect("scope");
        let tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(InventoryTestTool)]);
        let snapshot =
            ToolInventorySnapshot::new(Arc::clone(&tools), 7, scope).expect("valid snapshot");
        let source = Arc::new(StaticSource {
            snapshot,
            generation: AtomicU64::new(7),
        });
        let request = InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None)
            .expect("request");
        let guard = InventoryGuard::new(source.clone(), request, 7);
        assert!(guard.is_current(&ctx));
        assert!(!guard.is_current(&context(other.path().to_path_buf())));
        source.generation.store(8, Ordering::Release);
        assert!(!guard.is_current(&ctx));
    }

    #[test]
    fn unchanged_generation_rejects_replaced_handle_arc() {
        let dir = tempdir().expect("temporary workspace");
        let ctx = context(dir.path().to_path_buf());
        let scope = ToolCatalogScope::from_context(&ctx).expect("scope");
        let first_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(InventoryTestTool)]);
        let replacement_tools: Arc<Vec<Box<dyn Tool>>> =
            Arc::new(vec![Box::new(InventoryTestTool)]);
        let first = ToolInventorySnapshot::new(Arc::clone(&first_tools), 11, scope.clone())
            .expect("first snapshot");
        let replacement =
            ToolInventorySnapshot::new(replacement_tools, 11, scope).expect("replacement snapshot");
        let captured = CapturedInventory::owned(first);
        assert!(!captured.same_handles(&replacement));
    }

    #[test]
    fn history_binding_must_match_inventory_scope() {
        let dir = tempdir().expect("temporary workspace");
        let other = tempdir().expect("other workspace");
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(
                dir.path(),
                &crate::workspace_scope_id(dir.path()),
                "inventory-test-session",
            )
            .expect("history index"),
        ));
        let other_ctx = context(other.path().to_path_buf());
        let result = HistoryBindingRequest::from_index(
            index,
            ToolCatalogScope::from_context(&other_ctx).expect("scope"),
        );
        assert!(matches!(result, Err(InventoryError::InvalidScope(_))));
    }

    #[test]
    fn history_binding_checks_actual_journal_scope() {
        let dir = tempdir().expect("temporary workspace");
        let ctx = context(dir.path().to_path_buf());
        let scope = ToolCatalogScope::from_context(&ctx).expect("scope");
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(
                dir.path(),
                &workspace_scope_id(scope.workspace()),
                "different-session",
            )
            .expect("history index"),
        ));
        let result = HistoryBindingRequest::from_index(index, scope);
        assert!(matches!(result, Err(InventoryError::InvalidScope(_))));
    }

    #[test]
    fn retained_history_binding_rejects_replaced_workspace_scope() {
        let dir = tempdir().expect("temporary workspace");
        let replacement = tempdir().expect("replacement workspace");
        let ctx = context(dir.path().to_path_buf());
        let scope = ToolCatalogScope::from_context(&ctx).expect("scope");
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(
                dir.path(),
                &workspace_scope_id(scope.workspace()),
                scope.session_id(),
            )
            .expect("history index"),
        ));
        let binding = HistoryBindingRequest::from_index(index, scope).expect("binding");
        let replacement_ctx = context(replacement.path().to_path_buf());
        let result = InventoryRequest::for_context_with_binding(
            &replacement_ctx,
            ToolInventoryProfile::Interactive,
            Some(&binding),
        );
        assert!(matches!(result, Err(InventoryError::InvalidScope(_))));
    }
}
