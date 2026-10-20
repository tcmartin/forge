//! Shared CLI/ACP inventory assembly over retained native handles and live MCP
//! descriptors. Host policy is a ceiling; scoped requests may only narrow it.

use crate::tool_inventory::{
    InventoryError, InventoryGeneration, InventoryRequest, ToolInventoryProfile,
    ToolInventorySnapshot, ToolInventorySource,
};
use async_trait::async_trait;
use claurst_core::config::Config;
use claurst_core::types::ToolDefinition;
use claurst_tools::{
    tool_policy::{selector_matches, validate_selectors},
    ChildWriterCapability, ExecutionStatus, NativeBashExecutionWitness, PermissionLevel, Tool,
    ToolCatalogScope, ToolContext, ToolEffect, ToolResult,
};
use claurst_tools::{CatalogRevisionHeader, McpCatalogError, McpManager, McpToolDescriptor};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

const MAX_SCOPED_INVENTORIES: usize = 128;
const CATALOG_READY_TIMEOUT: Duration = Duration::from_secs(5);

/// A shared adapter keeps the exact host object alive when a filtered inventory
/// needs another owning vector. No schema or executable object is reconstructed.
struct SharedTool {
    inventory: Arc<Vec<Box<dyn Tool>>>,
    index: usize,
}

#[async_trait]
impl Tool for SharedTool {
    fn name(&self) -> &str {
        self.inventory[self.index].name()
    }
    fn description(&self) -> &str {
        self.inventory[self.index].description()
    }
    fn permission_level(&self) -> PermissionLevel {
        self.inventory[self.index].permission_level()
    }
    fn effect(&self) -> ToolEffect {
        self.inventory[self.index].effect()
    }
    fn child_writer_capability(&self) -> ChildWriterCapability {
        self.inventory[self.index].child_writer_capability()
    }
    fn self_gates(&self) -> bool {
        self.inventory[self.index].self_gates()
    }
    fn advanced(&self) -> bool {
        self.inventory[self.index].advanced()
    }
    fn catalog_source(&self) -> claurst_tools::ToolCatalogSource {
        self.inventory[self.index].catalog_source()
    }
    fn input_schema(&self) -> Value {
        self.inventory[self.index].input_schema()
    }
    fn to_definition(&self) -> ToolDefinition {
        self.inventory[self.index].to_definition()
    }
    fn native_bash_execution_witness(
        &self,
        tool_id: &str,
        result: &ToolResult,
        emitted_result: &str,
    ) -> Option<NativeBashExecutionWitness> {
        self.inventory[self.index].native_bash_execution_witness(tool_id, result, emitted_result)
    }
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        self.inventory[self.index].execute(input, ctx).await
    }
}

/// Filter actual handles, including custom tools, without rebuilding built-ins.
pub fn filter_shared_tools(
    inventory: Arc<Vec<Box<dyn Tool>>>,
    mut allowed: impl FnMut(&dyn Tool) -> bool,
) -> Arc<Vec<Box<dyn Tool>>> {
    let tools = inventory
        .iter()
        .enumerate()
        .filter(|(_, tool)| allowed(tool.as_ref()))
        .map(|(index, _)| {
            Box::new(SharedTool {
                inventory: Arc::clone(&inventory),
                index,
            }) as Box<dyn Tool>
        })
        .collect();
    Arc::new(tools)
}

/// A captured MCP descriptor remains bound to its original source, connection,
/// catalog generation and exact schema even when its displayed name is reused.
pub struct McpToolWrapper {
    descriptor: McpToolDescriptor,
    definition: Arc<ToolDefinition>,
    manager: Arc<McpManager>,
}

impl McpToolWrapper {
    pub fn new(descriptor: McpToolDescriptor, manager: Arc<McpManager>) -> Self {
        let definition = descriptor.definition();
        Self {
            descriptor,
            definition,
            manager,
        }
    }
}

/// Return an MCP descriptor only for the query-owned executable wrapper.
///
/// `Tool` is intentionally `Any`, but arbitrary inventory implementations
/// cannot opt into this authority by reporting catalog metadata. Filtered
/// inventories use `SharedTool`, which recurses into the original trusted
/// handle without exposing a descriptor hook on the public trait.
pub(crate) fn trusted_mcp_descriptor(
    tool: &dyn Tool,
    manager: &McpManager,
) -> Option<McpToolDescriptor> {
    if let Some(wrapper) = (tool as &dyn std::any::Any).downcast_ref::<McpToolWrapper>() {
        return std::ptr::eq(wrapper.manager.as_ref(), manager).then(|| wrapper.descriptor.clone());
    }
    if let Some(shared) = (tool as &dyn std::any::Any).downcast_ref::<SharedTool>() {
        return shared
            .inventory
            .get(shared.index)
            .and_then(|original| trusted_mcp_descriptor(original.as_ref(), manager));
    }
    None
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.definition.name
    }
    fn description(&self) -> &str {
        &self.definition.description
    }
    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }
    fn advanced(&self) -> bool {
        true
    }
    fn catalog_source(&self) -> claurst_tools::ToolCatalogSource {
        claurst_tools::ToolCatalogSource::DeferredMcp
    }
    fn self_gates(&self) -> bool {
        true
    }
    fn input_schema(&self) -> Value {
        self.definition.input_schema.clone()
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        if let Err(error) = ctx.check_permission(self.name(), "Run MCP tool", false) {
            return ToolResult::error(error.to_string())
                .with_execution_status(ExecutionStatus::NotRun);
        }
        let args = (!input.is_null()).then_some(input);
        match self
            .manager
            .call_tool_descriptor(&self.descriptor, args)
            .await
        {
            Ok(result) => claurst_tools::mcp_result_to_tool_result(result, ctx).await,
            Err(error) if error.downcast_ref::<McpCatalogError>().is_some() => ToolResult::error(
                "MCP tool was not run because its catalog is unavailable or changed.",
            )
            .with_execution_status(ExecutionStatus::NotRun),
            // A transport failure may follow a remote side effect. Never label
            // it NotRun or retry the captured call.
            Err(_) => ToolResult::error("MCP tool execution failed.")
                .with_execution_status(ExecutionStatus::Unknown),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct SelectorPolicy {
    allowed: Vec<String>,
    denied: Vec<String>,
}

impl SelectorPolicy {
    fn from_config(config: &Config) -> Result<Self, InventoryError> {
        Ok(Self {
            allowed: validate_selectors(&config.allowed_tools, "allowed_tools")
                .map_err(InventoryError::InvalidScope)?,
            denied: validate_selectors(&config.disallowed_tools, "disallowed_tools")
                .map_err(InventoryError::InvalidScope)?,
        })
    }

    fn allows(&self, name: &str) -> bool {
        !self
            .denied
            .iter()
            .any(|pattern| selector_matches(pattern, name))
            && (self.allowed.is_empty()
                || self
                    .allowed
                    .iter()
                    .any(|pattern| selector_matches(pattern, name)))
    }
}

#[derive(Clone, Copy)]
enum HostAccess {
    Full,
    ReadOnly,
    SearchOnly,
}

impl HostAccess {
    fn parse(access: Option<&str>) -> Result<Self, InventoryError> {
        match access {
            None | Some("full") => Ok(Self::Full),
            Some("read-only") => Ok(Self::ReadOnly),
            Some("search-only") => Ok(Self::SearchOnly),
            Some(_) => Err(InventoryError::InvalidScope(
                "unknown host tool access profile".into(),
            )),
        }
    }

    fn allows(self, tool: &dyn Tool) -> bool {
        match self {
            Self::Full => true,
            Self::ReadOnly => {
                matches!(
                    tool.permission_level(),
                    PermissionLevel::ReadOnly | PermissionLevel::None
                ) || tool.name() == "AskUserQuestion"
            }
            Self::SearchOnly => {
                ["Grep", "Glob", "Read", "WebSearch", "WebFetch"].contains(&tool.name())
            }
        }
    }
}

#[derive(Clone)]
struct InventoryKey {
    scope: ToolCatalogScope,
    profile: ToolInventoryProfile,
    policy: SelectorPolicy,
    history: Option<Arc<Mutex<crate::HistoryIndex>>>,
    mcp_headers: Vec<CatalogRevisionHeader>,
}

impl PartialEq for InventoryKey {
    fn eq(&self, other: &Self) -> bool {
        self.scope == other.scope
            && self.profile == other.profile
            && self.policy == other.policy
            && self.mcp_headers == other.mcp_headers
            && match (&self.history, &other.history) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                _ => false,
            }
    }
}

struct CacheEntry {
    key: InventoryKey,
    generation: InventoryGeneration,
    snapshot: Option<Arc<ToolInventorySnapshot>>,
}

#[derive(Default)]
struct InventoryCache {
    next_generation: InventoryGeneration,
    entries: VecDeque<CacheEntry>,
}

impl InventoryCache {
    fn entry(&mut self, key: InventoryKey) -> Result<&mut CacheEntry, InventoryError> {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            // Moving a live entry to the back does not change its generation.
            if let Some(entry) = self.entries.remove(index) {
                self.entries.push_back(entry);
            }
        } else {
            self.next_generation = self
                .next_generation
                .checked_add(1)
                .ok_or(InventoryError::InvalidGeneration)?;
            if self.entries.len() == MAX_SCOPED_INVENTORIES {
                self.entries.pop_front();
            }
            self.entries.push_back(CacheEntry {
                key,
                generation: self.next_generation,
                snapshot: None,
            });
        }
        self.entries
            .back_mut()
            .ok_or(InventoryError::InvalidGeneration)
    }
}

/// One host source can serve foreground, ACP, child and cron requests. The
/// retained native slice is assembled once; scoped history/MCP adapters are
/// rebuilt only after a relevant scope or catalog revision changes.
pub struct HostInventorySource {
    native_tools: Arc<Vec<Box<dyn Tool>>>,
    ceiling: SelectorPolicy,
    access: HostAccess,
    cache: Mutex<InventoryCache>,
}

impl HostInventorySource {
    pub fn new(
        native_tools: Arc<Vec<Box<dyn Tool>>>,
        ceiling: &Config,
        access: Option<&str>,
    ) -> Result<Arc<Self>, InventoryError> {
        if native_tools.len() > claurst_tools::tool_catalog::MAX_CATALOG_ENTRIES {
            return Err(InventoryError::Unavailable(
                "native tool count limit exceeded".into(),
            ));
        }
        let access = HostAccess::parse(access)?;
        Ok(Arc::new(Self {
            native_tools,
            ceiling: SelectorPolicy::from_config(ceiling)?,
            access,
            cache: Mutex::new(InventoryCache::default()),
        }))
    }

    /// Agent-mode changes narrow the same retained native handles and original
    /// policy ceiling, while revoking the previous scoped snapshot authority.
    pub fn with_access(&self, access: Option<&str>) -> Result<Arc<Self>, InventoryError> {
        Ok(Arc::new(Self {
            native_tools: Arc::clone(&self.native_tools),
            ceiling: self.ceiling.clone(),
            access: HostAccess::parse(access)?,
            cache: Mutex::new(InventoryCache::default()),
        }))
    }

    fn key(
        &self,
        request: &InventoryRequest,
        ctx: &ToolContext,
    ) -> Result<InventoryKey, InventoryError> {
        if !request.scope().matches_context(ctx)
            || request
                .history()
                .is_some_and(|history| history.scope() != request.scope())
        {
            return Err(InventoryError::InvalidScope(
                "inventory request is stale".into(),
            ));
        }
        let mcp_headers = ctx
            .mcp_manager
            .as_ref()
            .map(|manager| manager.catalog_state().revision_headers())
            .unwrap_or_default();
        Ok(InventoryKey {
            scope: request.scope().clone(),
            profile: request.profile().clone(),
            policy: SelectorPolicy::from_config(&ctx.config)?,
            history: request.history().map(|history| Arc::clone(history.index())),
            mcp_headers,
        })
    }

    fn allows(&self, key: &InventoryKey, tool: &dyn Tool) -> bool {
        self.ceiling.allows(tool.name())
            && key.policy.allows(tool.name())
            && self.access.allows(tool)
            && key.profile.allows_handle(tool)
            && !(matches!(key.profile, ToolInventoryProfile::AgentChild { .. })
                && tool.name() == "Agent")
    }
}

#[async_trait]
impl ToolInventorySource for HostInventorySource {
    fn current_generation(
        &self,
        request: &InventoryRequest,
        ctx: &ToolContext,
    ) -> Result<InventoryGeneration, InventoryError> {
        let key = self.key(request, ctx)?;
        Ok(self.cache.lock().entry(key)?.generation)
    }

    async fn snapshot(
        &self,
        request: &InventoryRequest,
        ctx: &ToolContext,
    ) -> Result<Arc<ToolInventorySnapshot>, InventoryError> {
        let before = self.key(request, ctx)?;
        if let Some(snapshot) = self.cache.lock().entry(before)?.snapshot.clone() {
            return Ok(snapshot);
        }
        // Dirty catalogs are observed immediately by current_generation, but
        // only a checked ready catalog can supply new executable descriptors.
        let mcp = if let Some(manager) = ctx.mcp_manager.as_ref() {
            Some(tokio::select! {
                biased;
                _ = ctx.cancel_token.cancelled() => {
                    return Err(InventoryError::Unavailable("inventory request cancelled".into()));
                }
                result = manager.wait_catalog_ready(CATALOG_READY_TIMEOUT) => {
                    result.map_err(|error| InventoryError::Unavailable(error.to_string()))?
                }
            })
        } else {
            None
        };
        let key = self.key(request, ctx)?;
        if mcp
            .as_ref()
            .is_some_and(|snapshot| snapshot.headers != key.mcp_headers)
        {
            return Err(InventoryError::Unavailable(
                "MCP catalog changed during inventory capture".into(),
            ));
        }
        {
            let mut cache = self.cache.lock();
            if let Some(snapshot) = cache.entry(key.clone())?.snapshot.clone() {
                return Ok(snapshot);
            }
        }
        let mut tools: Vec<Box<dyn Tool>> = self
            .native_tools
            .iter()
            .enumerate()
            .filter(|(_, tool)| self.allows(&key, tool.as_ref()))
            .map(|(index, _)| {
                Box::new(SharedTool {
                    inventory: Arc::clone(&self.native_tools),
                    index,
                }) as Box<dyn Tool>
            })
            .collect();
        if let (Some(manager), Some(mcp)) = (ctx.mcp_manager.as_ref(), mcp) {
            for descriptor in mcp.descriptors {
                let wrapper = McpToolWrapper::new(descriptor, Arc::clone(manager));
                if self.allows(&key, &wrapper) {
                    tools.push(Box::new(wrapper));
                }
            }
        }
        if let Some(history) = request.history() {
            let reader = crate::make_history_read_tool_for_workspace(
                Arc::clone(history.index()),
                request.workspace_root(),
            );
            if self.allows(&key, reader.as_ref()) {
                tools.push(reader);
            }
        }
        if tools.len() > claurst_tools::tool_catalog::MAX_CATALOG_ENTRIES {
            return Err(InventoryError::Unavailable(
                "tool catalog count limit exceeded".into(),
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        if tools
            .iter()
            .any(|tool| !names.insert(tool.name().to_ascii_lowercase()))
        {
            return Err(InventoryError::Unavailable(
                "tool catalog name collision".into(),
            ));
        }
        if matches!(request.profile(), ToolInventoryProfile::AgentChild { .. }) {
            tools = crate::child_policy::confine_child_tools(tools, request.workspace_root())
                .map_err(InventoryError::InvalidScope)?;
        }
        // No async work or lock is held during tool assembly. Recheck both
        // scope and source headers before publishing the whole owning snapshot.
        if key != self.key(request, ctx)? {
            return Err(InventoryError::Unavailable(
                "inventory changed during assembly".into(),
            ));
        }
        let mut cache = self.cache.lock();
        let entry = cache.entry(key)?;
        // Concurrent foreground/child captures may have published this exact
        // key already. Reuse its actual handles rather than replacing them.
        if let Some(snapshot) = entry.snapshot.as_ref() {
            return Ok(Arc::clone(snapshot));
        }
        let snapshot =
            ToolInventorySnapshot::new(Arc::new(tools), entry.generation, request.scope().clone())?;
        entry.snapshot = Some(Arc::clone(&snapshot));
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Allow;
    impl claurst_core::PermissionHandler for Allow {
        fn check_permission(
            &self,
            _: &claurst_core::PermissionRequest,
        ) -> claurst_core::PermissionDecision {
            claurst_core::PermissionDecision::Allow
        }
        fn request_permission(
            &self,
            _: &claurst_core::PermissionRequest,
        ) -> claurst_core::PermissionDecision {
            claurst_core::PermissionDecision::Allow
        }
    }

    fn context(workspace: &std::path::Path) -> ToolContext {
        ToolContext {
            working_dir: workspace.to_path_buf(),
            permission_mode: claurst_core::PermissionMode::Default,
            permission_handler: Arc::new(Allow),
            cost_tracker: claurst_core::CostTracker::new(),
            session_id: "host-inventory-test".into(),
            file_history: Arc::new(Mutex::new(claurst_core::file_history::FileHistory::new())),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            catalog_view: None,
            trusted_checker_catalog: Default::default(),
        }
    }

    struct WitnessTool {
        source: claurst_tools::ToolCatalogSource,
        witness: Option<NativeBashExecutionWitness>,
    }

    #[async_trait]
    impl Tool for WitnessTool {
        fn name(&self) -> &str {
            claurst_core::constants::TOOL_NAME_BASH
        }
        fn description(&self) -> &str {
            "source-backed witness fixture"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Execute
        }
        fn catalog_source(&self) -> claurst_tools::ToolCatalogSource {
            self.source
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type":"object"})
        }
        fn native_bash_execution_witness(
            &self,
            _tool_id: &str,
            _result: &ToolResult,
            _emitted_result: &str,
        ) -> Option<NativeBashExecutionWitness> {
            self.witness.clone()
        }
        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            ToolResult::success("fixture")
        }
    }

    fn fixture_witness() -> NativeBashExecutionWitness {
        NativeBashExecutionWitness {
            tool_id: "call-1".into(),
            displayed_result_sha256: "a".repeat(64),
            command_status: claurst_tools::NativeBashCommandStatus::Completed,
            exit_code: Some(0),
            partial: false,
        }
    }

    #[test]
    fn shared_native_wrapper_forwards_the_retained_bash_capability() {
        let native: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(claurst_tools::PtyBashTool)]);
        let filtered = filter_shared_tools(native, |_| true);
        let result = ToolResult::success("fixture").with_metadata(serde_json::json!({
            "command_status":"completed", "exit_code":0, "partial":false,
        }));
        let witness = crate::native_bash_execution_witness(
            filtered.as_ref(),
            claurst_core::constants::TOOL_NAME_BASH,
            "call-1",
            &result,
            "fixture",
        )
        .expect("the actual retained PtyBash capability survives filtering");
        assert_eq!(
            witness.displayed_result_sha256,
            claurst_core::crypto_utils::sha256_hex(b"fixture")
        );
        assert_eq!(witness.exit_code, Some(0));
        assert!(!witness.partial);
    }

    #[test]
    fn default_native_and_same_name_deferred_mcp_cannot_authorize() {
        for (source, witness) in [
            (claurst_tools::ToolCatalogSource::Native, None),
            (
                claurst_tools::ToolCatalogSource::DeferredMcp,
                Some(fixture_witness()),
            ),
        ] {
            let tools: Arc<Vec<Box<dyn Tool>>> =
                Arc::new(vec![Box::new(WitnessTool { source, witness })]);
            let wrapped = filter_shared_tools(tools, |_| true);
            assert!(crate::native_bash_execution_witness(
                wrapped.as_ref(),
                claurst_core::constants::TOOL_NAME_BASH,
                "call-1",
                &ToolResult::success("fixture"),
                "fixture",
            )
            .is_none());
        }
    }

    struct CustomObservation {
        name: &'static str,
        schemas: Arc<AtomicUsize>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for CustomObservation {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "retained custom observation"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn effect(&self) -> ToolEffect {
            ToolEffect::Observation
        }
        fn advanced(&self) -> bool {
            true
        }
        fn input_schema(&self) -> Value {
            self.schemas.fetch_add(1, Ordering::SeqCst);
            serde_json::json!({"type":"object","properties":{"value":{"type":"string"}}})
        }
        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            ToolResult::success("custom-observation-result")
        }
    }

    #[tokio::test]
    async fn repeated_snapshot_and_agent_filter_preserve_custom_handle_without_schema_evaluation() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = context(workspace.path());
        let schemas = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let native: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(CustomObservation {
            name: "CustomObservation",
            schemas: Arc::clone(&schemas),
            calls: Arc::clone(&calls),
        })]);
        let filtered = filter_shared_tools(native, |tool| tool.effect() == ToolEffect::Observation);
        let source = HostInventorySource::new(filtered, &ctx.config, Some("read-only")).unwrap();
        let request =
            InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None).unwrap();
        let first = source.snapshot(&request, &ctx).await.unwrap();
        let second = source.snapshot(&request, &ctx).await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            first.generation(),
            source.current_generation(&request, &ctx).unwrap()
        );
        assert_eq!(schemas.load(Ordering::SeqCst), 0);
        assert_eq!(first.tools()[0].name(), "CustomObservation");
        assert!(first.tools()[0].advanced());
        assert!(!first.tools()[0].execute(Value::Null, &ctx).await.is_error);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        first.tools()[0].input_schema();
        assert_eq!(schemas.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn host_ceiling_and_child_filter_intersect_current_policy() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = context(workspace.path());
        let native: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![
            Box::new(CustomObservation {
                name: "Allowed",
                schemas: Default::default(),
                calls: Default::default(),
            }),
            Box::new(CustomObservation {
                name: "Denied",
                schemas: Default::default(),
                calls: Default::default(),
            }),
        ]);
        let ceiling = Config {
            disallowed_tools: vec!["Denied".into()],
            ..Config::default()
        };
        let source = HostInventorySource::new(native, &ceiling, None).unwrap();
        let request = InventoryRequest::for_context(
            &ctx,
            ToolInventoryProfile::AgentChild {
                allowed_tools: Some(Arc::from(
                    vec!["Allowed".into(), "Denied".into()].into_boxed_slice(),
                )),
                capability: crate::tool_inventory::ChildInventoryCapability::ObservationOnly,
            },
            None,
        )
        .unwrap();
        let first = source.snapshot(&request, &ctx).await.unwrap();
        assert_eq!(
            first
                .tools()
                .iter()
                .map(|tool| tool.name())
                .collect::<Vec<_>>(),
            vec!["Allowed"]
        );
        ctx.config.disallowed_tools = vec!["Allowed".into()];
        let generation = source.current_generation(&request, &ctx).unwrap();
        assert_ne!(first.generation(), generation);
        assert!(source
            .snapshot(&request, &ctx)
            .await
            .unwrap()
            .tools()
            .is_empty());
    }

    #[tokio::test]
    async fn scope_history_and_profile_changes_receive_distinct_snapshots() {
        let workspace = tempfile::tempdir().unwrap();
        let history_root = tempfile::tempdir().unwrap();
        let mut ctx = context(workspace.path());
        let source = HostInventorySource::new(Arc::new(Vec::new()), &ctx.config, None).unwrap();
        let index = Arc::new(Mutex::new(
            crate::HistoryIndex::open(
                history_root.path(),
                &crate::workspace_scope_id(workspace.path()),
                &ctx.session_id,
            )
            .unwrap(),
        ));
        let parent =
            InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, Some(&index))
                .unwrap();
        let parent_snapshot = source.snapshot(&parent, &ctx).await.unwrap();
        assert_eq!(parent_snapshot.tools()[0].name(), "HistoryRead");
        ctx.session_id = "cron-fresh-session".into();
        assert!(
            InventoryRequest::for_context(&ctx, ToolInventoryProfile::Cron, Some(&index)).is_err()
        );
        let cron_index = Arc::new(Mutex::new(
            crate::HistoryIndex::open(
                history_root.path(),
                &crate::workspace_scope_id(workspace.path()),
                &ctx.session_id,
            )
            .unwrap(),
        ));
        let cron =
            InventoryRequest::for_context(&ctx, ToolInventoryProfile::Cron, Some(&cron_index))
                .unwrap();
        let cron_snapshot = source.snapshot(&cron, &ctx).await.unwrap();
        assert_ne!(parent_snapshot.generation(), cron_snapshot.generation());
        assert!(!parent_snapshot.scope().matches_context(&ctx));
        assert!(
            parent_snapshot.tools()[0]
                .execute(serde_json::json!({"query":"old"}), &ctx)
                .await
                .is_error
        );
        assert!(source.current_generation(&parent, &ctx).is_err());
    }

    #[tokio::test]
    async fn scoped_cache_is_finite_and_eviction_revokes_old_generation() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = context(workspace.path());
        let source = HostInventorySource::new(Arc::new(Vec::new()), &ctx.config, None).unwrap();
        let first_request =
            InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None).unwrap();
        let first = source.snapshot(&first_request, &ctx).await.unwrap();
        for index in 0..MAX_SCOPED_INVENTORIES {
            ctx.session_id = format!("cache-session-{index}");
            let request =
                InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None)
                    .unwrap();
            source.snapshot(&request, &ctx).await.unwrap();
        }
        assert_eq!(source.cache.lock().entries.len(), MAX_SCOPED_INVENTORIES);
        ctx.session_id = first_request.session_id().to_owned();
        assert_ne!(
            first.generation(),
            source.current_generation(&first_request, &ctx).unwrap()
        );
    }
}
