//! Shared CLI/ACP inventory assembly over retained native handles and live MCP
//! descriptors. Host policy is a ceiling; scoped requests may only narrow it.

use crate::tool_inventory::{
    InventoryError, InventoryGeneration, InventoryRequest, ToolInventoryProfile,
    ToolInventorySnapshot, ToolInventorySource,
};
use async_trait::async_trait;
use claurst_core::config::Config;
use claurst_core::crypto_utils::sha256_hex;
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
    effect: ToolEffect,
}

impl McpToolWrapper {
    pub fn new(descriptor: McpToolDescriptor, manager: Arc<McpManager>) -> Self {
        let definition = descriptor.definition();
        Self {
            descriptor,
            definition,
            manager,
            effect: ToolEffect::Unknown,
        }
    }

    fn with_effect(
        descriptor: McpToolDescriptor,
        manager: Arc<McpManager>,
        effect: ToolEffect,
    ) -> Self {
        let definition = descriptor.definition();
        Self {
            descriptor,
            definition,
            manager,
            effect,
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
    fn effect(&self) -> ToolEffect {
        self.effect
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
    grant_set_digest: String,
}

impl PartialEq for InventoryKey {
    fn eq(&self, other: &Self) -> bool {
        self.scope == other.scope
            && self.profile == other.profile
            && self.policy == other.policy
            && self.mcp_headers == other.mcp_headers
            && self.grant_set_digest == other.grant_set_digest
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
        let grant_set_digest = grant_set_digest(&ctx.config);
        Ok(InventoryKey {
            scope: request.scope().clone(),
            profile: request.profile().clone(),
            policy: SelectorPolicy::from_config(&ctx.config)?,
            history: request.history().map(|history| Arc::clone(history.index())),
            mcp_headers,
            grant_set_digest,
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
        let validated_grants = ctx.config.validated_mcp_observation_grants().ok();
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
                let effect = if validated_grants.as_ref().is_some_and(|grants| {
                    grants.iter().any(|grant| {
                        grant.server == descriptor.server_name().trim().to_ascii_lowercase()
                            && grant.raw_tool == descriptor.raw_name()
                            && grant.definition_digest == descriptor.definition_digest()
                            && manager
                                .source_key_for_descriptor(&descriptor)
                                .is_some_and(|source| source == grant.source_key)
                    })
                }) {
                    ToolEffect::Observation
                } else {
                    ToolEffect::Unknown
                };
                let wrapper = McpToolWrapper::with_effect(descriptor, Arc::clone(manager), effect);
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

fn frame_grant_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u32).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn grant_set_digest(config: &Config) -> String {
    let mut bytes = b"forge.mcp-observation-grant-set.v1".to_vec();
    bytes.push(0);
    let Ok(mut grants) = config.validated_mcp_observation_grants() else {
        bytes.push(0);
        return sha256_hex(&bytes);
    };
    bytes.push(1);
    grants.sort_by(|left, right| {
        (
            left.source_key.as_bytes(),
            left.server.as_bytes(),
            left.raw_tool.as_bytes(),
            left.definition_digest.as_bytes(),
            left.version.as_bytes(),
        )
            .cmp(&(
                right.source_key.as_bytes(),
                right.server.as_bytes(),
                right.raw_tool.as_bytes(),
                right.definition_digest.as_bytes(),
                right.version.as_bytes(),
            ))
    });
    bytes.extend_from_slice(&(grants.len() as u32).to_be_bytes());
    for grant in grants {
        frame_grant_string(&mut bytes, &grant.version);
        frame_grant_string(&mut bytes, &grant.source_key);
        frame_grant_string(&mut bytes, &grant.server);
        frame_grant_string(&mut bytes, &grant.raw_tool);
        frame_grant_string(&mut bytes, &grant.definition_digest);
    }
    sha256_hex(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn observation_grant_set_digest_is_order_and_case_stable() {
        let grant = |server: &str, key: char| claurst_core::config::McpObservationGrant {
            version: claurst_core::config::MCP_OBSERVATION_GRANT_VERSION.into(),
            source_key: key.to_string().repeat(64),
            server: server.into(),
            raw_tool: "read_file".into(),
            definition_digest: "b".repeat(64),
        };
        let left = Config {
            mcp_observation_grants: vec![grant("Demo", 'a'), grant("other", 'c')],
            ..Config::default()
        };
        let right = Config {
            mcp_observation_grants: vec![grant("OTHER", 'c'), grant(" demo ", 'a')],
            ..Config::default()
        };
        assert_eq!(grant_set_digest(&left), grant_set_digest(&right));
    }

    fn valid_grant(
        source_key: &str,
        server: &str,
        raw_tool: &str,
        definition_digest: &str,
    ) -> claurst_core::config::McpObservationGrant {
        claurst_core::config::McpObservationGrant {
            version: claurst_core::config::MCP_OBSERVATION_GRANT_VERSION.into(),
            source_key: source_key.into(),
            server: server.into(),
            raw_tool: raw_tool.into(),
            definition_digest: definition_digest.into(),
        }
    }

    #[test]
    fn observation_grant_set_digest_revokes_add_remove_and_definition_changes() {
        let source = "a".repeat(64);
        let digest = "b".repeat(64);
        let replacement = "c".repeat(64);
        let grant = valid_grant(&source, "demo", "read_file", &digest);
        let empty = Config::default();
        let mut added = empty.clone();
        added.mcp_observation_grants.push(grant.clone());
        let mut replaced = added.clone();
        replaced.mcp_observation_grants[0].definition_digest = replacement;
        let mut removed = replaced.clone();
        removed.mcp_observation_grants.clear();
        assert_ne!(grant_set_digest(&empty), grant_set_digest(&added));
        assert_ne!(grant_set_digest(&added), grant_set_digest(&replaced));
        assert_ne!(grant_set_digest(&replaced), grant_set_digest(&removed));
        assert_eq!(grant_set_digest(&empty), grant_set_digest(&removed));
    }

    #[test]
    fn observation_grant_set_digest_separates_valid_empty_and_invalid_state() {
        let valid_empty = Config::default();
        let mut invalid = valid_empty.clone();
        invalid.mcp_observation_grants.push(valid_grant(
            &"a".repeat(64),
            "demo",
            "read_file",
            &"z".repeat(64),
        ));
        invalid.mcp_observation_grants[0].source_key = "not-a-digest".into();
        let invalid_again = invalid.clone();
        assert_ne!(grant_set_digest(&valid_empty), grant_set_digest(&invalid));
        assert_eq!(grant_set_digest(&invalid), grant_set_digest(&invalid_again));
        let mut wrong_schema = valid_empty.clone();
        wrong_schema.mcp_observation_grants.push(valid_grant(
            &"a".repeat(64),
            "demo",
            "read_file",
            &"z".repeat(64),
        ));
        wrong_schema.mcp_observation_grants[0].version = "unknown-schema".into();
        assert_ne!(
            grant_set_digest(&valid_empty),
            grant_set_digest(&wrong_schema)
        );
    }

    async fn run_inventory_dispatch_batch(
        tools: &[Box<dyn Tool>],
        ctx: &ToolContext,
        active: &crate::active_schema::ActiveSchemaSet,
        view: &Arc<claurst_tools::ToolCatalogView>,
        guard: crate::tool_inventory::InventoryGuard,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> (Vec<ToolResult>, bool) {
        let effects = tools.iter().map(|tool| tool.effect()).collect();
        let futures = tools
            .iter()
            .map(|tool| {
                let name = tool.name().to_owned();
                let view = Arc::clone(view);
                let guard = guard.clone();
                async move {
                    crate::runner::tools::execute_tool_with_inventory(
                        &name,
                        &Value::Object(Default::default()),
                        tools,
                        ctx,
                        active,
                        &view,
                        Some(guard),
                    )
                    .await
                }
            })
            .collect::<Vec<_>>();
        crate::runner::tools::run_tool_batch_with_effects(futures, effects, cancel_token).await
    }

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

    struct UnknownDirectTool;

    #[async_trait]
    impl Tool for UnknownDirectTool {
        fn name(&self) -> &str {
            "UnknownDirectTool"
        }
        fn description(&self) -> &str {
            "untrusted direct fixture"
        }
        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Execute
        }
        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("must not be admitted")
        }
    }

    #[tokio::test]
    async fn direct_unknown_and_child_observation_profiles_fail_closed() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = context(workspace.path());
        let direct: Box<dyn Tool> = Box::new(UnknownDirectTool);
        assert_eq!(direct.effect(), ToolEffect::Unknown);
        let child = ToolInventoryProfile::AgentChild {
            allowed_tools: Some(Arc::from(
                vec!["UnknownDirectTool".into()].into_boxed_slice(),
            )),
            capability: crate::tool_inventory::ChildInventoryCapability::ObservationOnly,
        };
        assert!(!child.allows_handle(direct.as_ref()));
        let source =
            HostInventorySource::new(Arc::new(vec![direct]), &ctx.config, Some("read-only"))
                .unwrap();
        let request = InventoryRequest::for_context(&ctx, child, None).unwrap();
        assert!(source
            .snapshot(&request, &ctx)
            .await
            .unwrap()
            .tools()
            .is_empty());
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

    #[tokio::test]
    async fn grant_set_change_revokes_old_inventory_generation() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = context(workspace.path());
        let source = HostInventorySource::new(Arc::new(Vec::new()), &ctx.config, None).unwrap();
        let request =
            InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None).unwrap();
        let first = source.snapshot(&request, &ctx).await.unwrap();
        let guard = crate::tool_inventory::InventoryGuard::new(
            Arc::clone(&source) as Arc<dyn ToolInventorySource>,
            request.clone(),
            first.generation(),
        );
        ctx.config.mcp_observation_grants.push(valid_grant(
            &"a".repeat(64),
            "demo",
            "read_file",
            &"b".repeat(64),
        ));
        let revoked_generation = source.current_generation(&request, &ctx).unwrap();
        assert_ne!(first.generation(), revoked_generation);
        assert!(!guard.is_current(&ctx));
        let refreshed = source.snapshot(&request, &ctx).await.unwrap();
        assert_ne!(first.generation(), refreshed.generation());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_stdio_manager_inventory_runner_overlap_and_revocation_barrier() {
        use claurst_core::config::{McpServerConfig, McpServerOrigin};
        use std::collections::HashMap;

        const PHASE_TIMEOUT: Duration = Duration::from_secs(3);
        const CLEANUP_TIMEOUT: Duration = Duration::from_secs(3);
        // The fixture has two explicit modes: the first two calls rendezvous
        // before either result, then later calls complete serially. Every
        // request start and result end is persisted for an order proof.
        let fixture = r#"
import json, sys
log_path = sys.argv[1]
parallel_pending = []
call_count = 0

def log(event, name):
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(event + ":" + name + "\n")

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {
            "protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "observation-fixture", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [
            {"name": "read_a", "description": "fixture a", "inputSchema": {"type": "object"}},
            {"name": "read_b", "description": "fixture b", "inputSchema": {"type": "object"}},
        ]}
    elif method == "tools/call":
        name = request.get("params", {}).get("name", "unknown")
        log("request-start", name)
        call_count += 1
        if call_count <= 2:
            parallel_pending.append((request.get("id"), name))
            if len(parallel_pending) < 2:
                continue
            for pending_id, pending_name in parallel_pending:
                log("result-end", pending_name)
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": pending_id, "result": {
                    "content": [{"type": "text", "text": "fixture"}], "isError": False
                }}) + "\n")
            parallel_pending.clear()
            sys.stdout.flush()
            continue
        log("result-end", name)
        result = {"content": [{"type": "text", "text": "fixture"}], "isError": False}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}) + "\n")
    sys.stdout.flush()
"#;
        let workspace = tempfile::tempdir().unwrap();
        let events_path = workspace.path().join("mcp-events.log");
        let config = McpServerConfig {
            name: "fixture".into(),
            command: Some("python3".into()),
            args: vec![
                "-u".into(),
                "-c".into(),
                fixture.into(),
                events_path.to_string_lossy().into_owned(),
            ],
            env: HashMap::new(),
            url: None,
            server_type: "stdio".into(),
            origin: McpServerOrigin::User,
        };
        let manager = Arc::new(McpManager::connect_all(std::slice::from_ref(&config)).await);
        let batch_cancel = tokio_util::sync::CancellationToken::new();
        let exercise = async {
            let descriptors = manager
                .checked_catalog_snapshot()
                .map_err(|error| format!("stdio fixture catalog unavailable: {error}"))?
                .descriptors;
            if descriptors.len() != 2 {
                return Err(format!(
                    "expected two fixture descriptors, got {}",
                    descriptors.len()
                ));
            }
            let source_key = manager
                .source_key_for_descriptor(&descriptors[0])
                .ok_or_else(|| "stdio source key unavailable".to_string())?;
            let mut ctx = context(workspace.path());
            ctx.mcp_manager = Some(Arc::clone(&manager));
            ctx.config.mcp_observation_grants = descriptors
                .iter()
                .map(|descriptor| {
                    valid_grant(
                        &source_key,
                        descriptor.server_name(),
                        descriptor.raw_name(),
                        descriptor.definition_digest(),
                    )
                })
                .collect();
            let source = HostInventorySource::new(Arc::new(Vec::new()), &ctx.config, None)
                .map_err(|error| format!("host source setup failed: {error}"))?;
            let request =
                InventoryRequest::for_context(&ctx, ToolInventoryProfile::Interactive, None)
                    .map_err(|error| format!("inventory request setup failed: {error}"))?;
            let inventory = source
                .snapshot(&request, &ctx)
                .await
                .map_err(|error| format!("granted inventory failed: {error}"))?;
            if inventory.tools().len() != 2
                || inventory
                    .tools()
                    .iter()
                    .any(|tool| tool.effect() != ToolEffect::Observation)
            {
                return Err("granted inventory did not retain two Observation tools".into());
            }
            let guard = crate::tool_inventory::InventoryGuard::new(
                Arc::clone(&source) as Arc<dyn ToolInventorySource>,
                request.clone(),
                inventory.generation(),
            );
            let view = claurst_tools::ToolCatalogView::from_tools(&ctx, inventory.tools())
                .map_err(|error| format!("granted catalog view failed: {error}"))?;
            let active =
                crate::active_schema::ActiveSchemaSet::initial(&view, inventory.tools(), false)
                    .map_err(|error| format!("granted active schema failed: {error}"))?;
            let (results, cancelled) = tokio::time::timeout(
                PHASE_TIMEOUT,
                run_inventory_dispatch_batch(
                    inventory.tools(),
                    &ctx,
                    &active,
                    &view,
                    guard.clone(),
                    &batch_cancel,
                ),
            )
            .await
            .map_err(|_| "parallel MCP dispatch timed out; fixture rendezvous was not reached")?;
            if cancelled || results.len() != 2 || results.iter().any(|result| result.is_error) {
                return Err("granted MCP dispatch did not complete successfully".into());
            }
            let initial_events = std::fs::read_to_string(&events_path)
                .map_err(|error| format!("reading initial fixture events failed: {error}"))?;
            let initial_lines: Vec<&str> = initial_events.lines().collect();
            if initial_lines
                != vec![
                    "request-start:read_a",
                    "request-start:read_b",
                    "result-end:read_a",
                    "result-end:read_b",
                ]
            {
                return Err(format!(
                    "fixture parallel event proof was invalid: {initial_lines:?}"
                ));
            }

            ctx.config.mcp_observation_grants.clear();
            if guard.is_current(&ctx) {
                return Err("revoked inventory guard remained current".into());
            }
            let (stale_results, stale_cancelled) = tokio::time::timeout(
                PHASE_TIMEOUT,
                run_inventory_dispatch_batch(
                    inventory.tools(),
                    &ctx,
                    &active,
                    &view,
                    guard,
                    &batch_cancel,
                ),
            )
            .await
            .map_err(|_| "stale guarded dispatch timed out")?;
            if stale_cancelled
                || stale_results.len() != 2
                || stale_results
                    .iter()
                    .any(|result| result.execution_status() != Some(ExecutionStatus::NotRun))
            {
                return Err("stale guarded calls were not rejected as NotRun".into());
            }
            let after_stale_events = std::fs::read_to_string(&events_path)
                .map_err(|error| format!("reading stale fixture events failed: {error}"))?;
            if after_stale_events.lines().count() != 4 {
                return Err("stale guarded calls reached the MCP child".into());
            }

            let revoked = source
                .snapshot(&request, &ctx)
                .await
                .map_err(|error| format!("revoked inventory refresh failed: {error}"))?;
            if revoked.tools().len() != 2
                || revoked
                    .tools()
                    .iter()
                    .any(|tool| tool.effect() != ToolEffect::Unknown)
            {
                return Err("revoked inventory did not produce two Unknown tools".into());
            }
            let revoked_guard = crate::tool_inventory::InventoryGuard::new(
                Arc::clone(&source) as Arc<dyn ToolInventorySource>,
                request,
                revoked.generation(),
            );
            let revoked_view = claurst_tools::ToolCatalogView::from_tools(&ctx, revoked.tools())
                .map_err(|error| format!("revoked catalog view failed: {error}"))?;
            let revoked_active = crate::active_schema::ActiveSchemaSet::initial(
                &revoked_view,
                revoked.tools(),
                false,
            )
            .map_err(|error| format!("revoked active schema failed: {error}"))?;
            let (unknown_results, unknown_cancelled) = tokio::time::timeout(
                PHASE_TIMEOUT,
                run_inventory_dispatch_batch(
                    revoked.tools(),
                    &ctx,
                    &revoked_active,
                    &revoked_view,
                    revoked_guard,
                    &batch_cancel,
                ),
            )
            .await
            .map_err(|_| "serial Unknown dispatch timed out")?;
            if unknown_cancelled
                || unknown_results.len() != 2
                || unknown_results.iter().any(|result| result.is_error)
            {
                return Err("Unknown MCP dispatch did not complete successfully".into());
            }
            let final_events = std::fs::read_to_string(&events_path)
                .map_err(|error| format!("reading final fixture events failed: {error}"))?;
            let final_lines: Vec<&str> = final_events.lines().collect();
            if final_lines
                != vec![
                    "request-start:read_a",
                    "request-start:read_b",
                    "result-end:read_a",
                    "result-end:read_b",
                    "request-start:read_a",
                    "result-end:read_a",
                    "request-start:read_b",
                    "result-end:read_b",
                ]
            {
                return Err(format!(
                    "fixture serial event proof was invalid: {final_lines:?}"
                ));
            }
            Ok::<(), String>(())
        };
        let exercise_result = tokio::time::timeout(PHASE_TIMEOUT * 3, exercise).await;
        batch_cancel.cancel();
        let cleanup_result = tokio::time::timeout(
            CLEANUP_TIMEOUT,
            McpManager::reconcile_with_previous(&manager, &[]),
        )
        .await;
        drop(manager);
        let exercise_result = exercise_result
            .map_err(|_| "bounded MCP observation exercise timed out".to_string())
            .and_then(|result| result);
        assert!(
            cleanup_result.is_ok(),
            "MCP manager cleanup timed out before child shutdown"
        );
        let cleanup = cleanup_result.unwrap();
        assert!(
            cleanup.failures.is_empty(),
            "MCP manager cleanup reported failures: {:?}",
            cleanup.failures
        );
        drop(cleanup.manager);
        exercise_result.expect("real stdio observation exercise failed");
    }
}
