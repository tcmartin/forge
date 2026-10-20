//! Admission and capability policy for nested query agents.
//!
//! Child tool selection is deliberately stricter than the parent permission
//! mode.  A child that has not requested a successful worktree receives only
//! tools which explicitly declare [`ToolEffect::Observation`].  Permission
//! levels describe prompting and are not evidence that a tool is pure.

use async_trait::async_trait;
use claurst_core::config::Config;
use claurst_tools::{
    tool_policy, ChildWriterCapability, PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult,
};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

/// Maximum number of nested inference loops admitted at once.
pub(crate) const MAX_CHILD_INFERENCE: usize = 4;

// These tools operate on explicit filesystem paths and can be bounded by the
// child worktree context. Arbitrary command, process, network, MCP, task and
// UI tools remain unconfinable until an OS-backed writer adapter exists.
const TRUSTED_FILESYSTEM_WRITERS: &[&str] =
    &["Edit", "Write", "BatchEdit", "ApplyPatch", "NotebookEdit"];

static CHILD_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn admission() -> Arc<Semaphore> {
    CHILD_ADMISSION
        .get_or_init(|| Arc::new(Semaphore::new(MAX_CHILD_INFERENCE)))
        .clone()
}

/// Acquire a global child inference slot, observing parent cancellation while
/// waiting.  Foreground and background AgentTool calls use this same gate.
pub(crate) async fn acquire_child_admission(
    cancel: CancellationToken,
) -> Result<OwnedSemaphorePermit, String> {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err("Child agent admission cancelled before provider traffic".to_string()),
        permit = admission().acquire_owned() => permit
            .map_err(|_| "Child agent admission is unavailable".to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChildCapabilityProfile {
    /// Only tools explicitly marked as observation-only are exposed.
    ReadOnly,
    /// Mutation-capable tools are allowed, but only after a worktree exists.
    Mutating,
}

impl ChildCapabilityProfile {
    pub(crate) fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("read_only") {
            "read_only" | "readonly" => Ok(Self::ReadOnly),
            "mutating" => Ok(Self::Mutating),
            other => Err(format!(
                "unsupported child capability profile '{other}'; use 'read_only' or 'mutating'"
            )),
        }
    }

    fn allows(self, tool: &dyn Tool) -> bool {
        match self {
            Self::ReadOnly => tool.effect() == ToolEffect::Observation,
            Self::Mutating => {
                tool.effect() == ToolEffect::Observation
                    || (tool.effect() == ToolEffect::Mutation
                        && tool.child_writer_capability()
                            == ChildWriterCapability::ScopedFilesystem
                        && TRUSTED_FILESYSTEM_WRITERS.contains(&tool.name()))
            }
        }
    }
}

pub(crate) struct ChildToolPlan {
    pub(crate) tools: Vec<Box<dyn Tool>>,
    pub(crate) names: Vec<String>,
}

/// Build a child plan from the exact source-backed handles captured for this
/// child scope.  Forwarding adapters retain the source Arc while allowing the
/// existing child planner and worktree writer wrappers to own their vectors.
pub(crate) fn plan_tools_from_inventory(
    config: &Config,
    inventory: Arc<crate::tool_inventory::ToolInventorySnapshot>,
    requested: Option<&[String]>,
    profile: ChildCapabilityProfile,
    worktree_ready: bool,
) -> Result<ChildToolPlan, String> {
    if profile == ChildCapabilityProfile::Mutating && !worktree_ready {
        return Err(
            "mutating child capability requires a successfully created worktree".to_string(),
        );
    }

    let requested_set =
        requested.map(|names| names.iter().map(String::as_str).collect::<HashSet<_>>());
    let mut names = Vec::new();
    let mut tools = Vec::new();
    let inventory_tools = inventory.tools_arc();

    for (index, tool) in inventory_tools.iter().enumerate() {
        let name = tool.name();
        if requested_set
            .as_ref()
            .is_some_and(|requested| !requested.contains(name))
        {
            continue;
        }
        if !tool_policy::tool_allowed(config, name)
            .map_err(|error| format!("invalid parent tool policy: {error}"))?
        {
            continue;
        }
        if !profile.allows(tool.as_ref()) {
            if requested_set
                .as_ref()
                .is_some_and(|requested| requested.contains(name))
                && profile == ChildCapabilityProfile::Mutating
            {
                return Err(format!(
                    "child tool '{name}' cannot be safely confined to the worktree"
                ));
            }
            continue;
        }
        names.push(name.to_string());
        tools.push(Box::new(InventoryTool {
            inventory: Arc::clone(&inventory_tools),
            index,
        }) as Box<dyn Tool>);
    }

    if let Some(requested) = requested {
        for name in requested {
            if !names.iter().any(|selected| selected == name) {
                return Err(format!("unsupported child tool '{name}'"));
            }
        }
    }

    Ok(ChildToolPlan { tools, names })
}

struct InventoryTool {
    inventory: Arc<Vec<Box<dyn Tool>>>,
    index: usize,
}

#[async_trait]
impl Tool for InventoryTool {
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

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        self.inventory[self.index].execute(input, ctx).await
    }
}

/// Validate filesystem targets before invoking a trusted writer. This checks
/// the canonical existing ancestor (including symlinks) and rejects traversal
/// components. In-process checks cannot eliminate a concurrent rename race;
/// callers must still provide an OS-backed sandbox before treating this as a
/// race-proof boundary.
fn validate_target(root: &Path, ctx: &ToolContext, raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("child writer path must not be empty".to_string());
    }
    let raw_path = Path::new(raw);
    if raw_path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(format!("child writer path rejects traversal: {raw}"));
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("child worktree is unavailable: {error}"))?;
    let path = ctx.resolve_path(raw);
    let mut ancestor = path.clone();
    while !ancestor.exists() {
        if !ancestor.pop() {
            return Err(format!("child writer path has no existing ancestor: {raw}"));
        }
    }
    let canonical_ancestor = ancestor
        .canonicalize()
        .map_err(|error| format!("child writer path ancestor is unavailable: {error}"))?;
    if !canonical_ancestor.starts_with(&root) {
        return Err(format!("child writer path escapes worktree: {raw}"));
    }
    if path.exists() {
        let canonical_target = path
            .canonicalize()
            .map_err(|error| format!("child writer target is unavailable: {error}"))?;
        if !canonical_target.starts_with(&root) {
            return Err(format!("child writer path escapes worktree: {raw}"));
        }
    }
    Ok(path)
}

fn patch_targets(patch: &str) -> Result<Vec<String>, String> {
    let mut targets = Vec::new();
    for line in patch.lines() {
        if let Some(raw) = line.strip_prefix("+++ ") {
            let target = raw.split_whitespace().next().unwrap_or_default();
            if target == "/dev/null" {
                return Err("child ApplyPatch cannot target /dev/null".to_string());
            }
            targets.push(target.trim_start_matches("b/").to_string());
        }
    }
    if targets.is_empty() {
        return Err("child ApplyPatch contains no target paths".to_string());
    }
    Ok(targets)
}

fn validate_writer_input(
    name: &str,
    input: &Value,
    root: &Path,
    ctx: &ToolContext,
) -> Result<(), String> {
    match name {
        "Edit" | "Write" => {
            let path = input
                .get("file_path")
                .and_then(Value::as_str)
                .ok_or_else(|| "child writer input requires file_path".to_string())?;
            validate_target(root, ctx, path).map(|_| ())
        }
        "NotebookEdit" => {
            let path = input
                .get("notebook_path")
                .and_then(Value::as_str)
                .ok_or_else(|| "child writer input requires notebook_path".to_string())?;
            validate_target(root, ctx, path).map(|_| ())
        }
        "BatchEdit" => {
            let edits = input
                .get("edits")
                .and_then(Value::as_array)
                .ok_or_else(|| "child BatchEdit input requires edits".to_string())?;
            for edit in edits {
                let path = edit
                    .get("file_path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "child BatchEdit edit requires file_path".to_string())?;
                validate_target(root, ctx, path)?;
            }
            Ok(())
        }
        "ApplyPatch" => {
            let patch = input
                .get("patch")
                .and_then(Value::as_str)
                .ok_or_else(|| "child ApplyPatch input requires patch".to_string())?;
            for path in patch_targets(patch)? {
                validate_target(root, ctx, &path)?;
            }
            Ok(())
        }
        _ => Err(format!("child writer wrapper does not support '{name}'")),
    }
}

struct ChildWriterTool {
    inner: Box<dyn Tool>,
    root: PathBuf,
}

#[async_trait]
impl Tool for ChildWriterTool {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn permission_level(&self) -> PermissionLevel {
        self.inner.permission_level()
    }
    fn effect(&self) -> ToolEffect {
        self.inner.effect()
    }
    fn child_writer_capability(&self) -> ChildWriterCapability {
        self.inner.child_writer_capability()
    }
    fn self_gates(&self) -> bool {
        self.inner.self_gates()
    }
    fn advanced(&self) -> bool {
        self.inner.advanced()
    }
    fn catalog_source(&self) -> claurst_tools::ToolCatalogSource {
        self.inner.catalog_source()
    }
    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        if let Err(error) = validate_writer_input(self.name(), &input, &self.root, ctx) {
            return ToolResult::error(error);
        }
        self.inner.execute(input, ctx).await
    }
}

pub(crate) fn contain_writers(plan: ChildToolPlan, root: &Path) -> Result<ChildToolPlan, String> {
    let ChildToolPlan { tools, names } = plan;
    let tools = tools
        .into_iter()
        .map(|tool| {
            if TRUSTED_FILESYSTEM_WRITERS.contains(&tool.name()) {
                if tool.effect() != ToolEffect::Mutation
                    || tool.child_writer_capability() != ChildWriterCapability::ScopedFilesystem
                {
                    return Err(format!(
                        "child writer '{}' lacks the native scoped-filesystem capability",
                        tool.name()
                    ));
                }
                Ok(Box::new(ChildWriterTool {
                    inner: tool,
                    root: root.to_path_buf(),
                }) as Box<dyn Tool>)
            } else {
                Ok(tool)
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ChildToolPlan { tools, names })
}

/// Apply the child worktree writer wrappers to a source-owned snapshot before
/// the snapshot is published.  Dynamic inventory refreshes must use this
/// helper on every newly materialized child slice; otherwise a raw Write/Edit
/// handle could replace the contained handle after the initial plan.
pub(crate) fn confine_child_tools(
    tools: Vec<Box<dyn Tool>>,
    root: &Path,
) -> Result<Vec<Box<dyn Tool>>, String> {
    let names = tools.iter().map(|tool| tool.name().to_owned()).collect();
    contain_writers(ChildToolPlan { tools, names }, root).map(|plan| plan.tools)
}

/// Resolve the child inventory as the intersection of the parent config and
/// the requested names. Denial is handled by the shared selector policy, then
/// the child's capability profile applies an additional effect restriction.
pub(crate) fn plan_tools(
    config: &Config,
    requested: Option<&[String]>,
    profile: ChildCapabilityProfile,
    worktree_ready: bool,
) -> Result<ChildToolPlan, String> {
    if profile == ChildCapabilityProfile::Mutating && !worktree_ready {
        return Err(
            "mutating child capability requires a successfully created worktree".to_string(),
        );
    }

    let all = claurst_tools::all_tools();
    let known: HashSet<&str> = all.iter().map(|tool| tool.name()).collect();
    let requested_set =
        requested.map(|names| names.iter().map(String::as_str).collect::<HashSet<_>>());

    if let Some(names) = requested {
        for name in names {
            if name == claurst_core::constants::TOOL_NAME_AGENT {
                return Err("child AgentTool recursion is not supported".to_string());
            }
            if !known.contains(name.as_str()) {
                return Err(format!("unsupported child tool '{name}'"));
            }
        }
    }

    let mut names = Vec::new();
    let mut tools = Vec::new();
    for tool in all {
        let name = tool.name();
        if requested_set
            .as_ref()
            .is_some_and(|requested| !requested.contains(name))
        {
            continue;
        }
        if !tool_policy::tool_allowed(config, name)
            .map_err(|error| format!("invalid parent tool policy: {error}"))?
        {
            continue;
        }
        if !profile.allows(tool.as_ref()) {
            if requested_set
                .as_ref()
                .is_some_and(|requested| requested.contains(name))
                && profile == ChildCapabilityProfile::Mutating
            {
                return Err(format!(
                    "child tool '{name}' cannot be safely confined to the worktree"
                ));
            }
            continue;
        }
        names.push(name.to_string());
        tools.push(tool);
    }

    Ok(ChildToolPlan { tools, names })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct DynamicObservation;

    struct ForgedWrite;

    #[async_trait]
    impl Tool for DynamicObservation {
        fn name(&self) -> &str {
            "DynamicObservation"
        }

        fn description(&self) -> &str {
            "source-backed observation fixture"
        }

        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }

        fn effect(&self) -> ToolEffect {
            ToolEffect::Observation
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("dynamic")
        }
    }

    #[async_trait]
    impl Tool for ForgedWrite {
        fn name(&self) -> &str {
            "Write"
        }

        fn description(&self) -> &str {
            "untrusted same-name writer fixture"
        }

        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::Write
        }

        fn effect(&self) -> ToolEffect {
            ToolEffect::Mutation
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("must not run")
        }
    }

    struct AllowPermissionHandler;

    impl claurst_core::permissions::PermissionHandler for AllowPermissionHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }

        fn request_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }
    }

    fn writer_context(root: &Path) -> ToolContext {
        ToolContext {
            working_dir: root.to_path_buf(),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(AllowPermissionHandler),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "child-writer-test".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: CancellationToken::new(),
            catalog_view: None,
            trusted_checker_catalog: Arc::new(claurst_tools::TrustedCheckerCatalog::empty()),
        }
    }

    #[test]
    fn default_profile_is_trusted_observation_intersection() {
        let config = Config {
            allowed_tools: vec!["Read".into(), "Glob".into(), "Grep".into()],
            disallowed_tools: vec!["Glob".into()],
            ..Config::default()
        };
        let plan = plan_tools(&config, None, ChildCapabilityProfile::ReadOnly, false).unwrap();
        assert_eq!(plan.names, vec!["Read", "Grep"]);
    }

    #[test]
    fn read_profile_excludes_unknown_process_network_and_memory_tools() {
        let config = Config::default();
        let plan = plan_tools(&config, None, ChildCapabilityProfile::ReadOnly, false).unwrap();
        for name in [
            "PtyBash",
            "PowerShell",
            "WebFetch",
            "WebSearch",
            "LSP",
            "REPL",
            "Config",
            "Skill",
            "TodoWrite",
        ] {
            assert!(
                !plan.names.iter().any(|candidate| candidate == name),
                "{name}"
            );
        }
        assert!(plan.names.iter().any(|name| name == "Read"));
        assert!(plan.names.iter().any(|name| name == "EvidenceRead"));
    }

    #[test]
    fn requested_tools_are_intersected_with_parent_policy() {
        let config = Config {
            allowed_tools: vec!["Read".into(), "Grep".into()],
            disallowed_tools: vec!["Grep".into()],
            ..Config::default()
        };
        let requested = vec!["Read".into(), "Grep".into()];
        let plan = plan_tools(
            &config,
            Some(&requested),
            ChildCapabilityProfile::ReadOnly,
            false,
        )
        .unwrap();
        assert_eq!(plan.names, vec!["Read"]);
    }

    #[test]
    fn source_inventory_admits_dynamic_observation_and_rejects_missing_name() {
        let root = tempfile::tempdir().unwrap();
        let context = writer_context(root.path());
        let scope = claurst_tools::ToolCatalogScope::from_context(&context).unwrap();
        let tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(DynamicObservation)]);
        let snapshot =
            crate::tool_inventory::ToolInventorySnapshot::new(Arc::clone(&tools), 1, scope)
                .unwrap();
        let config = Config::default();

        let plan = plan_tools_from_inventory(
            &config,
            snapshot.clone(),
            Some(&["DynamicObservation".to_string()]),
            ChildCapabilityProfile::ReadOnly,
            false,
        )
        .unwrap();
        assert_eq!(plan.names, vec!["DynamicObservation"]);

        let missing = plan_tools_from_inventory(
            &config,
            snapshot,
            Some(&["MissingObservation".to_string()]),
            ChildCapabilityProfile::ReadOnly,
            false,
        );
        assert!(missing.is_err());
    }

    #[test]
    fn source_inventory_and_wrapper_reject_same_named_untrusted_writer() {
        let root = tempfile::tempdir().unwrap();
        let context = writer_context(root.path());
        let scope = claurst_tools::ToolCatalogScope::from_context(&context).unwrap();
        let tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(ForgedWrite)]);
        let snapshot = crate::tool_inventory::ToolInventorySnapshot::new(tools, 1, scope).unwrap();
        let requested = ["Write".to_string()];

        let error = match plan_tools_from_inventory(
            &Config::default(),
            snapshot,
            Some(&requested),
            ChildCapabilityProfile::Mutating,
            true,
        ) {
            Ok(_) => panic!("same-name writer must not enter a child inventory"),
            Err(error) => error,
        };
        assert!(error.contains("safely confined"));

        let error = match contain_writers(
            ChildToolPlan {
                tools: vec![Box::new(ForgedWrite)],
                names: vec!["Write".to_string()],
            },
            root.path(),
        ) {
            Ok(_) => panic!("wrapper must revalidate the exact live handle"),
            Err(error) => error,
        };
        assert!(error.contains("lacks the native scoped-filesystem capability"));
    }

    #[test]
    fn unsupported_requested_tools_and_profiles_fail_closed() {
        let config = Config::default();
        let requested = vec!["NotARealTool".into()];
        assert!(plan_tools(
            &config,
            Some(&requested),
            ChildCapabilityProfile::ReadOnly,
            false
        )
        .is_err());
        assert!(plan_tools(&config, None, ChildCapabilityProfile::Mutating, false).is_err());
        assert!(ChildCapabilityProfile::parse(Some("network"),).is_err());
    }

    #[test]
    fn mutating_profile_exposes_only_trusted_filesystem_writers() {
        let config = Config::default();
        let plan = plan_tools(&config, None, ChildCapabilityProfile::Mutating, true).unwrap();
        for expected in TRUSTED_FILESYSTEM_WRITERS {
            assert!(
                plan.names.iter().any(|name| name.as_str() == *expected),
                "missing native scoped writer {expected}"
            );
        }
        assert!(!plan.names.iter().any(|name| name == "Bash"));

        let requested = vec!["Bash".to_string()];
        let error = match plan_tools(
            &config,
            Some(&requested),
            ChildCapabilityProfile::Mutating,
            true,
        ) {
            Ok(_) => panic!("unconfinable command execution must fail explicitly"),
            Err(error) => error,
        };
        assert!(error.contains("safely confined"));
    }

    #[tokio::test]
    async fn real_write_rejects_absolute_parent_and_symlink_escape_before_execute() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let root = root.path().canonicalize().unwrap();
        let outside_file = outside.path().join("outside.txt");
        let context = writer_context(&root);
        let writer = ChildWriterTool {
            inner: Box::new(claurst_tools::FileWriteTool),
            root: root.clone(),
        };

        for path in [
            outside_file.display().to_string(),
            "../escape.txt".to_string(),
        ] {
            let result = writer
                .execute(
                    serde_json::json!({"file_path": path, "content": "sentinel"}),
                    &context,
                )
                .await;
            assert!(result.is_error, "path should be rejected: {path}");
        }
        assert!(!outside_file.exists());

        #[cfg(unix)]
        {
            let link = root.join("linked");
            std::os::unix::fs::symlink(outside.path(), &link).unwrap();
            let result = writer
                .execute(
                    serde_json::json!({"file_path": "linked/escape.txt", "content": "sentinel"}),
                    &context,
                )
                .await;
            assert!(result.is_error);
            assert!(!outside.path().join("escape.txt").exists());
        }
    }

    #[tokio::test]
    async fn admission_is_bounded_and_cancellation_while_waiting_is_observed() {
        let mut permits = Vec::new();
        for _ in 0..MAX_CHILD_INFERENCE {
            permits.push(
                acquire_child_admission(CancellationToken::new())
                    .await
                    .unwrap(),
            );
        }
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(acquire_child_admission(cancelled).await.is_err());
        drop(permits);
        assert!(acquire_child_admission(CancellationToken::new())
            .await
            .is_ok());
    }
}
