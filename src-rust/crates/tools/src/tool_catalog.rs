//! Immutable metadata-only tool discovery catalog.
//!
//! A catalog is built by the host from the exact tool handles authorized for a
//! query.  It intentionally never calls `Tool::input_schema`: deferred tools
//! stay metadata-only until query-owned activation has selected them.

use crate::{PermissionDecision, PermissionLevel, Tool, ToolContext};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const MAX_CATALOG_ENTRIES: usize = 4096;
pub const MAX_TOOL_NAME_BYTES: usize = 256;
pub const MAX_DESCRIPTION_CHARS: usize = 160;
pub const MAX_DISCOVERY_QUERY_BYTES: usize = 4096;
pub const MAX_DISCOVERY_TERMS: usize = 64;

static NEXT_CATALOG_ID: AtomicU64 = AtomicU64::new(1);

/// The immutable workspace/session binding for a catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCatalogScope {
    workspace: PathBuf,
    workspace_identity: WorkspaceIdentity,
    session_id: String,
}

/// Stable identity for the workspace directory represented by a catalog.
///
/// Unix device/inode pairs detect same-path replacement while remaining
/// unaffected by ordinary writes inside the directory. Platforms without a
/// stable standard-library file identity retain the canonical path binding;
/// their platform-specific identity APIs can be added without weakening the
/// Unix invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkspaceIdentity {
    #[cfg(unix)]
    Unix { device: u64, inode: u64 },
    #[cfg(not(unix))]
    CanonicalPath,
}

fn workspace_identity(path: &Path) -> Result<WorkspaceIdentity, ToolCatalogError> {
    let metadata = fs::metadata(path)
        .map_err(|error| ToolCatalogError::ScopeUnavailable(error.to_string()))?;
    if !metadata.is_dir() {
        return Err(ToolCatalogError::ScopeUnavailable(
            "catalog workspace is not a directory".to_string(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(WorkspaceIdentity::Unix {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Ok(WorkspaceIdentity::CanonicalPath)
    }
}

impl ToolCatalogScope {
    /// Capture the current host workspace and session identity. This metadata
    /// does not grant tool authority or construct a discovery receipt.
    pub fn from_context(ctx: &ToolContext) -> Result<Self, ToolCatalogError> {
        let workspace = crate::worktree::workspace_root(ctx)
            .canonicalize()
            .map_err(|error| ToolCatalogError::ScopeUnavailable(error.to_string()))?;
        let identity = workspace_identity(&workspace)?;
        if ctx.session_id.is_empty() || ctx.session_id.len() > 256 {
            return Err(ToolCatalogError::ScopeUnavailable(
                "catalog session id is empty or too long".to_string(),
            ));
        }
        Ok(Self {
            workspace,
            workspace_identity: identity,
            session_id: ctx.session_id.clone(),
        })
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn matches_context(&self, ctx: &ToolContext) -> bool {
        let Ok(current) = Self::from_context(ctx) else {
            return false;
        };
        current == self.clone()
    }
}

/// Source class used by query-owned wire assembly.  The descriptor contains
/// no schema or executable tool handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCatalogSource {
    Native,
    DeferredNative,
    DeferredMcp,
}

/// Bounded metadata for one authorized tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCatalogEntry {
    name: String,
    description: String,
    permission: PermissionLevel,
    effect: crate::ToolEffect,
    source: ToolCatalogSource,
}

impl ToolCatalogEntry {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn permission(&self) -> PermissionLevel {
        self.permission
    }

    pub fn effect(&self) -> crate::ToolEffect {
        self.effect
    }

    pub fn source(&self) -> ToolCatalogSource {
        self.source
    }

    pub fn deferred(&self) -> bool {
        !matches!(self.source, ToolCatalogSource::Native)
    }
}

/// Immutable metadata view bound to one workspace/session and catalog epoch.
#[derive(Debug, Clone)]
pub struct ToolCatalogView {
    catalog_id: u64,
    scope: ToolCatalogScope,
    entries: Arc<[ToolCatalogEntry]>,
}

impl ToolCatalogView {
    /// Build from the exact final tool handles selected by the host.
    ///
    /// This performs policy and capability filtering but does not prompt. An
    /// `Ask` capability remains discoverable and is checked again at execute
    /// time. Denied entries are omitted. Input schemas are deliberately not
    /// evaluated here, including for deferred or MCP tools.
    pub fn from_tools(
        ctx: &ToolContext,
        tools: &[Box<dyn Tool>],
    ) -> Result<Arc<Self>, ToolCatalogError> {
        let scope = ToolCatalogScope::from_context(ctx)?;
        if tools.len() > MAX_CATALOG_ENTRIES {
            return Err(ToolCatalogError::CatalogTooLarge {
                actual: tools.len(),
                limit: MAX_CATALOG_ENTRIES,
            });
        }

        let mut entries = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = tool.name();
            if name.is_empty()
                || name.len() > MAX_TOOL_NAME_BYTES
                || name.chars().any(char::is_control)
            {
                return Err(ToolCatalogError::InvalidToolName(name.to_string()));
            }
            if !crate::tool_policy::tool_allowed(&ctx.config, name)
                .map_err(ToolCatalogError::InvalidPolicy)?
            {
                continue;
            }
            let is_read_only = matches!(
                tool.permission_level(),
                PermissionLevel::None | PermissionLevel::ReadOnly
            );
            if matches!(
                ctx.check_capability(name, tool.description(), is_read_only),
                PermissionDecision::Deny | PermissionDecision::DenyPermanently
            ) {
                continue;
            }
            let description = compact_description(tool.description());
            let source = tool.catalog_source();
            entries.push(ToolCatalogEntry {
                name: name.to_string(),
                description,
                permission: tool.permission_level(),
                effect: tool.effect(),
                source,
            });
        }

        entries.sort_by(|left, right| left.name.cmp(&right.name));
        let mut canonical_names = BTreeSet::new();
        for entry in &entries {
            if !canonical_names.insert(entry.name.to_ascii_lowercase()) {
                return Err(ToolCatalogError::DuplicateTool(entry.name.clone()));
            }
        }
        Ok(Arc::new(Self {
            catalog_id: NEXT_CATALOG_ID.fetch_add(1, Ordering::Relaxed),
            scope,
            entries: entries.into(),
        }))
    }

    pub fn catalog_id(&self) -> u64 {
        self.catalog_id
    }

    pub fn scope(&self) -> &ToolCatalogScope {
        &self.scope
    }

    pub fn entries(&self) -> &[ToolCatalogEntry] {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&ToolCatalogEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn matches_context(&self, ctx: &ToolContext) -> bool {
        self.scope.matches_context(ctx)
    }

    pub fn contains_name(&self, name: &str) -> bool {
        self.entry(name).is_some()
    }
}

/// Private host-only authority produced by a successful ToolSearch call.
///
/// It is intentionally not serializable. Query code must validate it against
/// the same catalog view before resolving selected names to executable handles.
#[derive(Clone)]
pub struct DiscoveryReceipt {
    catalog_id: u64,
    scope: ToolCatalogScope,
    selected_names: Arc<[String]>,
}

impl fmt::Debug for DiscoveryReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryReceipt")
            .field("catalog_id", &self.catalog_id)
            .field("scope", &self.scope)
            .field("selected_count", &self.selected_names.len())
            .finish()
    }
}

impl DiscoveryReceipt {
    pub(crate) fn new(view: &ToolCatalogView, selected_names: Vec<String>) -> Self {
        Self {
            catalog_id: view.catalog_id,
            scope: view.scope.clone(),
            selected_names: selected_names.into(),
        }
    }

    pub fn selected_names(&self) -> &[String] {
        &self.selected_names
    }

    pub fn matches_view(&self, view: &ToolCatalogView, ctx: &ToolContext) -> bool {
        self.catalog_id == view.catalog_id
            && self.scope == view.scope().clone()
            && view.matches_context(ctx)
            && self
                .selected_names
                .iter()
                .all(|name| view.contains_name(name))
    }

    /// Validate and clone the selected canonical names for query-owned handle
    /// resolution. This does not expose or mint an executable tool handle.
    pub fn resolve_names(
        &self,
        view: &ToolCatalogView,
        ctx: &ToolContext,
    ) -> Result<Vec<String>, ToolCatalogError> {
        if !self.matches_view(view, ctx) {
            return Err(ToolCatalogError::ReceiptScopeMismatch);
        }
        Ok(self.selected_names.to_vec())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCatalogError {
    ScopeUnavailable(String),
    InvalidPolicy(String),
    CatalogTooLarge { actual: usize, limit: usize },
    InvalidToolName(String),
    DuplicateTool(String),
    SelectionUnavailable(Vec<String>),
    MissingCatalog,
    ScopeMismatch,
    QueryTooLarge { bytes: usize, limit: usize },
    TooManyTerms { actual: usize, limit: usize },
    SelectionTooLarge { actual: usize, limit: usize },
    EmptySelection,
    ReceiptScopeMismatch,
}

impl fmt::Display for ToolCatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScopeUnavailable(reason) => {
                write!(formatter, "tool catalog scope unavailable: {reason}")
            }
            Self::InvalidPolicy(reason) => {
                write!(formatter, "tool catalog policy invalid: {reason}")
            }
            Self::CatalogTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "tool catalog has {actual} entries; limit is {limit}"
                )
            }
            Self::InvalidToolName(name) => write!(formatter, "invalid tool name: {name}"),
            Self::DuplicateTool(name) => write!(formatter, "duplicate tool name: {name}"),
            Self::SelectionUnavailable(names) => {
                write!(
                    formatter,
                    "requested tools are unavailable: {}",
                    names.join(", ")
                )
            }
            Self::MissingCatalog => write!(
                formatter,
                "tool discovery catalog is not bound to this query"
            ),
            Self::ScopeMismatch => write!(
                formatter,
                "tool discovery catalog belongs to another workspace or session"
            ),
            Self::QueryTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "tool discovery query is {bytes} bytes; limit is {limit}"
                )
            }
            Self::TooManyTerms { actual, limit } => {
                write!(
                    formatter,
                    "tool discovery query has {actual} terms; limit is {limit}"
                )
            }
            Self::SelectionTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "tool discovery selection has {actual} names; limit is {limit}"
                )
            }
            Self::EmptySelection => write!(formatter, "tool discovery selection is empty"),
            Self::ReceiptScopeMismatch => {
                write!(formatter, "tool discovery receipt is stale or out of scope")
            }
        }
    }
}

impl std::error::Error for ToolCatalogError {}

pub(crate) fn compact_description(description: &str) -> String {
    let mut output = String::new();
    let mut used = 0usize;
    for word in description.split_whitespace() {
        if used == MAX_DESCRIPTION_CHARS {
            break;
        }
        if used > 0 {
            output.push(' ');
            used += 1;
            if used == MAX_DESCRIPTION_CHARS {
                break;
            }
        }
        let remaining = MAX_DESCRIPTION_CHARS - used;
        let mut consumed = 0usize;
        let mut truncated = false;
        for character in word.chars() {
            if consumed == remaining {
                truncated = true;
                break;
            }
            output.push(character);
            consumed += 1;
        }
        used += consumed;
        if truncated {
            break;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    struct PanicSchemaTool;

    #[async_trait::async_trait]
    impl Tool for PanicSchemaTool {
        fn name(&self) -> &str {
            "Deferred"
        }
        fn description(&self) -> &str {
            "deferred tool"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn advanced(&self) -> bool {
            true
        }
        fn input_schema(&self) -> serde_json::Value {
            panic!("schema must stay deferred")
        }
        async fn execute(&self, _: serde_json::Value, _: &ToolContext) -> crate::ToolResult {
            crate::ToolResult::success("ok")
        }
    }

    #[test]
    fn catalog_never_eagerly_reads_schema_and_marks_deferred() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(PanicSchemaTool)];
        let view = ToolCatalogView::from_tools(&ctx, &tools).expect("catalog");
        assert_eq!(view.entries().len(), 1);
        assert!(view.entries()[0].deferred());
    }

    #[test]
    fn receipt_is_bound_to_catalog_scope_and_selection() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(PanicSchemaTool)];
        let view = ToolCatalogView::from_tools(&ctx, &tools).expect("catalog");
        let receipt = DiscoveryReceipt::new(&view, vec!["Deferred".to_string()]);
        assert_eq!(
            receipt.resolve_names(&view, &ctx).unwrap(),
            vec!["Deferred"]
        );
        let mut other = ctx.clone();
        other.session_id = "other".to_string();
        assert_eq!(
            receipt.resolve_names(&view, &other),
            Err(ToolCatalogError::ReceiptScopeMismatch)
        );
    }

    #[test]
    fn catalog_filters_denied_and_bounds_metadata() {
        let mut ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        ctx.config.disallowed_tools = vec!["Denied".to_string()];
        struct Simple;
        #[async_trait::async_trait]
        impl Tool for Simple {
            fn name(&self) -> &str {
                "Denied"
            }
            fn description(&self) -> &str {
                "x"
            }
            fn permission_level(&self) -> PermissionLevel {
                PermissionLevel::ReadOnly
            }
            fn input_schema(&self) -> serde_json::Value {
                json!({})
            }
            async fn execute(&self, _: serde_json::Value, _: &ToolContext) -> crate::ToolResult {
                crate::ToolResult::success("")
            }
        }
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(Simple)];
        let view = ToolCatalogView::from_tools(&ctx, &tools).expect("catalog");
        assert!(view.entries().is_empty());
        assert_eq!(
            compact_description(&"x ".repeat(100_000)).chars().count(),
            160
        );
    }

    #[test]
    fn mcp_name_alone_does_not_mark_an_entry_deferred() {
        struct PrefixTool;
        #[async_trait::async_trait]
        impl Tool for PrefixTool {
            fn name(&self) -> &str {
                "mcp__fixture_allowed_echo"
            }
            fn description(&self) -> &str {
                "fixture"
            }
            fn permission_level(&self) -> PermissionLevel {
                PermissionLevel::ReadOnly
            }
            fn input_schema(&self) -> serde_json::Value {
                json!({})
            }
            async fn execute(&self, _: serde_json::Value, _: &ToolContext) -> crate::ToolResult {
                crate::ToolResult::success("")
            }
        }
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(PrefixTool)];
        let view = ToolCatalogView::from_tools(&ctx, &tools).expect("catalog");
        assert_eq!(view.entries()[0].source(), ToolCatalogSource::Native);
    }

    struct NamedReadOnly(String);

    #[async_trait::async_trait]
    impl Tool for NamedReadOnly {
        fn name(&self) -> &str {
            &self.0
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn input_schema(&self) -> serde_json::Value {
            json!({})
        }
        async fn execute(&self, _: serde_json::Value, _: &ToolContext) -> crate::ToolResult {
            crate::ToolResult::success("")
        }
    }

    #[test]
    fn catalog_rejects_case_fold_ambiguous_names_and_controls() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let ambiguous: Vec<Box<dyn Tool>> = vec![
            Box::new(NamedReadOnly("A".to_string())),
            Box::new(NamedReadOnly("B".to_string())),
            Box::new(NamedReadOnly("a".to_string())),
        ];
        assert!(matches!(
            ToolCatalogView::from_tools(&ctx, &ambiguous),
            Err(ToolCatalogError::DuplicateTool(_))
        ));
        let controls: Vec<Box<dyn Tool>> = vec![Box::new(NamedReadOnly("line\nfeed".to_string()))];
        assert!(matches!(
            ToolCatalogView::from_tools(&ctx, &controls),
            Err(ToolCatalogError::InvalidToolName(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn catalog_scope_rejects_same_path_workspace_replacement() {
        let root = tempfile::tempdir().expect("temporary root");
        let workspace = root.path().join("workspace");
        fs::create_dir(&workspace).expect("workspace");
        let ctx = crate::test_support::allow_all_context(workspace.clone());
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(PanicSchemaTool)];
        let view = ToolCatalogView::from_tools(&ctx, &tools).expect("catalog");
        let receipt = DiscoveryReceipt::new(&view, vec!["Deferred".to_string()]);

        fs::remove_dir(&workspace).expect("remove original workspace");
        fs::create_dir(&workspace).expect("replace workspace");
        assert!(!view.matches_context(&ctx));
        assert_eq!(
            receipt.resolve_names(&view, &ctx),
            Err(ToolCatalogError::ReceiptScopeMismatch)
        );
    }
}
