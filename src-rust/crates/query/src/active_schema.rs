//! Query-owned active tool schemas for progressive tool disclosure.
//!
//! The host catalog is metadata-only.  This module resolves input schemas from
//! the exact borrowed tool slice only when a tool is initially disclosed or a
//! host discovery receipt activates it.

use claurst_api::ApiToolDefinition;
use claurst_core::crypto_utils::sha256_hex;
use claurst_core::types::ToolDefinition;
use claurst_tools::{DiscoveryReceipt, Tool, ToolCatalogError, ToolCatalogView, ToolContext};
use std::collections::BTreeMap;
use std::fmt;

pub(crate) const MAX_ACTIVE_SCHEMAS: usize = 128;
pub(crate) const MAX_SCHEMA_BYTES: usize = 64 * 1024;
pub(crate) const MAX_ACTIVE_SCHEMA_BYTES: usize = 1024 * 1024;

/// The query-owned immutable set of schemas disclosed on the current request.
#[derive(Debug, Clone)]
pub(crate) struct ActiveSchemaSet {
    catalog_id: u64,
    active_revision: u64,
    definitions: BTreeMap<String, ToolDefinition>,
    digest: String,
}

impl ActiveSchemaSet {
    /// Build the initial set from the final host-authorized tool slice.
    pub(crate) fn initial(
        view: &ToolCatalogView,
        tools: &[Box<dyn Tool>],
        search_authorized: bool,
    ) -> Result<Self, ActiveSchemaError> {
        let mut names = Vec::new();
        for entry in view.entries() {
            if !search_authorized || !entry.deferred() || entry.name() == "ToolSearch" {
                names.push(entry.name().to_string());
            }
        }
        Self::from_names(view, tools, names, 0)
    }

    /// Transactionally merge one captured batch of discovery receipts.
    ///
    /// The current set is unchanged when any receipt is stale, out of scope,
    /// missing, or would exceed a finite schema cap.
    pub(crate) fn activate_receipts(
        &self,
        view: &ToolCatalogView,
        ctx: &ToolContext,
        tools: &[Box<dyn Tool>],
        receipts: &[DiscoveryReceipt],
    ) -> Result<Self, ActiveSchemaError> {
        if self.catalog_id != view.catalog_id() || !view.matches_context(ctx) {
            return Err(ActiveSchemaError::CatalogChanged);
        }

        let mut definitions = self.definitions.clone();
        let before = definitions.len();
        for receipt in receipts {
            for name in receipt.resolve_names(view, ctx)? {
                if let std::collections::btree_map::Entry::Vacant(entry) = definitions.entry(name) {
                    let definition = Self::resolve_definition(view, tools, entry.key())?;
                    entry.insert(definition);
                }
            }
        }
        if definitions.len() == before {
            return Ok(self.clone());
        }
        Self::from_definitions(view, definitions, self.active_revision.saturating_add(1))
    }

    fn from_names(
        view: &ToolCatalogView,
        tools: &[Box<dyn Tool>],
        names: Vec<String>,
        active_revision: u64,
    ) -> Result<Self, ActiveSchemaError> {
        if names.len() > MAX_ACTIVE_SCHEMAS {
            return Err(ActiveSchemaError::ActiveCountExceeded {
                actual: names.len(),
                limit: MAX_ACTIVE_SCHEMAS,
            });
        }

        let mut definitions = BTreeMap::new();
        for name in names {
            let definition = Self::resolve_definition(view, tools, &name)?;
            definitions.insert(name, definition);
        }
        Self::from_definitions(view, definitions, active_revision)
    }

    fn resolve_definition(
        view: &ToolCatalogView,
        tools: &[Box<dyn Tool>],
        name: &str,
    ) -> Result<ToolDefinition, ActiveSchemaError> {
        let Some(entry) = view.entry(name) else {
            return Err(ActiveSchemaError::ToolNotInCatalog(name.to_string()));
        };
        let Some(tool) = tools.iter().find(|tool| tool.name() == name) else {
            return Err(ActiveSchemaError::ToolNotInSlice(name.to_string()));
        };
        let definition = tool.to_definition();
        if entry.name() != definition.name {
            return Err(ActiveSchemaError::DefinitionNameMismatch {
                expected: entry.name().to_string(),
                actual: definition.name,
            });
        }
        Ok(definition)
    }

    fn from_definitions(
        view: &ToolCatalogView,
        definitions: BTreeMap<String, ToolDefinition>,
        active_revision: u64,
    ) -> Result<Self, ActiveSchemaError> {
        if definitions.len() > MAX_ACTIVE_SCHEMAS {
            return Err(ActiveSchemaError::ActiveCountExceeded {
                actual: definitions.len(),
                limit: MAX_ACTIVE_SCHEMAS,
            });
        }

        let mut total_definition_bytes = 0usize;
        for (name, definition) in &definitions {
            if view.entry(name).is_none() {
                return Err(ActiveSchemaError::ToolNotInCatalog(name.clone()));
            }
            if name != &definition.name {
                return Err(ActiveSchemaError::DefinitionNameMismatch {
                    expected: name.clone(),
                    actual: definition.name.clone(),
                });
            }
            let schema_bytes = serde_json::to_vec(&definition.input_schema)
                .map_err(|error| ActiveSchemaError::SchemaEncoding(error.to_string()))?;
            if schema_bytes.len() > MAX_SCHEMA_BYTES {
                return Err(ActiveSchemaError::SchemaTooLarge {
                    name: name.clone(),
                    actual: schema_bytes.len(),
                    limit: MAX_SCHEMA_BYTES,
                });
            }
            let encoded_definition = serde_json::to_vec(definition)
                .map_err(|error| ActiveSchemaError::SchemaEncoding(error.to_string()))?;
            total_definition_bytes = total_definition_bytes
                .checked_add(encoded_definition.len())
                .ok_or(ActiveSchemaError::AggregateSchemaTooLarge {
                    actual: usize::MAX,
                    limit: MAX_ACTIVE_SCHEMA_BYTES,
                })?;
            if total_definition_bytes > MAX_ACTIVE_SCHEMA_BYTES {
                return Err(ActiveSchemaError::AggregateSchemaTooLarge {
                    actual: total_definition_bytes,
                    limit: MAX_ACTIVE_SCHEMA_BYTES,
                });
            }
        }

        let digest_input = definitions
            .values()
            .map(|definition| {
                serde_json::to_string(definition)
                    .map(|encoded| format!("{}\0{}\n", definition.name, encoded))
                    .map_err(|error| ActiveSchemaError::SchemaEncoding(error.to_string()))
            })
            .collect::<Result<String, _>>()?;

        Ok(Self {
            catalog_id: view.catalog_id(),
            active_revision,
            definitions,
            digest: sha256_hex(digest_input.as_bytes()),
        })
    }

    pub(crate) fn catalog_id(&self) -> u64 {
        self.catalog_id
    }

    pub(crate) fn active_revision(&self) -> u64 {
        self.active_revision
    }

    pub(crate) fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn is_disclosed(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    pub(crate) fn definitions(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.definitions.values()
    }

    pub(crate) fn api_definitions(&self) -> Vec<ApiToolDefinition> {
        self.definitions
            .values()
            .map(ApiToolDefinition::from)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ActiveSchemaError {
    CatalogChanged,
    ToolNotInCatalog(String),
    ToolNotInSlice(String),
    DefinitionNameMismatch {
        expected: String,
        actual: String,
    },
    ActiveCountExceeded {
        actual: usize,
        limit: usize,
    },
    SchemaTooLarge {
        name: String,
        actual: usize,
        limit: usize,
    },
    AggregateSchemaTooLarge {
        actual: usize,
        limit: usize,
    },
    SchemaEncoding(String),
    Catalog(ToolCatalogError),
}

impl From<ToolCatalogError> for ActiveSchemaError {
    fn from(error: ToolCatalogError) -> Self {
        Self::Catalog(error)
    }
}

impl fmt::Display for ActiveSchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CatalogChanged => formatter.write_str("tool schema catalog changed"),
            Self::ToolNotInCatalog(name) => write!(formatter, "tool is not in the catalog: {name}"),
            Self::ToolNotInSlice(name) => {
                write!(formatter, "tool is not in the retained tool set: {name}")
            }
            Self::DefinitionNameMismatch { .. } => formatter.write_str("tool schema name mismatch"),
            Self::ActiveCountExceeded { actual, limit } => {
                write!(
                    formatter,
                    "active tool schema count {actual} exceeds limit {limit}"
                )
            }
            Self::SchemaTooLarge { actual, limit, .. } => {
                write!(formatter, "tool schema is {actual} bytes; limit is {limit}")
            }
            Self::AggregateSchemaTooLarge { actual, limit } => {
                write!(
                    formatter,
                    "active tool schemas total {actual} bytes; limit is {limit}"
                )
            }
            Self::SchemaEncoding(_) => formatter.write_str("tool schema could not be encoded"),
            Self::Catalog(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ActiveSchemaError {}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use claurst_tools::{ToolResult, ToolSearchTool};
    use serde_json::{json, Value};
    use std::sync::Arc;

    struct AllowAll;

    impl claurst_core::permissions::PermissionHandler for AllowAll {
        fn check_permission(
            &self,
            _: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }

        fn request_permission(
            &self,
            _: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Allow
        }
    }

    fn allow_all_context() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(AllowAll),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "active-schema-test".to_string(),
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

    struct TestTool {
        name: String,
        deferred: bool,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn permission_level(&self) -> claurst_tools::PermissionLevel {
            claurst_tools::PermissionLevel::ReadOnly
        }
        fn advanced(&self) -> bool {
            self.deferred
        }
        fn input_schema(&self) -> serde_json::Value {
            json!({"type":"object"})
        }
        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    #[test]
    fn initial_set_omits_deferred_tools_when_search_is_available() {
        let ctx = allow_all_context();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(TestTool {
                name: "Normal".into(),
                deferred: false,
            }),
            Box::new(TestTool {
                name: "Deferred".into(),
                deferred: true,
            }),
        ];
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&view, &tools, true).unwrap();
        assert!(active.is_disclosed("Normal"));
        assert!(!active.is_disclosed("Deferred"));
    }

    #[tokio::test]
    async fn receipt_activation_is_bounded_and_deterministic() {
        let ctx = allow_all_context();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(TestTool {
                name: "Normal".into(),
                deferred: false,
            }),
            Box::new(TestTool {
                name: "Deferred".into(),
                deferred: true,
            }),
        ];
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&view, &tools, true).unwrap();
        let mut search_ctx = ctx.clone();
        search_ctx.catalog_view = Some(view.clone());
        let mut search_result = ToolSearchTool
            .execute(json!({"query": "select:Deferred"}), &search_ctx)
            .await;
        let receipt = search_result
            .take_discovery_receipt()
            .expect("search receipt");
        let expanded = active
            .activate_receipts(&view, &ctx, &tools, std::slice::from_ref(&receipt))
            .unwrap();
        assert!(expanded.is_disclosed("Deferred"));
        assert_eq!(expanded.active_revision(), 1);
        assert_ne!(expanded.digest(), active.digest());
        let unchanged = expanded
            .activate_receipts(&view, &ctx, &tools, std::slice::from_ref(&receipt))
            .unwrap();
        assert_eq!(unchanged.active_revision(), expanded.active_revision());
        assert_eq!(unchanged.digest(), expanded.digest());

        let stale_view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let stale = active.activate_receipts(&stale_view, &ctx, &tools, &[receipt]);
        assert!(matches!(stale, Err(ActiveSchemaError::CatalogChanged)));
        assert!(!active.is_disclosed("Deferred"));
    }

    #[test]
    fn thousand_deferred_tools_do_not_enter_initial_wire_schema_set() {
        let ctx = allow_all_context();
        let mut tools: Vec<Box<dyn Tool>> = vec![Box::new(ToolSearchTool)];
        tools.extend((0..1_000).map(|index| {
            Box::new(TestTool {
                name: format!("Deferred{index}"),
                deferred: true,
            }) as Box<dyn Tool>
        }));
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&view, &tools, true).unwrap();
        assert_eq!(active.definitions().count(), 1);
        assert!(active.is_disclosed("ToolSearch"));
        assert!(!active.is_disclosed("Deferred999"));
        assert!(matches!(
            ActiveSchemaSet::initial(&view, &tools, false),
            Err(ActiveSchemaError::ActiveCountExceeded { .. })
        ));
    }

    #[test]
    fn small_inventory_without_tool_search_remains_fully_disclosed() {
        let ctx = allow_all_context();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(TestTool {
                name: "ReadOnlyA".into(),
                deferred: false,
            }),
            Box::new(TestTool {
                name: "ReadOnlyB".into(),
                deferred: false,
            }),
        ];
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&view, &tools, false).unwrap();

        assert_eq!(active.definitions().count(), 2);
        assert!(active.is_disclosed("ReadOnlyA"));
        assert!(active.is_disclosed("ReadOnlyB"));
    }

    #[test]
    fn schema_order_and_digest_are_stable_for_reordered_tool_slices() {
        let ctx = allow_all_context();
        let first_tools: Vec<Box<dyn Tool>> = vec![
            Box::new(TestTool {
                name: "Beta".into(),
                deferred: false,
            }),
            Box::new(TestTool {
                name: "Alpha".into(),
                deferred: false,
            }),
        ];
        let second_tools: Vec<Box<dyn Tool>> = vec![
            Box::new(TestTool {
                name: "Alpha".into(),
                deferred: false,
            }),
            Box::new(TestTool {
                name: "Beta".into(),
                deferred: false,
            }),
        ];
        let first_view = ToolCatalogView::from_tools(&ctx, &first_tools).unwrap();
        let second_view = ToolCatalogView::from_tools(&ctx, &second_tools).unwrap();
        let first = ActiveSchemaSet::initial(&first_view, &first_tools, false).unwrap();
        let second = ActiveSchemaSet::initial(&second_view, &second_tools, false).unwrap();

        assert_eq!(
            serde_json::to_vec(&first.api_definitions()).unwrap(),
            serde_json::to_vec(&second.api_definitions()).unwrap()
        );
        assert_eq!(first.digest(), second.digest());
        assert_eq!(
            first
                .definitions()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Alpha", "Beta"]
        );
    }

    #[test]
    fn aggregate_cap_counts_full_definition_bytes() {
        let ctx = allow_all_context();
        let tools: Vec<Box<dyn Tool>> = (0..128)
            .map(|index| {
                Box::new(TestTool {
                    name: format!("Large{index}"),
                    deferred: false,
                }) as Box<dyn Tool>
            })
            .collect();
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let definitions = view
            .entries()
            .iter()
            .map(|entry| {
                (
                    entry.name().to_string(),
                    ToolDefinition {
                        name: entry.name().to_string(),
                        description: "description ".repeat(1_000),
                        input_schema: json!({"type": "object"}),
                    },
                )
            })
            .collect();
        assert!(matches!(
            ActiveSchemaSet::from_definitions(&view, definitions, 0),
            Err(ActiveSchemaError::AggregateSchemaTooLarge { .. })
        ));
    }

    async fn select_receipt(
        ctx: &ToolContext,
        view: &Arc<ToolCatalogView>,
        names: &str,
    ) -> DiscoveryReceipt {
        let mut scoped = ctx.clone();
        scoped.catalog_view = Some(view.clone());
        ToolSearchTool
            .execute(json!({"query": format!("select:{names}")}), &scoped)
            .await
            .take_discovery_receipt()
            .expect("authorized scoped selection")
    }

    #[tokio::test]
    async fn joined_receipts_merge_once_and_a_stale_member_rejects_the_whole_proposal() {
        let ctx = allow_all_context();
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(ToolSearchTool),
            Box::new(TestTool {
                name: "First".into(),
                deferred: true,
            }),
            Box::new(TestTool {
                name: "Second".into(),
                deferred: true,
            }),
        ];
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&view, &tools, true).unwrap();
        let first = select_receipt(&ctx, &view, "First").await;
        let second = select_receipt(&ctx, &view, "Second,First").await;
        let merged = active
            .activate_receipts(&view, &ctx, &tools, &[first.clone(), second])
            .unwrap();
        assert_eq!(merged.active_revision(), 1);
        assert_eq!(merged.definitions().count(), 3);

        let foreign = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let stale = select_receipt(&ctx, &foreign, "Second").await;
        let digest = active.digest().to_owned();
        assert!(active
            .activate_receipts(&view, &ctx, &tools, &[first, stale])
            .is_err());
        assert_eq!(active.digest(), digest);
        assert!(!active.is_disclosed("First"));
        assert!(!active.is_disclosed("Second"));
    }

    struct OversizedSchema;

    #[async_trait]
    impl Tool for OversizedSchema {
        fn name(&self) -> &str {
            "Oversized"
        }
        fn description(&self) -> &str {
            "oversized deferred fixture"
        }
        fn permission_level(&self) -> claurst_tools::PermissionLevel {
            claurst_tools::PermissionLevel::ReadOnly
        }
        fn advanced(&self) -> bool {
            true
        }
        fn input_schema(&self) -> Value {
            json!({"value": "x".repeat(MAX_SCHEMA_BYTES)})
        }
        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            panic!("not invoked")
        }
    }

    #[tokio::test]
    async fn schema_and_active_count_caps_leave_the_previous_set_intact() {
        let ctx = allow_all_context();
        let mut tools: Vec<Box<dyn Tool>> =
            vec![Box::new(ToolSearchTool), Box::new(OversizedSchema)];
        tools.extend((0..127).map(|index| {
            Box::new(TestTool {
                name: format!("Normal{index}"),
                deferred: false,
            }) as Box<dyn Tool>
        }));
        tools.push(Box::new(TestTool {
            name: "Deferred".into(),
            deferred: true,
        }));
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let active = ActiveSchemaSet::initial(&view, &tools, true).unwrap();
        assert_eq!(active.definitions().count(), MAX_ACTIVE_SCHEMAS);
        let count_receipt = select_receipt(&ctx, &view, "Deferred").await;
        assert!(matches!(
            active.activate_receipts(&view, &ctx, &tools, &[count_receipt]),
            Err(ActiveSchemaError::ActiveCountExceeded { .. })
        ));
        assert!(!active.is_disclosed("Deferred"));

        let sparse: Vec<Box<dyn Tool>> = vec![
            Box::new(ToolSearchTool),
            Box::new(OversizedSchema),
            Box::new(TestTool {
                name: "Valid".into(),
                deferred: true,
            }),
        ];
        let sparse_view = ToolCatalogView::from_tools(&ctx, &sparse).unwrap();
        let sparse_active = ActiveSchemaSet::initial(&sparse_view, &sparse, true).unwrap();
        let receipt = select_receipt(&ctx, &sparse_view, "Valid,Oversized").await;
        assert!(matches!(
            sparse_active.activate_receipts(&sparse_view, &ctx, &sparse, &[receipt]),
            Err(ActiveSchemaError::SchemaTooLarge { .. })
        ));
        assert!(!sparse_active.is_disclosed("Valid"));
        assert!(!sparse_active.is_disclosed("Oversized"));
    }

    #[test]
    fn nested_schema_object_insertion_order_does_not_change_wire_or_digest() {
        let ctx = allow_all_context();
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(TestTool {
            name: "Normal".into(),
            deferred: false,
        })];
        let view = ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let build = |keys: [&str; 2]| {
            let mut properties = serde_json::Map::new();
            for key in keys {
                properties.insert(key.into(), json!({"type": "string"}));
            }
            ActiveSchemaSet::from_definitions(
                &view,
                BTreeMap::from([(
                    "Normal".into(),
                    ToolDefinition {
                        name: "Normal".into(),
                        description: "test".into(),
                        input_schema: json!({"type": "object", "properties": properties}),
                    },
                )]),
                0,
            )
            .unwrap()
        };
        let first = build(["zeta", "alpha"]);
        let second = build(["alpha", "zeta"]);
        assert_eq!(first.digest(), second.digest());
        assert_eq!(
            serde_json::to_vec(&first.api_definitions()).unwrap(),
            serde_json::to_vec(&second.api_definitions()).unwrap()
        );
    }
}
