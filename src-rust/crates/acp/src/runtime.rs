//! Shared agent runtime owned by the ACP server.
//!
//! Built once on startup and reused for every session. Per-session state
//! (cwd, transcript, cancellation token, permission queue) is layered on
//! top via `sessions::SessionState`.

use std::path::PathBuf;
use std::sync::Arc;

use claurst_core::config::{Config, Settings};
use claurst_core::permissions::PermissionManager;
use claurst_core::CostTracker;
use claurst_query::QueryConfig;
use claurst_tools::Tool;

/// Snapshot of the global agent runtime — built at server startup, cloned
/// (cheaply, via Arc) into each session.
#[derive(Clone)]
pub struct AgentRuntime {
    pub config: Config,
    pub settings: Settings,
    pub api_client: Arc<claurst_api::AnthropicClient>,
    pub provider_registry: Arc<claurst_api::ProviderRegistry>,
    pub tools: Arc<Vec<Box<dyn Tool>>>,
    pub cost_tracker: Arc<CostTracker>,
    pub query_config: QueryConfig,
    pub mcp_manager: Option<Arc<claurst_mcp::McpManager>>,
    pub permission_manager: Arc<std::sync::Mutex<PermissionManager>>,
    pub working_dir: PathBuf,
    session_owned_mcp: bool,
}

impl AgentRuntime {
    /// Build the runtime from on-disk settings, env vars, and a working
    /// directory. Mirrors the headless startup path but with ACP-specific
    /// defaults (non-interactive, permission decisions routed back to the
    /// connected client).
    pub async fn build(working_dir: PathBuf) -> anyhow::Result<Self> {
        let settings = Settings::load_sync()
            .map_err(|_| anyhow::anyhow!("ACP settings could not be loaded"))?;
        let mut config = settings.effective_config();
        // Plan mode requires interactive UI — fall back to Default so the
        // ACP permission bridge can route decisions to the client.
        if config.permission_mode == claurst_core::PermissionMode::Plan {
            config.permission_mode = claurst_core::PermissionMode::Default;
        }
        config.project_dir = Some(working_dir.clone());

        let active_provider = config.selected_provider_id().to_string();
        let (api_key, use_bearer_auth) = if active_provider == "anthropic" {
            config
                .resolve_anthropic_auth_async()
                .await
                .unwrap_or_default()
        } else {
            (String::new(), false)
        };

        let client_config = claurst_api::client::ClientConfig {
            api_key: api_key.clone(),
            api_base: config.resolve_anthropic_api_base(),
            use_bearer_auth,
            ..Default::default()
        };
        let api_client = Arc::new(claurst_api::AnthropicClient::new(client_config.clone())?);
        let provider_registry = Arc::new(claurst_api::ProviderRegistry::from_config(
            &config,
            client_config,
        ));

        let permission_manager = Arc::new(std::sync::Mutex::new(PermissionManager::new(
            config.permission_mode.clone(),
            &settings,
        )));

        let cost_tracker = CostTracker::new();

        // MCP servers from settings connect once for all sessions. Validated
        // client-supplied stdio rosters are added by each session runtime.
        let mcp_manager = build_mcp_manager(&config, &settings, &working_dir).await;

        // Retain native handles once. The shared source adds current MCP
        // descriptors and binds each prompt to its session workspace.
        let mut tools: Vec<Box<dyn Tool>> = claurst_tools::all_tools();
        tools.push(Box::new(claurst_query::AgentTool));
        let tools = Arc::new(tools);

        let mut query_config = QueryConfig::from_config(&config);
        query_config.working_directory = Some(working_dir.display().to_string());
        query_config.provider_registry = Some(provider_registry.clone());
        query_config.inventory_profile = claurst_query::ToolInventoryProfile::Acp;
        query_config.inventory_source =
            Some(claurst_query::host_inventory::HostInventorySource::new(
                Arc::clone(&tools),
                &config,
                None,
            )?);

        Ok(Self {
            config,
            settings,
            api_client,
            provider_registry,
            tools,
            cost_tracker,
            query_config,
            mcp_manager,
            permission_manager,
            working_dir,
            session_owned_mcp: false,
        })
    }

    /// Reuse API/native handles, but own a session-scoped inventory cache so
    /// its MCP wrappers cannot be retained by another session's cache.
    pub async fn for_session(
        &self,
        working_dir: PathBuf,
        servers: &[claurst_core::config::McpServerConfig],
        admission: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    ) -> anyhow::Result<Arc<Self>> {
        let mut scoped = self.clone();
        scoped.session_owned_mcp = false;
        scoped.config.project_dir = Some(working_dir.clone());
        scoped.working_dir = working_dir.clone();
        scoped.query_config.working_directory = Some(working_dir.display().to_string());
        scoped.query_config.inventory_source =
            Some(claurst_query::host_inventory::HostInventorySource::new(
                Arc::clone(&self.tools),
                &scoped.config,
                None,
            )?);
        if !servers.is_empty() {
            let admission =
                admission.ok_or_else(|| anyhow::anyhow!("Missing session MCP admission"))?;
            scoped.mcp_manager = Some(
                claurst_mcp::McpManager::connect_session_stdio(
                    self.mcp_manager.clone(),
                    servers,
                    &working_dir,
                    admission,
                )
                .await?,
            );
            scoped.session_owned_mcp = true;
        }
        Ok(Arc::new(scoped))
    }

    pub async fn shutdown_session_mcp(&self) -> anyhow::Result<()> {
        if self.session_owned_mcp {
            if let Some(manager) = self.mcp_manager.as_ref() {
                manager.shutdown_owned().await?;
            }
        }
        Ok(())
    }
}

async fn build_mcp_manager(
    config: &Config,
    settings: &Settings,
    working_dir: &std::path::Path,
) -> Option<Arc<claurst_mcp::McpManager>> {
    if config.mcp_servers.is_empty() {
        return None;
    }
    // SECURITY (issue #123): never auto-launch project-defined MCP servers in
    // this non-interactive runtime unless they have been trusted. The ACP
    // runtime loads only global settings today (so all servers are user-origin
    // and pass through), but gating here keeps the invariant if project config
    // is ever merged in.
    let project_root = claurst_core::mcp_trust::project_root_for(working_dir);
    let store = claurst_core::mcp_trust::McpTrustStore::load();
    let decision = claurst_core::mcp_trust::partition_mcp_servers(
        &config.mcp_servers,
        project_root.as_deref(),
        settings.trust_project_mcp_servers,
        &std::collections::HashSet::new(),
        &store,
    );
    if !decision.pending.is_empty() {
        tracing::warn!(
            count = decision.pending.len(),
            "Skipping untrusted project-defined MCP server(s) in ACP runtime"
        );
    }
    if decision.allowed.is_empty() {
        return None;
    }
    let mgr = Arc::new(claurst_mcp::McpManager::connect_all(&decision.allowed).await);
    mgr.clone().spawn_notification_poll_loop();
    Some(mgr)
}
