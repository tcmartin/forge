//! Additive session views retain native global connections and catalog leaves.

use crate::{catalog_state::CatalogState, McpClient, McpManager};
use claurst_core::config::McpServerConfig;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;

/// Notification refreshes can temporarily retain the manager. Setup failure
/// must close owned clients even while such a task still has a strong handle.
struct StartupCleanup {
    manager: Option<Arc<McpManager>>,
}

impl Drop for StartupCleanup {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.take() {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    let _ = manager.shutdown_owned().await;
                });
            }
        }
    }
}

impl McpManager {
    /// Caller validates the complete client roster and reserves its capacity.
    /// Values are already explicit ACP arguments/environment; do not expand
    /// them against the agent's inherited environment or launch via a shell.
    pub async fn connect_session_stdio(
        inherited: Option<Arc<Self>>,
        configs: &[McpServerConfig],
        working_dir: &Path,
        admission: Arc<OwnedSemaphorePermit>,
    ) -> anyhow::Result<Arc<Self>> {
        let mut names = HashSet::new();
        if configs.len() > 8 || !working_dir.is_absolute() {
            anyhow::bail!("Invalid session MCP roster");
        }
        for config in configs {
            if config.server_type != "stdio"
                || !names.insert(config.name.to_lowercase())
                || inherited.as_ref().is_some_and(|parent| {
                    parent
                        .server_configs
                        .keys()
                        .any(|name| name.to_lowercase() == config.name.to_lowercase())
                })
            {
                anyhow::bail!("Invalid session MCP roster");
            }
        }
        let mut manager = Self::new();
        if let Some(parent) = inherited.as_ref() {
            manager.clients = parent.clients.clone();
            manager.server_configs = parent.server_configs.clone();
            manager.failed_servers = parent.failed_servers.clone();
        }
        manager.catalog = Arc::new(CatalogState::inherit(
            inherited.as_ref().map(|parent| parent.catalog.as_ref()),
        ));
        manager.inherited = inherited;
        for config in configs {
            let client = McpClient::connect_stdio_scoped(
                config,
                Arc::clone(&manager.catalog),
                working_dir,
                Arc::clone(&admission),
            )
            .await?;
            manager
                .server_configs
                .insert(config.name.clone(), config.clone());
            manager
                .clients
                .insert(config.name.clone(), Arc::new(client));
        }
        let manager = Arc::new(manager);
        let mut cleanup = StartupCleanup {
            manager: Some(Arc::clone(&manager)),
        };
        manager.clone().spawn_notification_poll_loop();
        manager
            .wait_catalog_ready(std::time::Duration::from_secs(3))
            .await?;
        cleanup.manager = None;
        Ok(manager)
    }

    /// A global client's single notification reader belongs to its original
    /// manager. Subscriptions must be registered with that same owner.
    pub(super) fn resource_owner(&self, server_name: &str) -> &Self {
        let mut owner = self;
        while let Some(parent) = owner.inherited.as_ref() {
            if parent.client_for_name(server_name).is_none() {
                break;
            }
            owner = parent;
        }
        owner
    }

    /// Close only newly owned clients, leaving inherited connections intact.
    /// This is explicit rather than relying on inventory/cache Arc lifetimes.
    pub async fn shutdown_owned(&self) -> anyhow::Result<()> {
        let clients = self
            .clients
            .iter()
            .filter(|(name, _)| {
                !self
                    .inherited
                    .as_ref()
                    .is_some_and(|parent| parent.client_for_name(name).is_some())
            })
            .map(|(_, client)| client);
        let results = futures::future::join_all(clients.map(|client| client.shutdown())).await;
        if results.iter().any(Result::is_err) {
            anyhow::bail!("MCP transport shutdown failed");
        }
        Ok(())
    }
}
