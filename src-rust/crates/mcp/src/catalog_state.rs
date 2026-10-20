//! Shared, generation-checked MCP tool catalog state.
//!
//! Transport notifications invalidate a catalog synchronously.  Descriptors
//! are immutable capability handles: they are only admitted when their source
//! identity, connection generation, catalog generation, and definition all
//! still match the live catalog.

use crate::types::McpTool;
use claurst_core::crypto_utils::sha256_hex;
use claurst_core::types::ToolDefinition;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::time::Duration;
use tokio::sync::watch;
use uuid::Uuid;

pub const MAX_TOOLS: usize = 4096;
pub const MAX_PAGES: usize = 128;
pub const MAX_NAME_BYTES: usize = 256;
pub const MAX_SCHEMA_BYTES: usize = 64 * 1024;
pub const MAX_DEFINITION_BYTES: usize = 128 * 1024;
pub const MAX_CATALOG_BYTES: usize = 8 * 1024 * 1024;

fn lock<'a, T>(mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn canonical_server_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogAvailability {
    Dirty,
    Available,
    Unavailable,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum McpCatalogError {
    #[error("MCP catalog for server '{server}' is unavailable")]
    Unavailable { server: String },
    #[error("MCP catalog for server '{server}' is dirty")]
    Dirty { server: String },
    #[error("MCP tool '{callable_name}' is not in the current catalog")]
    UnknownTool { callable_name: String },
    #[error("MCP tool descriptor is stale")]
    StaleDescriptor,
    #[error("MCP catalog refresh raced with a newer generation")]
    StaleRefresh,
    #[error("MCP catalog has ambiguous tool name '{name}'")]
    NameCollision { name: String },
    #[error("MCP catalog limit exceeded: {0}")]
    LimitExceeded(String),
    #[error("MCP catalog contains an invalid tool name")]
    InvalidName,
}

#[derive(Clone)]
pub struct McpToolDescriptor {
    server_name: String,
    raw_name: String,
    callable_name: String,
    connection_generation: u64,
    catalog_generation: u64,
    source_identity: Uuid,
    definition: Arc<ToolDefinition>,
    definition_digest: String,
    definition_bytes: usize,
}

impl fmt::Debug for McpToolDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpToolDescriptor")
            .field("server_name", &self.server_name)
            .field("raw_name", &self.raw_name)
            .field("callable_name", &self.callable_name)
            .field("connection_generation", &self.connection_generation)
            .field("catalog_generation", &self.catalog_generation)
            .field("definition_digest", &self.definition_digest)
            .finish()
    }
}

impl McpToolDescriptor {
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn raw_name(&self) -> &str {
        &self.raw_name
    }

    pub fn callable_name(&self) -> &str {
        &self.callable_name
    }

    /// Preserve the checked source identity when a descriptor is projected
    /// into the query-owned native diagnostic envelope.
    pub fn source_identity(&self) -> Uuid {
        self.source_identity
    }

    pub fn connection_generation(&self) -> u64 {
        self.connection_generation
    }

    pub fn catalog_generation(&self) -> u64 {
        self.catalog_generation
    }

    pub fn definition(&self) -> Arc<ToolDefinition> {
        Arc::clone(&self.definition)
    }

    pub fn definition_digest(&self) -> &str {
        &self.definition_digest
    }

    pub fn definition_bytes(&self) -> usize {
        self.definition_bytes
    }
}

#[derive(Debug, Clone)]
pub struct CatalogSnapshot {
    pub server_name: String,
    pub connection_generation: u64,
    pub catalog_generation: u64,
    pub availability: CatalogAvailability,
    pub tools: Vec<McpToolDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogRevisionHeader {
    pub server_name: String,
    pub source_identity: Uuid,
    pub connection_generation: u64,
    pub catalog_generation: u64,
    pub availability: CatalogAvailability,
    pub tool_count: usize,
    pub definition_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CombinedCatalogSnapshot {
    pub headers: Vec<CatalogRevisionHeader>,
    pub descriptors: Vec<McpToolDescriptor>,
    pub definition_bytes: usize,
}

struct ServerState {
    server_name: String,
    connection_generation: u64,
    catalog_generation: u64,
    availability: CatalogAvailability,
    source_identity: Uuid,
    tools: HashMap<String, McpToolDescriptor>,
    definition_bytes: usize,
}

#[derive(Clone)]
pub struct ServerCatalogState {
    inner: Arc<Mutex<ServerState>>,
    global_names: Arc<Mutex<HashMap<String, String>>>,
    readiness: watch::Sender<u64>,
}

impl fmt::Debug for ServerCatalogState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = lock(&self.inner);
        formatter
            .debug_struct("ServerCatalogState")
            .field("server_name", &state.server_name)
            .field("connection_generation", &state.connection_generation)
            .field("catalog_generation", &state.catalog_generation)
            .field("availability", &state.availability)
            .field("tool_count", &state.tools.len())
            .finish()
    }
}

impl ServerCatalogState {
    pub(crate) fn new(server_name: impl Into<String>) -> Self {
        Self::with_global_names(server_name.into(), Arc::new(Mutex::new(HashMap::new())))
    }

    fn with_global_names(
        server_name: String,
        global_names: Arc<Mutex<HashMap<String, String>>>,
    ) -> Self {
        let (readiness, _) = watch::channel(0);
        Self {
            inner: Arc::new(Mutex::new(ServerState {
                server_name,
                connection_generation: 1,
                catalog_generation: 0,
                availability: CatalogAvailability::Unavailable,
                source_identity: Uuid::new_v4(),
                tools: HashMap::new(),
                definition_bytes: 0,
            })),
            global_names,
            readiness,
        }
    }

    pub fn server_name(&self) -> String {
        lock(&self.inner).server_name.clone()
    }

    pub fn connection_generation(&self) -> u64 {
        lock(&self.inner).connection_generation
    }

    pub fn catalog_generation(&self) -> u64 {
        lock(&self.inner).catalog_generation
    }

    pub fn availability(&self) -> CatalogAvailability {
        lock(&self.inner).availability
    }

    /// Wait for a dirty catalog to become available, with a finite deadline.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<CatalogSnapshot, McpCatalogError> {
        let mut changes = self.readiness.subscribe();
        let wait = async {
            loop {
                let snapshot = self.snapshot();
                match snapshot.availability {
                    CatalogAvailability::Available => return Ok(snapshot),
                    CatalogAvailability::Unavailable => {
                        return Err(McpCatalogError::Unavailable {
                            server: snapshot.server_name,
                        })
                    }
                    CatalogAvailability::Dirty => {
                        changes
                            .changed()
                            .await
                            .map_err(|_| McpCatalogError::Unavailable {
                                server: self.server_name(),
                            })?
                    }
                }
            }
        };
        tokio::time::timeout(timeout, wait)
            .await
            .unwrap_or_else(|_| {
                Err(McpCatalogError::Dirty {
                    server: self.server_name(),
                })
            })
    }

    /// Mark the source dirty before a list-changed notification is queued.
    pub fn mark_dirty(&self) -> u64 {
        let generation = {
            let mut state = lock(&self.inner);
            state.catalog_generation = state.catalog_generation.saturating_add(1);
            state.availability = CatalogAvailability::Dirty;
            state.catalog_generation
        };
        let _ = self.readiness.send(generation);
        generation
    }

    pub fn mark_dirty_if_connection(&self, connection_generation: u64) -> Option<u64> {
        let generation = {
            let mut state = lock(&self.inner);
            if state.connection_generation != connection_generation {
                return None;
            }
            state.catalog_generation = state.catalog_generation.saturating_add(1);
            state.availability = CatalogAvailability::Dirty;
            state.catalog_generation
        };
        let _ = self.readiness.send(generation);
        Some(generation)
    }

    pub fn mark_unavailable(&self) -> u64 {
        let generation = {
            let mut names = lock(&self.global_names);
            let mut state = lock(&self.inner);
            remove_global_names(&mut names, &state);
            state.catalog_generation = state.catalog_generation.saturating_add(1);
            state.availability = CatalogAvailability::Unavailable;
            state.tools.clear();
            state.definition_bytes = 0;
            state.catalog_generation
        };
        let _ = self.readiness.send(generation);
        generation
    }

    pub fn mark_unavailable_if_connection(&self, connection_generation: u64) -> Option<u64> {
        let generation = {
            let mut names = lock(&self.global_names);
            let mut state = lock(&self.inner);
            if state.connection_generation != connection_generation {
                return None;
            }
            remove_global_names(&mut names, &state);
            state.catalog_generation = state.catalog_generation.saturating_add(1);
            state.availability = CatalogAvailability::Unavailable;
            state.tools.clear();
            state.definition_bytes = 0;
            state.catalog_generation
        };
        let _ = self.readiness.send(generation);
        Some(generation)
    }

    /// Start a new transport connection. Descriptors from the previous
    /// connection remain stale even when the server returns identical tools.
    pub fn begin_connection(&self) -> u64 {
        let connection_generation = {
            let mut names = lock(&self.global_names);
            let mut state = lock(&self.inner);
            remove_global_names(&mut names, &state);
            state.connection_generation = state.connection_generation.saturating_add(1);
            state.catalog_generation = state.catalog_generation.saturating_add(1);
            state.availability = CatalogAvailability::Unavailable;
            state.tools.clear();
            state.definition_bytes = 0;
            state.connection_generation
        };
        let _ = self.readiness.send(self.catalog_generation());
        connection_generation
    }

    pub fn begin_refresh(&self) -> u64 {
        self.mark_dirty()
    }

    pub fn begin_refresh_if_connection(
        &self,
        connection_generation: u64,
    ) -> Result<u64, McpCatalogError> {
        self.mark_dirty_if_connection(connection_generation)
            .ok_or(McpCatalogError::StaleRefresh)
    }

    pub fn install_tools(
        &self,
        connection_generation: u64,
        tools: &[McpTool],
    ) -> Result<CatalogSnapshot, McpCatalogError> {
        let (server_name, catalog_generation, source_identity) = {
            let state = lock(&self.inner);
            (
                state.server_name.clone(),
                state.catalog_generation,
                state.source_identity,
            )
        };
        let descriptors = match build_descriptors(
            &server_name,
            connection_generation,
            catalog_generation,
            source_identity,
            tools,
        ) {
            Ok(descriptors) => descriptors,
            Err(error) => {
                self.fail_if_current(connection_generation, catalog_generation);
                return Err(error);
            }
        };
        self.commit_prepared(
            CatalogAvailability::Unavailable,
            catalog_generation,
            connection_generation,
            descriptors,
        )
    }

    pub fn commit_refresh(
        &self,
        starting_generation: u64,
        connection_generation: u64,
        tools: &[McpTool],
    ) -> Result<CatalogSnapshot, McpCatalogError> {
        let (server_name, source_identity) = {
            let state = lock(&self.inner);
            (state.server_name.clone(), state.source_identity)
        };
        let descriptors = build_descriptors(
            &server_name,
            connection_generation,
            starting_generation,
            source_identity,
            tools,
        )
        .inspect_err(|_| {
            self.fail_if_current(connection_generation, starting_generation);
        })?;
        self.commit_prepared(
            CatalogAvailability::Dirty,
            starting_generation,
            connection_generation,
            descriptors,
        )
    }

    pub fn fail_refresh_if_current(
        &self,
        connection_generation: u64,
        catalog_generation: u64,
    ) -> Option<u64> {
        self.fail_if_current(connection_generation, catalog_generation)
    }

    fn fail_if_current(&self, connection_generation: u64, catalog_generation: u64) -> Option<u64> {
        let generation = {
            let mut names = lock(&self.global_names);
            let mut state = lock(&self.inner);
            if state.connection_generation != connection_generation
                || state.catalog_generation != catalog_generation
            {
                return None;
            }
            remove_global_names(&mut names, &state);
            state.catalog_generation = state.catalog_generation.saturating_add(1);
            state.availability = CatalogAvailability::Unavailable;
            state.tools.clear();
            state.definition_bytes = 0;
            state.catalog_generation
        };
        let _ = self.readiness.send(generation);
        Some(generation)
    }

    fn commit_prepared(
        &self,
        expected_availability: CatalogAvailability,
        expected_catalog_generation: u64,
        connection_generation: u64,
        descriptors: Vec<McpToolDescriptor>,
    ) -> Result<CatalogSnapshot, McpCatalogError> {
        let mut names = lock(&self.global_names);
        let mut state = lock(&self.inner);
        if state.connection_generation != connection_generation
            || state.catalog_generation != expected_catalog_generation
            || state.availability != expected_availability
        {
            return Err(McpCatalogError::StaleRefresh);
        }

        let mut seen = HashSet::new();
        for descriptor in &descriptors {
            let folded = descriptor.callable_name.to_lowercase();
            if !seen.insert(folded.clone()) {
                return Err(McpCatalogError::NameCollision {
                    name: descriptor.callable_name.clone(),
                });
            }
            if let Some(owner) = names.get(&folded) {
                if owner != &state.server_name {
                    remove_global_names(&mut names, &state);
                    state.catalog_generation = state.catalog_generation.saturating_add(1);
                    state.availability = CatalogAvailability::Unavailable;
                    state.tools.clear();
                    state.definition_bytes = 0;
                    let generation = state.catalog_generation;
                    drop(state);
                    let _ = self.readiness.send(generation);
                    return Err(McpCatalogError::NameCollision {
                        name: descriptor.callable_name.clone(),
                    });
                }
            }
        }

        remove_global_names(&mut names, &state);
        for descriptor in &descriptors {
            names.insert(
                descriptor.callable_name.to_lowercase(),
                state.server_name.clone(),
            );
        }
        state.tools = descriptors
            .iter()
            .map(|descriptor| (descriptor.callable_name.clone(), descriptor.clone()))
            .collect();
        state.definition_bytes = descriptors
            .iter()
            .map(McpToolDescriptor::definition_bytes)
            .sum();
        state.availability = CatalogAvailability::Available;
        let snapshot = snapshot_locked(&state);
        let generation = state.catalog_generation;
        drop(state);
        let _ = self.readiness.send(generation);
        Ok(snapshot)
    }

    pub fn descriptor(&self, callable_name: &str) -> Result<McpToolDescriptor, McpCatalogError> {
        let state = lock(&self.inner);
        match state.availability {
            CatalogAvailability::Dirty => {
                return Err(McpCatalogError::Dirty {
                    server: state.server_name.clone(),
                });
            }
            CatalogAvailability::Unavailable => {
                return Err(McpCatalogError::Unavailable {
                    server: state.server_name.clone(),
                });
            }
            CatalogAvailability::Available => {}
        }
        state
            .tools
            .get(callable_name)
            .cloned()
            .ok_or_else(|| McpCatalogError::UnknownTool {
                callable_name: callable_name.to_string(),
            })
    }

    pub fn validate(&self, descriptor: &McpToolDescriptor) -> Result<(), McpCatalogError> {
        let state = lock(&self.inner);
        if state.availability != CatalogAvailability::Available
            || state.connection_generation != descriptor.connection_generation
            || state.catalog_generation != descriptor.catalog_generation
            || state.source_identity != descriptor.source_identity
            || state.server_name != descriptor.server_name
        {
            return Err(McpCatalogError::StaleDescriptor);
        }
        let current = state
            .tools
            .get(&descriptor.callable_name)
            .ok_or(McpCatalogError::StaleDescriptor)?;
        if current.raw_name != descriptor.raw_name
            || current.callable_name != descriptor.callable_name
            || current.definition_digest != descriptor.definition_digest
        {
            return Err(McpCatalogError::StaleDescriptor);
        }
        Ok(())
    }

    pub fn snapshot(&self) -> CatalogSnapshot {
        snapshot_locked(&lock(&self.inner))
    }

    pub fn revision_header(&self) -> CatalogRevisionHeader {
        let state = lock(&self.inner);
        revision_header_locked(&state)
    }

    pub fn checked_snapshot(&self) -> Result<CatalogSnapshot, McpCatalogError> {
        let snapshot = self.snapshot();
        if snapshot.availability != CatalogAvailability::Available {
            return Err(match snapshot.availability {
                CatalogAvailability::Dirty => McpCatalogError::Dirty {
                    server: snapshot.server_name,
                },
                CatalogAvailability::Unavailable => McpCatalogError::Unavailable {
                    server: snapshot.server_name,
                },
                CatalogAvailability::Available => unreachable!(),
            });
        }
        Ok(snapshot)
    }
}

pub struct CatalogState {
    servers: Mutex<HashMap<String, Arc<ServerCatalogState>>>,
    global_names: Arc<Mutex<HashMap<String, String>>>,
    revision: AtomicU64,
}

impl fmt::Debug for CatalogState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CatalogState")
            .field("server_count", &lock(&self.servers).len())
            .finish()
    }
}

impl Default for CatalogState {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogState {
    /// Reuse the actual leaf identities in a private aggregate view. Private
    /// servers keep a separate name registry; the combined snapshot checks
    /// collisions without adding their names to the inherited catalog.
    pub(crate) fn inherit(source: Option<&Self>) -> Self {
        Self {
            servers: Mutex::new(
                source
                    .map(|source| lock(&source.servers).clone())
                    .unwrap_or_default(),
            ),
            global_names: Arc::new(Mutex::new(HashMap::new())),
            revision: AtomicU64::new(0),
        }
    }

    /// Reuse only the selected source leaves in a private aggregate view.
    /// Removed or replaced sources are intentionally absent so an unavailable
    /// leaf cannot poison readiness for unchanged sources.
    pub(crate) fn inherit_selected<'a>(
        source: &Self,
        server_names: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let selected: HashSet<String> = server_names
            .into_iter()
            .map(canonical_server_name)
            .collect();
        let servers = lock(&source.servers)
            .iter()
            .filter(|(name, _)| selected.contains(&canonical_server_name(name)))
            .map(|(name, server)| (name.clone(), Arc::clone(server)))
            .collect();
        Self {
            servers: Mutex::new(servers),
            global_names: Arc::new(Mutex::new(HashMap::new())),
            revision: AtomicU64::new(0),
        }
    }

    pub fn new() -> Self {
        Self {
            servers: Mutex::new(HashMap::new()),
            global_names: Arc::new(Mutex::new(HashMap::new())),
            revision: AtomicU64::new(0),
        }
    }

    pub fn server(&self, server_name: &str) -> Result<Arc<ServerCatalogState>, McpCatalogError> {
        validate_name(server_name)?;
        let mut servers = lock(&self.servers);
        if let Some((_, server)) = servers
            .iter()
            .find(|(name, _)| canonical_server_name(name) == canonical_server_name(server_name))
        {
            return Ok(Arc::clone(server));
        }
        let server = Arc::new(ServerCatalogState::with_global_names(
            server_name.to_string(),
            Arc::clone(&self.global_names),
        ));
        servers.insert(server_name.to_string(), Arc::clone(&server));
        self.revision.fetch_add(1, Ordering::AcqRel);
        Ok(server)
    }

    pub fn lookup_server(&self, server_name: &str) -> Option<Arc<ServerCatalogState>> {
        let servers = lock(&self.servers);
        servers.get(server_name).cloned().or_else(|| {
            servers
                .iter()
                .find(|(name, _)| canonical_server_name(name) == canonical_server_name(server_name))
                .map(|(_, server)| Arc::clone(server))
        })
    }

    /// Remove a source leaf after a failed or revoked connection. Marking it
    /// unavailable first invalidates all captured descriptors and releases
    /// its callable names before the aggregate forgets the leaf.
    pub(crate) fn remove_server(&self, server_name: &str) -> Option<Arc<ServerCatalogState>> {
        let mut servers = lock(&self.servers);
        let key = servers
            .keys()
            .find(|name| canonical_server_name(name) == canonical_server_name(server_name))
            .cloned();
        let server = key.and_then(|key| servers.remove(&key));
        if let Some(server) = &server {
            server.mark_unavailable();
            self.revision.fetch_add(1, Ordering::AcqRel);
        }
        server
    }

    pub fn snapshot(&self, server_name: &str) -> CatalogSnapshot {
        self.lookup_server(server_name)
            .map(|server| server.snapshot())
            .unwrap_or_else(|| CatalogSnapshot {
                server_name: server_name.to_string(),
                connection_generation: 0,
                catalog_generation: 0,
                availability: CatalogAvailability::Unavailable,
                tools: Vec::new(),
            })
    }

    /// Wait for one configured source to finish a dirty refresh without
    /// creating an unknown source as a side effect.
    pub async fn wait_ready(
        &self,
        server_name: &str,
        timeout: Duration,
    ) -> Result<CatalogSnapshot, McpCatalogError> {
        self.lookup_server(server_name)
            .ok_or_else(|| McpCatalogError::Unavailable {
                server: server_name.to_string(),
            })?
            .wait_ready(timeout)
            .await
    }

    pub fn revision_headers(&self) -> Vec<CatalogRevisionHeader> {
        let mut headers: Vec<_> = lock(&self.servers)
            .values()
            .map(|server| server.revision_header())
            .collect();
        headers.sort_by(|left, right| left.server_name.cmp(&right.server_name));
        headers
    }

    pub async fn wait_all_ready(
        &self,
        timeout: Duration,
    ) -> Result<CombinedCatalogSnapshot, McpCatalogError> {
        let wait = async {
            loop {
                let revision = self.revision.load(Ordering::Acquire);
                let servers: Vec<_> = lock(&self.servers).values().cloned().collect();
                let mut changes = Vec::with_capacity(servers.len());
                for server in &servers {
                    changes.push(server.readiness.subscribe());
                }
                match self.checked_snapshot() {
                    Ok(snapshot) if self.revision.load(Ordering::Acquire) == revision => {
                        return Ok(snapshot)
                    }
                    Err(McpCatalogError::Unavailable { .. }) => {
                        return Err(McpCatalogError::Unavailable {
                            server: "combined catalog".into(),
                        })
                    }
                    Err(McpCatalogError::Dirty { .. }) | Err(McpCatalogError::StaleRefresh) => {}
                    Err(error) => return Err(error),
                    Ok(_) => {}
                }
                if changes.is_empty() {
                    return self.checked_snapshot();
                }
                let waits = changes.into_iter().map(|mut change| {
                    Box::pin(async move { change.changed().await.map_err(|_| ()) })
                        as futures::future::BoxFuture<'static, Result<(), ()>>
                });
                let _ = futures::future::select_all(waits).await;
            }
        };
        tokio::time::timeout(timeout, wait)
            .await
            .unwrap_or_else(|_| {
                Err(McpCatalogError::Dirty {
                    server: "combined catalog".into(),
                })
            })
    }

    pub async fn wait_ready_combined(
        &self,
        timeout: Duration,
    ) -> Result<CombinedCatalogSnapshot, McpCatalogError> {
        self.wait_all_ready(timeout).await
    }

    pub fn checked_snapshot(&self) -> Result<CombinedCatalogSnapshot, McpCatalogError> {
        let initial_revision = self.revision.load(Ordering::Acquire);
        let servers: Vec<Arc<ServerCatalogState>> = lock(&self.servers).values().cloned().collect();
        let mut headers = Vec::with_capacity(servers.len());
        let mut descriptors = Vec::new();
        let mut definition_bytes = 0usize;
        let mut names = HashSet::new();
        for server in servers {
            let before = server.revision_header();
            let snapshot = server.checked_snapshot()?;
            let header = server.revision_header();
            if before != header {
                return Err(McpCatalogError::StaleRefresh);
            }
            definition_bytes = definition_bytes
                .checked_add(header.definition_bytes)
                .ok_or_else(|| McpCatalogError::LimitExceeded("catalog bytes".into()))?;
            if definition_bytes > MAX_CATALOG_BYTES {
                return Err(McpCatalogError::LimitExceeded("catalog bytes".into()));
            }
            for descriptor in &snapshot.tools {
                if !names.insert(descriptor.callable_name().to_lowercase()) {
                    return Err(McpCatalogError::NameCollision {
                        name: descriptor.callable_name().to_string(),
                    });
                }
            }
            if descriptors
                .len()
                .checked_add(snapshot.tools.len())
                .is_none_or(|count| count > MAX_TOOLS)
            {
                return Err(McpCatalogError::LimitExceeded("tool count".into()));
            }
            descriptors.extend(snapshot.tools);
            headers.push(header);
        }
        if descriptors.len() > MAX_TOOLS {
            return Err(McpCatalogError::LimitExceeded("tool count".into()));
        }
        headers.sort_by(|left, right| left.server_name.cmp(&right.server_name));
        descriptors.sort_by(|left, right| left.callable_name().cmp(right.callable_name()));
        if self.revision.load(Ordering::Acquire) != initial_revision {
            return Err(McpCatalogError::StaleRefresh);
        }
        Ok(CombinedCatalogSnapshot {
            headers,
            descriptors,
            definition_bytes,
        })
    }
}

fn remove_global_names(names: &mut HashMap<String, String>, state: &ServerState) {
    names.retain(|_, owner| owner != &state.server_name);
}

fn snapshot_locked(state: &ServerState) -> CatalogSnapshot {
    let mut tools = if state.availability == CatalogAvailability::Available {
        state.tools.values().cloned().collect()
    } else {
        Vec::new()
    };
    tools.sort_by(|left: &McpToolDescriptor, right: &McpToolDescriptor| {
        left.callable_name.cmp(&right.callable_name)
    });
    CatalogSnapshot {
        server_name: state.server_name.clone(),
        connection_generation: state.connection_generation,
        catalog_generation: state.catalog_generation,
        availability: state.availability,
        tools,
    }
}

fn revision_header_locked(state: &ServerState) -> CatalogRevisionHeader {
    CatalogRevisionHeader {
        server_name: state.server_name.clone(),
        source_identity: state.source_identity,
        connection_generation: state.connection_generation,
        catalog_generation: state.catalog_generation,
        availability: state.availability,
        tool_count: if state.availability == CatalogAvailability::Available {
            state.tools.len()
        } else {
            0
        },
        definition_bytes: if state.availability == CatalogAvailability::Available {
            state.definition_bytes
        } else {
            0
        },
    }
}

fn validate_name(name: &str) -> Result<(), McpCatalogError> {
    if name.is_empty() || name.len() > MAX_NAME_BYTES || name.chars().any(char::is_control) {
        return Err(McpCatalogError::InvalidName);
    }
    Ok(())
}

fn build_descriptors(
    server_name: &str,
    connection_generation: u64,
    catalog_generation: u64,
    source_identity: Uuid,
    tools: &[McpTool],
) -> Result<Vec<McpToolDescriptor>, McpCatalogError> {
    validate_name(server_name)?;
    if tools.len() > MAX_TOOLS {
        return Err(McpCatalogError::LimitExceeded("tool count".into()));
    }
    let mut seen = HashSet::new();
    let mut total_bytes = 0usize;
    let mut descriptors = Vec::with_capacity(tools.len());
    for tool in tools {
        validate_name(&tool.name)?;
        let folded = tool.name.to_lowercase();
        if !seen.insert(folded) {
            return Err(McpCatalogError::NameCollision {
                name: tool.name.clone(),
            });
        }
        let schema_bytes = serde_json::to_vec(&tool.input_schema)
            .map_err(|error| McpCatalogError::LimitExceeded(error.to_string()))?;
        if schema_bytes.len() > MAX_SCHEMA_BYTES {
            return Err(McpCatalogError::LimitExceeded("input schema bytes".into()));
        }
        let callable_name = format!("{server_name}_{}", tool.name);
        validate_name(&callable_name)?;
        let definition = ToolDefinition {
            name: callable_name.clone(),
            description: format!(
                "[{server_name}] {}",
                tool.description.as_deref().unwrap_or_default()
            ),
            input_schema: tool.input_schema.clone(),
        };
        let encoded = serde_json::to_vec(&definition)
            .map_err(|error| McpCatalogError::LimitExceeded(error.to_string()))?;
        if encoded.len() > MAX_DEFINITION_BYTES {
            return Err(McpCatalogError::LimitExceeded(
                "tool definition bytes".into(),
            ));
        }
        total_bytes = total_bytes.saturating_add(encoded.len());
        if total_bytes > MAX_CATALOG_BYTES {
            return Err(McpCatalogError::LimitExceeded("catalog bytes".into()));
        }
        descriptors.push(McpToolDescriptor {
            server_name: server_name.to_string(),
            raw_name: tool.name.clone(),
            callable_name,
            connection_generation,
            catalog_generation,
            source_identity,
            definition: Arc::new(definition),
            definition_digest: sha256_hex(&encoded),
            definition_bytes: encoded.len(),
        });
    }
    Ok(descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> McpTool {
        McpTool {
            name: name.into(),
            description: Some("test".into()),
            input_schema: serde_json::json!({"type":"object"}),
        }
    }

    #[test]
    fn session_view_keeps_native_identity_and_observes_parent_revocation() {
        let parent = CatalogState::new();
        let server = parent.server("configured").unwrap();
        server.install_tools(1, &[tool("echo")]).unwrap();
        let session = CatalogState::inherit(Some(&parent));
        assert!(Arc::ptr_eq(
            &server,
            &session.lookup_server("configured").unwrap()
        ));
        let descriptor = session
            .checked_snapshot()
            .unwrap()
            .descriptors
            .pop()
            .unwrap();
        assert_eq!(server.validate(&descriptor), Ok(()));
        server.mark_unavailable();
        assert!(session.checked_snapshot().is_err());
        assert_eq!(
            server.validate(&descriptor),
            Err(McpCatalogError::StaleDescriptor)
        );
    }

    #[test]
    fn private_collision_fails_combined_view_without_poisoning_global_names() {
        let parent = CatalogState::new();
        parent
            .server("a")
            .unwrap()
            .install_tools(1, &[tool("b_c")])
            .unwrap();
        let session = CatalogState::inherit(Some(&parent));
        session
            .server("a_b")
            .unwrap()
            .install_tools(1, &[tool("c")])
            .unwrap();
        assert!(matches!(
            session.checked_snapshot(),
            Err(McpCatalogError::NameCollision { .. })
        ));
        assert_eq!(parent.checked_snapshot().unwrap().descriptors.len(), 1);
        assert!(parent.lookup_server("a_b").is_none());
    }

    #[test]
    fn same_named_private_servers_cannot_exchange_descriptor_authority() {
        let first = CatalogState::inherit(None);
        let second = CatalogState::inherit(None);
        let first_server = first.server("session").unwrap();
        let second_server = second.server("session").unwrap();
        first_server.install_tools(1, &[tool("echo")]).unwrap();
        second_server.install_tools(1, &[tool("echo")]).unwrap();
        let descriptor = first.checked_snapshot().unwrap().descriptors.pop().unwrap();
        assert_eq!(
            second_server.validate(&descriptor),
            Err(McpCatalogError::StaleDescriptor)
        );
    }

    #[test]
    fn dirty_generation_rejects_old_descriptor_and_hides_stale_tools() {
        let state = ServerCatalogState::new("server");
        state.install_tools(1, &[tool("echo")]).unwrap();
        let descriptor = state.descriptor("server_echo").unwrap();
        state.mark_dirty();
        assert!(state.snapshot().tools.is_empty());
        assert_eq!(
            state.validate(&descriptor),
            Err(McpCatalogError::StaleDescriptor)
        );
    }

    #[test]
    fn case_fold_collision_is_rejected() {
        let state = ServerCatalogState::new("server");
        let error = state
            .install_tools(1, &[tool("Echo"), tool("echo")])
            .unwrap_err();
        assert!(matches!(error, McpCatalogError::NameCollision { .. }));
    }

    #[test]
    fn failed_refresh_clears_old_callables() {
        let state = ServerCatalogState::new("server");
        state.install_tools(1, &[tool("echo")]).unwrap();
        let generation = state.begin_refresh();
        assert!(state
            .commit_refresh(generation, 1, &[tool("Echo"), tool("echo")])
            .is_err());
        assert!(matches!(
            state.descriptor("server_echo"),
            Err(McpCatalogError::Unavailable { .. })
        ));
    }

    #[test]
    fn reconnect_invalidates_same_definition() {
        let state = ServerCatalogState::new("server");
        state.install_tools(1, &[tool("echo")]).unwrap();
        let old = state.descriptor("server_echo").unwrap();
        let generation = state.begin_connection();
        state.install_tools(generation, &[tool("echo")]).unwrap();
        assert_eq!(state.validate(&old), Err(McpCatalogError::StaleDescriptor));
    }

    #[test]
    fn stale_connection_cannot_dirty_or_clear_replacement() {
        let state = ServerCatalogState::new("server");
        state.install_tools(1, &[tool("echo")]).unwrap();
        let old_connection = state.connection_generation();
        let new_connection = state.begin_connection();
        state
            .install_tools(new_connection, &[tool("echo")])
            .unwrap();
        assert_eq!(state.mark_dirty_if_connection(old_connection), None);
        assert_eq!(state.mark_unavailable_if_connection(old_connection), None);
        assert_eq!(state.availability(), CatalogAvailability::Available);
        assert_eq!(
            state.revision_header().connection_generation,
            new_connection
        );
    }

    #[test]
    fn stale_refresh_failure_cannot_clear_new_catalog() {
        let state = ServerCatalogState::new("server");
        state.install_tools(1, &[tool("echo")]).unwrap();
        let old_generation = state.begin_refresh();
        let new_connection = state.begin_connection();
        state
            .install_tools(new_connection, &[tool("echo")])
            .unwrap();
        assert_eq!(state.fail_refresh_if_current(1, old_generation), None);
        assert_eq!(state.availability(), CatalogAvailability::Available);
        assert!(state.descriptor("server_echo").is_ok());
    }

    #[test]
    fn combined_snapshot_keeps_typed_descriptors_and_headers() {
        let catalog = CatalogState::new();
        let server = catalog.server("server").unwrap();
        server.install_tools(1, &[tool("echo")]).unwrap();
        let combined = catalog.checked_snapshot().unwrap();
        assert_eq!(combined.headers.len(), 1);
        assert_eq!(combined.descriptors.len(), 1);
        assert_eq!(combined.descriptors[0].callable_name(), "server_echo");
        assert_eq!(combined.headers[0].tool_count, 1);
        assert_eq!(
            combined.definition_bytes,
            combined.headers[0].definition_bytes
        );
    }

    #[test]
    fn separate_sources_reject_same_generation_descriptor() {
        let first = ServerCatalogState::new("server");
        let second = ServerCatalogState::new("server");
        first.install_tools(1, &[tool("echo")]).unwrap();
        second.install_tools(1, &[tool("echo")]).unwrap();
        let descriptor = first.descriptor("server_echo").unwrap();
        assert_eq!(
            second.validate(&descriptor),
            Err(McpCatalogError::StaleDescriptor)
        );
    }

    #[test]
    fn shared_catalog_rejects_callable_collisions_without_inserting_unknown_servers() {
        let catalog = CatalogState::new();
        let first = catalog.server("a_b").unwrap();
        let second = catalog.server("a").unwrap();
        first.install_tools(1, &[tool("tool-c")]).unwrap();
        assert!(matches!(
            second.install_tools(1, &[tool("b_tool-c")]),
            Err(McpCatalogError::NameCollision { .. })
        ));
        assert!(catalog.lookup_server("missing").is_none());
        assert!(catalog.server("bad\nname").is_err());
    }

    #[test]
    fn inherit_selected_reuses_unchanged_leaf_and_excludes_revoked_source() {
        let source = CatalogState::new();
        let keep = source.server("Keep").unwrap();
        let revoke = source.server("revoke").unwrap();
        keep.install_tools(1, &[tool("echo")]).unwrap();
        revoke.install_tools(1, &[tool("echo")]).unwrap();
        let selected = CatalogState::inherit_selected(&source, ["keep"]);
        let selected_keep = selected.lookup_server("KEEP").unwrap();
        assert!(Arc::ptr_eq(&keep, &selected_keep));
        assert!(selected.lookup_server("revoke").is_none());
        assert_eq!(selected.checked_snapshot().unwrap().descriptors.len(), 1);
    }

    #[test]
    fn remove_server_invalidates_leaf_before_forgetting_it() {
        let catalog = CatalogState::new();
        let server = catalog.server("revoke").unwrap();
        server.install_tools(1, &[tool("echo")]).unwrap();
        let descriptor = server.descriptor("revoke_echo").unwrap();
        let removed = catalog.remove_server("revoke").unwrap();
        assert!(Arc::ptr_eq(&server, &removed));
        assert!(catalog.lookup_server("revoke").is_none());
        assert_eq!(removed.availability(), CatalogAvailability::Unavailable);
        assert_eq!(
            removed.validate(&descriptor),
            Err(McpCatalogError::StaleDescriptor)
        );
    }

    #[tokio::test]
    async fn readiness_wait_is_finite_and_terminal_unavailable_is_immediate() {
        let state = ServerCatalogState::new("server");
        let started = std::time::Instant::now();
        assert!(matches!(
            state.wait_ready(Duration::from_secs(1)).await,
            Err(McpCatalogError::Unavailable { .. })
        ));
        assert!(started.elapsed() < Duration::from_millis(100));

        state.install_tools(1, &[tool("echo")]).unwrap();
        state.mark_dirty();
        assert!(matches!(
            state.wait_ready(Duration::from_millis(1)).await,
            Err(McpCatalogError::Dirty { .. })
        ));
    }

    #[tokio::test]
    async fn combined_wait_rechecks_dirty_sources_before_returning() {
        let catalog = Arc::new(CatalogState::new());
        let server = catalog.server("server").unwrap();
        server.install_tools(1, &[tool("echo")]).unwrap();
        let generation = server.begin_refresh();
        let waiting = {
            let catalog = Arc::clone(&catalog);
            tokio::spawn(async move { catalog.wait_all_ready(Duration::from_secs(1)).await })
        };
        tokio::task::yield_now().await;
        server
            .commit_refresh(generation, 1, &[tool("echo-v2")])
            .unwrap();
        let snapshot = waiting.await.unwrap().unwrap();
        assert_eq!(snapshot.descriptors.len(), 1);
        assert_eq!(snapshot.descriptors[0].raw_name(), "echo-v2");
        assert_eq!(
            snapshot.headers[0].availability,
            CatalogAvailability::Available
        );
    }
}
