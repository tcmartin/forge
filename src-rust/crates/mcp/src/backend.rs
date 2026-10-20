use crate::catalog_state::ServerCatalogState;
use crate::mcp_operation_receipt::McpOperationContext;
use crate::types::{
    CallToolResult, GetPromptResult, McpPrompt, McpResource, McpTool, ResourceContents,
    ServerCapabilities, ServerInfo,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Backend implementation used by an `McpClient` instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpBackendKind {
    /// rmcp-backed implementation.
    Rmcp,
}

/// Immutable view of the server state that upper layers depend on.
#[derive(Debug, Clone)]
pub struct McpClientSnapshot {
    pub server_name: String,
    pub server_info: Option<ServerInfo>,
    pub capabilities: ServerCapabilities,
    pub tools: Vec<McpTool>,
    pub resources: Vec<McpResource>,
    pub prompts: Vec<McpPrompt>,
    pub instructions: Option<String>,
}

impl McpClientSnapshot {
    /// Create an empty snapshot for a newly constructed client.
    pub fn empty(server_name: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
            server_info: None,
            capabilities: ServerCapabilities::default(),
            tools: vec![],
            resources: vec![],
            prompts: vec![],
            instructions: None,
        }
    }
}

/// Protocol/session backend used by `McpClient`.
///
/// The goal is to keep upper layers stable (`McpManager`, CLI, tools, TUI)
/// while allowing transport implementations to migrate independently.
// async-trait emits a bare `#[must_use]` on its boxed futures;
// Rust 1.99 Clippy sees the boxed future's own marker and reports a duplicate.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait McpClientBackend: Send + Sync {
    fn kind(&self) -> McpBackendKind;

    /// Return the latest server snapshot for populating `McpClient` fields.
    fn snapshot(&self) -> McpClientSnapshot;

    /// Shared generation state used by notifications and exact tool admission.
    fn catalog_state(&self) -> Arc<ServerCatalogState>;

    /// Return complete, ID-bound stdio operation contexts when capture is
    /// enabled.  Backends without the opt-in observer return an empty list;
    /// they never synthesize operation authority from typed results alone.
    fn operation_contexts(&self) -> anyhow::Result<Vec<McpOperationContext>> {
        Ok(Vec::new())
    }

    /// Join transport shutdown. In-memory backends have no transport to join.
    async fn shutdown(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn list_tools(&self) -> anyhow::Result<Vec<McpTool>>;

    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> anyhow::Result<CallToolResult>;

    async fn list_resources(&self) -> anyhow::Result<Vec<McpResource>>;

    /// Read every resource content entry in protocol order.  MCP permits a
    /// response to contain multiple entries; callers must not silently drop
    /// all but the first one.
    async fn read_resource(&self, uri: &str) -> anyhow::Result<Vec<ResourceContents>>;

    async fn list_prompts(&self) -> anyhow::Result<Vec<McpPrompt>>;

    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<HashMap<String, String>>,
    ) -> anyhow::Result<GetPromptResult>;

    async fn subscribe_resource(&self, uri: &str) -> anyhow::Result<()>;

    async fn unsubscribe_resource(&self, uri: &str) -> anyhow::Result<()>;

    /// Subscribe to raw notification payloads emitted by the backend.
    fn subscribe_to_notifications(&self) -> BoxStream<'static, anyhow::Result<Value>>;
}

/// Pick the preferred backend for a configured transport type.
pub fn preferred_backend_kind(_server_type: &str) -> McpBackendKind {
    McpBackendKind::Rmcp
}
