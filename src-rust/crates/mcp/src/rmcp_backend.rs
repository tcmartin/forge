use crate::backend::{McpBackendKind, McpClientBackend, McpClientSnapshot};
use crate::bounded_transport::{self, BoundedChildProcess, BoundedStreamableHttpTransport};
use crate::catalog_state::{CatalogState, ServerCatalogState};
use crate::mcp_operation_receipt::{
    bind_call_tool_receipt, current_query_operation_identity, current_query_operation_observer,
    send_call_tool_with_request_id, McpOperationBackend, McpOperationContext,
    McpOperationContextStore,
};
use crate::raw_mcp_observer::{RawMcpObserver, RawMcpReceipt, RawMcpReceiptSink};
use crate::transport;
use crate::types::{
    CallToolResult, GetPromptResult, McpContent, McpPrompt, McpPromptArgument, McpResource,
    McpTool, PromptMessage, PromptMessageContent, PromptsCapability, ResourceContents,
    ResourcesCapability, ServerCapabilities, ServerInfo, ToolsCapability,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use rmcp::model as rmcp_model;
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

const MAX_CURSOR_BYTES: usize = 64 * 1024;
const MAX_INCOMING_MESSAGES: usize = 256;
const MAX_NOTIFICATION_BYTES: usize = 32 * 1024;
const SERVICE_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Lock a `std::sync::Mutex`, recovering from poisoning instead of panicking.
///
/// A poisoned mutex only means some other thread panicked while holding the
/// lock; the guarded state is still valid to read/write here, so panicking
/// again on every subsequent lock would just cascade one earlier failure
/// into a permanently unusable client.
fn lock_recover<T>(mutex: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Own the pre-backend receipt path across every await in connection setup.
///
/// A caller can cancel `connect_with_transport` while rmcp is waiting for a
/// discovery response.  In that case no backend `Drop` runs, so this guard is
/// the only bounded, sink-owned path that can retain the observed partial
/// receipt.  It is disarmed only after the `RunningService` has been moved into
/// the backend monitor.
struct PendingRawReceipt {
    sink: Option<RawMcpReceiptSink>,
    observer: Option<RawMcpObserver>,
}

impl PendingRawReceipt {
    fn new(sink: Option<&RawMcpReceiptSink>, observer: Option<&RawMcpObserver>) -> Self {
        Self {
            sink: sink.cloned(),
            observer: observer.cloned(),
        }
    }

    fn disarm(&mut self) {
        self.sink = None;
        self.observer = None;
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        let sink = self.sink.take();
        let observer = self.observer.take();
        write_raw_mcp_receipt(sink.as_ref(), observer.as_ref())
    }
}

impl Drop for PendingRawReceipt {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

pub struct RmcpNotificationClient {
    notifications_tx: mpsc::Sender<Value>,
    client_info: rmcp_model::ClientInfo,
    catalog: Arc<ServerCatalogState>,
    connection_generation: u64,
}

impl RmcpNotificationClient {
    pub fn new(
        notifications_tx: mpsc::Sender<Value>,
        client_info: rmcp_model::ClientInfo,
        catalog: Arc<ServerCatalogState>,
        connection_generation: u64,
    ) -> Self {
        Self {
            notifications_tx,
            client_info,
            catalog,
            connection_generation,
        }
    }

    fn send_notification(&self, method: &str, params: Option<Value>) {
        let payload = match params {
            Some(params) => json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }),
            None => json!({
                "jsonrpc": "2.0",
                "method": method,
            }),
        };
        let serialized_bytes = match serde_json::to_vec(&payload) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.catalog
                    .mark_unavailable_if_connection(self.connection_generation);
                tracing::warn!(%error, method, "MCP notification could not be serialized; catalog marked unavailable");
                return;
            }
        };
        if serialized_bytes.len() > MAX_NOTIFICATION_BYTES {
            self.catalog
                .mark_unavailable_if_connection(self.connection_generation);
            tracing::warn!(
                method,
                bytes = serialized_bytes.len(),
                limit = MAX_NOTIFICATION_BYTES,
                "MCP notification exceeds bounded queue payload; catalog marked unavailable"
            );
            return;
        }
        if let Err(error) = self.notifications_tx.try_send(payload) {
            self.catalog
                .mark_unavailable_if_connection(self.connection_generation);
            tracing::warn!(%error, method, "MCP notification queue unavailable; catalog marked unavailable");
        }
    }

    fn mark_tools_dirty_and_notify(&self) {
        if self
            .catalog
            .mark_dirty_if_connection(self.connection_generation)
            .is_some()
        {
            self.send_notification("notifications/tools/list_changed", Some(json!({})));
        }
    }
}

impl ClientHandler for RmcpNotificationClient {
    async fn on_resource_updated(
        &self,
        params: rmcp_model::ResourceUpdatedNotificationParam,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) {
        self.send_notification(
            "notifications/resources/updated",
            Some(json!({ "uri": params.uri })),
        );
    }

    async fn on_resource_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) {
        self.send_notification("notifications/resources/list_changed", Some(json!({})));
    }

    async fn on_tool_list_changed(&self, _context: rmcp::service::NotificationContext<RoleClient>) {
        self.mark_tools_dirty_and_notify();
    }

    async fn on_prompt_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) {
        self.send_notification("notifications/prompts/list_changed", Some(json!({})));
    }

    fn get_info(&self) -> rmcp_model::ClientInfo {
        self.client_info.clone()
    }
}

pub struct RmcpClientBackend {
    snapshot: McpClientSnapshot,
    peer: rmcp::Peer<RoleClient>,
    notifications_rx: Arc<Mutex<mpsc::Receiver<Value>>>,
    catalog: Arc<ServerCatalogState>,
    connection_generation: u64,
    cancellation: StdMutex<Option<rmcp::service::RunningServiceCancellationToken>>,
    monitor: JoinHandle<()>,
    service_done: tokio_util::sync::CancellationToken,
    service_error: Arc<StdMutex<Option<String>>>,
    raw_observer: Option<RawMcpObserver>,
    raw_receipt_sink: Option<RawMcpReceiptSink>,
    raw_receipt_written: StdMutex<bool>,
    operation_contexts: McpOperationContextStore,
}

impl RmcpClientBackend {
    pub async fn connect_stdio(
        config: &claurst_core::config::McpServerConfig,
        catalog: Arc<CatalogState>,
    ) -> anyhow::Result<Self> {
        Self::connect_stdio_scoped(config, catalog, None, None).await
    }

    pub async fn connect_stdio_scoped(
        config: &claurst_core::config::McpServerConfig,
        catalog: Arc<CatalogState>,
        working_dir: Option<&std::path::Path>,
        admission: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    ) -> anyhow::Result<Self> {
        Self::connect_stdio_scoped_with_receipt_sink(config, catalog, working_dir, admission, None)
            .await
    }

    pub async fn connect_stdio_scoped_with_receipt_sink(
        config: &claurst_core::config::McpServerConfig,
        catalog: Arc<CatalogState>,
        working_dir: Option<&std::path::Path>,
        admission: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
        receipt_sink: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let command = config.command.clone().ok_or_else(|| {
            anyhow::anyhow!("MCP server '{}' has no command configured", config.name)
        })?;
        let receipt_sink = receipt_sink
            .map(RawMcpReceiptSink::new)
            .transpose()
            .map_err(|error| anyhow::anyhow!("invalid raw MCP receipt sink: {error}"))?;

        // Raw MCP capture is an explicit controller opt-in.  The absence of
        // a validated sink disables the observer before child startup, so no
        // frames or sidecar receipt are retained on ordinary connections.
        let capture_raw = receipt_sink.is_some();
        let mut command_builder = Command::new(&command);
        command_builder.args(&config.args).envs(&config.env);
        if let Some(working_dir) = working_dir {
            command_builder.current_dir(working_dir);
        }
        let transport = match admission {
            Some(admission) => BoundedChildProcess::spawn_admitted_with_raw_observation(
                command_builder,
                Some(admission),
                capture_raw,
            ),
            None => BoundedChildProcess::spawn_admitted_with_raw_observation(
                command_builder,
                None,
                capture_raw,
            ),
        }
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to spawn rmcp stdio child for '{}': {}",
                config.name,
                e
            )
        })?;
        let raw_observer = capture_raw.then(|| transport.raw_observer());

        let client_info = build_client_info(rmcp_model::ProtocolVersion::default());
        Self::connect_with_transport(
            config,
            transport,
            client_info,
            "stdio",
            catalog,
            None,
            (raw_observer, receipt_sink),
        )
        .await
    }

    pub async fn connect_http(
        config: &claurst_core::config::McpServerConfig,
        auth_token: Option<String>,
        protocol_version: rmcp_model::ProtocolVersion,
        catalog: Arc<CatalogState>,
    ) -> anyhow::Result<Self> {
        let endpoint = config
            .url
            .clone()
            .ok_or_else(|| anyhow::anyhow!("MCP server '{}' has no URL configured", config.name))?;

        let transport = BoundedStreamableHttpTransport::new(endpoint, auth_token)?;
        let close_error = transport.close_error_handle();
        let client_info = build_client_info(protocol_version);
        Self::connect_with_transport(
            config,
            transport,
            client_info,
            "http",
            catalog,
            Some(close_error),
            (None, None),
        )
        .await
    }

    pub async fn connect_legacy_sse(
        config: &claurst_core::config::McpServerConfig,
        auth_token: Option<String>,
        catalog: Arc<CatalogState>,
    ) -> anyhow::Result<Self> {
        // Legacy SSE servers still expose the real POST endpoint via
        // `event: endpoint`; keep a thin compatibility shim here and hand the
        // session and capability flow back to rmcp once connected.
        let transport = LegacySseRmcpTransport::connect(config, auth_token).await?;
        let client_info = build_client_info(rmcp_model::ProtocolVersion::V_2024_11_05);
        Self::connect_with_transport(
            config,
            transport,
            client_info,
            "sse",
            catalog,
            None,
            (None, None),
        )
        .await
    }

    async fn connect_with_transport<T, E, A>(
        config: &claurst_core::config::McpServerConfig,
        transport: T,
        client_info: rmcp_model::ClientInfo,
        transport_label: &str,
        catalog: Arc<CatalogState>,
        transport_error: Option<Arc<StdMutex<Option<String>>>>,
        raw_capture: (Option<RawMcpObserver>, Option<RawMcpReceiptSink>),
    ) -> anyhow::Result<Self>
    where
        T: rmcp::transport::IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let (raw_observer, raw_receipt_sink) = raw_capture;
        let mut pending_receipt =
            PendingRawReceipt::new(raw_receipt_sink.as_ref(), raw_observer.as_ref());
        let (notifications_tx, notifications_rx) = mpsc::channel(256);
        let server_catalog = catalog
            .server(&config.name)
            .map_err(|error| anyhow::anyhow!(error))?;
        // Bind the actual immutable launch snapshot only after transport
        // construction succeeds. A failed stdio spawn or HTTP/SSE transport
        // constructor therefore cannot relabel a still-live prior client.
        let connection_generation =
            server_catalog.begin_connection_with_source_key(crate::mcp_launch_source_key(config));
        let handler = RmcpNotificationClient::new(
            notifications_tx,
            client_info,
            Arc::clone(&server_catalog),
            connection_generation,
        );
        let running = match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            handler.serve(transport),
        )
        .await
        {
            Ok(Ok(running)) => running,
            Ok(Err(error)) => {
                let primary = anyhow::anyhow!(
                    "rmcp {} client '{}' failed to initialize: {}",
                    transport_label,
                    config.name,
                    error
                );
                let receipt_error = pending_receipt.flush().err();
                return match receipt_error {
                    Some(receipt_error) => Err(anyhow::anyhow!("{primary}; {receipt_error}")),
                    None => Err(primary),
                };
            }
            Err(_) => {
                let primary = anyhow::anyhow!(
                    "rmcp {} client '{}' initialization timed out",
                    transport_label,
                    config.name
                );
                let receipt_error = pending_receipt.flush().err();
                return match receipt_error {
                    Some(receipt_error) => Err(anyhow::anyhow!("{primary}; {receipt_error}")),
                    None => Err(primary),
                };
            }
        };

        let peer = running.peer().clone();
        let cancellation = running.cancellation_token();
        let snapshot = match build_snapshot(
            config,
            &peer,
            Arc::clone(&server_catalog),
            connection_generation,
        )
        .await
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                server_catalog.mark_unavailable_if_connection(connection_generation);
                cancellation.cancel();
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(2), running.cancel()).await;
                let receipt_error = pending_receipt.flush().err();
                return match receipt_error {
                    Some(receipt_error) => Err(anyhow::anyhow!("{error}; {receipt_error}")),
                    None => Err(error),
                };
            }
        };
        let monitor_catalog = Arc::clone(&server_catalog);
        let service_done = tokio_util::sync::CancellationToken::new();
        let monitor_done = service_done.clone();
        let service_error = Arc::new(StdMutex::new(None));
        let monitor_service_error = Arc::clone(&service_error);
        let monitor_transport_error = transport_error.clone();
        let monitor = tokio::spawn(async move {
            // Keep the RunningService owned by this task. `waiting()` joins
            // rmcp's actual service task; polling `is_closed()` can miss a
            // transport that terminates between polls and leaves ownership
            // ambiguous during teardown.
            let wait_result = running.waiting().await;
            let service_failed = matches!(
                wait_result,
                Ok(rmcp::service::QuitReason::JoinError(_)) | Err(_)
            );
            let transport_error =
                monitor_transport_error.and_then(|sink| lock_recover(&sink).clone());
            if let Some(error) = select_service_teardown_error(service_failed, transport_error) {
                *lock_recover(&monitor_service_error) = Some(error);
            }
            monitor_catalog.mark_unavailable_if_connection(connection_generation);
            monitor_done.cancel();
        });

        pending_receipt.disarm();
        Ok(Self {
            snapshot,
            peer,
            notifications_rx: Arc::new(Mutex::new(notifications_rx)),
            catalog: server_catalog,
            connection_generation,
            cancellation: StdMutex::new(Some(cancellation)),
            monitor,
            service_done,
            service_error,
            raw_observer,
            raw_receipt_sink,
            raw_receipt_written: StdMutex::new(false),
            operation_contexts: McpOperationContextStore::default(),
        })
    }
}

impl RmcpClientBackend {
    pub(crate) fn raw_mcp_receipt(&self) -> Option<RawMcpReceipt> {
        self.raw_observer.as_ref().map(RawMcpObserver::receipt)
    }

    /// Return transport operation contexts for a later query/HMAC adapter.
    /// Query identities remain unset until an exact request-ID join is made.
    pub(crate) fn mcp_operation_contexts(&self) -> anyhow::Result<Vec<McpOperationContext>> {
        self.operation_contexts.snapshot()
    }

    fn flush_raw_mcp_receipt(&self) -> anyhow::Result<()> {
        let Some(sink) = self.raw_receipt_sink.as_ref() else {
            return Ok(());
        };
        let Some(observer) = self.raw_observer.as_ref() else {
            return Err(anyhow::anyhow!("raw MCP receipt sink has no observer"));
        };
        let mut written = lock_recover(&self.raw_receipt_written);
        if *written {
            return Ok(());
        }
        let contexts = self
            .operation_contexts
            .snapshot()?
            .into_iter()
            .map(|context| {
                serde_json::to_value(context).map_err(|error| {
                    anyhow::anyhow!("MCP operation context serialization failed: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        sink.write_with_operation_contexts(&observer.receipt(), &contexts)
            .map_err(|error| anyhow::anyhow!("raw MCP receipt write failed: {error}"))?;
        *written = true;
        Ok(())
    }
}

fn write_raw_mcp_receipt(
    sink: Option<&RawMcpReceiptSink>,
    observer: Option<&RawMcpObserver>,
) -> anyhow::Result<()> {
    match (sink, observer) {
        (Some(sink), Some(observer)) => sink
            .write(&observer.receipt())
            .map_err(|error| anyhow::anyhow!("raw MCP receipt write failed: {error}")),
        (Some(_), None) => Err(anyhow::anyhow!("raw MCP receipt sink has no observer")),
        (None, _) => Ok(()),
    }
}

impl Drop for RmcpClientBackend {
    fn drop(&mut self) {
        self.catalog
            .mark_unavailable_if_connection(self.connection_generation);
        if let Some(cancellation) = lock_recover(&self.cancellation).take() {
            cancellation.cancel();
        }
        self.monitor.abort();
        let _ = self.flush_raw_mcp_receipt();
    }
}

fn select_service_teardown_error(
    service_failed: bool,
    transport_error: Option<String>,
) -> Option<String> {
    transport_error
        .or_else(|| service_failed.then(|| "MCP rmcp service teardown failed".to_owned()))
}

fn build_client_info(protocol_version: rmcp_model::ProtocolVersion) -> rmcp_model::ClientInfo {
    let mut client_info = rmcp_model::ClientInfo::default();
    client_info.protocol_version = protocol_version;
    client_info.capabilities.roots = Some(rmcp_model::RootsCapabilities {
        list_changed: Some(false),
    });
    client_info.client_info = rmcp_model::Implementation::new(
        claurst_core::constants::APP_NAME,
        claurst_core::constants::APP_VERSION,
    );
    client_info
}

#[derive(Debug, thiserror::Error)]
enum LegacySseTransportError {
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

struct LegacySseRmcpTransport {
    server_name: String,
    sse_url: String,
    client: reqwest::Client,
    auth_token: Option<String>,
    post_endpoint: Arc<StdMutex<Option<String>>>,
    incoming_tx: mpsc::Sender<rmcp::service::RxJsonRpcMessage<RoleClient>>,
    incoming_rx: Arc<Mutex<mpsc::Receiver<rmcp::service::RxJsonRpcMessage<RoleClient>>>>,
    background_tasks: Arc<StdMutex<Vec<JoinHandle<()>>>>,
}

impl LegacySseRmcpTransport {
    async fn connect(
        config: &claurst_core::config::McpServerConfig,
        auth_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let sse_url = config
            .url
            .clone()
            .ok_or_else(|| anyhow::anyhow!("MCP server '{}' has no URL configured", config.name))?;
        let client = reqwest::Client::new();
        let (incoming_tx, incoming_rx) = mpsc::channel(MAX_INCOMING_MESSAGES);
        let transport = Self {
            server_name: config.name.clone(),
            sse_url,
            client,
            auth_token,
            post_endpoint: Arc::new(StdMutex::new(None)),
            incoming_tx,
            incoming_rx: Arc::new(Mutex::new(incoming_rx)),
            background_tasks: Arc::new(StdMutex::new(Vec::new())),
        };
        transport.start_sse_listener().await?;
        Ok(transport)
    }

    async fn start_sse_listener(&self) -> anyhow::Result<()> {
        let (endpoint_tx, endpoint_rx) = oneshot::channel::<anyhow::Result<String>>();
        let endpoint_tx = Arc::new(StdMutex::new(Some(endpoint_tx)));

        let mut request = self
            .client
            .get(&self.sse_url)
            .header(reqwest::header::ACCEPT, "text/event-stream");
        if let Some(token) = &self.auth_token {
            request = request.header(
                reqwest::header::AUTHORIZATION,
                transport::bearer_header_value(token)?,
            );
        }
        let response = request.send().await.map_err(|e| {
            anyhow::anyhow!(
                "MCP server '{}': failed to open legacy SSE stream '{}': {}",
                self.server_name,
                self.sse_url,
                e
            )
        })?;
        if !response.status().is_success() {
            let status = response.status();
            let body = bounded_transport::read_response_body(
                response,
                bounded_transport::MAX_HTTP_BODY_BYTES,
            )
            .await
            .unwrap_or_else(|error| error.to_string());
            anyhow::bail!(
                "MCP server '{}': legacy SSE stream returned HTTP {}: {}",
                self.server_name,
                status,
                body
            );
        }

        let server_name = self.server_name.clone();
        let sse_url = self.sse_url.clone();
        let post_endpoint = Arc::clone(&self.post_endpoint);
        let incoming_tx = self.incoming_tx.clone();
        let endpoint_tx_for_task = Arc::clone(&endpoint_tx);
        let task = tokio::spawn(async move {
            let result = transport::process_sse_response(response, |event, data| {
                if matches!(event, Some("endpoint")) {
                    let endpoint = transport::resolve_legacy_endpoint(&sse_url, data)?;
                    *lock_recover(&post_endpoint) = Some(endpoint.clone());
                    if let Some(tx) = lock_recover(&endpoint_tx_for_task).take() {
                        let _ = tx.send(Ok(endpoint));
                    }
                    return Ok(());
                }

                if data.trim().is_empty() {
                    return Ok(());
                }

                let message = parse_server_message(&server_name, data)?;
                incoming_tx.try_send(message).map_err(|error| {
                    anyhow::anyhow!("legacy SSE incoming queue is full or closed: {error}")
                })?;
                Ok(())
            })
            .await;

            if let Err(e) = result {
                tracing::warn!(server = %server_name, error = %e, "Legacy SSE stream closed with error");
                if let Some(tx) = lock_recover(&endpoint_tx_for_task).take() {
                    let _ = tx.send(Err(anyhow::anyhow!(e.to_string())));
                }
            } else if let Some(tx) = lock_recover(&endpoint_tx_for_task).take() {
                let _ = tx.send(Err(anyhow::anyhow!(
                    "legacy SSE stream closed before announcing endpoint"
                )));
            }
        });
        lock_recover(&self.background_tasks).push(task);

        let endpoint = tokio::time::timeout(std::time::Duration::from_secs(10), endpoint_rx)
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "MCP server '{}': timed out waiting for legacy SSE endpoint event",
                    self.server_name
                )
            })?
            .map_err(|_| {
                anyhow::anyhow!(
                    "MCP server '{}': legacy SSE stream ended before endpoint discovery",
                    self.server_name
                )
            })??;
        *lock_recover(&self.post_endpoint) = Some(endpoint);
        Ok(())
    }
}

impl Drop for LegacySseRmcpTransport {
    fn drop(&mut self) {
        // Some upper-layer paths drop the backend instead of calling close()
        // explicitly. Abort the listener tasks again here so legacy SSE does
        // not outlive the connection teardown.
        let mut tasks = lock_recover(&self.background_tasks);
        for handle in tasks.drain(..) {
            handle.abort();
        }
    }
}

impl rmcp::transport::Transport<RoleClient> for LegacySseRmcpTransport {
    type Error = LegacySseTransportError;

    fn send(
        &mut self,
        item: rmcp::service::TxJsonRpcMessage<RoleClient>,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send + 'static {
        let endpoint = lock_recover(&self.post_endpoint).clone();
        let client = self.client.clone();
        let auth_token = self.auth_token.clone();
        let server_name = self.server_name.clone();
        let incoming_tx = self.incoming_tx.clone();
        let background_tasks = Arc::clone(&self.background_tasks);
        async move {
            let endpoint = endpoint.ok_or_else(|| {
                anyhow::anyhow!(
                    "MCP server '{}': legacy SSE POST endpoint has not been discovered",
                    server_name
                )
            })?;

            let mut request = client
                .post(endpoint)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .json(&item);
            if let Some(token) = &auth_token {
                request = request.header(
                    reqwest::header::AUTHORIZATION,
                    transport::bearer_header_value(token)?,
                );
            }

            let response = request.send().await.map_err(|e| {
                anyhow::anyhow!(
                    "MCP server '{}': legacy SSE POST request failed: {}",
                    server_name,
                    e
                )
            })?;

            handle_legacy_sse_http_response(server_name, response, incoming_tx, background_tasks)
                .await
                .map_err(Into::into)
        }
    }

    fn receive(
        &mut self,
    ) -> impl std::future::Future<Output = Option<rmcp::service::RxJsonRpcMessage<RoleClient>>> + Send
    {
        let incoming_rx = Arc::clone(&self.incoming_rx);
        async move {
            let mut rx = incoming_rx.lock().await;
            rx.recv().await
        }
    }

    fn close(&mut self) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let background_tasks = Arc::clone(&self.background_tasks);
        async move {
            let mut tasks = lock_recover(&background_tasks);
            for handle in tasks.drain(..) {
                handle.abort();
            }
            Ok(())
        }
    }
}

fn parse_server_message(
    server_name: &str,
    data: &str,
) -> anyhow::Result<rmcp::service::RxJsonRpcMessage<RoleClient>> {
    let raw: Value = serde_json::from_str(data).map_err(|e| {
        anyhow::anyhow!(
            "MCP server '{}': failed to parse legacy SSE JSON payload: {}",
            server_name,
            e
        )
    })?;
    crate::bounded_transport::reject_ambiguous_resource_payload(&raw)
        .map_err(|error| anyhow::anyhow!("MCP server '{}': {}", server_name, error))?;
    serde_json::from_value(raw).map_err(|e| {
        anyhow::anyhow!(
            "MCP server '{}': failed to parse legacy SSE JSON payload: {}",
            server_name,
            e
        )
    })
}

async fn handle_legacy_sse_http_response(
    server_name: String,
    response: reqwest::Response,
    incoming_tx: mpsc::Sender<rmcp::service::RxJsonRpcMessage<RoleClient>>,
    background_tasks: Arc<StdMutex<Vec<JoinHandle<()>>>>,
) -> anyhow::Result<()> {
    let status = response.status();
    if !status.is_success() && status != reqwest::StatusCode::ACCEPTED {
        let body =
            bounded_transport::read_response_body(response, bounded_transport::MAX_HTTP_BODY_BYTES)
                .await
                .unwrap_or_else(|error| error.to_string());
        anyhow::bail!(
            "MCP server '{}': HTTP {} from legacy SSE transport: {}",
            server_name,
            status,
            body
        );
    }

    if status == reqwest::StatusCode::ACCEPTED {
        return Ok(());
    }

    if transport::is_event_stream_response(&response) {
        let server_name_for_task = server_name.clone();
        let task = tokio::spawn(async move {
            if let Err(e) = transport::process_sse_response(response, |_, data| {
                if data.trim().is_empty() {
                    return Ok(());
                }
                let message = parse_server_message(&server_name_for_task, data)?;
                incoming_tx.try_send(message).map_err(|error| {
                    anyhow::anyhow!("legacy SSE incoming queue is full or closed: {error}")
                })?;
                Ok(())
            })
            .await
            {
                tracing::warn!(server = %server_name_for_task, error = %e, "legacy SSE POST stream closed with error");
            }
        });
        lock_recover(&background_tasks).push(task);
        return Ok(());
    }

    let text =
        bounded_transport::read_response_body(response, bounded_transport::MAX_HTTP_BODY_BYTES)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "MCP server '{}': failed to read legacy SSE HTTP response body: {}",
                    server_name,
                    e
                )
            })?;
    if text.trim().is_empty() {
        return Ok(());
    }

    let message = parse_server_message(&server_name, &text)?;
    incoming_tx
        .try_send(message)
        .map_err(|error| anyhow::anyhow!("legacy SSE incoming queue is full or closed: {error}"))?;
    Ok(())
}

#[async_trait]
impl McpClientBackend for RmcpClientBackend {
    async fn shutdown(&self) -> anyhow::Result<()> {
        self.catalog
            .mark_unavailable_if_connection(self.connection_generation);
        let cancellation = { lock_recover(&self.cancellation).take() };
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
        }
        let service_wait =
            tokio::time::timeout(SERVICE_SHUTDOWN_TIMEOUT, self.service_done.cancelled()).await;
        if service_wait.is_err() {
            // The monitor owns RunningService.  Abort it only after the
            // bounded wait, then flush the partial observer snapshot and
            // report cleanup as failed instead of hanging the caller.
            self.monitor.abort();
        }
        let receipt_error = self.flush_raw_mcp_receipt().err();
        let service_error = if service_wait.is_err() {
            Some("MCP service shutdown deadline exceeded".to_owned())
        } else {
            lock_recover(&self.service_error).clone()
        };
        match (service_error, receipt_error) {
            (Some(service), Some(receipt)) => Err(anyhow::anyhow!(
                "MCP service teardown failed: {service}; raw receipt write failed: {receipt}"
            )),
            (Some(service), None) => Err(anyhow::anyhow!(service)),
            (None, Some(receipt)) => Err(receipt),
            (None, None) => Ok(()),
        }
    }

    fn kind(&self) -> McpBackendKind {
        McpBackendKind::Rmcp
    }

    fn snapshot(&self) -> McpClientSnapshot {
        self.snapshot.clone()
    }

    fn catalog_state(&self) -> Arc<ServerCatalogState> {
        Arc::clone(&self.catalog)
    }

    fn operation_contexts(&self) -> anyhow::Result<Vec<McpOperationContext>> {
        self.mcp_operation_contexts()
    }

    async fn list_tools(&self) -> anyhow::Result<Vec<McpTool>> {
        let starting_generation = self
            .catalog
            .begin_refresh_if_connection(self.connection_generation)
            .map_err(|error| anyhow::anyhow!(error))?;
        let tools = match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            bounded_list_tools(&self.peer),
        )
        .await
        {
            Ok(Ok(tools)) => tools,
            Ok(Err(error)) => {
                self.catalog
                    .fail_refresh_if_current(self.connection_generation, starting_generation);
                return Err(error);
            }
            Err(_) => {
                self.catalog
                    .fail_refresh_if_current(self.connection_generation, starting_generation);
                return Err(anyhow::anyhow!("rmcp list_tools timed out"));
            }
        };
        self.catalog
            .commit_refresh(starting_generation, self.connection_generation, &tools)
            .map_err(|error| anyhow::anyhow!(error))?;
        Ok(tools)
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> anyhow::Result<CallToolResult> {
        let query_identity = current_query_operation_identity();
        let query_observer = current_query_operation_observer();
        let backend = self.raw_observer.as_ref().and_then(|_| {
            self.catalog
                .snapshot()
                .tools
                .iter()
                .find(|descriptor| descriptor.raw_name() == name)
                .map(McpOperationBackend::from_descriptor)
        });
        let arguments_for_binding = arguments.clone();
        let mut params = rmcp_model::CallToolRequestParams::new(name.to_string());
        if let Some(arguments) = arguments {
            params = params.with_arguments(json_value_to_object(arguments)?);
        }
        let result = if self.raw_observer.is_none() {
            // Preserve the default no-sink path and its existing typed rmcp
            // convenience call exactly.
            self.peer
                .call_tool(params)
                .await
                .map_err(|e| anyhow::anyhow!("rmcp call_tool '{}' failed: {}", name, e))?
        } else {
            let (result, request_id) = send_call_tool_with_request_id(&self.peer, params).await?;
            let receipt = self.raw_mcp_receipt().ok_or_else(|| {
                anyhow::anyhow!(
                    "MCP operation capture was enabled but no raw receipt observer exists"
                )
            })?;
            let mut context = bind_call_tool_receipt(
                &request_id,
                name,
                arguments_for_binding.as_ref(),
                &result,
                &receipt,
            )?;
            let identity = query_identity.unwrap_or_default();
            let matched =
                backend.is_some() && identity.expected_backend.as_ref() == backend.as_ref();
            context.backend = backend;
            // Direct host-only identity scopes remain metadata. Query observers
            // require an exact executable/backend match and a completed raw pair.
            if matched || (identity.expected_backend.is_none() && query_observer.is_none()) {
                context.query_identity = identity;
            }
            self.operation_contexts.record(context.clone())?;
            if matched {
                if let Some(observer) = query_observer {
                    observer(&context);
                }
            }
            result
        };
        Ok(convert_call_tool_result(result))
    }

    async fn list_resources(&self) -> anyhow::Result<Vec<McpResource>> {
        let resources = self
            .peer
            .list_all_resources()
            .await
            .map_err(|e| anyhow::anyhow!("rmcp list_resources failed: {}", e))?;
        Ok(resources.into_iter().map(convert_resource).collect())
    }

    async fn read_resource(&self, uri: &str) -> anyhow::Result<Vec<ResourceContents>> {
        let result = self
            .peer
            .read_resource(rmcp_model::ReadResourceRequestParams::new(uri.to_string()))
            .await
            .map_err(|e| anyhow::anyhow!("rmcp read_resource '{}' failed: {}", uri, e))?;
        if result.contents.is_empty() {
            return Err(anyhow::anyhow!(
                "rmcp read_resource '{}' returned no contents",
                uri
            ));
        }
        Ok(result
            .contents
            .into_iter()
            .map(convert_resource_contents)
            .collect())
    }

    async fn list_prompts(&self) -> anyhow::Result<Vec<McpPrompt>> {
        let prompts = self
            .peer
            .list_all_prompts()
            .await
            .map_err(|e| anyhow::anyhow!("rmcp list_prompts failed: {}", e))?;
        Ok(prompts.into_iter().map(convert_prompt).collect())
    }

    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<HashMap<String, String>>,
    ) -> anyhow::Result<GetPromptResult> {
        let mut params = rmcp_model::GetPromptRequestParams::new(name.to_string());
        if let Some(arguments) = arguments {
            let args = arguments
                .into_iter()
                .map(|(key, value)| (key, Value::String(value)))
                .collect();
            params = params.with_arguments(args);
        }
        let result = self
            .peer
            .get_prompt(params)
            .await
            .map_err(|e| anyhow::anyhow!("rmcp get_prompt '{}' failed: {}", name, e))?;
        Ok(convert_get_prompt_result(result))
    }

    async fn subscribe_resource(&self, uri: &str) -> anyhow::Result<()> {
        self.peer
            .subscribe(rmcp_model::SubscribeRequestParams::new(uri.to_string()))
            .await
            .map_err(|e| anyhow::anyhow!("rmcp subscribe '{}' failed: {}", uri, e))
    }

    async fn unsubscribe_resource(&self, uri: &str) -> anyhow::Result<()> {
        self.peer
            .unsubscribe(rmcp_model::UnsubscribeRequestParams::new(uri.to_string()))
            .await
            .map_err(|e| anyhow::anyhow!("rmcp unsubscribe '{}' failed: {}", uri, e))
    }

    fn subscribe_to_notifications(&self) -> BoxStream<'static, anyhow::Result<Value>> {
        let notifications_rx = Arc::clone(&self.notifications_rx);
        let (tx, rx) = mpsc::channel::<anyhow::Result<Value>>(100);
        tokio::spawn(async move {
            loop {
                let next = {
                    let mut receiver = notifications_rx.lock().await;
                    receiver.recv().await
                };
                match next {
                    Some(value) => {
                        if tx.send(Ok(value)).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                }
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }
}

async fn build_snapshot(
    config: &claurst_core::config::McpServerConfig,
    peer: &rmcp::Peer<RoleClient>,
    server_catalog: Arc<ServerCatalogState>,
    connection_generation: u64,
) -> anyhow::Result<McpClientSnapshot> {
    let server_info = peer.peer_info().map(|info| (*info).clone());

    let (server_info_value, capabilities, instructions) = match server_info {
        Some(info) => (
            Some(ServerInfo {
                name: info.server_info.name,
                version: info.server_info.version,
            }),
            convert_server_capabilities(info.capabilities),
            info.instructions,
        ),
        None => (None, ServerCapabilities::default(), None),
    };

    let tools = if capabilities.tools.is_some() {
        let tools =
            match tokio::time::timeout(std::time::Duration::from_secs(5), bounded_list_tools(peer))
                .await
            {
                Ok(Ok(tools)) => tools,
                Ok(Err(error)) => {
                    server_catalog.mark_unavailable_if_connection(connection_generation);
                    return Err(error);
                }
                Err(_) => {
                    server_catalog.mark_unavailable_if_connection(connection_generation);
                    return Err(anyhow::anyhow!("rmcp initial tools/list timed out"));
                }
            };
        server_catalog
            .install_tools(connection_generation, &tools)
            .map_err(|error| anyhow::anyhow!(error))?;
        tools
    } else {
        server_catalog
            .install_tools(connection_generation, &[])
            .map_err(|error| anyhow::anyhow!(error))?;
        vec![]
    };

    let resources = if capabilities.resources.is_some() {
        (match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            bounded_list_resources(peer),
        )
        .await
        {
            Ok(Ok(resources)) => resources,
            Ok(Err(error)) => {
                server_catalog.mark_unavailable_if_connection(connection_generation);
                return Err(anyhow::anyhow!("rmcp resources/list failed: {}", error));
            }
            Err(_) => {
                server_catalog.mark_unavailable_if_connection(connection_generation);
                return Err(anyhow::anyhow!("rmcp initial resources/list timed out"));
            }
        })
        .into_iter()
        .map(convert_resource)
        .collect()
    } else {
        vec![]
    };

    let prompts = if capabilities.prompts.is_some() {
        (match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            bounded_list_prompts(peer),
        )
        .await
        {
            Ok(Ok(prompts)) => prompts,
            Ok(Err(error)) => {
                server_catalog.mark_unavailable_if_connection(connection_generation);
                return Err(anyhow::anyhow!("rmcp prompts/list failed: {}", error));
            }
            Err(_) => {
                server_catalog.mark_unavailable_if_connection(connection_generation);
                return Err(anyhow::anyhow!("rmcp initial prompts/list timed out"));
            }
        })
        .into_iter()
        .map(convert_prompt)
        .collect()
    } else {
        vec![]
    };

    Ok(McpClientSnapshot {
        server_name: config.name.clone(),
        server_info: server_info_value,
        capabilities,
        tools,
        resources,
        prompts,
        instructions,
    })
}

async fn bounded_list_tools(peer: &rmcp::Peer<RoleClient>) -> anyhow::Result<Vec<McpTool>> {
    let mut cursor = None;
    let mut cursors = std::collections::HashSet::new();
    let mut tools = Vec::new();
    let mut total_bytes = 0usize;
    for _ in 0..crate::catalog_state::MAX_PAGES {
        if tools.len() >= crate::catalog_state::MAX_TOOLS {
            anyhow::bail!("MCP tools/list exceeded the tool count limit");
        }
        let page = peer
            .list_tools(Some(
                rmcp_model::PaginatedRequestParams::default().with_cursor(cursor.clone()),
            ))
            .await
            .map_err(|error| anyhow::anyhow!("rmcp list_tools failed: {error}"))?;
        for tool in page.tools {
            let encoded = serde_json::to_vec(&tool)
                .map_err(|error| anyhow::anyhow!("invalid MCP tool definition: {error}"))?;
            if encoded.len() > crate::catalog_state::MAX_DEFINITION_BYTES {
                anyhow::bail!("MCP tool definition exceeds the byte limit");
            }
            total_bytes = total_bytes.saturating_add(encoded.len());
            if total_bytes > crate::catalog_state::MAX_CATALOG_BYTES {
                anyhow::bail!("MCP tools/list snapshot exceeds the byte limit");
            }
            tools.push(convert_tool(tool));
            if tools.len() > crate::catalog_state::MAX_TOOLS {
                anyhow::bail!("MCP tools/list exceeded the tool count limit");
            }
        }
        cursor = checked_next_cursor(page.next_cursor, &mut cursors, "tools/list")?;
        if cursor.is_none() {
            return Ok(tools);
        }
    }
    anyhow::bail!("MCP tools/list exceeded the page limit")
}

async fn bounded_list_resources(
    peer: &rmcp::Peer<RoleClient>,
) -> anyhow::Result<Vec<rmcp_model::Resource>> {
    let mut cursor = None;
    let mut cursors = HashSet::new();
    let mut resources = Vec::new();
    let mut total_bytes = 0usize;
    for _ in 0..crate::catalog_state::MAX_PAGES {
        if resources.len() >= crate::catalog_state::MAX_TOOLS {
            anyhow::bail!("MCP resources/list exceeded the item count limit");
        }
        let page = peer
            .list_resources(Some(
                rmcp_model::PaginatedRequestParams::default().with_cursor(cursor.clone()),
            ))
            .await
            .map_err(|error| anyhow::anyhow!("rmcp list_resources failed: {error}"))?;
        for resource in page.resources {
            let encoded = serde_json::to_vec(&resource)
                .map_err(|error| anyhow::anyhow!("invalid MCP resource definition: {error}"))?;
            if encoded.len() > crate::catalog_state::MAX_DEFINITION_BYTES {
                anyhow::bail!("MCP resource definition exceeds the byte limit");
            }
            total_bytes = total_bytes.checked_add(encoded.len()).ok_or_else(|| {
                anyhow::anyhow!("MCP resources/list snapshot exceeds the byte limit")
            })?;
            if total_bytes > crate::catalog_state::MAX_CATALOG_BYTES {
                anyhow::bail!("MCP resources/list snapshot exceeds the byte limit");
            }
            resources.push(resource);
            if resources.len() > crate::catalog_state::MAX_TOOLS {
                anyhow::bail!("MCP resources/list exceeded the item count limit");
            }
        }
        cursor = checked_next_cursor(page.next_cursor, &mut cursors, "resources/list")?;
        if cursor.is_none() {
            return Ok(resources);
        }
    }
    anyhow::bail!("MCP resources/list exceeded the page limit")
}

async fn bounded_list_prompts(
    peer: &rmcp::Peer<RoleClient>,
) -> anyhow::Result<Vec<rmcp_model::Prompt>> {
    let mut cursor = None;
    let mut cursors = HashSet::new();
    let mut prompts = Vec::new();
    let mut total_bytes = 0usize;
    for _ in 0..crate::catalog_state::MAX_PAGES {
        if prompts.len() >= crate::catalog_state::MAX_TOOLS {
            anyhow::bail!("MCP prompts/list exceeded the item count limit");
        }
        let page = peer
            .list_prompts(Some(
                rmcp_model::PaginatedRequestParams::default().with_cursor(cursor.clone()),
            ))
            .await
            .map_err(|error| anyhow::anyhow!("rmcp list_prompts failed: {error}"))?;
        for prompt in page.prompts {
            let encoded = serde_json::to_vec(&prompt)
                .map_err(|error| anyhow::anyhow!("invalid MCP prompt definition: {error}"))?;
            if encoded.len() > crate::catalog_state::MAX_DEFINITION_BYTES {
                anyhow::bail!("MCP prompt definition exceeds the byte limit");
            }
            total_bytes = total_bytes.checked_add(encoded.len()).ok_or_else(|| {
                anyhow::anyhow!("MCP prompts/list snapshot exceeds the byte limit")
            })?;
            if total_bytes > crate::catalog_state::MAX_CATALOG_BYTES {
                anyhow::bail!("MCP prompts/list snapshot exceeds the byte limit");
            }
            prompts.push(prompt);
            if prompts.len() > crate::catalog_state::MAX_TOOLS {
                anyhow::bail!("MCP prompts/list exceeded the item count limit");
            }
        }
        cursor = checked_next_cursor(page.next_cursor, &mut cursors, "prompts/list")?;
        if cursor.is_none() {
            return Ok(prompts);
        }
    }
    anyhow::bail!("MCP prompts/list exceeded the page limit")
}

fn checked_next_cursor(
    cursor: Option<String>,
    seen: &mut HashSet<String>,
    resource: &str,
) -> anyhow::Result<Option<String>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    if cursor.len() > MAX_CURSOR_BYTES {
        anyhow::bail!("MCP {resource} cursor exceeds the byte limit");
    }
    if !seen.insert(cursor.clone()) {
        anyhow::bail!("MCP {resource} returned a cursor cycle");
    }
    Ok(Some(cursor))
}

fn convert_server_capabilities(capabilities: rmcp_model::ServerCapabilities) -> ServerCapabilities {
    ServerCapabilities {
        tools: capabilities.tools.map(|tools| ToolsCapability {
            list_changed: tools.list_changed.unwrap_or(false),
        }),
        resources: capabilities.resources.map(|resources| ResourcesCapability {
            subscribe: resources.subscribe.unwrap_or(false),
            list_changed: resources.list_changed.unwrap_or(false),
        }),
        prompts: capabilities.prompts.map(|prompts| PromptsCapability {
            list_changed: prompts.list_changed.unwrap_or(false),
        }),
        logging: capabilities.logging.map(Value::Object),
    }
}

fn convert_tool(tool: rmcp_model::Tool) -> McpTool {
    McpTool {
        name: tool.name.into_owned(),
        description: tool.description.map(|description| description.into_owned()),
        input_schema: Value::Object((*tool.input_schema).clone()),
    }
}

fn convert_resource(resource: rmcp_model::Resource) -> McpResource {
    McpResource {
        uri: resource.uri.clone(),
        name: resource.name.clone(),
        description: resource.description.clone(),
        mime_type: resource.mime_type.clone(),
        annotations: resource
            .annotations
            .as_ref()
            .and_then(|annotations| serde_json::to_value(annotations).ok()),
    }
}

fn convert_prompt(prompt: rmcp_model::Prompt) -> McpPrompt {
    McpPrompt {
        name: prompt.name,
        description: prompt.description,
        arguments: prompt
            .arguments
            .unwrap_or_default()
            .into_iter()
            .map(|argument| McpPromptArgument {
                name: argument.name,
                description: argument.description,
                required: argument.required.unwrap_or(false),
            })
            .collect(),
    }
}

fn convert_resource_contents(resource: rmcp_model::ResourceContents) -> ResourceContents {
    match resource {
        rmcp_model::ResourceContents::TextResourceContents {
            uri,
            mime_type,
            text,
            ..
        } => ResourceContents {
            uri,
            mime_type,
            text: Some(text),
            blob: None,
        },
        rmcp_model::ResourceContents::BlobResourceContents {
            uri,
            mime_type,
            blob,
            ..
        } => ResourceContents {
            uri,
            mime_type,
            text: None,
            blob: Some(blob),
        },
    }
}

fn convert_prompt_message(message: rmcp_model::PromptMessage) -> PromptMessage {
    PromptMessage {
        role: match message.role {
            rmcp_model::PromptMessageRole::User => "user".to_string(),
            rmcp_model::PromptMessageRole::Assistant => "assistant".to_string(),
        },
        content: match message.content {
            rmcp_model::PromptMessageContent::Text { text } => PromptMessageContent::Text { text },
            rmcp_model::PromptMessageContent::Image { image } => PromptMessageContent::Image {
                data: image.data.clone(),
                mime_type: image.mime_type.clone(),
            },
            rmcp_model::PromptMessageContent::Resource { resource } => {
                PromptMessageContent::Resource {
                    resource: serde_json::to_value(resource).unwrap_or(Value::Null),
                }
            }
            rmcp_model::PromptMessageContent::ResourceLink { link } => {
                PromptMessageContent::Resource {
                    resource: serde_json::to_value(link).unwrap_or(Value::Null),
                }
            }
        },
    }
}

fn convert_get_prompt_result(result: rmcp_model::GetPromptResult) -> GetPromptResult {
    GetPromptResult {
        description: result.description,
        messages: result
            .messages
            .into_iter()
            .map(convert_prompt_message)
            .collect(),
    }
}

fn convert_call_tool_result(result: rmcp_model::CallToolResult) -> CallToolResult {
    CallToolResult {
        content: result.content.into_iter().map(convert_content).collect(),
        is_error: result.is_error.unwrap_or(false),
    }
}

fn convert_content(content: rmcp_model::Content) -> McpContent {
    match content.raw {
        rmcp_model::RawContent::Text(text) => McpContent::Text { text: text.text },
        rmcp_model::RawContent::Image(image) => McpContent::Image {
            data: image.data.clone(),
            mime_type: image.mime_type.clone(),
        },
        rmcp_model::RawContent::Resource(resource) => McpContent::Resource {
            resource: convert_resource_contents(resource.resource),
        },
        rmcp_model::RawContent::ResourceLink(link) => McpContent::ResourceLink {
            uri: link.uri,
            name: link.name,
            description: link.description,
            mime_type: link.mime_type,
        },
        rmcp_model::RawContent::Audio(audio) => McpContent::Audio {
            data: audio.data,
            mime_type: audio.mime_type,
        },
    }
}

fn json_value_to_object(value: Value) -> anyhow::Result<rmcp_model::JsonObject> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(anyhow::anyhow!(
            "rmcp tool arguments must be a JSON object, got {}",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::config::McpServerConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::time::{timeout, Duration};

    fn test_sse_config(url: String) -> McpServerConfig {
        McpServerConfig {
            name: "test-sse".to_string(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: Some(url),
            server_type: "sse".to_string(),
            origin: Default::default(),
        }
    }

    async fn serve_bounded_json_peer(
        io: tokio::io::DuplexStream,
        list_calls: Arc<AtomicUsize>,
        tool_calls: Arc<AtomicUsize>,
    ) -> bool {
        let (reader, mut writer) = tokio::io::split(io);
        let mut reader = tokio::io::BufReader::new(reader);
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = match reader.read_until(b'\n', &mut line).await {
                Ok(read) => read,
                Err(_) => return false,
            };
            if read == 0 {
                return true;
            }
            let request: serde_json::Value = match serde_json::from_slice(&line) {
                Ok(request) => request,
                Err(_) => return false,
            };
            let Some(method) = request.get("method").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(id) = request.get("id") else {
                continue;
            };
            let result = match method {
                "initialize" => serde_json::json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"}
                }),
                "tools/list" => {
                    list_calls.fetch_add(1, Ordering::SeqCst);
                    serde_json::json!({
                        "tools": [{
                            "name": "echo",
                            "description": "echo test",
                            "inputSchema": {"type": "object"}
                        }],
                        "nextCursor": null
                    })
                }
                "tools/call" => {
                    tool_calls.fetch_add(1, Ordering::SeqCst);
                    serde_json::json!({
                        "content": [{"type": "text", "text": "ok"}],
                        "isError": false
                    })
                }
                _ => serde_json::json!({}),
            };
            let mut response = serde_json::to_vec(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result
            }))
            .expect("serialize loopback response");
            response.push(b'\n');
            if writer.write_all(&response).await.is_err() {
                return false;
            }
        }
    }

    async fn serve_single_response(status_line: &str, content_type: &str, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");
        let response = format!(
            "HTTP/1.1 {status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept test connection");
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await;
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write test response");
            let _ = stream.shutdown().await;
        });
        format!("http://{addr}")
    }

    async fn fetch_response(
        status_line: &str,
        content_type: &str,
        body: &str,
    ) -> reqwest::Response {
        let url = serve_single_response(status_line, content_type, body).await;
        reqwest::Client::new()
            .get(url)
            .send()
            .await
            .expect("fetch test response")
    }

    #[test]
    fn lock_recover_returns_guard_after_poisoning() {
        let mutex = Arc::new(StdMutex::new(0));
        let poisoned = Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.lock().expect("lock for poisoning");
            panic!("poison the mutex");
        })
        .join();

        assert!(mutex.is_poisoned());
        let guard = lock_recover(&mutex);
        assert_eq!(*guard, 0);
    }

    #[test]
    fn build_client_info_sets_expected_protocol_and_identity() {
        let info = build_client_info(rmcp_model::ProtocolVersion::V_2024_11_05);
        assert_eq!(
            info.protocol_version,
            rmcp_model::ProtocolVersion::V_2024_11_05
        );
        assert_eq!(info.client_info.name, claurst_core::constants::APP_NAME);
        assert_eq!(
            info.client_info.version,
            claurst_core::constants::APP_VERSION
        );
        assert_eq!(
            info.capabilities
                .roots
                .as_ref()
                .and_then(|roots| roots.list_changed),
            Some(false)
        );
    }

    #[test]
    fn bounded_cursor_rejects_cycles_and_oversized_values() {
        let mut seen = HashSet::new();
        assert_eq!(
            checked_next_cursor(Some("page-1".to_string()), &mut seen, "tools/list")
                .unwrap()
                .as_deref(),
            Some("page-1")
        );
        assert!(checked_next_cursor(Some("page-1".to_string()), &mut seen, "tools/list").is_err());
        assert!(checked_next_cursor(
            Some("x".repeat(MAX_CURSOR_BYTES + 1)),
            &mut seen,
            "tools/list"
        )
        .is_err());
    }

    #[test]
    fn resource_conversion_preserves_ordered_text_and_blob_entries() {
        let resources = vec![
            rmcp_model::ResourceContents::TextResourceContents {
                uri: "file:///one.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                text: "one".to_string(),
                meta: None,
            },
            rmcp_model::ResourceContents::BlobResourceContents {
                uri: "file:///two.png".to_string(),
                mime_type: Some("image/png".to_string()),
                blob: "aGVsbG8=".to_string(),
                meta: None,
            },
        ];
        let converted = resources
            .into_iter()
            .map(convert_resource_contents)
            .collect::<Vec<_>>();
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].text.as_deref(), Some("one"));
        assert_eq!(converted[1].blob.as_deref(), Some("aGVsbG8="));
        assert_eq!(converted[0].uri, "file:///one.txt");
        assert_eq!(converted[1].uri, "file:///two.png");
    }

    #[tokio::test]
    async fn legacy_sse_transport_connect_discovers_endpoint_event() {
        let body = "event: endpoint\ndata: /messages\n\n";
        let url = serve_single_response("200 OK", "text/event-stream", body).await;
        let config = test_sse_config(url.clone());

        let transport = LegacySseRmcpTransport::connect(&config, None)
            .await
            .expect("connect legacy sse transport");

        let endpoint = transport
            .post_endpoint
            .lock()
            .expect("endpoint mutex poisoned")
            .clone();
        let expected = format!("{url}/messages");
        assert_eq!(endpoint.as_deref(), Some(expected.as_str()));
    }

    #[tokio::test]
    async fn handle_legacy_sse_http_response_queues_json_message() {
        let response = fetch_response(
            "200 OK",
            "application/json",
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
        )
        .await;
        let (tx, mut rx) = mpsc::channel(8);
        let tasks = Arc::new(StdMutex::new(Vec::new()));

        handle_legacy_sse_http_response("test".to_string(), response, tx, Arc::clone(&tasks))
            .await
            .expect("handle json response");

        let message = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("json response timeout")
            .expect("json response message");
        assert!(matches!(
            message,
            rmcp::service::RxJsonRpcMessage::<RoleClient>::Response(_)
        ));
    }

    #[tokio::test]
    async fn handle_legacy_sse_http_response_queues_sse_message() {
        let response = fetch_response(
            "200 OK",
            "text/event-stream",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n",
        )
        .await;
        let (tx, mut rx) = mpsc::channel(8);
        let tasks = Arc::new(StdMutex::new(Vec::new()));

        handle_legacy_sse_http_response("test".to_string(), response, tx, Arc::clone(&tasks))
            .await
            .expect("handle sse response");

        let message = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("sse response timeout")
            .expect("sse response message");
        assert!(matches!(
            message,
            rmcp::service::RxJsonRpcMessage::<RoleClient>::Response(_)
        ));
    }

    #[tokio::test]
    async fn legacy_sse_transport_drop_aborts_background_tasks() {
        let task = tokio::spawn(async move {
            loop {
                tokio::task::yield_now().await;
            }
        });

        let transport = LegacySseRmcpTransport {
            server_name: "test".to_string(),
            sse_url: "http://localhost/sse".to_string(),
            client: reqwest::Client::new(),
            auth_token: None,
            post_endpoint: Arc::new(StdMutex::new(None)),
            incoming_tx: mpsc::channel(MAX_INCOMING_MESSAGES).0,
            incoming_rx: Arc::new(Mutex::new(mpsc::channel(MAX_INCOMING_MESSAGES).1)),
            background_tasks: Arc::new(StdMutex::new(vec![task])),
        };

        let background_tasks = Arc::clone(&transport.background_tasks);
        drop(transport);

        let is_empty = background_tasks
            .lock()
            .expect("task mutex poisoned")
            .is_empty();
        assert!(is_empty);
    }

    #[tokio::test]
    async fn tool_list_notification_invalidates_before_delivery() {
        let catalog = Arc::new(ServerCatalogState::new("test"));
        let tool = McpTool {
            name: "echo".into(),
            description: None,
            input_schema: serde_json::json!({"type": "object"}),
        };
        catalog
            .install_tools(1, &[tool])
            .expect("install test catalog");
        let (tx, mut rx) = mpsc::channel(8);
        let handler = RmcpNotificationClient::new(
            tx,
            rmcp_model::ClientInfo::default(),
            Arc::clone(&catalog),
            1,
        );

        handler.mark_tools_dirty_and_notify();
        assert_eq!(
            catalog.availability(),
            crate::catalog_state::CatalogAvailability::Dirty
        );
        let notification = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("notification timeout")
            .expect("notification payload");
        assert_eq!(notification["method"], "notifications/tools/list_changed");
        assert!(catalog.snapshot().tools.is_empty());
    }

    #[test]
    fn notification_queue_overflow_and_payload_size_fail_closed() {
        let catalog = Arc::new(ServerCatalogState::new("test"));
        catalog
            .install_tools(
                1,
                &[McpTool {
                    name: "echo".into(),
                    description: None,
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            )
            .expect("install test catalog");
        let (tx, mut rx) = mpsc::channel(1);
        let handler = RmcpNotificationClient::new(
            tx,
            rmcp_model::ClientInfo::default(),
            Arc::clone(&catalog),
            1,
        );
        for _ in 0..=MAX_INCOMING_MESSAGES {
            handler.send_notification("notifications/resources/list_changed", None);
        }
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_err());
        assert_eq!(
            catalog.availability(),
            crate::catalog_state::CatalogAvailability::Unavailable
        );
        assert!(catalog.snapshot().tools.is_empty());

        let oversized_catalog = Arc::new(ServerCatalogState::new("oversized"));
        oversized_catalog
            .install_tools(
                1,
                &[McpTool {
                    name: "echo".into(),
                    description: None,
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            )
            .expect("install oversized test catalog");
        let (oversized_tx, mut oversized_rx) = mpsc::channel(1);
        let oversized_handler = RmcpNotificationClient::new(
            oversized_tx,
            rmcp_model::ClientInfo::default(),
            Arc::clone(&oversized_catalog),
            1,
        );
        oversized_handler.send_notification(
            "notifications/resources/updated",
            Some(json!({"uri": "x".repeat(MAX_NOTIFICATION_BYTES)})),
        );
        assert!(oversized_rx.try_recv().is_err());
        assert_eq!(
            oversized_catalog.availability(),
            crate::catalog_state::CatalogAvailability::Unavailable
        );
        assert!(oversized_catalog.snapshot().tools.is_empty());
    }

    #[tokio::test]
    async fn old_notification_cannot_dirty_replacement_connection() {
        let catalog = Arc::new(ServerCatalogState::new("test"));
        catalog
            .install_tools(
                1,
                &[McpTool {
                    name: "echo".into(),
                    description: None,
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            )
            .expect("install initial catalog");
        let (tx, mut rx) = mpsc::channel(8);
        let old_handler = RmcpNotificationClient::new(
            tx,
            rmcp_model::ClientInfo::default(),
            Arc::clone(&catalog),
            1,
        );
        let replacement = catalog.begin_connection();
        catalog
            .install_tools(
                replacement,
                &[McpTool {
                    name: "echo".into(),
                    description: None,
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            )
            .expect("install replacement catalog");
        old_handler.mark_tools_dirty_and_notify();
        assert_eq!(
            catalog.availability(),
            crate::catalog_state::CatalogAvailability::Available
        );
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn teardown_error_selection_prefers_sanitized_transport_category() {
        let transport_error =
            Some("MCP HTTP reader teardown timed out or remained unresolved".to_owned());
        assert_eq!(
            select_service_teardown_error(true, transport_error),
            Some("MCP HTTP reader teardown timed out or remained unresolved".to_owned())
        );
        assert_eq!(
            select_service_teardown_error(true, None),
            Some("MCP rmcp service teardown failed".to_owned())
        );
        assert_eq!(select_service_teardown_error(false, None), None);
    }

    #[tokio::test]
    async fn loopback_backend_catalogs_and_calls_counter_server() {
        let list_calls = Arc::new(AtomicUsize::new(0));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let (server_io, client_io) = tokio::io::duplex(16 * 1024);
        let server_list_calls = Arc::clone(&list_calls);
        let server_tool_calls = Arc::clone(&tool_calls);
        let server_task = tokio::spawn(async move {
            serve_bounded_json_peer(server_io, server_list_calls, server_tool_calls).await
        });

        let config = McpServerConfig {
            name: "loopback".to_string(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: None,
            server_type: "duplex".to_string(),
            origin: Default::default(),
        };
        let backend = RmcpClientBackend::connect_with_transport(
            &config,
            client_io,
            build_client_info(rmcp_model::ProtocolVersion::default()),
            "duplex",
            Arc::new(CatalogState::new()),
            None,
            (None, None),
        )
        .await
        .expect("loopback backend connection");
        assert_eq!(list_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.catalog_state().snapshot().tools.len(), 1);

        backend
            .call_tool("echo", Some(json!({})))
            .await
            .expect("loopback tool call");
        assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
        drop(backend);
        assert!(
            timeout(Duration::from_secs(1), server_task)
                .await
                .expect("server shutdown timeout")
                .expect("server task join"),
            "server did not observe client EOF"
        );
    }

    #[tokio::test]
    async fn loopback_http_backend_shutdown_propagates_reader_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (get_seen_tx, get_seen_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let mut initialized = false;
            let mut notification = false;
            let mut get = false;
            let mut deleted = false;
            let mut get_seen_tx = Some(get_seen_tx);
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut chunk = [0_u8; 1024];
                    let read = stream.read(&mut chunk).await.unwrap();
                    assert!(read > 0, "HTTP request ended before headers");
                    request.extend_from_slice(&chunk[..read]);
                    assert!(request.len() <= 8192, "fixture request exceeds bound");
                    if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let headers = std::str::from_utf8(&request[..header_end]).unwrap();
                let is_get = headers.starts_with("GET ");
                let is_delete = headers.starts_with("DELETE ");
                let body_len = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                assert!(header_end + body_len <= 8192, "fixture body exceeds bound");
                while request.len() < header_end + body_len {
                    let mut chunk = [0_u8; 1024];
                    let read = stream.read(&mut chunk).await.unwrap();
                    assert!(read > 0, "HTTP request ended before body");
                    request.extend_from_slice(&chunk[..read]);
                    assert!(request.len() <= 8192, "fixture request exceeds bound");
                }
                let response = if is_get {
                    assert!(!get);
                    get = true;
                    "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned()
                } else if is_delete {
                    assert!(!deleted);
                    deleted = true;
                    "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_owned()
                } else {
                    let body: serde_json::Value =
                        serde_json::from_slice(&request[header_end..header_end + body_len])
                            .unwrap();
                    match body["method"].as_str().unwrap() {
                        "initialize" => {
                            assert!(!initialized);
                            initialized = true;
                            let result = json!({
                                "jsonrpc": "2.0", "id": body["id"],
                                "result": {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "serverInfo": {"name": "fixture", "version": "1"}}
                            })
                            .to_string();
                            format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: test-session\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{result}", result.len())
                        }
                        "notifications/initialized" => {
                            assert!(!notification);
                            notification = true;
                            "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned()
                        }
                        method => panic!("unexpected fixture method: {method}"),
                    }
                };
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.shutdown().await.unwrap();
                if is_get {
                    get_seen_tx.take().unwrap().send(()).unwrap();
                }
            }
            assert!(initialized && notification && get && deleted);
        });
        let endpoint = format!("http://{address}/mcp");
        let config = McpServerConfig {
            name: "http-reader-failure".to_owned(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: Some(endpoint),
            server_type: "http".to_owned(),
            origin: Default::default(),
        };
        let transport =
            BoundedStreamableHttpTransport::new(config.url.clone().unwrap(), None).unwrap();
        let force_reader_failure = transport.test_force_reader_failure_handle();
        let close_error = transport.close_error_handle();
        let backend = RmcpClientBackend::connect_with_transport(
            &config,
            transport,
            build_client_info(rmcp_model::ProtocolVersion::default()),
            "http",
            Arc::new(CatalogState::new()),
            Some(close_error),
            (None, None),
        )
        .await
        .expect("loopback HTTP backend connection");
        timeout(Duration::from_secs(5), get_seen_rx)
            .await
            .expect("HTTP GET observed")
            .unwrap();
        force_reader_failure.store(true, Ordering::SeqCst);

        let error = backend
            .shutdown()
            .await
            .expect_err("reader teardown failure must reach backend shutdown");
        assert_eq!(error.to_string(), "MCP HTTP reader teardown failed");
        assert!(!error.to_string().contains("127.0.0.1"));
        timeout(Duration::from_secs(5), server)
            .await
            .expect("HTTP fixture completed")
            .unwrap();
    }

    #[tokio::test]
    async fn loopback_initialize_timeout_is_bounded() {
        let (server_io, client_io) = tokio::io::duplex(16 * 1024);
        let server_task = tokio::spawn(async move {
            let mut server_io = server_io;
            let mut request = [0_u8; 4096];
            let _ = server_io.read(&mut request).await;
            loop {
                match server_io.read(&mut request).await {
                    Ok(0) => return true,
                    Ok(_) => {}
                    Err(_) => return false,
                }
            }
        });
        let config = McpServerConfig {
            name: "stalling-loopback".to_string(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: None,
            server_type: "duplex".to_string(),
            origin: Default::default(),
        };
        let result = timeout(
            Duration::from_secs(12),
            RmcpClientBackend::connect_with_transport(
                &config,
                client_io,
                build_client_info(rmcp_model::ProtocolVersion::default()),
                "duplex",
                Arc::new(CatalogState::new()),
                None,
                (None, None),
            ),
        )
        .await
        .expect("internal initialization deadline did not fire");
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("stalled initialization unexpectedly succeeded"),
        };
        assert!(error.to_string().contains("initialization timed out"));
        assert!(timeout(Duration::from_secs(1), server_task)
            .await
            .expect("timed-out client retained its transport")
            .expect("server task join"));
    }
}
