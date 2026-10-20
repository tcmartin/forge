//! Top-level ACP request / notification dispatcher.

use std::sync::Arc;

use agent_client_protocol_schema as acp;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::connection::{Connection, Inbound, QueuedInbound};
use crate::runtime::AgentRuntime;
use crate::sessions::{SessionRegistry, SessionState};

/// The ACP agent: owns the connection, the runtime, and the session registry.
pub struct AgentServer {
    pub connection: Arc<Connection>,
    pub runtime: Arc<AgentRuntime>,
    pub sessions: Arc<SessionRegistry>,
    pub client_capabilities: parking_lot::RwLock<acp::ClientCapabilities>,
    session_mcp_capacity: Arc<tokio::sync::Semaphore>,
}

/// A connected roster still belongs to setup until persistence/replay and
/// registry activation finish. Refresh tasks must not extend that ownership.
struct SessionSetupCleanup {
    runtime: Option<Arc<AgentRuntime>>,
}

impl Drop for SessionSetupCleanup {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            if let Ok(executor) = tokio::runtime::Handle::try_current() {
                executor.spawn(async move {
                    let _ = runtime.shutdown_session_mcp().await;
                });
            }
        }
    }
}

impl AgentServer {
    pub fn new(connection: Arc<Connection>, runtime: Arc<AgentRuntime>) -> Arc<Self> {
        Arc::new(Self {
            connection,
            runtime,
            sessions: Arc::new(SessionRegistry::new()),
            client_capabilities: parking_lot::RwLock::new(acp::ClientCapabilities::default()),
            session_mcp_capacity: Arc::new(tokio::sync::Semaphore::new(
                crate::session_mcp::MAX_SESSION_MCP_CONNECTIONS as usize,
            )),
        })
    }

    /// Dispatch under the server loop's task cap, retaining its input charge
    /// until the handler and response finish.
    pub(crate) async fn dispatch(self: &Arc<Self>, queued: QueuedInbound) {
        let QueuedInbound { message, _budget } = queued;
        match message {
            Inbound::Request { id, method, params } => {
                let response = self.handle_request(&method, params).await;
                let result = match response {
                    Ok(value) => self.connection.send_response(id, value).await,
                    Err(err) => self.connection.send_error_response(id, err).await,
                };
                if result.is_err() {
                    warn!("ACP: failed to send response");
                }
            }
            Inbound::Notification { method, params } => {
                self.handle_notification(&method, params).await;
            }
        }
    }

    async fn handle_request(
        self: &Arc<Self>,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, acp::Error> {
        debug!("ACP: dispatch request");
        match method {
            "initialize" => {
                let req: acp::InitializeRequest = parse_params(params)?;
                let result = self.on_initialize(req).await?;
                serde_json::to_value(result).map_err(|_| acp::Error::internal_error())
            }
            "authenticate" => {
                let _req: acp::AuthenticateRequest = parse_params(params)?;
                // Claurst uses local credentials; clients don't need to authenticate.
                serde_json::to_value(acp::AuthenticateResponse::default())
                    .map_err(|_| acp::Error::internal_error())
            }
            "session/new" => {
                let req: acp::NewSessionRequest = parse_params(params)?;
                let result = self.on_new_session(req).await?;
                serde_json::to_value(result).map_err(|_| acp::Error::internal_error())
            }
            "session/load" => {
                let req: acp::LoadSessionRequest = parse_params(params)?;
                let result = self.on_load_session(req).await?;
                serde_json::to_value(result).map_err(|_| acp::Error::internal_error())
            }
            "session/prompt" => {
                let req: acp::PromptRequest = parse_params(params)?;
                let result = self.on_prompt(req).await?;
                serde_json::to_value(result).map_err(|_| acp::Error::internal_error())
            }
            _ => {
                warn!("ACP: method not found");
                Err(acp::Error::method_not_found())
            }
        }
    }

    async fn handle_notification(self: &Arc<Self>, method: &str, params: Option<Value>) {
        debug!("ACP: dispatch notification");
        match method {
            "session/cancel" => {
                let parsed: Result<acp::CancelNotification, _> = params
                    .map(serde_json::from_value)
                    .unwrap_or(Err(serde::de::Error::custom("missing params")));
                match parsed {
                    Ok(notif) => {
                        if let Some(session) = self.sessions.get(&notif.session_id) {
                            info!("ACP: cancelling session");
                            session.cancel_current_turn();
                        }
                    }
                    Err(_) => warn!("ACP: malformed session/cancel notification"),
                }
            }
            _ => {
                warn!("ACP: ignoring unknown notification");
            }
        }
    }

    async fn on_initialize(
        self: &Arc<Self>,
        req: acp::InitializeRequest,
    ) -> Result<acp::InitializeResponse, acp::Error> {
        info!("ACP: initialize");
        *self.client_capabilities.write() = req.client_capabilities.clone();

        let agent_info = acp::Implementation::new("forge", env!("CARGO_PKG_VERSION"))
            .title(Some("Forge".to_string()));

        let mut response = acp::InitializeResponse::new(acp::ProtocolVersion::V1)
            .agent_capabilities(agent_capabilities());
        response = response.agent_info(Some(agent_info));
        Ok(response)
    }

    async fn on_new_session(
        self: &Arc<Self>,
        req: acp::NewSessionRequest,
    ) -> Result<acp::NewSessionResponse, acp::Error> {
        let servers =
            crate::session_mcp::validate(&req.mcp_servers, &self.runtime.config.mcp_servers)?;
        if !req.cwd.is_absolute() {
            return Err(acp::Error::invalid_params().data(Some(
                serde_json::json!({ "reason": "cwd must be absolute" }),
            )));
        }
        let session_id = acp::SessionId::new(format!("acp-{}", uuid::Uuid::new_v4()));
        let state = SessionState::new(session_id.clone(), req.cwd.clone()).map_err(|_| {
            acp::Error::invalid_params()
                .data(Some(serde_json::json!({ "reason": "cwd is unavailable" })))
        })?;
        info!("ACP: new session");

        let runtime = self.session_runtime(state.cwd.clone(), &servers).await?;
        *state.runtime.lock() = Some(Arc::clone(&runtime));
        let mut cleanup = SessionSetupCleanup {
            runtime: Some(runtime),
        };

        state
            .persist()
            .await
            .map_err(|_| acp::Error::internal_error())?;
        self.sessions.insert(state);
        cleanup.runtime = None;
        Ok(acp::NewSessionResponse::new(session_id))
    }

    async fn on_load_session(
        self: &Arc<Self>,
        req: acp::LoadSessionRequest,
    ) -> Result<acp::LoadSessionResponse, acp::Error> {
        let servers =
            crate::session_mcp::validate(&req.mcp_servers, &self.runtime.config.mcp_servers)?;
        if !req.cwd.is_absolute() {
            return Err(acp::Error::invalid_params().data(Some(
                serde_json::json!({ "reason": "cwd must be absolute" }),
            )));
        }
        let reservation = self.sessions.reserve_load(&req.session_id).ok_or_else(|| {
            acp::Error::invalid_params().data(Some(
                serde_json::json!({ "reason": "session is already active or loading" }),
            ))
        })?;
        let state = SessionState::load(req.session_id.clone(), req.cwd)
            .await
            .map_err(|_| {
                acp::Error::invalid_params().data(Some(serde_json::json!({
                    "reason": "session could not be loaded for this workspace"
                })))
            })?;
        let runtime = self.session_runtime(state.cwd.clone(), &servers).await?;
        *state.runtime.lock() = Some(Arc::clone(&runtime));
        let mut cleanup = SessionSetupCleanup {
            runtime: Some(runtime),
        };
        crate::prompt::replay_history(&self.connection, &state).await?;
        reservation
            .activate(state)
            .map_err(|_| acp::Error::internal_error())?;
        cleanup.runtime = None;
        info!("ACP: loaded session");
        Ok(acp::LoadSessionResponse::new())
    }

    async fn on_prompt(
        self: &Arc<Self>,
        req: acp::PromptRequest,
    ) -> Result<acp::PromptResponse, acp::Error> {
        let session = match self.sessions.get(&req.session_id) {
            Some(s) => s,
            None => {
                return Err(acp::Error::invalid_params().data(Some(serde_json::json!({
                    "reason": "unknown session",
                    "sessionId": req.session_id,
                }))));
            }
        };
        let runtime = session
            .runtime
            .lock()
            .clone()
            .unwrap_or_else(|| self.runtime.clone());
        crate::prompt::handle(runtime, self.connection.clone(), session, req).await
    }

    async fn session_runtime(
        &self,
        cwd: std::path::PathBuf,
        servers: &[claurst_core::config::McpServerConfig],
    ) -> Result<Arc<AgentRuntime>, acp::Error> {
        let admission = if servers.is_empty() {
            None
        } else {
            Some(Arc::new(
                self.session_mcp_capacity
                    .clone()
                    .try_acquire_many_owned(servers.len() as u32)
                    .map_err(|_| {
                        crate::session_mcp::unavailable("connection_capacity_exhausted")
                    })?,
            ))
        };
        let disconnected = self.connection.disconnected();
        tokio::select! {
            biased;
            // Empty rosters only clone local state. Finish accepted metadata
            // requests after input EOF; only child setup needs cancellation.
            _ = disconnected.cancelled(), if !servers.is_empty() => Err(crate::session_mcp::unavailable("disconnected")),
            result = tokio::time::timeout(std::time::Duration::from_secs(10), self.runtime.for_session(cwd, servers, admission)) => {
                match result {
                    Ok(Ok(runtime)) => Ok(runtime),
                    Ok(Err(_)) => Err(crate::session_mcp::unavailable("initialization_failed")),
                    Err(_) => Err(crate::session_mcp::unavailable("initialization_timeout")),
                }
            }
        }
    }

    pub(crate) async fn shutdown(&self) -> anyhow::Result<()> {
        let shutdown = async {
            let runtimes = self.sessions.runtimes();
            let sessions = futures::future::join_all(
                runtimes
                    .iter()
                    .map(|runtime| runtime.shutdown_session_mcp()),
            );
            let global = async {
                if let Some(manager) = self.runtime.mcp_manager.as_ref() {
                    manager.shutdown_owned().await?;
                }
                Ok::<_, anyhow::Error>(())
            };
            let (sessions, global) = tokio::join!(sessions, global);
            // Permits live in the actual transports until their children are
            // reaped, including children from a cancelled setup future.
            let _all = self
                .session_mcp_capacity
                .clone()
                .acquire_many_owned(crate::session_mcp::MAX_SESSION_MCP_CONNECTIONS)
                .await?;
            if global.is_err() || sessions.iter().any(Result::is_err) {
                anyhow::bail!("ACP MCP transport shutdown failed");
            }
            Ok::<_, anyhow::Error>(())
        };
        tokio::time::timeout(std::time::Duration::from_secs(3), shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("ACP MCP shutdown deadline exceeded"))?
    }
}

fn agent_capabilities() -> acp::AgentCapabilities {
    acp::AgentCapabilities::new()
        .load_session(true)
        .prompt_capabilities(acp::PromptCapabilities::new())
        .mcp_capabilities(acp::McpCapabilities::new())
}

fn parse_params<T: serde::de::DeserializeOwned>(params: Option<Value>) -> Result<T, acp::Error> {
    let value = params.ok_or_else(acp::Error::invalid_params)?;
    serde_json::from_value(value).map_err(|e| {
        acp::Error::invalid_params().data(Some(
            serde_json::json!({ "deserialize_error": e.to_string() }),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Params {
        name: String,
    }

    #[test]
    fn parse_params_missing_returns_invalid_params_error() {
        let result: Result<Params, acp::Error> = parse_params(None);
        let err = result.expect_err("None params should be rejected");
        assert_eq!(err.code, acp::ErrorCode::InvalidParams);
    }

    #[test]
    fn parse_params_deserializes_matching_shape() {
        let value = serde_json::json!({ "name": "claurst" });
        let result: Params = parse_params(Some(value)).expect("valid params should parse");
        assert_eq!(
            result,
            Params {
                name: "claurst".to_string()
            }
        );
    }

    #[test]
    fn parse_params_mismatched_shape_returns_invalid_params_error_with_detail() {
        let value = serde_json::json!({ "wrong_field": 1 });
        let result: Result<Params, acp::Error> = parse_params(Some(value));
        let err = result.expect_err("mismatched shape should be rejected");
        assert_eq!(err.code, acp::ErrorCode::InvalidParams);
        assert!(err.data.is_some());
    }

    #[test]
    fn capabilities_advertise_durable_session_loading() {
        assert!(agent_capabilities().load_session);
    }
}
