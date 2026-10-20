//! Agent Client Protocol (ACP) server for Forge.
//!
//! ACP is the open protocol pioneered by Zed for standardizing communication
//! between AI coding agents and editors (Zed, Neovim, JetBrains, VS Code, …).
//! Spec: <https://agentclientprotocol.com>
//!
//! This crate turns the local `forge` binary into a compliant ACP agent
//! over newline-delimited JSON-RPC 2.0 on stdio. Editors launch `forge acp`
//! as a subprocess and drive it through the protocol's standard methods:
//!
//! | Method                       | Direction  | Notes                                       |
//! |------------------------------|------------|---------------------------------------------|
//! | `initialize`                 | C → A      | Capability negotiation                      |
//! | `authenticate`               | C → A      | No-op (Claurst uses local credentials)      |
//! | `session/new`                | C → A      | Create a session with cwd + MCP roster      |
//! | `session/prompt`             | C → A      | Run a turn; streams `session/update` events |
//! | `session/cancel`             | C → A (no resp) | Cancel an in-flight prompt             |
//! | `session/update`             | A → C (no resp) | Streamed text/tool deltas              |
//! | `session/request_permission` | A → C      | Tool approval dialog                        |
//!
//! Client stdio MCP rosters are validated and added to a private session view
//! of configured connections. Runtime credentials are not saved in transcripts.

mod connection;
mod permission;
mod prompt;
mod runtime;
mod server;
mod session_mcp;
mod sessions;

use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{error, info, warn};

use connection::{Inbound, QueuedInbound, MAX_INBOUND_QUEUE};

const MAX_REQUEST_HANDLERS: usize = 8;

#[cfg(test)]
mod bounded_transport_tests;

pub use connection::Connection;
pub use runtime::AgentRuntime;
pub use server::AgentServer;

/// Run the ACP server on the current process' stdin / stdout. Returns when
/// stdin reaches EOF or when the runtime fails to initialize.
pub async fn run_acp_server() -> anyhow::Result<()> {
    // We must NEVER write to stdout outside the protocol — every byte on
    // stdout is parsed by the client as JSON-RPC. Force logs to stderr.
    install_stderr_tracing();

    let working_dir = std::env::current_dir()?;
    info!(version = env!("CARGO_PKG_VERSION"), "ACP: starting server");

    let runtime = AgentRuntime::build(working_dir).await?;
    let runtime = Arc::new(runtime);
    let connection = Connection::new(tokio::io::stdout());
    let server = AgentServer::new(connection.clone(), runtime);

    let (tx, rx) = mpsc::channel(MAX_INBOUND_QUEUE);
    let reader_fut = connection::run_reader(connection.clone(), tokio::io::stdin(), tx);
    let dispatch_server = server.clone();
    let dispatch_fut = dispatch_loop(connection, rx, move |message| {
        let server = dispatch_server.clone();
        async move { server.dispatch(message).await }
    });

    let (reader_res, _) = tokio::join!(reader_fut, dispatch_fut);
    if reader_res.is_err() {
        error!("ACP: reader loop failed");
    }
    let shutdown_res = server.shutdown().await;
    reader_res?;
    shutdown_res?;

    info!("ACP: server shutdown");
    Ok(())
}

async fn dispatch_loop<F, Fut>(
    connection: Arc<Connection>,
    mut rx: mpsc::Receiver<QueuedInbound>,
    mut dispatch: F,
) where
    F: FnMut(QueuedInbound) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let mut tasks = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            biased;
            Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                if result.is_err() { warn!("ACP: request handler failed"); }
            }
            message = rx.recv() => {
                let Some(message) = message else { break };
                if let Inbound::Request { id, .. } = &message.message {
                    if tasks.len() >= MAX_REQUEST_HANDLERS {
                        let error = agent_client_protocol_schema::Error::internal_error().data(Some(
                            serde_json::json!({"reason":"ACP request handler limit reached"})
                        ));
                        if connection.send_error_response(id.clone(), error).await.is_err() {
                            warn!("ACP: failed to report request saturation");
                        }
                    } else {
                        tasks.spawn(dispatch(message));
                    }
                } else {
                    // Notifications only update local cancellation state. They
                    // never wait for a request-handler slot or start a new task.
                    dispatch(message).await;
                }
            }
        }
    }
    while let Some(result) = tasks.join_next().await {
        if result.is_err() {
            warn!("ACP: request handler failed during shutdown");
        }
    }
}

fn install_stderr_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_env("CLAURST_ACP_LOG").unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();
}
