//! `session/prompt` handler — drives the Claurst query loop and forwards
//! every meaningful event back to the ACP client as a `session/update`
//! notification.

use std::sync::Arc;

use agent_client_protocol_schema as acp;
use claurst_api::streaming::{AnthropicStreamEvent, ContentDelta};
use claurst_core::types::{ContentBlock, Message, MessageContent, Role, ToolResultContent};
use claurst_query::event_channel::{bounded_events, EventDeliveryState, QueryEventReceiver};
use claurst_query::{QueryEvent, QueryOutcome};
use claurst_tools::ToolContext;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, warn};

use crate::connection::{input_charge, Connection};
use crate::permission::AcpPermissionHandler;
use crate::runtime::AgentRuntime;
use crate::sessions::SessionState;

const MAX_RETAINED_EVENTS: usize = 128;
const EVENT_BYTE_BUDGET: usize = 8 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 4 * 1024 * 1024;

/// Handle one `session/prompt` JSON-RPC call.
///
/// Drives the full Claurst query loop with the runtime's tools, MCP servers,
/// and provider registry, while streaming every text delta, thinking delta,
/// and tool invocation back as `session/update` notifications. Returns the
/// final `PromptResponse` with the appropriate `StopReason`.
pub async fn handle(
    runtime: Arc<AgentRuntime>,
    connection: Arc<Connection>,
    session: Arc<SessionState>,
    params: acp::PromptRequest,
) -> Result<acp::PromptResponse, acp::Error> {
    let disconnected = connection.disconnected();
    let _prompt_guard = tokio::select! {
        biased;
        _ = disconnected.cancelled() => return Ok(acp::PromptResponse::new(acp::StopReason::Cancelled)),
        guard = session.prompt_gate.lock() => guard,
    };

    // Convert prompt content blocks → a single user message in Claurst's
    // internal format.
    let user_text = render_prompt_blocks(&params.prompt);
    if user_text.trim().is_empty() {
        return Err(acp::Error::invalid_params());
    }

    // Append the user turn to the session transcript.
    let mut messages: Vec<Message> = {
        let guard = session.messages.lock();
        guard.clone()
    };
    messages.push(Message::user(user_text));

    // Reset the session's cancellation token for this new turn.
    let cancel = session.cancel_token_for_turn();

    // Build per-session ToolContext.
    let permission_handler: Arc<dyn claurst_core::PermissionHandler> =
        Arc::new(AcpPermissionHandler);
    let tool_ctx = ToolContext {
        working_dir: session.cwd.clone(),
        permission_mode: runtime.config.permission_mode.clone(),
        permission_handler,
        cost_tracker: runtime.cost_tracker.clone(),
        session_id: session.session_id.0.to_string(),
        file_history: session.file_history.clone(),
        current_turn: session.current_turn.clone(),
        non_interactive: false, // ACP routes permissions via the bridge
        mcp_manager: runtime.mcp_manager.clone(),
        config: runtime.config.clone(),
        managed_agent_config: runtime.config.managed_agents.clone(),
        completion_notifier: None,
        pending_permissions: Some(session.pending_permissions.clone()),
        permission_manager: Some(runtime.permission_manager.clone()),
        user_question_tx: None,
        // Bind to this turn's cancel token so the parallel tool executor and any
        // sub-agents observe cancellation (issue #218). `run_query_loop` also
        // rebinds this to the token it is driven by.
        cancel_token: cancel.clone(),
        catalog_view: None,
        trusted_checker_catalog: Default::default(),
    };

    // Each turn owns its delivery budget and actual query cancellation token.
    let (ev_tx, ev_rx, delivery) = bounded_events(
        cancel.clone(),
        MAX_RETAINED_EVENTS,
        EVENT_BYTE_BUDGET,
        MAX_EVENT_BYTES,
        event_charge,
    )
    .map_err(|_| acp::Error::internal_error())?;

    // Spawn the permission drainer for this turn.
    session
        .pending_permissions
        .lock()
        .reopen(Some(crate::permission::MAX_QUEUED_PERMISSIONS));
    let drainer_cancel = cancel.child_token();
    let drainer = crate::permission::spawn_drainer(
        connection.clone(),
        session.session_id.clone(),
        session.pending_permissions.clone(),
        drainer_cancel.clone(),
    );

    // Event channel + forwarder.
    let forwarder = tokio::spawn(forward_events(
        connection.clone(),
        session.session_id.clone(),
        ev_rx,
        delivery.clone(),
    ));

    // Run the query loop.
    let query = claurst_query::run_query_loop_with_events(
        runtime.api_client.as_ref(),
        &mut messages,
        runtime.tools.as_slice(),
        &tool_ctx,
        &runtime.query_config,
        runtime.cost_tracker.clone(),
        Some(ev_tx),
        cancel.clone(),
        None,
    );
    let outcome = run_with_disconnect(query, &disconnected, &cancel).await;

    // Tear down forwarder + drainer.
    drainer_cancel.cancel();
    let _ = drainer.await;
    // Forwarder finishes when ev_tx is dropped at end of run_query_loop.
    if !matches!(forwarder.await, Ok(Ok(()))) {
        delivery.forwarding_failed();
    }

    // Persist the updated transcript.
    {
        let mut guard = session.messages.lock();
        *guard = messages;
    }
    if session.persist().await.is_err() {
        error!("ACP: failed to persist session transcript");
        return Err(acp::Error::internal_error());
    }

    // A late forwarding failure can happen after inference completes. Preserve
    // the transcript first; never report success or replay uncertain output.
    if let Some(failure) = delivery.failure() {
        return Err(acp::Error::internal_error().data(Some(serde_json::json!({
            "kind": "event_delivery",
            "reason": failure.as_str(),
        }))));
    }

    let stop_reason = match outcome {
        QueryOutcome::EndTurn { .. } => acp::StopReason::EndTurn,
        QueryOutcome::MaxTokens { .. } => acp::StopReason::MaxTokens,
        QueryOutcome::Cancelled => acp::StopReason::Cancelled,
        QueryOutcome::BudgetExceeded { .. } => acp::StopReason::MaxTurnRequests,
        QueryOutcome::Error(_) => {
            error!("ACP: query loop errored");
            acp::StopReason::Refusal
        }
    };

    Ok(acp::PromptResponse::new(stop_reason))
}

/// Signal cancellation and await the query's cleanup instead of dropping it.
async fn run_with_disconnect<F: std::future::Future>(
    query: F,
    disconnected: &CancellationToken,
    cancel: &CancellationToken,
) -> F::Output {
    tokio::pin!(query);
    tokio::select! {
        biased;
        _ = disconnected.cancelled() => {
            cancel.cancel();
            query.await
        }
        outcome = &mut query => outcome,
    }
}

/// Replay a persisted ACP transcript before `session/load` returns.
pub(crate) async fn replay_history(
    connection: &Arc<Connection>,
    session: &Arc<SessionState>,
) -> Result<(), acp::Error> {
    let messages = session.messages.lock().clone();
    for message in &messages {
        let updates = replay_updates(message);
        for update in updates {
            let notification = acp::SessionNotification::new(session.session_id.clone(), update);
            connection
                .send_notification("session/update", notification)
                .await
                .map_err(|_| acp::Error::internal_error())?;
        }
    }
    Ok(())
}

fn replay_updates(message: &Message) -> Vec<acp::SessionUpdate> {
    match &message.content {
        MessageContent::Text(text) => vec![role_text_update(&message.role, text)],
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| replay_block(&message.role, block))
            .collect(),
    }
}

fn role_text_update(role: &Role, text: &str) -> acp::SessionUpdate {
    let chunk = acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(text)));
    match role {
        Role::User => acp::SessionUpdate::UserMessageChunk(chunk),
        Role::Assistant => acp::SessionUpdate::AgentMessageChunk(chunk),
    }
}

fn replay_block(role: &Role, block: &ContentBlock) -> Option<acp::SessionUpdate> {
    match block {
        ContentBlock::Text { text } => Some(role_text_update(role, text)),
        ContentBlock::Thinking { thinking, .. } => {
            let chunk =
                acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(thinking)));
            Some(acp::SessionUpdate::AgentThoughtChunk(chunk))
        }
        ContentBlock::RedactedThinking { .. } => None,
        ContentBlock::ToolUse {
            id, name, input, ..
        } => {
            let call = acp::ToolCall::new(
                acp::ToolCallId::new(id.as_str()),
                tool_title(name, Some(input)),
            )
            .kind(classify_tool_kind(name))
            .status(acp::ToolCallStatus::InProgress)
            .raw_input(Some(input.clone()));
            Some(acp::SessionUpdate::ToolCall(call))
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let text = replay_tool_result_text(content);
            let status = if is_error.unwrap_or(false) {
                acp::ToolCallStatus::Failed
            } else {
                acp::ToolCallStatus::Completed
            };
            let output =
                acp::Content::new(acp::ContentBlock::Text(acp::TextContent::new(text.clone())));
            let fields = acp::ToolCallUpdateFields::new()
                .status(status)
                .content(vec![acp::ToolCallContent::Content(output)])
                .raw_output(Some(serde_json::Value::String(text)));
            Some(acp::SessionUpdate::ToolCallUpdate(
                acp::ToolCallUpdate::new(acp::ToolCallId::new(tool_use_id.as_str()), fields),
            ))
        }
        ContentBlock::UserLocalCommandOutput { command, output } => {
            Some(role_text_update(role, &format!("!{command}\n{output}")))
        }
        ContentBlock::UserCommand { name, args } => {
            Some(role_text_update(role, &format!("/{name} {args}")))
        }
        ContentBlock::UserMemoryInput { key, value } => {
            Some(role_text_update(role, &format!("# {key}: {value}")))
        }
        ContentBlock::SystemAPIError { message, .. } => Some(role_text_update(role, message)),
        ContentBlock::CollapsedReadSearch {
            tool_name,
            paths,
            n_hidden,
        } => Some(role_text_update(
            role,
            &format!("{tool_name}: {} path(s), {n_hidden} hidden", paths.len()),
        )),
        ContentBlock::TaskAssignment {
            id,
            subject,
            description,
        } => Some(role_text_update(
            role,
            &format!("Task {id}: {subject}\n{description}"),
        )),
        ContentBlock::Image { .. } | ContentBlock::Document { .. } => None,
    }
}

fn replay_tool_result_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(text) => text.clone(),
        ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Concatenate text from prompt content blocks. Image / Audio / embedded
/// resources are dropped here (they require additional prompt capabilities
/// which v1 does not advertise) but are tracked for telemetry.
fn render_prompt_blocks(blocks: &[acp::ContentBlock]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in blocks {
        match block {
            acp::ContentBlock::Text(t) => parts.push(t.text.clone()),
            acp::ContentBlock::ResourceLink(link) => {
                parts.push(format!("[resource link: {}]", link.uri));
            }
            acp::ContentBlock::Resource(res) => {
                // Best-effort: emit any embedded text.
                let json = serde_json::to_value(res).unwrap_or_default();
                if let Some(text) = json
                    .get("resource")
                    .and_then(|r| r.get("text"))
                    .and_then(|t| t.as_str())
                {
                    parts.push(text.to_string());
                }
            }
            acp::ContentBlock::Image(_) | acp::ContentBlock::Audio(_) => {
                warn!("ACP: ignoring multimedia content block (capability not advertised)");
            }
            _ => {
                warn!("ACP: ignoring unknown content block variant");
            }
        }
    }
    parts.join("\n\n")
}

/// Charge retained strings, projected copies, worst-case JSON escaping and the
/// parsed tool JSON arena before admitting an event. The scanner allocates no
/// JSON tree. Only variants this adapter already ignored are filtered out.
fn event_charge(event: &QueryEvent) -> Option<usize> {
    let (capacity, json) = match event {
        QueryEvent::Stream(AnthropicStreamEvent::ContentBlockDelta {
            delta: ContentDelta::TextDelta { text },
            ..
        }) => (text.capacity(), None),
        QueryEvent::Stream(AnthropicStreamEvent::ContentBlockDelta {
            delta: ContentDelta::ThinkingDelta { thinking },
            ..
        }) => (thinking.capacity(), None),
        QueryEvent::ToolStart {
            tool_name,
            tool_id,
            input_json,
        } => (
            tool_name
                .capacity()
                .saturating_add(tool_id.capacity())
                .saturating_add(input_json.capacity()),
            Some(input_json.as_bytes()),
        ),
        QueryEvent::ToolEnd {
            tool_name,
            tool_id,
            result,
            native_operation_receipt,
            ..
        } => {
            let capacity = tool_name
                .capacity()
                .saturating_add(tool_id.capacity())
                .saturating_add(result.capacity());
            let charge = capacity.saturating_mul(8).saturating_add(1024);
            if charge > MAX_EVENT_BYTES {
                return Some(usize::MAX);
            }
            let result_charge = input_charge(result.as_bytes()).ok().unwrap_or(usize::MAX);
            let receipt_charge = native_operation_receipt
                .as_ref()
                .map(|receipt| {
                    serde_json::to_vec(receipt)
                        .ok()
                        .and_then(|bytes| input_charge(&bytes).ok())
                        .unwrap_or(usize::MAX)
                })
                .unwrap_or(0);
            return Some(
                charge
                    .saturating_add(result_charge)
                    .saturating_add(receipt_charge),
            );
        }
        QueryEvent::Error(message) => (message.capacity(), None),
        QueryEvent::McpDiscovery(_) | QueryEvent::McpInvocation { .. } => {
            return claurst_query::event_channel::diagnostic_event_charge(event);
        }
        _ => return None,
    };
    let charge = capacity.saturating_mul(8).saturating_add(1024);
    if charge > MAX_EVENT_BYTES {
        return Some(usize::MAX);
    }
    Some(match json {
        Some(bytes) => input_charge(bytes)
            .map(|arena| charge.saturating_add(arena))
            .unwrap_or(usize::MAX),
        None => charge,
    })
}

/// Pump QueryEvents → `session/update` SessionNotifications. Keep each lease
/// alive until its write finishes, including when stdout stalls.
async fn forward_events(
    connection: Arc<Connection>,
    session_id: acp::SessionId,
    mut rx: QueryEventReceiver,
    delivery: EventDeliveryState,
) -> Result<(), acp::Error> {
    while let Some(event) = rx.recv().await {
        let result = match event.event() {
            QueryEvent::Stream(AnthropicStreamEvent::ContentBlockDelta { delta, .. }) => {
                match delta {
                    ContentDelta::TextDelta { text } => {
                        send_text_chunk(&connection, &session_id, text, false).await
                    }
                    ContentDelta::ThinkingDelta { thinking } => {
                        send_text_chunk(&connection, &session_id, thinking, true).await
                    }
                    _ => Ok(()),
                }
            }
            QueryEvent::ToolStart {
                tool_name,
                tool_id,
                input_json,
            } => {
                let kind = classify_tool_kind(tool_name);
                let raw_input = serde_json::from_str::<serde_json::Value>(input_json).ok();
                let title = tool_title(tool_name, raw_input.as_ref());
                let mut tool_call =
                    acp::ToolCall::new(acp::ToolCallId::new(tool_id.as_str()), title)
                        .kind(kind)
                        .status(acp::ToolCallStatus::InProgress);
                if let Some(input) = raw_input {
                    tool_call = tool_call.raw_input(Some(input));
                }
                send_session_update(
                    &connection,
                    &session_id,
                    acp::SessionUpdate::ToolCall(tool_call),
                )
                .await
            }
            QueryEvent::ToolEnd {
                tool_name: _,
                tool_id,
                result,
                is_error,
                native_operation_receipt: _,
                execution_witness: _,
            } => {
                let status = if *is_error {
                    acp::ToolCallStatus::Failed
                } else {
                    acp::ToolCallStatus::Completed
                };
                let content = vec![acp::ToolCallContent::Content(acp::Content::new(
                    acp::ContentBlock::Text(acp::TextContent::new(result.clone())),
                ))];
                let raw_output = serde_json::from_str::<serde_json::Value>(result)
                    .ok()
                    .or_else(|| Some(serde_json::Value::String(result.clone())));
                let mut fields = acp::ToolCallUpdateFields::new()
                    .status(status)
                    .content(content);
                if let Some(out) = raw_output {
                    fields = fields.raw_output(Some(out));
                }
                let update =
                    acp::ToolCallUpdate::new(acp::ToolCallId::new(tool_id.as_str()), fields);
                send_session_update(
                    &connection,
                    &session_id,
                    acp::SessionUpdate::ToolCallUpdate(update),
                )
                .await
            }
            QueryEvent::McpDiscovery(_) | QueryEvent::McpInvocation { .. } => Ok(()),
            QueryEvent::Error(msg) => {
                send_text_chunk(
                    &connection,
                    &session_id,
                    &format!("\n[error: {}]", msg),
                    false,
                )
                .await
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            delivery.forwarding_failed();
            return Err(error);
        }
    }
    Ok(())
}

async fn send_text_chunk(
    connection: &Arc<Connection>,
    session_id: &acp::SessionId,
    text: &str,
    is_thought: bool,
) -> Result<(), acp::Error> {
    let chunk = acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(text)));
    let update = if is_thought {
        acp::SessionUpdate::AgentThoughtChunk(chunk)
    } else {
        acp::SessionUpdate::AgentMessageChunk(chunk)
    };
    send_session_update(connection, session_id, update).await
}

async fn send_session_update(
    connection: &Arc<Connection>,
    session_id: &acp::SessionId,
    update: acp::SessionUpdate,
) -> Result<(), acp::Error> {
    let notif = acp::SessionNotification::new(session_id.clone(), update);
    connection
        .send_notification("session/update", notif)
        .await
        .map_err(|_| acp::Error::internal_error())?;
    debug!("ACP: sent session/update");
    Ok(())
}

fn classify_tool_kind(tool_name: &str) -> acp::ToolKind {
    match tool_name {
        "Read" | "FileRead" => acp::ToolKind::Read,
        "Edit" | "FileEdit" | "Write" | "FileWrite" | "BatchEdit" | "ApplyPatch" => {
            acp::ToolKind::Edit
        }
        "Bash" | "Shell" | "Execute" => acp::ToolKind::Execute,
        "WebFetch" | "WebSearch" => acp::ToolKind::Fetch,
        "Glob" | "Grep" | "GlobTool" => acp::ToolKind::Search,
        "Delete" | "Rm" => acp::ToolKind::Delete,
        "Move" | "Rename" => acp::ToolKind::Move,
        "Think" | "Sequential" => acp::ToolKind::Think,
        _ => acp::ToolKind::Other,
    }
}

/// Compose a short, human-readable title for a tool call. Falls back to the
/// tool's bare name if no descriptive field is present.
fn tool_title(tool_name: &str, raw_input: Option<&serde_json::Value>) -> String {
    if let Some(input) = raw_input {
        // Prefer path-like fields for file tools.
        for key in &["file_path", "path", "filename", "url", "pattern", "command"] {
            if let Some(v) = input.get(*key).and_then(|x| x.as_str()) {
                return format!("{}: {}", tool_name, v);
            }
        }
    }
    tool_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_query::event_channel::EventDeliveryFailure;
    use tokio::io::{duplex, AsyncReadExt};

    #[tokio::test]
    async fn stalled_forwarding_retains_charge_and_failure_cancels_query() {
        let (mut reader, writer) = duplex(1);
        let connection = Connection::new(writer);
        let cancel = CancellationToken::new();
        let (tx, rx, delivery) =
            bounded_events(cancel.clone(), 1, 4096, 4096, event_charge).unwrap();
        tx.send(QueryEvent::Error("first".to_string())).unwrap();
        let task = tokio::spawn(forward_events(
            connection,
            acp::SessionId::new("stalled"),
            rx,
            delivery.clone(),
        ));
        let mut first = [0];
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            reader.read_exact(&mut first),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(tx.send(QueryEvent::Error("second".to_string())).is_err());
        assert!(cancel.is_cancelled());
        assert_eq!(delivery.failure(), Some(EventDeliveryFailure::CountLimit));
        drop(reader);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        assert_eq!(delivery.failure(), Some(EventDeliveryFailure::CountLimit));
    }

    #[tokio::test]
    async fn broken_output_after_producer_finishes_is_reported() {
        let (reader, writer) = duplex(4096);
        let connection = Connection::new(writer);
        let cancel = CancellationToken::new();
        let (tx, rx, delivery) =
            bounded_events(cancel.clone(), 8, 4096, 4096, event_charge).unwrap();
        tx.send(QueryEvent::Error("finished producer".to_string()))
            .unwrap();
        drop(tx);
        drop(reader);
        assert!(forward_events(
            connection,
            acp::SessionId::new("broken"),
            rx,
            delivery.clone()
        )
        .await
        .is_err());
        assert_eq!(
            delivery.failure(),
            Some(EventDeliveryFailure::ForwardingFailed)
        );
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn event_admission_rejects_structural_amplification_before_json_projection() {
        let input_json = format!("[{}0]", "0,".repeat(35_000));
        let (tx, _rx, delivery) = bounded_events(
            CancellationToken::new(),
            8,
            EVENT_BYTE_BUDGET,
            MAX_EVENT_BYTES,
            event_charge,
        )
        .unwrap();
        assert!(tx
            .send(QueryEvent::ToolStart {
                tool_name: "Write".to_string(),
                tool_id: "dense".to_string(),
                input_json
            })
            .is_err());
        assert_eq!(
            delivery.failure(),
            Some(EventDeliveryFailure::EventSizeLimit)
        );
        assert_eq!(
            event_charge(&QueryEvent::Status("ignored".to_string())),
            None
        );
    }

    #[tokio::test]
    async fn event_forwarding_preserves_text_and_tool_pair_order() {
        let (mut reader, writer) = duplex(16384);
        let connection = Connection::new(writer);
        let cancel = CancellationToken::new();
        let (tx, rx, delivery) = bounded_events(
            cancel.clone(),
            MAX_RETAINED_EVENTS,
            EVENT_BYTE_BUDGET,
            MAX_EVENT_BYTES,
            event_charge,
        )
        .unwrap();
        tx.send(QueryEvent::Stream(
            AnthropicStreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::TextDelta {
                    text: "hello".to_string(),
                },
            },
        ))
        .unwrap();
        tx.send(QueryEvent::ToolStart {
            tool_name: "Write".to_string(),
            tool_id: "call-1".to_string(),
            input_json: "{\"file_path\":\"fixture.txt\"}".to_string(),
        })
        .unwrap();
        tx.send(QueryEvent::ToolEnd {
            tool_name: "Write".to_string(),
            tool_id: "call-1".to_string(),
            result: "{\"mutation\":\"committed\"}".to_string(),
            is_error: false,
            native_operation_receipt: None,
            execution_witness: None,
        })
        .unwrap();
        drop(tx);
        forward_events(
            connection.clone(),
            acp::SessionId::new("fixture"),
            rx,
            delivery.clone(),
        )
        .await
        .unwrap();
        assert!(delivery.failure().is_none());
        assert!(!cancel.is_cancelled());
        drop(connection);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await.unwrap();
        let updates = std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(updates.len(), 3);
        assert_eq!(updates[0]["params"]["update"]["content"]["text"], "hello");
        assert_eq!(updates[1]["params"]["update"]["sessionUpdate"], "tool_call");
        assert_eq!(updates[1]["params"]["update"]["toolCallId"], "call-1");
        assert_eq!(updates[2]["params"]["update"]["toolCallId"], "call-1");
        assert_eq!(updates[2]["params"]["update"]["status"], "completed");
        assert_eq!(
            updates[2]["params"]["update"]["rawOutput"]["mutation"],
            "committed"
        );
    }

    #[tokio::test]
    async fn disconnect_signals_cancellation_and_awaits_cleanup() {
        let disconnected = CancellationToken::new();
        let cancel = CancellationToken::new();
        let cleaning = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let observed_cancel = cancel.clone();
        let observed_cleaning = cleaning.clone();
        let cleanup_release = release.clone();
        let connection_closed = disconnected.clone();
        let task = tokio::spawn(async move {
            let query = async move {
                observed_cancel.cancelled().await;
                observed_cleaning.notify_one();
                cleanup_release.acquire().await.unwrap().forget();
                42
            };
            run_with_disconnect(query, &connection_closed, &cancel).await
        });
        disconnected.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), cleaning.notified())
            .await
            .unwrap();
        assert!(
            !task.is_finished(),
            "query was dropped before cleanup completed"
        );
        release.add_permits(1);
        assert_eq!(task.await.unwrap(), 42);
    }

    #[test]
    fn render_prompt_blocks_joins_text_blocks_with_blank_line() {
        let blocks = vec![
            acp::ContentBlock::Text(acp::TextContent::new("first")),
            acp::ContentBlock::Text(acp::TextContent::new("second")),
        ];
        assert_eq!(render_prompt_blocks(&blocks), "first\n\nsecond");
    }

    #[test]
    fn render_prompt_blocks_renders_resource_links() {
        let blocks = vec![acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
            "notes.md",
            "file:///tmp/notes.md",
        ))];
        assert_eq!(
            render_prompt_blocks(&blocks),
            "[resource link: file:///tmp/notes.md]"
        );
    }

    #[test]
    fn render_prompt_blocks_drops_images_without_panicking() {
        let blocks = vec![
            acp::ContentBlock::Text(acp::TextContent::new("caption")),
            acp::ContentBlock::Image(acp::ImageContent::new("base64data", "image/png")),
        ];
        assert_eq!(render_prompt_blocks(&blocks), "caption");
    }

    #[test]
    fn render_prompt_blocks_empty_input_is_empty_string() {
        assert_eq!(render_prompt_blocks(&[]), "");
    }

    #[test]
    fn classify_tool_kind_maps_known_names() {
        assert_eq!(classify_tool_kind("Read"), acp::ToolKind::Read);
        assert_eq!(classify_tool_kind("Edit"), acp::ToolKind::Edit);
        assert_eq!(classify_tool_kind("Bash"), acp::ToolKind::Execute);
        assert_eq!(classify_tool_kind("WebFetch"), acp::ToolKind::Fetch);
        assert_eq!(classify_tool_kind("Grep"), acp::ToolKind::Search);
        assert_eq!(classify_tool_kind("Rm"), acp::ToolKind::Delete);
        assert_eq!(classify_tool_kind("Rename"), acp::ToolKind::Move);
        assert_eq!(classify_tool_kind("Sequential"), acp::ToolKind::Think);
    }

    #[test]
    fn classify_tool_kind_unknown_name_is_other() {
        assert_eq!(classify_tool_kind("SomeMcpTool"), acp::ToolKind::Other);
    }

    #[test]
    fn tool_title_falls_back_to_bare_name_without_input() {
        assert_eq!(tool_title("Bash", None), "Bash");
    }

    #[test]
    fn tool_title_prefers_path_like_fields_in_priority_order() {
        let input = serde_json::json!({ "path": "/a", "file_path": "/b" });
        // file_path is checked before path in the priority list.
        assert_eq!(tool_title("Edit", Some(&input)), "Edit: /b");
    }

    #[test]
    fn tool_title_uses_command_field_for_bash() {
        let input = serde_json::json!({ "command": "ls -la" });
        assert_eq!(tool_title("Bash", Some(&input)), "Bash: ls -la");
    }

    #[test]
    fn tool_title_falls_back_when_input_has_no_known_field() {
        let input = serde_json::json!({ "unrelated": 42 });
        assert_eq!(tool_title("CustomTool", Some(&input)), "CustomTool");
    }

    #[test]
    fn replay_preserves_user_agent_and_paired_tool_order() {
        let messages = [
            Message::user("inspect"),
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({"path":"notes.md"}),
                thought_signature: None,
            }]),
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: ToolResultContent::Text("contents".to_string()),
                is_error: Some(false),
            }]),
            Message::assistant("done"),
        ];
        let updates = messages
            .iter()
            .flat_map(replay_updates)
            .map(|update| serde_json::to_value(update).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(updates.len(), 4);
        assert_eq!(updates[0]["sessionUpdate"], "user_message_chunk");
        assert_eq!(updates[0]["content"]["text"], "inspect");
        assert_eq!(updates[1]["sessionUpdate"], "tool_call");
        assert_eq!(updates[1]["toolCallId"], "call-1");
        assert_eq!(updates[2]["sessionUpdate"], "tool_call_update");
        assert_eq!(updates[2]["toolCallId"], "call-1");
        assert_eq!(updates[2]["status"], "completed");
        assert_eq!(updates[3]["sessionUpdate"], "agent_message_chunk");
        assert_eq!(updates[3]["content"]["text"], "done");
    }

    #[test]
    fn replay_omits_redacted_thinking_payloads() {
        let message = Message::assistant_blocks(vec![ContentBlock::RedactedThinking {
            data: "opaque-secret".to_string(),
        }]);
        assert!(replay_updates(&message).is_empty());
    }

    #[tokio::test]
    async fn replay_writes_ordered_session_update_notifications() {
        let root = std::env::temp_dir().join(format!("forge-acp-replay-{}", uuid::Uuid::new_v4()));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let session = SessionState::new(acp::SessionId::new("acp-replay"), workspace).unwrap();
        *session.messages.lock() = vec![Message::user("one"), Message::assistant("two")];
        let (mut reader, writer) = duplex(8192);
        let connection = Connection::new(writer);

        replay_history(&connection, &session).await.unwrap();
        drop(connection);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await.unwrap();
        let messages = std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["method"], "session/update");
        assert_eq!(
            messages[0]["params"]["update"]["sessionUpdate"],
            "user_message_chunk"
        );
        assert_eq!(
            messages[1]["params"]["update"]["sessionUpdate"],
            "agent_message_chunk"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
