//! Scoped, read-only access to omitted conversation history.

use crate::context_history::{HistoryIndex, HistoryRef};
use async_trait::async_trait;
use claurst_core::types::ToolDefinition;
use claurst_tools::{PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_HISTORY_RESULT_BYTES: usize = 24 * 1024;
const MAX_PAGE_SIZE: usize = 64;
const MAX_HISTORY_BYTE_PAGE: usize = 4 * 1024;

/// A history reader bound to one immutable workspace/session scope.
///
/// The index is shared for synchronous journal reads only. The captured scope
/// is checked against every caller context before any record is read.
pub struct HistoryReadTool {
    index: Arc<parking_lot::Mutex<HistoryIndex>>,
    workspace: PathBuf,
    session: String,
}

impl HistoryReadTool {
    pub fn new(index: Arc<parking_lot::Mutex<HistoryIndex>>) -> Self {
        let (workspace, session) = {
            let index = index.lock();
            let scope = index.ledger().journal().scope();
            // A production scope is a one-way workspace hash.  Keep this
            // compatibility constructor fail-closed until its caller passes
            // the immutable canonical workspace to `new_with_workspace`.
            let workspace = if scope.workspace.starts_with("workspace-") {
                PathBuf::new()
            } else {
                PathBuf::from(&scope.workspace)
            };
            (workspace, scope.session.clone())
        };
        Self::from_parts(index, workspace, session)
    }

    /// Bind a reader to the canonical workspace that was used to open the
    /// journal.  The journal stores only `workspace_scope_id`, which cannot be
    /// converted back into a filesystem path during retrieval.
    pub fn new_with_workspace(
        index: Arc<parking_lot::Mutex<HistoryIndex>>,
        workspace: impl AsRef<Path>,
    ) -> Self {
        let session = index.lock().ledger().journal().scope().session.clone();
        let workspace = workspace
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| workspace.as_ref().to_path_buf());
        Self::from_parts(index, workspace, session)
    }

    fn from_parts(
        index: Arc<parking_lot::Mutex<HistoryIndex>>,
        workspace: PathBuf,
        session: String,
    ) -> Self {
        Self {
            index,
            workspace,
            session,
        }
    }

    fn scope_matches(&self, ctx: &ToolContext) -> bool {
        if ctx.session_id != self.session {
            return false;
        }
        same_path(
            &self.workspace,
            &claurst_tools::worktree::workspace_root(ctx),
        )
    }
}

/// Construct the scoped tool for callers building a policy-filtered tool set.
/// Callers should add it only when the configured allow-list admits the exact
/// `HistoryRead` name.
pub fn make_history_read_tool(index: Arc<parking_lot::Mutex<HistoryIndex>>) -> Box<dyn Tool> {
    Box::new(HistoryReadTool::new(index))
}

/// Construct a reader for a production hashed scope.  `workspace` must be the
/// same immutable canonical workspace supplied while opening the index.
pub fn make_history_read_tool_for_workspace(
    index: Arc<parking_lot::Mutex<HistoryIndex>>,
    workspace: impl AsRef<Path>,
) -> Box<dyn Tool> {
    Box::new(HistoryReadTool::new_with_workspace(index, workspace))
}

fn same_path(expected: &Path, actual: &Path) -> bool {
    let expected = expected
        .canonicalize()
        .unwrap_or_else(|_| expected.to_path_buf());
    let actual = actual
        .canonicalize()
        .unwrap_or_else(|_| actual.to_path_buf());
    expected == actual
}

fn bounded_json(value: Value) -> Result<String, String> {
    let encoded = serde_json::to_string(&value).map_err(|error| error.to_string())?;
    if encoded.len() > MAX_HISTORY_RESULT_BYTES {
        return Err(format!(
            "history page exceeds the {} byte result limit; request a smaller page",
            MAX_HISTORY_RESULT_BYTES
        ));
    }
    Ok(encoded)
}

fn utf8_byte_slice(value: &str, offset: usize, limit: usize) -> (&str, usize, usize) {
    let start = offset.min(value.len());
    let start = (0..=start)
        .rev()
        .find(|index| value.is_char_boundary(*index))
        .unwrap_or(0);
    let requested_end = start.saturating_add(limit).min(value.len());
    let end = (0..=requested_end)
        .rev()
        .find(|index| value.is_char_boundary(*index))
        .unwrap_or(start);
    (&value[start..end], start, end)
}

#[async_trait]
impl Tool for HistoryReadTool {
    fn name(&self) -> &str {
        "HistoryRead"
    }

    fn description(&self) -> &str {
        "Read bounded historical or superseded conversation groups by retrieval ID."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Observation
    }

    fn self_gates(&self) -> bool {
        true
    }

    fn advanced(&self) -> bool {
        true
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "ref_id": {"type": "string"},
                "query": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1024},
                "limit": {"type": "integer", "minimum": 1, "maximum": 64},
                "byte_offset": {"type": "integer", "minimum": 0},
                "byte_limit": {"type": "integer", "minimum": 1, "maximum": 4096}
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        if let Err(error) = ctx.check_permission(
            self.name(),
            "Read an omitted historical conversation group",
            true,
        ) {
            return ToolResult::error(error.to_string());
        }
        if !self.scope_matches(ctx) {
            return ToolResult::error("HistoryRead is scoped to the owning workspace and session.");
        }

        let ref_id = input
            .get("ref_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let query = input.get("query").and_then(Value::as_str).unwrap_or("");
        let offset = input
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(1024) as usize;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(16)
            .clamp(1, MAX_PAGE_SIZE as u64) as usize;
        let byte_offset = input
            .get("byte_offset")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(usize::MAX as u64) as usize;
        let byte_limit = input
            .get("byte_limit")
            .and_then(Value::as_u64)
            .unwrap_or(MAX_HISTORY_BYTE_PAGE as u64)
            .clamp(1, MAX_HISTORY_BYTE_PAGE as u64) as usize;

        let index = self.index.lock();
        if let Some(ref_id) = ref_id {
            let group = match index.retrieve_group(ref_id) {
                Ok(group) => group,
                Err(error) => return ToolResult::error(error.to_string()),
            };
            let query_lower = query.trim().to_ascii_lowercase();
            let matching: Vec<&claurst_core::types::Message> = group
                .messages
                .iter()
                .filter(|message| {
                    query_lower.is_empty()
                        || serde_json::to_string(message)
                            .map(|encoded| encoded.to_ascii_lowercase().contains(&query_lower))
                            .unwrap_or(false)
                })
                .collect();
            let total_messages = matching.len();

            let start = offset.min(total_messages);
            let page_end = start.saturating_add(limit).min(total_messages);

            // A single retained message can exceed the tool result limit even
            // when the group itself is valid. Page its serialized UTF-8 bytes
            // explicitly rather than shrinking the page to zero messages.
            // Only inspect messages at or after the requested message offset;
            // an earlier oversized message must never hijack a later page.
            let oversized_index = (start..page_end).find(|index| {
                matching
                    .get(*index)
                    .and_then(|message| serde_json::to_string(message).ok())
                    .is_some_and(|encoded| encoded.len() > MAX_HISTORY_RESULT_BYTES)
            });
            if oversized_index == Some(start) {
                let message_index = start;
                let message = matching[message_index];
                let encoded = match serde_json::to_string(message) {
                    Ok(encoded) => encoded,
                    Err(error) => return ToolResult::error(error.to_string()),
                };
                let search_offset = if query_lower.is_empty() {
                    0
                } else {
                    encoded.to_ascii_lowercase().find(&query_lower).unwrap_or(0)
                };
                let requested_offset = input
                    .get("byte_offset")
                    .and_then(Value::as_u64)
                    .map(|_| byte_offset)
                    .unwrap_or(search_offset);
                let mut page_limit = byte_limit;
                loop {
                    let (chunk, start, end) =
                        utf8_byte_slice(&encoded, requested_offset, page_limit);
                    let has_more_bytes = end < encoded.len();
                    let has_more_messages = message_index + 1 < total_messages;
                    let output = json!({
                        "retrieval_status": "historical_bytes",
                        "authority": group.history_ref.authority,
                        "ref": group.history_ref,
                        "query": query,
                        "message_id": message.uuid,
                        "message_index": message_index,
                        "byte_offset": start,
                        "byte_limit": end.saturating_sub(start),
                        "total_bytes": encoded.len(),
                        "next_message_offset": if has_more_bytes { message_index } else { message_index + 1 },
                        "next_byte_offset": if has_more_bytes { end } else { 0 },
                        "has_more": has_more_bytes || has_more_messages,
                        "chunk": chunk,
                        "reactivates_requirements": false,
                    });
                    match bounded_json(output) {
                        Ok(output) => return ToolResult::success(output),
                        Err(_error) if page_limit > 256 => page_limit /= 2,
                        Err(error) => return ToolResult::error(error),
                    }
                }
            }
            let mut page: Vec<&claurst_core::types::Message> = matching
                .iter()
                .skip(start)
                .take(page_end.saturating_sub(start))
                .copied()
                .collect();
            if let Some(oversized_index) = oversized_index {
                page.truncate(oversized_index.saturating_sub(start));
            }
            loop {
                let next_message_offset = start.saturating_add(page.len());
                let has_more = next_message_offset < total_messages;
                let output = json!({
                    "retrieval_status": "historical",
                    "authority": group.history_ref.authority,
                    "ref": group.history_ref,
                    "query": query,
                    "offset": start,
                    "limit": limit,
                    "returned": page.len(),
                    "total_messages": total_messages,
                    "next_message_offset": next_message_offset,
                    "next_byte_offset": 0,
                    "has_more": has_more,
                    "messages": page,
                    "reactivates_requirements": false,
                });
                match bounded_json(output) {
                    Ok(output) => return ToolResult::success(output),
                    Err(_error) if page.len() > 1 => {
                        page.pop();
                    }
                    Err(error) => return ToolResult::error(error),
                }
            }
        }

        let mut refs: Vec<HistoryRef> = index.search_omitted_refs(query, offset, limit);
        loop {
            let output = json!({
                "retrieval_status": "index",
                "query": query,
                "offset": offset,
                "limit": limit,
                "refs": refs,
                "reactivates_requirements": false,
            });
            match bounded_json(output) {
                Ok(output) => return ToolResult::success(output),
                Err(_error) if refs.len() > 1 => {
                    refs.pop();
                }
                Err(error) => return ToolResult::error(error),
            }
        }
    }

    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            input_schema: self.input_schema(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
    use claurst_tools::ToolContext;
    use std::sync::Arc;

    struct AllowAll;

    impl PermissionHandler for AllowAll {
        fn check_permission(&self, _: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }

        fn request_permission(&self, _: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    fn context(workspace: &Path) -> ToolContext {
        ToolContext {
            working_dir: workspace.to_path_buf(),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(AllowAll),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "history-read-session".to_owned(),
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
            catalog_view: None,
            trusted_checker_catalog: Arc::new(claurst_tools::TrustedCheckerCatalog::empty()),
        }
    }

    #[test]
    fn hashed_production_scope_retrieval_uses_captured_workspace() {
        let config = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let scope = crate::workspace_scope_id(workspace.path());
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(config.path(), &scope, "history-read-session").unwrap(),
        ));
        let reader = HistoryReadTool::new_with_workspace(index, workspace.path());

        assert_eq!(reader.session, "history-read-session");
        assert!(reader.scope_matches(&context(workspace.path())));
        assert!(!same_path(Path::new(&scope), workspace.path()));
    }

    #[test]
    fn hashed_scope_compatibility_constructor_fails_closed() {
        let config = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let scope = crate::workspace_scope_id(workspace.path());
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(config.path(), &scope, "history-read-session").unwrap(),
        ));
        let reader = HistoryReadTool::new(index);

        assert!(!reader.scope_matches(&context(workspace.path())));
    }

    #[tokio::test]
    async fn historical_ref_supports_bounded_unicode_pages() {
        let config = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let scope = crate::workspace_scope_id(workspace.path());
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(config.path(), &scope, "history-read-session").unwrap(),
        ));
        let mut messages = Vec::new();
        for (index, text) in ["old-canary 雪事实", "middle", "tail"]
            .into_iter()
            .enumerate()
        {
            let mut message = claurst_core::types::Message::user(text);
            message.uuid = Some(format!("history-{index}"));
            messages.push(message);
        }
        let reference = index
            .lock()
            .persist_omitted_group(
                "omitted-middle-unicode",
                1,
                messages,
                crate::context_history::OmitReason::Budget,
            )
            .unwrap();
        let reader = HistoryReadTool::new_with_workspace(index, workspace.path());
        let result = reader
            .execute(
                json!({
                    "ref_id": reference.id,
                    "query": "雪事实",
                    "offset": 0,
                    "limit": 1,
                }),
                &context(workspace.path()),
            )
            .await;
        assert!(!result.is_error);
        let value: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(value["returned"], 1);
        assert_eq!(value["total_messages"], 1);
        assert!(value["messages"][0].to_string().contains("雪事实"));
    }

    #[tokio::test]
    async fn historical_ref_pages_a_large_message_by_utf8_bytes() {
        let config = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let scope = crate::workspace_scope_id(workspace.path());
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(config.path(), &scope, "history-read-session").unwrap(),
        ));
        let mut message = claurst_core::types::Message::user(format!(
            "{} CANARY-TAIL 雪",
            "prefix-".repeat(6_000)
        ));
        message.uuid = Some("large-history-message".into());
        let reference = index
            .lock()
            .persist_omitted_group(
                "large-omitted-message",
                1,
                vec![message],
                crate::context_history::OmitReason::Budget,
            )
            .unwrap();
        let reader = HistoryReadTool::new_with_workspace(index, workspace.path());
        let head = reader
            .execute(
                json!({
                    "ref_id": reference.id,
                    "byte_offset": 0,
                    "byte_limit": 2048,
                }),
                &context(workspace.path()),
            )
            .await;
        assert!(
            !head.is_error,
            "large history head failed: {}",
            head.content
        );
        let head_value: Value = serde_json::from_str(&head.content).unwrap();
        assert_eq!(head_value["retrieval_status"], "historical_bytes");
        assert_eq!(head_value["byte_offset"], 0);
        assert!(head_value["has_more"].as_bool().unwrap());
        let result = reader
            .execute(
                json!({
                    "ref_id": reference.id,
                    "query": "CANARY-TAIL",
                    "byte_limit": 2048,
                }),
                &context(workspace.path()),
            )
            .await;
        assert!(
            !result.is_error,
            "large history read failed: {}",
            result.content
        );
        let value: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(value["retrieval_status"], "historical_bytes");
        assert!(value["byte_offset"].as_u64().unwrap() > 20_000);
        assert!(value["chunk"].as_str().unwrap().contains("CANARY-TAIL"));
        assert!(!value["has_more"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn historical_ref_pages_mixed_members_from_requested_offsets() {
        let config = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let scope = crate::workspace_scope_id(workspace.path());
        let index = Arc::new(parking_lot::Mutex::new(
            HistoryIndex::open(config.path(), &scope, "history-read-session").unwrap(),
        ));
        let mut small = claurst_core::types::Message::user("small-member");
        small.uuid = Some("small-member-id".into());
        let mut first_large = claurst_core::types::Message::user(format!(
            "FIRST-LARGE-HEAD {} FIRST-LARGE-TAIL",
            "first-segment-".repeat(3_000)
        ));
        first_large.uuid = Some("first-large-id".into());
        let mut second_large = claurst_core::types::Message::user(format!(
            "SECOND-LARGE-HEAD {} SECOND-LARGE-TAIL",
            "second-segment-".repeat(3_000)
        ));
        second_large.uuid = Some("second-large-id".into());
        let reference = index
            .lock()
            .persist_omitted_group(
                "mixed-large-members",
                1,
                vec![small, first_large, second_large],
                crate::context_history::OmitReason::Budget,
            )
            .unwrap();
        let reader = HistoryReadTool::new_with_workspace(index, workspace.path());
        let mut offset = 0usize;
        let mut byte_offset = 0usize;
        let mut serialized = [String::new(), String::new(), String::new()];

        loop {
            let result = reader
                .execute(
                    json!({
                        "ref_id": reference.id,
                        "offset": offset,
                        "limit": 64,
                        "byte_offset": byte_offset,
                        "byte_limit": 2048,
                    }),
                    &context(workspace.path()),
                )
                .await;
            assert!(
                !result.is_error,
                "mixed history read failed: {}",
                result.content
            );
            let value: Value = serde_json::from_str(&result.content).unwrap();
            if value["retrieval_status"] == "historical" {
                assert_eq!(value["returned"], 1);
                assert_eq!(value["messages"][0]["uuid"], "small-member-id");
                serialized[0] = value["messages"][0].to_string();
            } else {
                assert_eq!(value["retrieval_status"], "historical_bytes");
                let message_index = value["message_index"].as_u64().unwrap() as usize;
                assert!(message_index > 0 && message_index < serialized.len());
                serialized[message_index].push_str(value["chunk"].as_str().unwrap());
                let expected_id = if message_index == 1 {
                    "first-large-id"
                } else {
                    "second-large-id"
                };
                assert_eq!(value["message_id"], expected_id);
            }

            offset = value["next_message_offset"].as_u64().unwrap() as usize;
            byte_offset = value["next_byte_offset"].as_u64().unwrap() as usize;
            if !value["has_more"].as_bool().unwrap() {
                break;
            }
        }

        assert!(serialized[0].contains("small-member-id"));
        assert!(serialized[1].contains("FIRST-LARGE-TAIL"));
        assert!(serialized[2].contains("SECOND-LARGE-TAIL"));
        assert_eq!(offset, 3);
        assert_eq!(byte_offset, 0);
    }
}
