// monitor_tool.rs — Monitor background tasks
//
// Provides a "monitor" tool that lets the agent inspect background tasks
// started via BashTool with run_in_background=true.  Supports listing all
// tasks, checking the status or output of a specific task, and cancelling a
// running task.

use crate::{agent_tasks, background_processes, PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

pub struct MonitorTool;

#[derive(Deserialize)]
struct MonitorInput {
    #[serde(default)]
    action: MonitorAction,
    #[serde(default)]
    task_id: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum MonitorAction {
    #[default]
    List,
    Status,
    Output,
    Cancel,
}

#[async_trait]
impl Tool for MonitorTool {
    fn name(&self) -> &str {
        "monitor"
    }

    fn description(&self) -> &str {
        "Monitor background tasks started with run_in_background=true. \
        List all tasks, check status, retrieve output, or cancel a running task."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "status", "output", "cancel"],
                    "description": "Action to perform. 'list' shows all tasks, 'status'/'output' inspect a specific task, 'cancel' terminates a running task.",
                    "default": "list"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID to inspect or cancel. Required for status, output, cancel actions."
                }
            }
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let parsed: MonitorInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        match parsed.action {
            MonitorAction::List => {
                let mut tasks = background_processes::list(ctx).await;
                tasks.extend(agent_tasks::list(ctx).await);
                if tasks.is_empty() {
                    return ToolResult::success("No background tasks.");
                }
                let mut lines = vec!["Background tasks:".to_string()];
                for t in &tasks {
                    lines.push(format!(
                        "  {} [{}] {}",
                        t["id"].as_str().unwrap_or("unknown"),
                        t["status"].as_str().unwrap_or("unknown"),
                        t["description"].as_str().unwrap_or("unknown")
                    ));
                }
                ToolResult::success(lines.join("\n"))
            }

            MonitorAction::Status => {
                let id = match parsed.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for status action"),
                };
                let status = if agent_tasks::contains(&id) {
                    agent_tasks::status(&id, ctx).await
                } else {
                    background_processes::status(&id, ctx).await
                };
                match status {
                    Ok(task) => {
                        ToolResult::success(serde_json::to_string_pretty(&task).unwrap_or_default())
                    }
                    Err(error) => ToolResult::error(error),
                }
            }

            MonitorAction::Output => {
                let id = match parsed.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for output action"),
                };
                let output = if agent_tasks::contains(&id) {
                    agent_tasks::output(&id, ctx, true).await
                } else {
                    background_processes::output(&id, ctx, true).await
                };
                match output {
                    Ok(task) => ToolResult::success(
                        task["task"]["output"]
                            .as_str()
                            .filter(|output| !output.is_empty())
                            .unwrap_or("(no output yet)"),
                    ),
                    Err(error) => ToolResult::error(error),
                }
            }

            MonitorAction::Cancel => {
                let id = match parsed.task_id {
                    Some(id) => id,
                    None => return ToolResult::error("task_id required for cancel action"),
                };
                let stop = if agent_tasks::contains(&id) {
                    agent_tasks::stop(&id, ctx).await
                } else {
                    background_processes::stop(&id, ctx)
                };
                match stop {
                    Ok(()) => ToolResult::success(format!("Task {} cancellation requested.", id)),
                    Err(error) => ToolResult::error(error),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_tool_name() {
        assert_eq!(MonitorTool.name(), "monitor");
    }

    #[test]
    fn monitor_requires_execute_permission() {
        assert_eq!(MonitorTool.permission_level(), PermissionLevel::Execute);
    }

    #[test]
    fn monitor_schema_has_action_and_task_id() {
        let schema = MonitorTool.input_schema();
        let props = &schema["properties"];
        assert!(
            props["action"].is_object(),
            "schema should have 'action' property"
        );
        assert!(
            props["task_id"].is_object(),
            "schema should have 'task_id' property"
        );
    }

    #[test]
    fn monitor_schema_is_object() {
        let schema = MonitorTool.input_schema();
        assert!(schema.is_object());
        assert!(schema.get("properties").is_some());
    }

    #[tokio::test]
    async fn monitor_list_empty() {
        // The global registry is shared across tests, so we just verify the
        // tool runs without panicking and returns a success result.
        let tool = MonitorTool;
        let input = json!({"action": "list"});
        let ctx = make_test_ctx();
        let result = tool.execute(input, &ctx).await;
        // Either "No background tasks." or a list — both are successes.
        assert!(
            !result.is_error,
            "list action should not return an error: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn monitor_status_missing_task_id() {
        let tool = MonitorTool;
        let input = json!({"action": "status"});
        let ctx = make_test_ctx();
        let result = tool.execute(input, &ctx).await;
        assert!(result.is_error);
        assert!(result.content.contains("task_id required"));
    }

    #[tokio::test]
    async fn monitor_output_unknown_task() {
        let tool = MonitorTool;
        let input = json!({"action": "output", "task_id": "nonexistent-uuid-1234"});
        let ctx = make_test_ctx();
        let result = tool.execute(input, &ctx).await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn monitor_cancel_unknown_task() {
        let tool = MonitorTool;
        let input = json!({"action": "cancel", "task_id": "nonexistent-uuid-5678"});
        let ctx = make_test_ctx();
        let result = tool.execute(input, &ctx).await;
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn monitor_scopes_list_output_and_cancel_to_owner() {
        let workspace = tempfile::tempdir().unwrap();
        let mut owner = make_test_ctx();
        owner.working_dir = workspace.path().to_path_buf();
        owner.session_id = "monitor-owner".to_string();
        let mut foreign = make_test_ctx();
        foreign.working_dir = workspace.path().to_path_buf();
        foreign.session_id = "monitor-foreign".to_string();

        let id = crate::background_processes::spawn(
            "sleep 10".to_string(),
            workspace.path().to_path_buf(),
            30_000,
            &owner,
        )
        .expect("owner can register a background process");
        let tool = MonitorTool;

        let listed = tool.execute(json!({"action": "list"}), &foreign).await;
        assert!(!listed.is_error);
        assert!(!listed.content.contains(&id));

        let output = tool
            .execute(json!({"action": "output", "task_id": id}), &foreign)
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("not available"));

        let cancel = tool
            .execute(json!({"action": "cancel", "task_id": id}), &foreign)
            .await;
        assert!(cancel.is_error);
        assert!(cancel.content.contains("not available"));

        crate::background_processes::stop(&id, &owner).unwrap();
        let _ = crate::background_processes::output(&id, &owner, true).await;
    }

    fn make_test_ctx() -> ToolContext {
        use claurst_core::config::Config;
        use claurst_core::permissions::AutoPermissionHandler;
        use std::path::PathBuf;
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let handler = Arc::new(AutoPermissionHandler {
            mode: claurst_core::config::PermissionMode::Default,
        });
        ToolContext {
            working_dir: PathBuf::from("."),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "test-monitor".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            trusted_checker_catalog: Default::default(),
            catalog_view: None,
        }
    }
}
