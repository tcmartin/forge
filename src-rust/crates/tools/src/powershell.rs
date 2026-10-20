// PowerShell tool: execute PowerShell commands (Windows-native).
//
// On Windows, PowerShell provides richer scripting than cmd.exe.
// On non-Windows platforms, attempts to use `pwsh` (PowerShell Core).
//
// Security model
// ──────────────
// Before any execution the command is passed through `classify_ps_command`.
// The resulting `PsRiskLevel` drives the permission gate:
//
//   Critical → always blocked (hard error, never executed)
//   High     → requires explicit user approval (once / session / deny)
//   Medium   → requires approval only when ctx.require_confirmation is set
//   Low      → executes directly

#[cfg(windows)]
use crate::mutation;
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use claurst_core::ps_classifier::{classify_ps_command, PsRiskLevel};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

pub struct PowerShellTool;

#[derive(Debug, Deserialize)]
struct PowerShellInput {
    command: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_timeout")]
    timeout: u64,
    /// When true, Medium-risk commands also prompt for approval.
    #[serde(default)]
    require_confirmation: bool,
}

fn default_timeout() -> u64 {
    120_000
}

/// Add typed process outcome fields while retaining evidence and existing
/// metadata such as execution status.
fn with_command_metadata(
    mut result: ToolResult,
    command_status: &str,
    exit_code: Option<i32>,
    partial: bool,
) -> ToolResult {
    let mut metadata = match result.metadata.take() {
        Some(Value::Object(metadata)) => metadata,
        Some(other) => Map::from_iter([("tool_metadata".into(), other)]),
        None => Map::new(),
    };
    metadata.insert(
        "command_status".into(),
        Value::String(command_status.into()),
    );
    metadata.insert(
        "exit_code".into(),
        exit_code.map_or(Value::Null, Value::from),
    );
    metadata.insert("partial".into(), Value::Bool(partial));
    result.metadata = Some(Value::Object(metadata));
    result
}

fn not_run_result(message: impl Into<String>) -> ToolResult {
    with_command_metadata(ToolResult::error(message), "not_run", None, false)
}

struct CapturedPowerShellOutput<'a> {
    stdout: &'a [u8],
    stderr: &'a [u8],
    exit_code: Option<i32>,
    partial: bool,
    timed_out: bool,
    cancelled: bool,
    error: Option<&'a str>,
}

/// Normalize one completed pipe capture, then retain the entire normalized
/// observation before any model-facing excerpt is created. Keeping this step
/// independent from process spawning makes the Windows output contract
/// testable on every host.
fn retained_capture_result(
    ctx: &ToolContext,
    captured: CapturedPowerShellOutput<'_>,
    timeout_ms: u64,
) -> ToolResult {
    let partial =
        captured.partial || captured.timed_out || captured.cancelled || captured.error.is_some();
    let mut output = String::from_utf8_lossy(captured.stdout).into_owned();
    if !captured.stderr.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str("STDERR:\n");
        output.push_str(&String::from_utf8_lossy(captured.stderr));
    }
    if output.is_empty() {
        output = "(no output)".to_string();
    }
    let output = if captured.cancelled {
        format!("PowerShell command cancelled; capture incomplete\n{output}")
    } else if let Some(error) = captured.error {
        format!("PowerShell capture failed: {error}; capture incomplete\n{output}")
    } else if captured.timed_out {
        format!("PowerShell command timed out after {timeout_ms}ms; capture incomplete\n{output}")
    } else if captured.partial {
        format!("[Capture incomplete: stream capture limit reached]\n{output}")
    } else {
        output
    };
    let output = if !captured.timed_out
        && !captured.cancelled
        && captured.error.is_none()
        && captured.exit_code.is_some_and(|code| code != 0)
    {
        format!(
            "PowerShell exited with code {}\n{output}",
            captured.exit_code.expect("checked above")
        )
    } else {
        output
    };
    let result = crate::evidence::retain_output(
        ctx,
        output,
        captured.timed_out
            || captured.cancelled
            || captured.error.is_some()
            || captured.exit_code.is_some_and(|code| code != 0),
        partial,
    );
    let (result, status) = if captured.cancelled {
        (
            result.with_execution_status(crate::ExecutionStatus::Unknown),
            "cancelled",
        )
    } else if captured.timed_out {
        (
            result.with_execution_status(crate::ExecutionStatus::Unknown),
            "timed_out",
        )
    } else if captured.error.is_some() || captured.exit_code.is_none() {
        (
            result.with_execution_status(crate::ExecutionStatus::Unknown),
            "unknown",
        )
    } else {
        (result, "completed")
    };
    with_command_metadata(result, status, captured.exit_code, partial)
}

// ---------------------------------------------------------------------------
// Risk-label helpers (used in messages shown to the user)
// ---------------------------------------------------------------------------

fn risk_label(level: PsRiskLevel) -> &'static str {
    match level {
        PsRiskLevel::Critical => "Critical",
        PsRiskLevel::High => "High",
        PsRiskLevel::Medium => "Medium",
        PsRiskLevel::Low => "Low",
    }
}

fn risk_explanation(level: PsRiskLevel, command: &str) -> String {
    match level {
        PsRiskLevel::Critical => format!(
            "PowerShell command classified as CRITICAL risk — execution blocked.\n\
             Reason: the command contains destructive or remote-code-execution patterns.\n\
             Command: {}",
            command
        ),
        PsRiskLevel::High => {
            "[High risk] This may modify system-wide security policy, the registry (HKLM), user accounts, or firewall rules.".to_string()
        }
        PsRiskLevel::Medium => {
            "[Medium risk] This may delete files, control services, or make network requests.".to_string()
        }
        PsRiskLevel::Low => String::new(), // never shown
    }
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Tool for PowerShellTool {
    // Gates itself: calls `ctx.check_permission*` in `execute()` (#210).
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "PowerShell"
    }

    fn description(&self) -> &str {
        "Execute a PowerShell command. Use for Windows-native operations, .NET APIs, \
         registry access, and Windows-specific system administration."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The PowerShell command or script to execute"
                },
                "description": {
                    "type": "string",
                    "description": "Description of what this command does"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in ms (default 120000, max 600000)"
                },
                "require_confirmation": {
                    "type": "boolean",
                    "description": "When true, Medium-risk commands also prompt for approval"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: PowerShellInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return not_run_result(format!("Invalid input: {}", e)),
        };

        // ── Step 1: classify the command ─────────────────────────────────────
        let risk = classify_ps_command(&params.command);

        // ── Step 2: apply the risk gate ──────────────────────────────────────
        match risk {
            PsRiskLevel::Critical => {
                // Hard block — never executed regardless of permission mode.
                return not_run_result(risk_explanation(PsRiskLevel::Critical, &params.command));
            }

            PsRiskLevel::High => {
                // Require explicit user permission (same once/session/deny
                // pattern as BashTool: delegate to ctx.check_permission which
                // in interactive mode shows the TUI dialog).
                let desc = format!(
                    "[{} risk] {}",
                    risk_label(risk),
                    params.description.as_deref().unwrap_or(&params.command)
                );
                let details = risk_explanation(PsRiskLevel::High, &params.command);
                if let Err(e) = ctx.check_permission_with_details_and_path(
                    self.name(),
                    &desc,
                    &details,
                    std::path::PathBuf::from(&params.command),
                    false,
                ) {
                    return not_run_result(e.to_string());
                }
            }

            PsRiskLevel::Medium => {
                // Only gate if the caller set require_confirmation, or if the
                // context permission mode is Default (non-bypass, non-accept).
                let needs_gate = params.require_confirmation
                    || matches!(
                        ctx.permission_mode,
                        claurst_core::config::PermissionMode::Default
                            | claurst_core::config::PermissionMode::Plan
                    );

                if needs_gate {
                    let desc = format!(
                        "[{} risk] {}",
                        risk_label(risk),
                        params.description.as_deref().unwrap_or(&params.command)
                    );
                    let details = risk_explanation(PsRiskLevel::Medium, &params.command);
                    if let Err(e) = ctx.check_permission_with_details_and_path(
                        self.name(),
                        &desc,
                        &details,
                        std::path::PathBuf::from(&params.command),
                        false,
                    ) {
                        return not_run_result(e.to_string());
                    }
                }
            }

            PsRiskLevel::Low => {
                // Standard (non-risk-gated) permission check — honours bypass
                // and plan-mode rules, but does not show a dialog.
                let reason = params
                    .description
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("This will execute a PowerShell command.")
                    .to_string();

                if let Err(e) = ctx.check_permission_for_path(
                    self.name(),
                    &reason,
                    std::path::PathBuf::from(&params.command),
                    false,
                ) {
                    return not_run_result(e.to_string());
                }
            }
        }

        // ── Step 3: execute ──────────────────────────────────────────────────
        let (exe, args) = if cfg!(windows) {
            (
                "powershell",
                vec!["-NoProfile", "-NonInteractive", "-Command"],
            )
        } else {
            ("pwsh", vec!["-NoProfile", "-NonInteractive", "-Command"])
        };

        debug!(
            command_bytes = params.command.len(),
            risk    = ?risk,
            "Executing PowerShell command"
        );

        let timeout_ms = params.timeout.min(600_000);
        let timeout_dur = Duration::from_millis(timeout_ms);

        #[cfg(windows)]
        let pending = match mutation::register_pending(
            mutation::scope_for_context(ctx),
            "PowerShell shell",
        ) {
            Ok(ticket) => ticket,
            Err(error) => return not_run_result(error.to_string()),
        };

        #[cfg(windows)]
        let (child, mut group) = {
            let mut command = Command::new(exe);
            command
                .args(&args)
                .arg(&params.command)
                .current_dir(&ctx.working_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .stdin(Stdio::null())
                .kill_on_drop(true);
            match crate::command_capture::spawn_owned_command_checked(command) {
                Ok(pair) => pair,
                Err(crate::command_capture::SpawnOwnedError::Spawn(error)) => {
                    let _ = pending.confirm_terminal();
                    return not_run_result(format!("Failed to spawn PowerShell: {error}"));
                }
                Err(crate::command_capture::SpawnOwnedError::Arm { child, error }) => {
                    let cleanup = crate::command_capture::cleanup_unarmed_child(child).await;
                    return with_command_metadata(
                        ToolResult::error(format!(
                            "Failed to arm PowerShell ownership: {error}; cleanup remains unknown ({cleanup})"
                        )),
                        "unknown",
                        None,
                        true,
                    );
                }
            }
        };
        #[cfg(not(windows))]
        let child = match Command::new(exe)
            .args(&args)
            .arg(&params.command)
            .current_dir(&ctx.working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return not_run_result(format!("Failed to spawn PowerShell: {}", e)),
        };

        #[cfg(windows)]
        let captured = match crate::command_capture::capture_owned_with_cancel(
            child,
            timeout_dur,
            &ctx.cancel_token,
            &mut group,
        )
        .await
        {
            Ok(output) => output,
            Err(e) => {
                return with_command_metadata(
                    ToolResult::error(format!("Failed to capture PowerShell output: {e}")),
                    "unknown",
                    None,
                    true,
                )
            }
        };
        #[cfg(not(windows))]
        let captured = match crate::command_capture::capture_with_cancel(
            child,
            timeout_dur,
            &ctx.cancel_token,
        )
        .await
        {
            Ok(output) => output,
            Err(e) => {
                return with_command_metadata(
                    ToolResult::error(format!("Failed to capture PowerShell output: {e}")),
                    "unknown",
                    None,
                    true,
                )
            }
        };
        #[cfg(windows)]
        {
            if captured.cleanup != crate::command_capture::ProcessCleanup::ConfirmedClean {
                return with_command_metadata(
                    ToolResult::error("PowerShell process-tree cleanup remains unknown"),
                    "unknown",
                    None,
                    true,
                );
            }
            if let Err(error) = pending.confirm_terminal() {
                return with_command_metadata(
                    ToolResult::error(format!(
                        "PowerShell cleanup was clean but pending mutation confirmation failed: {error}"
                    )),
                    "unknown",
                    None,
                    true,
                );
            }
        }
        retained_capture_result(
            ctx,
            CapturedPowerShellOutput {
                stdout: &captured.stdout,
                stderr: &captured.stderr,
                exit_code: captured.status.and_then(|status| status.code()),
                partial: captured.partial,
                timed_out: captured.timed_out,
                cancelled: captured.cancelled,
                error: captured.error.as_deref(),
            },
            timeout_ms,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn normalized_large_failure_is_exactly_retrievable_without_powershell_runtime() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = crate::test_support::allow_all_context(workspace.path().to_path_buf());
        ctx.session_id = format!("powershell-output-{}", uuid::Uuid::new_v4());
        let marker = "DISTINCTIVE-POWERSHELL-MIDDLE";
        let stdout = format!(
            "HEAD:{}:{marker}:{}",
            "h".repeat(40_000),
            "t".repeat(40_000)
        );
        let stderr = "tail-failure";

        let result = retained_capture_result(
            &ctx,
            CapturedPowerShellOutput {
                stdout: stdout.as_bytes(),
                stderr: stderr.as_bytes(),
                exit_code: Some(7),
                partial: false,
                timed_out: false,
                cancelled: false,
                error: None,
            },
            120_000,
        );

        assert!(result.is_error);
        assert!(!result.content.contains(marker));
        assert!(result.content.contains(stderr));
        let metadata = result
            .metadata
            .as_ref()
            .expect("PowerShell evidence metadata");
        assert_eq!(metadata["command_status"], "completed");
        assert_eq!(metadata["exit_code"], 7);
        assert_eq!(metadata["partial"], false);
        let id = metadata["evidence"]["id"].as_str().expect("evidence id");

        let matches = crate::evidence::EvidenceReadTool
            .execute(json!({"id": id, "query": marker}), &ctx)
            .await;
        assert!(!matches.is_error, "{}", matches.content);
        let matches: Value = serde_json::from_str(&matches.content).unwrap();
        let offset = matches["match_offsets"][0].as_u64().unwrap() as usize;
        let middle = crate::evidence::EvidenceReadTool
            .execute(
                json!({"id": id, "offset": offset, "limit": marker.len()}),
                &ctx,
            )
            .await;
        assert_eq!(middle.content, marker);

        let (_, exact) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert_eq!(
            exact,
            format!("PowerShell exited with code 7\n{stdout}\nSTDERR:\n{stderr}")
        );
        let tail_offset = exact.len() - stderr.len();
        let tail = crate::evidence::EvidenceReadTool
            .execute(json!({"id": id, "offset": tail_offset}), &ctx)
            .await;
        assert_eq!(tail.content, stderr);
    }

    #[tokio::test]
    async fn cancellation_retains_available_streams_as_explicit_partial_evidence() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = crate::test_support::allow_all_context(workspace.path().to_path_buf());
        ctx.session_id = format!("powershell-cancel-{}", uuid::Uuid::new_v4());
        let result = retained_capture_result(
            &ctx,
            CapturedPowerShellOutput {
                stdout: b"stdout-before-cancel",
                stderr: b"stderr-before-cancel",
                exit_code: None,
                partial: true,
                timed_out: false,
                cancelled: true,
                error: Some("pipe closure unconfirmed"),
            },
            1_000,
        );

        assert!(result.is_error);
        assert_eq!(
            result.execution_status(),
            Some(crate::ExecutionStatus::Unknown)
        );
        let metadata = result
            .metadata
            .as_ref()
            .expect("PowerShell evidence metadata");
        assert_eq!(metadata["command_status"], "cancelled");
        assert_eq!(metadata["partial"], true);
        assert_eq!(metadata["evidence"]["partial"], true);
        let id = metadata["evidence"]["id"].as_str().expect("evidence id");
        let (_, exact) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert!(exact.starts_with("PowerShell command cancelled; capture incomplete\n"));
        assert!(exact.contains("stdout-before-cancel\nSTDERR:\nstderr-before-cancel"));
    }

    #[test]
    fn command_metadata_preserves_existing_evidence() {
        let result = ToolResult::success("output").with_metadata(json!({
            "evidence": {"id": "evidence-1", "partial": false},
            "execution_status": "unknown"
        }));
        let result = with_command_metadata(result, "unknown", None, true);
        let metadata = result.metadata.unwrap();

        assert_eq!(metadata["evidence"]["id"], "evidence-1");
        assert_eq!(metadata["execution_status"], "unknown");
        assert_eq!(metadata["command_status"], "unknown");
        assert_eq!(metadata["exit_code"], Value::Null);
        assert_eq!(metadata["partial"], true);
    }
}
