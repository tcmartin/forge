//! Owned foreground execution for exact, immutable host checker registrations.

use crate::command_capture::{capture_owned_with_cancel, ProcessCleanup, ProcessGroupGuard};
use crate::mutation::{self, PendingMutationTicket};
use crate::trusted_checker::{TrustedCheckerRegistration, TrustedCheckerShell};
use crate::verification_receipts::{CheckerReceipt, MutationReceipt, ReceiptBundle};
use crate::{execute_authorized, PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult};
use async_trait::async_trait;
use claurst_core::bash_classifier::{classify_bash_command, BashRiskLevel};
use claurst_core::ps_classifier::{classify_ps_command, PsRiskLevel};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckerInput {
    command: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default)]
    run_in_background: bool,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    require_confirmation: bool,
}

fn default_timeout() -> u64 {
    120_000
}

/// The model selects an invocation, never a registration or its authority.
/// Ambiguous registrations and unsupported input shapes stay on the ordinary
/// shell route without acquiring a trusted receipt.
pub(crate) fn matching_registration(
    ctx: &ToolContext,
    mutation: &MutationReceipt,
    check_tool: &str,
    input: &Value,
) -> Option<TrustedCheckerRegistration> {
    if mutation.scope() != &mutation::scope_for_context(ctx) {
        return None;
    }
    let input: CheckerInput = serde_json::from_value(input.clone()).ok()?;
    if input.run_in_background || input.timeout == 0 || input.timeout > 300_000 {
        return None;
    }
    let shell = match check_tool {
        "Bash" => TrustedCheckerShell::Bash,
        "PowerShell" => TrustedCheckerShell::PowerShell,
        _ => return None,
    };
    let cwd = match input.cwd {
        Some(path) if shell == TrustedCheckerShell::Bash => ctx.resolve_path(&path),
        Some(_) => return None,
        None => ctx.working_dir.clone(),
    }
    .canonicalize()
    .ok()?;
    let mut candidates = ctx.trusted_checker_catalog.registrations().filter(|entry| {
        entry.workspace() == mutation.scope().workspace
            && entry.session_id() == mutation.scope().session
            && entry.target() == mutation.path()
            && entry.operation() == mutation.operation()
            && entry.invocation().shell() == shell
            && entry.invocation().command() == input.command
            && entry.invocation().cwd() == cwd
            && entry.invocation().timeout() == Duration::from_millis(input.timeout)
            && entry.invocation().is_foreground()
            && entry.invocation().program().is_some()
    });
    let matched = candidates.next()?;
    if candidates.next().is_some() {
        return None;
    }
    Some(matched.clone())
}

pub(crate) async fn execute_registered_check(
    registration: TrustedCheckerRegistration,
    receipts: ReceiptBundle,
    input: Value,
    ctx: &ToolContext,
) -> ToolResult {
    execute_authorized(
        &RegisteredChecker {
            registration,
            receipts,
        },
        input,
        ctx,
    )
    .await
}

struct RegisteredChecker {
    registration: TrustedCheckerRegistration,
    receipts: ReceiptBundle,
}

fn not_run(message: impl Into<String>) -> ToolResult {
    ToolResult::error(message)
        .with_execution_status(crate::ExecutionStatus::NotRun)
        .with_metadata(json!({
            "command_status": "not_run", "exit_code": null, "partial": false,
            "check_trust": "unsupported", "adaptive_eligible": false,
            "execution_status": "not_run"
        }))
}

async fn confirm_pending(ticket: Arc<PendingMutationTicket>) -> Result<(), String> {
    tokio::time::timeout(
        Duration::from_secs(6),
        tokio::task::spawn_blocking(move || ticket.confirm_terminal()),
    )
    .await
    .map_err(|_| "Checker cleanup confirmation timed out".to_string())?
    .map_err(|_| "Checker cleanup confirmation task failed".to_string())?
}

#[async_trait]
impl Tool for RegisteredChecker {
    fn name(&self) -> &str {
        match self.registration.invocation().shell() {
            TrustedCheckerShell::Bash => "Bash",
            TrustedCheckerShell::PowerShell => "PowerShell",
        }
    }

    fn description(&self) -> &str {
        "Run the exact foreground checker registered by the host."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Mutation
    }

    fn self_gates(&self) -> bool {
        true
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        if matching_registration(ctx, self.receipts.mutation(), self.name(), &input).as_ref()
            != Some(&self.registration)
        {
            return not_run("Trusted checker registration does not match this invocation");
        }
        let params: CheckerInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(_) => return not_run("Invalid registered checker input"),
        };
        let reason = params
            .description
            .as_deref()
            .filter(|description| !description.trim().is_empty())
            .unwrap_or("Run registered foreground checker");
        let permission = match self.registration.invocation().shell() {
            TrustedCheckerShell::Bash => {
                if classify_bash_command(&params.command) == BashRiskLevel::Critical {
                    return not_run("Registered checker blocked by the Bash security classifier");
                }
                ctx.check_permission_for_path(
                    self.name(),
                    reason,
                    PathBuf::from(&params.command),
                    false,
                )
            }
            TrustedCheckerShell::PowerShell => {
                let risk = classify_ps_command(&params.command);
                if risk == PsRiskLevel::Critical {
                    return not_run(
                        "Registered checker blocked by the PowerShell security classifier",
                    );
                }
                let needs_details = risk == PsRiskLevel::High
                    || risk == PsRiskLevel::Medium
                        && (params.require_confirmation
                            || matches!(
                                ctx.permission_mode,
                                claurst_core::config::PermissionMode::Default
                                    | claurst_core::config::PermissionMode::Plan
                            ));
                if needs_details {
                    ctx.check_permission_with_details_and_path(
                        self.name(),
                        reason,
                        "Registered PowerShell checker requires command approval",
                        PathBuf::from(&params.command),
                        false,
                    )
                } else {
                    ctx.check_permission_for_path(
                        self.name(),
                        reason,
                        PathBuf::from(&params.command),
                        false,
                    )
                }
            }
        };
        if let Err(error) = permission {
            return not_run(error.to_string());
        }
        if ctx.cancel_token.is_cancelled() {
            return not_run("Registered checker cancelled before it started");
        }
        let start_stamp = mutation::current_stamp(ctx);
        if self.receipts.mutation().stamp().epoch.checked_add(1) != Some(start_stamp.epoch)
            || self.receipts.mutation().stamp().scope != start_stamp.scope
            || self.receipts.mutation().stamp().generation != start_stamp.generation
        {
            return not_run("Registered checker mutation receipt is stale");
        }
        let Some(program) = self.registration.invocation().program() else {
            return not_run("Registered checker interpreter is unavailable");
        };
        let scope = mutation::scope_for_context(ctx);
        let pending = match tokio::time::timeout(
            Duration::from_secs(6),
            tokio::task::spawn_blocking(move || {
                mutation::register_pending(scope, "trusted-checker")
            }),
        )
        .await
        {
            Ok(Ok(Ok(ticket))) => Arc::new(ticket),
            _ => return not_run("Registered checker pending ownership could not be recorded"),
        };
        if ctx.cancel_token.is_cancelled() {
            let _ = confirm_pending(pending).await;
            return not_run("Registered checker cancelled before process creation");
        }
        #[cfg(windows)]
        let (child, mut group) = {
            let mut command = Command::new(program);
            match self.registration.invocation().shell() {
                TrustedCheckerShell::Bash => {
                    command.args(["--noprofile", "--norc", "-c"]);
                }
                TrustedCheckerShell::PowerShell => {
                    command.args(["-NoProfile", "-NonInteractive", "-Command"]);
                }
            }
            command
                .arg(self.registration.invocation().command())
                .current_dir(self.registration.invocation().cwd())
                .env_clear()
                .envs(self.registration.invocation().environment())
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            match crate::command_capture::spawn_owned_command_checked(command) {
                Ok(pair) => pair,
                Err(crate::command_capture::SpawnOwnedError::Spawn(_)) => {
                    let _ = confirm_pending(pending).await;
                    return not_run("Registered checker process could not be started");
                }
                Err(crate::command_capture::SpawnOwnedError::Arm { child, error }) => {
                    let cleanup = crate::command_capture::cleanup_unarmed_child(child).await;
                    return ToolResult::error(format!(
                        "Registered checker ownership failed: {error}; cleanup remains unknown ({cleanup})"
                    ))
                    .with_execution_status(crate::ExecutionStatus::Unknown);
                }
            }
        };
        #[cfg(not(windows))]
        let (child, mut group) = {
            let mut command = Command::new(program);
            match self.registration.invocation().shell() {
                TrustedCheckerShell::Bash => {
                    command.args(["--noprofile", "--norc", "-c"]);
                }
                TrustedCheckerShell::PowerShell => {
                    command.args(["-NoProfile", "-NonInteractive", "-Command"]);
                }
            }
            command
                .arg(self.registration.invocation().command())
                .current_dir(self.registration.invocation().cwd())
                .env_clear()
                .envs(self.registration.invocation().environment())
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            #[cfg(unix)]
            command.process_group(0);
            let child = match command.spawn() {
                Ok(child) => child,
                Err(_) => {
                    let _ = confirm_pending(pending).await;
                    return not_run("Registered checker process could not be started");
                }
            };
            let group = ProcessGroupGuard::new(child.id());
            (child, group)
        };
        let captured = match capture_owned_with_cancel(
            child,
            self.registration.invocation().timeout(),
            &ctx.cancel_token,
            &mut group,
        )
        .await
        {
            Ok(captured) => captured,
            Err(_) => {
                return ToolResult::error("Registered checker capture failed; cleanup is uncertain")
                    .with_execution_status(crate::ExecutionStatus::Unknown)
            }
        };
        let confirmed = if captured.cleanup == ProcessCleanup::ConfirmedClean {
            confirm_pending(pending).await.is_ok()
        } else {
            false
        };
        let exit_code = captured
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code);
        let partial = captured.partial || captured.error.is_some() || !confirmed;
        let complete = confirmed && !partial && !captured.timed_out && !captured.cancelled;
        let mut output = String::from_utf8_lossy(&captured.stdout).into_owned();
        if !captured.stderr.is_empty() {
            output.push_str("\n[stderr]\n");
            output.push_str(&String::from_utf8_lossy(&captured.stderr));
        }
        if output.is_empty() {
            output.push_str("(no output)");
        }
        let status = if captured.cancelled {
            "cancelled"
        } else if captured.timed_out {
            "timed_out"
        } else if complete && exit_code.is_some() {
            "completed"
        } else {
            "unknown"
        };
        let mut result =
            crate::evidence::retain_output(ctx, output, !complete || exit_code != Some(0), partial);
        let retained = result.metadata.as_ref().is_some_and(|metadata| {
            metadata.get("evidence").is_some_and(|evidence| {
                evidence.get("partial").and_then(Value::as_bool) == Some(false)
            })
        });
        let checker = if complete && retained && exit_code == Some(0) {
            CheckerReceipt::completed(
                ctx,
                &self.registration,
                self.receipts.mutation(),
                &start_stamp,
                &captured,
            )
            .await
            .ok()
        } else {
            None
        };
        let trusted = checker.is_some();
        let mut metadata = result.metadata.take().unwrap_or_else(|| json!({}));
        if let Some(metadata) = metadata.as_object_mut() {
            metadata.insert("command_status".into(), json!(status));
            metadata.insert(
                "process_cleanup_confirmed".into(),
                json!(captured.cleanup == ProcessCleanup::ConfirmedClean),
            );
            metadata.insert("pending_cleanup_confirmed".into(), json!(confirmed));
            metadata.insert("exit_code".into(), json!(exit_code));
            metadata.insert("partial".into(), json!(partial));
            metadata.insert(
                "check_trust".into(),
                json!(if trusted { "registered" } else { "unsupported" }),
            );
            // The application must finalize receipts after the complete tool
            // batch and revalidate again at the inference transport boundary.
            metadata.insert("adaptive_eligible".into(), json!(false));
        }
        result.metadata = Some(metadata);
        if !complete {
            result = result.with_execution_status(crate::ExecutionStatus::Unknown);
        }
        if let Some(checker) = checker {
            result =
                result.with_application_receipts(self.receipts.clone().attach_checker(checker));
        }
        result
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use crate::verification_receipts::{finalize_proof, ReceiptError};
    use crate::{
        EditCheckTool, FileWriteTool, RegistrationSpec, TrustedCheckerCatalogBuilder,
        TrustedCheckerInvocation, TrustedCheckerOperation, TrustedCheckerSource,
    };
    use std::path::Path;

    fn context(root: &Path, operation: TrustedCheckerOperation, command: &str) -> ToolContext {
        let mut ctx = allow_all_context(root.to_path_buf());
        let entry = TrustedCheckerRegistration::new(RegistrationSpec {
            id: "fixture".into(),
            revision: 1,
            source: TrustedCheckerSource::Application,
            workspace: root.to_path_buf(),
            session_id: ctx.session_id.clone(),
            target: root.join("target"),
            operation,
            invocation: TrustedCheckerInvocation::foreground(
                TrustedCheckerShell::Bash,
                command,
                root,
                Duration::from_secs(5),
            )
            .unwrap(),
        })
        .unwrap();
        let mut builder = TrustedCheckerCatalogBuilder::new();
        builder.register(entry).unwrap();
        ctx.trusted_checker_catalog = Arc::new(builder.build());
        ctx
    }

    async fn edit_check(
        ctx: &ToolContext,
        operation: TrustedCheckerOperation,
        command: &str,
    ) -> ToolResult {
        let edit = match operation {
            TrustedCheckerOperation::Write => json!({"file_path": "target", "content": "after"}),
            TrustedCheckerOperation::Edit => {
                json!({"file_path": "target", "old_string": "before", "new_string": "after"})
            }
        };
        execute_authorized(&EditCheckTool, json!({
            "operation": match operation { TrustedCheckerOperation::Write => "Write", TrustedCheckerOperation::Edit => "Edit" },
            "edit": edit,
            "check": {"tool": "Bash", "input": {"command": command, "timeout": 5000}},
        }), ctx).await
    }

    async fn successful_bundle(root: &Path) -> (ToolContext, ReceiptBundle) {
        let command = "test \"$(cat target)\" = after; printf x >> checker-runs";
        let ctx = context(root, TrustedCheckerOperation::Write, command);
        let mut result = edit_check(&ctx, TrustedCheckerOperation::Write, command).await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(std::fs::read(root.join("checker-runs")).unwrap(), b"x");
        assert!(!mutation::pending_for_workspace(&ctx));
        (
            ctx,
            result
                .take_application_receipts()
                .expect("trusted receipt bundle"),
        )
    }

    #[tokio::test]
    async fn authentic_write_and_edit_finalize_after_broker_release() {
        for operation in [
            TrustedCheckerOperation::Write,
            TrustedCheckerOperation::Edit,
        ] {
            let root = tempfile::tempdir().unwrap();
            std::fs::write(root.path().join("target"), b"before").unwrap();
            let command = "test \"$(cat target)\" = after; printf x >> checker-runs";
            let ctx = context(root.path(), operation, command);
            let mut result = edit_check(&ctx, operation, command).await;
            assert!(!result.is_error, "{}", result.content);
            let bundle = result.take_application_receipts().expect("typed bundle");
            let cloned = bundle.clone();
            let proof = finalize_proof(&ctx, &bundle)
                .await
                .expect("finalized proof");
            proof.revalidate(&ctx).await.unwrap();
            proof.clone().revalidate(&ctx).await.unwrap();
            assert_eq!(
                finalize_proof(&ctx, &cloned).await.unwrap_err(),
                ReceiptError::AlreadyConsumed
            );
            assert_eq!(
                std::fs::read(root.path().join("checker-runs")).unwrap(),
                b"x"
            );
        }
    }

    #[tokio::test]
    async fn pinned_checker_ignores_session_shell_exports_cwd_and_startup_hooks() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let command = "test -z \"${CHECKER_POISON-}\" && test \"$(cat target)\" = after";
        let ctx = context(root.path(), TrustedCheckerOperation::Write, command);
        let hook = root.path().join("startup-hook");
        std::fs::write(&hook, "touch startup-hook-ran").unwrap();
        {
            let state = crate::context_shell_state(&ctx);
            let mut state = state.lock();
            state.cwd = Some(elsewhere.path().to_path_buf());
            state
                .env_vars
                .insert("CHECKER_POISON".into(), "CANARY_SESSION_SECRET".into());
            state
                .env_vars
                .insert("BASH_ENV".into(), hook.to_string_lossy().into_owned());
        }
        let mut result = edit_check(&ctx, TrustedCheckerOperation::Write, command).await;
        assert!(!result.is_error, "{}", result.content);
        assert!(!root.path().join("startup-hook-ran").exists());
        let bundle = result.take_application_receipts().unwrap();
        finalize_proof(&ctx, &bundle).await.unwrap();
    }

    #[tokio::test]
    async fn identical_atomic_replacement_invalidates_receipt_identity() {
        let root = tempfile::tempdir().unwrap();
        let (ctx, bundle) = successful_bundle(root.path()).await;
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, b"after").unwrap();
        std::fs::rename(replacement, root.path().join("target")).unwrap();
        assert_eq!(
            finalize_proof(&ctx, &bundle).await.unwrap_err(),
            ReceiptError::TargetChanged
        );
    }

    #[tokio::test]
    async fn invocation_mismatches_cannot_select_the_registered_checker() {
        let root = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let ctx = context(root.path(), TrustedCheckerOperation::Write, "true");
        let mut write = execute_authorized(
            &FileWriteTool,
            json!({
                "file_path": "target", "content": "after",
            }),
            &ctx,
        )
        .await;
        let bundle = write.take_application_receipts().unwrap();
        let valid = json!({"command": "true", "timeout": 5000});
        assert!(matching_registration(&ctx, bundle.mutation(), "Bash", &valid).is_some());
        for input in [
            json!({"command": "true", "timeout": 5001}),
            json!({"command": "true", "timeout": 0}),
            json!({"command": "true", "timeout": 300001}),
            json!({"command": "true", "timeout": 5000, "run_in_background": true}),
            json!({"command": "true", "timeout": 5000, "cwd": elsewhere.path()}),
            json!({"command": "true", "timeout": 5000, "env": {"BASH_ENV": "forged"}}),
            json!({"command": "echo true", "timeout": 5000}),
        ] {
            assert!(
                matching_registration(&ctx, bundle.mutation(), "Bash", &input).is_none(),
                "{input}"
            );
        }
        assert!(matching_registration(&ctx, bundle.mutation(), "PowerShell", &valid).is_none());
        let mut foreign = ctx.clone();
        foreign.session_id = "different-session".into();
        assert!(matching_registration(&foreign, bundle.mutation(), "Bash", &valid).is_none());
    }

    #[tokio::test]
    async fn configured_checker_deny_is_enforced_before_process_creation() {
        let root = tempfile::tempdir().unwrap();
        let command = "touch should-not-run";
        let mut ctx = context(root.path(), TrustedCheckerOperation::Write, command);
        ctx.config.disallowed_tools = vec!["Bash".into()];
        let mut result = edit_check(&ctx, TrustedCheckerOperation::Write, command).await;
        assert!(result.is_error);
        assert!(!root.path().join("should-not-run").exists());
        assert!(result.take_application_receipts().is_none());
        assert!(!mutation::pending_for_workspace(&ctx));
    }

    #[tokio::test]
    async fn foreign_pending_owner_and_catalog_revision_invalidate_receipts() {
        let root = tempfile::tempdir().unwrap();
        let (ctx, bundle) = successful_bundle(root.path()).await;
        let ticket = mutation::register_pending(
            mutation::scope_for_workspace(root.path(), "foreign-owner"),
            "fixture",
        )
        .unwrap();
        assert_eq!(
            finalize_proof(&ctx, &bundle).await.unwrap_err(),
            ReceiptError::StampMismatch
        );
        ticket.confirm_terminal().unwrap();
        let root = tempfile::tempdir().unwrap();
        let (mut ctx, bundle) = successful_bundle(root.path()).await;
        ctx.trusted_checker_catalog = Default::default();
        assert_eq!(
            finalize_proof(&ctx, &bundle).await.unwrap_err(),
            ReceiptError::CheckerNotRegistered
        );
    }

    #[tokio::test]
    async fn foreign_same_byte_write_before_outer_release_invalidates_captured_vector() {
        let root = tempfile::tempdir().unwrap();
        let command = "test \"$(cat target)\" = after";
        let ctx = context(root.path(), TrustedCheckerOperation::Write, command);
        let outer = match mutation::admit(&ctx, &EditCheckTool).unwrap() {
            mutation::MutationAdmission::Acquired(lease) => lease,
            _ => panic!("outer lease required"),
        };
        let bundle = mutation::with_lease(&outer, async {
            let mut result = edit_check(&ctx, TrustedCheckerOperation::Write, command).await;
            assert!(!result.is_error, "{}", result.content);
            let bundle = result.take_application_receipts().unwrap();
            assert_eq!(
                finalize_proof(&ctx, &bundle).await.unwrap_err(),
                ReceiptError::StampMismatch
            );
            let mut foreign = ctx.clone();
            foreign.session_id = "foreign-same-byte".into();
            let result = execute_authorized(
                &FileWriteTool,
                json!({
                    "file_path": "target", "content": "after",
                }),
                &foreign,
            )
            .await;
            assert!(!result.is_error, "{}", result.content);
            assert_eq!(result.metadata.unwrap()["mutation"], "no_op");
            bundle
        })
        .await;
        drop(outer);
        assert_eq!(
            finalize_proof(&ctx, &bundle).await.unwrap_err(),
            ReceiptError::StampMismatch
        );
    }

    #[tokio::test]
    async fn failed_and_cancelled_registered_checks_supply_no_receipts() {
        let root = tempfile::tempdir().unwrap();
        let ctx = context(root.path(), TrustedCheckerOperation::Write, "exit 7");
        let mut result = edit_check(&ctx, TrustedCheckerOperation::Write, "exit 7").await;
        assert!(result.is_error);
        assert!(result.take_application_receipts().is_none());
        assert!(!mutation::pending_for_workspace(&ctx));
        let root = tempfile::tempdir().unwrap();
        let command = "printf ready > checker-ready; exec sleep 10";
        let ctx = context(root.path(), TrustedCheckerOperation::Write, command);
        let task_ctx = ctx.clone();
        let task = tokio::spawn(async move {
            edit_check(&task_ctx, TrustedCheckerOperation::Write, command).await
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while !root.path().join("checker-ready").exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        ctx.cancel_token.cancel();
        let mut result = tokio::time::timeout(Duration::from_secs(8), task)
            .await
            .unwrap()
            .unwrap();
        assert!(result.is_error);
        assert!(result.take_application_receipts().is_none());
        let metadata = result
            .metadata
            .as_ref()
            .expect("checker cleanup diagnostics");
        assert_eq!(
            metadata["check_process_cleanup_confirmed"], true,
            "{result:?}"
        );
        assert_eq!(
            metadata["check_pending_cleanup_confirmed"], true,
            "{result:?}"
        );
        assert!(!mutation::pending_for_workspace(&ctx), "{result:?}");
    }
}
