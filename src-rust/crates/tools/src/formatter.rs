//! Run a configured file formatter after writes/edits.

use crate::command_capture::{capture_owned_with_cancel, spawn_owned_argv};
use crate::evidence::{EvidenceRef, EvidenceStore};
use crate::{tool_policy, ToolContext};
use claurst_core::permissions::PermissionDecision;
use serde::Serialize;
use std::path::Path;
use std::time::Duration;

const FORMATTER_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatterStatus {
    NotRun,
    Succeeded,
    Failed,
    Cancelled,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct FormatterOutcome {
    pub status: FormatterStatus,
    pub exit_code: Option<i32>,
    pub evidence: Option<EvidenceRef>,
    pub partial: bool,
    pub error: Option<String>,
    pub timed_out: bool,
    pub cancelled: bool,
}

impl FormatterOutcome {
    fn no_formatter() -> Self {
        Self {
            status: FormatterStatus::NotRun,
            exit_code: None,
            evidence: None,
            partial: false,
            error: None,
            timed_out: false,
            cancelled: false,
        }
    }

    fn not_run(error: impl Into<String>) -> Self {
        Self {
            status: FormatterStatus::NotRun,
            exit_code: None,
            evidence: None,
            partial: false,
            error: Some(error.into()),
            timed_out: false,
            cancelled: false,
        }
    }
}

fn formatter_text(stdout: &[u8], stderr: &[u8]) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    if stderr.is_empty() {
        stdout.into_owned()
    } else if stdout.is_empty() {
        format!("[stderr]\n{stderr}")
    } else {
        format!("{stdout}\n[stderr]\n{stderr}")
    }
}

fn retain_diagnostics(
    ctx: &ToolContext,
    text: String,
    partial: bool,
) -> Result<EvidenceRef, String> {
    EvidenceStore::for_context(ctx)
        .and_then(|store| store.capture(&text, partial))
        .map_err(|error| format!("Exact formatter evidence was not retained: {error}"))
}

fn admit_formatter(ctx: &ToolContext, path: &str) -> Result<(), String> {
    match tool_policy::tool_allowed(&ctx.config, "Bash") {
        Ok(true) => {}
        Ok(false) => return Err("Formatter blocked by configured Bash tool policy".into()),
        Err(error) => {
            return Err(format!(
                "Formatter blocked by invalid Bash tool policy: {error}"
            ))
        }
    }
    match ctx.check_capability(
        "Bash",
        &format!("Run configured formatter for {path}"),
        false,
    ) {
        PermissionDecision::Deny | PermissionDecision::DenyPermanently => {
            Err("Formatter blocked by central Bash capability policy".into())
        }
        PermissionDecision::Allow
        | PermissionDecision::AllowPermanently
        | PermissionDecision::Ask { .. } => Ok(()),
    }
}

async fn run_formatter_argv(
    argv: &[String],
    ctx: &ToolContext,
    timeout: Duration,
) -> FormatterOutcome {
    let (child, mut group) = match spawn_owned_argv(argv, &ctx.working_dir) {
        Ok(value) => value,
        Err(error) => {
            return FormatterOutcome {
                status: FormatterStatus::Failed,
                exit_code: None,
                evidence: None,
                partial: false,
                error: Some(format!("Failed to spawn formatter: {error}")),
                timed_out: false,
                cancelled: false,
            }
        }
    };
    let captured =
        match capture_owned_with_cancel(child, timeout, &ctx.cancel_token, &mut group).await {
            Ok(captured) => captured,
            Err(error) => {
                group.kill_now();
                return FormatterOutcome {
                    status: FormatterStatus::Unknown,
                    exit_code: None,
                    evidence: None,
                    partial: true,
                    error: Some(format!("Formatter output capture failed: {error}")),
                    timed_out: false,
                    cancelled: false,
                };
            }
        };
    // A formatter is a finite owned operation. Even an exit-0 direct child
    // may have left a redirected descendant in the same process group.
    group.kill_now();

    let partial = captured.partial || captured.timed_out || captured.cancelled;
    let diagnostics = formatter_text(&captured.stdout, &captured.stderr);
    let evidence = retain_diagnostics(ctx, diagnostics, partial);
    let evidence_error = evidence.as_ref().err().cloned();
    let exit_code = captured
        .status
        .as_ref()
        .and_then(std::process::ExitStatus::code);
    let error = if captured.cancelled {
        Some("Formatter cancelled before completion".into())
    } else if captured.timed_out {
        Some(format!(
            "Formatter timed out after {} seconds",
            timeout.as_secs()
        ))
    } else if let Some(capture_error) = captured.error.as_deref() {
        Some(format!("Formatter output capture failed: {capture_error}"))
    } else if partial {
        Some("Formatter output capture incomplete; formatter result is unknown".into())
    } else {
        captured
            .status
            .as_ref()
            .filter(|status| !status.success())
            .map(|status| format!("Formatter exited with code {}", status.code().unwrap_or(-1)))
    };
    let error = match (error, evidence_error) {
        (Some(error), Some(evidence_error)) => Some(format!("{error}; {evidence_error}")),
        (None, evidence_error) => evidence_error,
        (error, None) => error,
    };
    let status = if captured.cancelled {
        FormatterStatus::Cancelled
    } else if captured.timed_out || captured.error.is_some() || captured.status.is_none() || partial
    {
        FormatterStatus::Unknown
    } else if captured.status.is_some_and(|status| status.success()) {
        FormatterStatus::Succeeded
    } else {
        FormatterStatus::Failed
    };
    FormatterOutcome {
        status,
        exit_code,
        evidence: evidence.ok(),
        partial,
        error,
        timed_out: captured.timed_out,
        cancelled: captured.cancelled,
    }
}

/// Try the first configured formatter matching the file extension.
/// A formatter inherits the already-authorized file mutation and never
/// prompts recursively. Bash selector and capability deny checks still apply.
pub(crate) async fn try_format_file(path: &str, ctx: &ToolContext) -> FormatterOutcome {
    let effective = match crate::worktree::execution_context(ctx) {
        Ok(context) => context,
        Err(error) => return FormatterOutcome::not_run(error),
    };
    if effective.cancel_token.is_cancelled() {
        return FormatterOutcome::not_run("Formatter was cancelled before it started");
    }

    let ext = Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    let Some(formatter) = effective.config.formatter.values().find(|formatter| {
        !formatter.disabled
            && formatter
                .command
                .first()
                .is_some_and(|program| !program.trim().is_empty())
            && formatter
                .extensions
                .iter()
                .any(|candidate| candidate == &ext)
    }) else {
        return FormatterOutcome::no_formatter();
    };
    if let Err(error) = admit_formatter(&effective, path) {
        return FormatterOutcome::not_run(error);
    }

    let mut argv = Vec::with_capacity(formatter.command.len() + 1);
    argv.push(formatter.command[0].clone());
    let mut file_injected = false;
    for argument in &formatter.command[1..] {
        if argument == "$FILE" || argument == "{file}" {
            argv.push(path.to_string());
            file_injected = true;
        } else {
            argv.push(argument.clone());
        }
    }
    if !file_injected {
        argv.push(path.to_string());
    }
    run_formatter_argv(&argv, &effective, FORMATTER_TIMEOUT).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use claurst_core::permissions::PermissionHandler;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn configured_context(dir: &Path, command: Vec<String>) -> ToolContext {
        let mut ctx = allow_all_context(dir.to_path_buf());
        ctx.config.formatter.insert(
            "test".into(),
            claurst_core::config::FormatterConfig {
                command,
                extensions: vec![".txt".into()],
                disabled: false,
            },
        );
        ctx
    }

    #[tokio::test]
    async fn no_configured_formatter_is_clean_not_run() {
        let dir = tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let outcome = try_format_file("input.txt", &ctx).await;
        assert_eq!(outcome.status, FormatterStatus::NotRun);
        assert!(outcome.error.is_none());
        assert!(outcome.evidence.is_none());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn formatter_passes_literal_arguments_and_effective_cwd() {
        let dir = tempdir().unwrap();
        let command = vec![
            "/bin/sh".into(),
            "-c".into(),
            "pwd > cwd; printf '%s\\n' \"$@\" > args".into(),
            "formatter".into(),
            "literal; touch should-not-exist".into(),
            "$FILE".into(),
        ];
        let ctx = configured_context(dir.path(), command);
        let outcome = try_format_file("input.txt", &ctx).await;
        assert_eq!(outcome.status, FormatterStatus::Succeeded);
        assert_eq!(outcome.exit_code, Some(0));
        let actual_cwd = std::fs::read_to_string(dir.path().join("cwd")).unwrap();
        assert_eq!(
            Path::new(actual_cwd.trim()),
            dir.path().canonicalize().unwrap()
        );
        let args = std::fs::read_to_string(dir.path().join("args")).unwrap();
        assert!(args.contains("literal; touch should-not-exist"));
        assert!(args.contains("input.txt"));
        assert!(!dir.path().join("should-not-exist").exists());
        assert!(outcome.evidence.is_some());
    }

    #[tokio::test]
    async fn denied_selector_and_capability_do_not_spawn() {
        let dir = tempdir().unwrap();
        let marker = dir.path().join("spawned");
        let command = vec![
            "sh".into(),
            "-c".into(),
            format!("touch '{}'", marker.display()),
            "formatter".into(),
        ];
        let mut selector_ctx = configured_context(dir.path(), command.clone());
        selector_ctx.config.allowed_tools = vec!["Edit".into()];
        let outcome = try_format_file("input.txt", &selector_ctx).await;
        assert_eq!(outcome.status, FormatterStatus::NotRun);
        assert!(!marker.exists());

        let mut invalid_ctx = configured_context(dir.path(), command);
        invalid_ctx.config.allowed_tools = vec!["Bash(git:*)".into()];
        let outcome = try_format_file("input.txt", &invalid_ctx).await;
        assert_eq!(outcome.status, FormatterStatus::NotRun);
        assert!(!marker.exists());

        struct Deny;
        impl PermissionHandler for Deny {
            fn check_permission(
                &self,
                _: &claurst_core::permissions::PermissionRequest,
            ) -> PermissionDecision {
                PermissionDecision::Deny
            }
            fn request_permission(
                &self,
                _: &claurst_core::permissions::PermissionRequest,
            ) -> PermissionDecision {
                PermissionDecision::Deny
            }
        }
        let mut capability_ctx = configured_context(
            dir.path(),
            vec![
                "sh".into(),
                "-c".into(),
                format!("touch '{}'", marker.display()),
                "formatter".into(),
            ],
        );
        capability_ctx.permission_handler = Arc::new(Deny);
        let outcome = try_format_file("input.txt", &capability_ctx).await;
        assert_eq!(outcome.status, FormatterStatus::NotRun);
        assert!(!marker.exists());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn exit_failure_retains_evidence_and_exit_code() {
        let dir = tempdir().unwrap();
        let ctx = configured_context(
            dir.path(),
            vec![
                "sh".into(),
                "-c".into(),
                "printf formatter-error >&2; exit 7".into(),
                "formatter".into(),
            ],
        );
        let outcome = try_format_file("input.txt", &ctx).await;
        assert_eq!(outcome.status, FormatterStatus::Failed);
        assert_eq!(outcome.exit_code, Some(7));
        assert!(!outcome.partial);
        let evidence = outcome.evidence.expect("formatter diagnostics retained");
        let store = EvidenceStore::for_context(&ctx).unwrap();
        let (_, text) = store.read(&evidence.id).unwrap();
        assert!(text.contains("formatter-error"));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn partial_diagnostics_are_unknown_even_after_exit_zero() {
        let dir = tempdir().unwrap();
        let ctx = configured_context(
            dir.path(),
            vec![
                "sh".into(),
                "-c".into(),
                "head -c 16777217 /dev/zero".into(),
                "formatter".into(),
            ],
        );
        let outcome = run_formatter_argv(
            &ctx.config.formatter["test"].command,
            &ctx,
            Duration::from_secs(10),
        )
        .await;

        assert_eq!(outcome.status, FormatterStatus::Unknown);
        assert_eq!(outcome.exit_code, Some(0));
        assert!(outcome.partial);
        assert!(outcome
            .error
            .as_deref()
            .is_some_and(|error| error.contains("incomplete")));
        let evidence = outcome.evidence.expect("partial diagnostics retained");
        let store = EvidenceStore::for_context(&ctx).unwrap();
        let (reference, text) = store.read(&evidence.id).unwrap();
        assert!(reference.partial);
        assert_eq!(text.len(), 16 * 1024 * 1024);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn timeout_and_cancel_preserve_partial_output() {
        let dir = tempdir().unwrap();
        let descendant_marker = dir.path().join("descendant-survived");
        let timeout_gate = dir.path().join("timeout-write-gate");
        let timeout_script = format!(
            "printf before-timeout; (while [ ! -e '{}' ]; do sleep 0.01; done; touch '{}') & wait",
            timeout_gate.display(),
            descendant_marker.display()
        );
        let ctx = configured_context(
            dir.path(),
            vec!["sh".into(), "-c".into(), timeout_script, "formatter".into()],
        );
        let outcome = run_formatter_argv(
            &ctx.config.formatter["test"].command,
            &ctx,
            Duration::from_millis(100),
        )
        .await;
        assert_eq!(outcome.status, FormatterStatus::Unknown);
        assert!(outcome.timed_out && outcome.partial);
        assert!(outcome.evidence.is_some());
        std::fs::write(&timeout_gate, b"release").unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !descendant_marker.exists(),
            "owned formatter descendant survived timeout"
        );

        let cancel_ctx = configured_context(
            dir.path(),
            vec![
                "sh".into(),
                "-c".into(),
                "printf before-cancel; sleep 10".into(),
                "formatter".into(),
            ],
        );
        let trigger = cancel_ctx.cancel_token.clone();
        let task = tokio::spawn(async move {
            run_formatter_argv(
                &cancel_ctx.config.formatter["test"].command,
                &cancel_ctx,
                Duration::from_secs(10),
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
        let outcome = task.await.unwrap();
        assert_eq!(outcome.status, FormatterStatus::Cancelled);
        assert!(outcome.cancelled && outcome.partial && outcome.evidence.is_some());

        let drop_marker = dir.path().join("dropped-descendant-survived");
        let drop_gate = dir.path().join("dropped-write-gate");
        let drop_script = format!(
            "(while [ ! -e '{}' ]; do sleep 0.01; done; touch '{}') & wait",
            drop_gate.display(),
            drop_marker.display()
        );
        let drop_ctx = configured_context(
            dir.path(),
            vec!["sh".into(), "-c".into(), drop_script, "formatter".into()],
        );
        let command = drop_ctx.config.formatter["test"].command.clone();
        let drop_task = tokio::spawn(async move {
            run_formatter_argv(&command, &drop_ctx, Duration::from_secs(10)).await
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!drop_marker.exists());
        drop_task.abort();
        let _ = drop_task.await;
        std::fs::write(&drop_gate, b"release").unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !drop_marker.exists(),
            "owned formatter descendant survived dropped future"
        );

        let success_marker = dir.path().join("successful-descendant-survived");
        let success_gate = dir.path().join("successful-write-gate");
        let success_script = format!(
            "(while [ ! -e '{}' ]; do sleep 0.01; done; touch '{}') >/dev/null 2>&1 & exit 0",
            success_gate.display(),
            success_marker.display()
        );
        let success_ctx = configured_context(
            dir.path(),
            vec!["sh".into(), "-c".into(), success_script, "formatter".into()],
        );
        let command = success_ctx.config.formatter["test"].command.clone();
        let outcome = run_formatter_argv(&command, &success_ctx, Duration::from_secs(2)).await;
        assert_eq!(outcome.exit_code, Some(0));
        if outcome.status == FormatterStatus::Unknown {
            assert!(outcome.partial);
            assert!(outcome.error.as_deref().is_some_and(
                |error| error.contains("Owned process cleanup could not be confirmed")
            ));
        } else {
            assert_eq!(outcome.status, FormatterStatus::Succeeded);
        }
        assert!(!success_marker.exists());
        std::fs::write(&success_gate, b"release").unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !success_marker.exists(),
            "successful formatter descendant survived normal completion"
        );
    }
}
