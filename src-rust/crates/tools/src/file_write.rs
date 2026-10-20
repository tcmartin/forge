// FileWrite tool: write/create files.

use crate::formatter::FormatterStatus;
use crate::mutation::MutationStatus;
use crate::trusted_checker::TrustedCheckerOperation;
use crate::verification_receipts::MutationReceipt;
use crate::{ChildWriterCapability, PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult};
use async_trait::async_trait;
use claurst_core::crypto_utils::sha256_hex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use tracing::debug;

pub struct FileWriteTool;

#[derive(Debug, Deserialize)]
struct FileWriteInput {
    file_path: String,
    content: String,
}

#[async_trait]
impl Tool for FileWriteTool {
    // Gates itself: calls `ctx.check_permission_for_path` in `execute()` (#210).
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        claurst_core::constants::TOOL_NAME_FILE_WRITE
    }

    fn description(&self) -> &str {
        "Writes a file to the local filesystem. This tool will overwrite the existing \
         file if there is one. Prefer the Edit tool for modifying existing files. \
         Only use this tool to create new files or for complete rewrites."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Mutation
    }

    fn child_writer_capability(&self) -> ChildWriterCapability {
        ChildWriterCapability::ScopedFilesystem
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: FileWriteInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => {
                return mutation_error(MutationStatus::Failed, format!("Invalid input: {}", e))
            }
        };

        let path = ctx.resolve_path(&params.file_path);
        debug!(path = %path.display(), "Writing file");

        // Permission check
        if let Err(e) = ctx.check_permission_for_path(
            self.name(),
            &format!("Write {}", path.display()),
            path.clone(),
            false,
        ) {
            return mutation_error(MutationStatus::Failed, e.to_string());
        }

        if ctx.cancel_token.is_cancelled() {
            return mutation_error(
                MutationStatus::Cancelled,
                "File mutation cancelled before commit; verification not_run",
            )
            .with_execution_status(crate::ExecutionStatus::NotRun);
        }

        // Ensure parent directories exist
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return mutation_error(
                        MutationStatus::Failed,
                        format!("Failed to create directory {}: {}", parent.display(), e),
                    );
                }
            }
        }

        let existed = path.exists();
        let before_content = if existed {
            match tokio::fs::read(&path).await {
                Ok(bytes) => bytes,
                Err(e) => {
                    return mutation_error(
                        MutationStatus::Failed,
                        format!("Failed to read existing file {}: {}", path.display(), e),
                    )
                }
            }
        } else {
            Vec::new()
        };
        let is_new = !existed;

        commit_file_change(
            &path,
            &before_content,
            params.content.as_bytes(),
            !is_new,
            self.name(),
            ctx,
        )
        .await
    }
}

pub(crate) fn mutation_error(status: MutationStatus, message: impl Into<String>) -> ToolResult {
    ToolResult::error(message).with_metadata(json!({
        "mutation": status,
        "verification": "not_run",
        "formatter": { "status": "not_run" },
    }))
}

/// Finish one deterministic file mutation and record the actual final bytes.
/// A cancelled awaiting task is uncertain until readback: write_atomic owns
/// its commit transaction, and configured formatters own their process group.
pub(crate) async fn commit_file_change(
    path: &Path,
    before: &[u8],
    desired: &[u8],
    existed: bool,
    tool_name: &str,
    ctx: &ToolContext,
) -> ToolResult {
    let operation_id = uuid::Uuid::new_v4().to_string();
    if ctx.cancel_token.is_cancelled() {
        return mutation_error(
            MutationStatus::Cancelled,
            "File mutation cancelled before commit; verification not_run",
        )
        .with_execution_status(crate::ExecutionStatus::NotRun);
    }
    let before_hash = sha256_hex(before);
    if !existed || before != desired {
        if let Err(error) = crate::write_atomic(path, desired).await {
            let observed = tokio::fs::read(path).await.ok();
            if let Some(bytes) = &observed {
                ctx.record_file_change(path.to_path_buf(), before, bytes, tool_name);
            }
            return ToolResult::error(format!("File commit could not be confirmed for {}: {error}; mutation unknown; verification not_run", path.display()))
                .with_metadata(json!({
                    "operation_id": operation_id, "file_path": path,
                    "mutation": MutationStatus::Unknown, "verification": "not_run",
                    "formatter": {"status": "not_run"}, "before_hash": before_hash,
                    "after_hash": observed.as_deref().map(sha256_hex),
                }));
        }
    }
    let formatter = crate::try_format_file(&path.to_string_lossy(), ctx).await;
    let after = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) => return ToolResult::error(format!(
            "File was written but final bytes are unavailable for {}: {error}; mutation unknown; verification not_run", path.display()))
            .with_metadata(json!({
                "operation_id": operation_id, "file_path": path,
                "mutation": MutationStatus::Unknown, "verification": "not_run",
                "formatter": formatter, "before_hash": before_hash, "after_hash": null,
            })),
    };
    let actual_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    ctx.record_file_change(actual_path.clone(), before, &after, tool_name);
    let uncertain = formatter.partial
        || !matches!(
            formatter.status,
            FormatterStatus::Succeeded | FormatterStatus::NotRun
        );
    let mutation = if uncertain {
        MutationStatus::Unknown
    } else if existed && before == after {
        MutationStatus::NoOp
    } else {
        MutationStatus::Committed
    };
    let line_count = String::from_utf8_lossy(&after).lines().count();
    let mut message = format!(
        "File mutation {}: {} ({} lines, {} bytes). Formatter: {:?}. Verification: not_run.",
        mutation.as_str(),
        actual_path.display(),
        line_count,
        after.len(),
        formatter.status
    );
    if let Some(error) = &formatter.error {
        message.push_str(&format!(" {error}"));
    }
    if let Some(evidence) = &formatter.evidence {
        message.push_str(&format!(
            " Formatter diagnostics: EvidenceRead id={}",
            evidence.id
        ));
    }
    let result = if uncertain || formatter.error.is_some() {
        ToolResult::error(message)
    } else {
        ToolResult::success(message)
    };
    let after_hash = sha256_hex(&after);
    let operation = match tool_name {
        claurst_core::constants::TOOL_NAME_FILE_WRITE => Some(TrustedCheckerOperation::Write),
        claurst_core::constants::TOOL_NAME_FILE_EDIT => Some(TrustedCheckerOperation::Edit),
        _ => None,
    };
    // The default catalog stays inexpensive. Only a host-declared matching
    // target needs the additional bounded identity/hash readback for a receipt.
    let scope = crate::mutation::scope_for_context(ctx);
    let eligible = mutation == MutationStatus::Committed
        && !result.is_error
        && !ctx.cancel_token.is_cancelled()
        && operation.is_some_and(|operation| {
            ctx.trusted_checker_catalog.registrations().any(|entry| {
                entry.workspace() == scope.workspace
                    && entry.session_id() == scope.session
                    && entry.target() == actual_path
                    && entry.operation() == operation
            })
        });
    let receipt = match (eligible, operation) {
        (true, Some(operation)) => MutationReceipt::committed(
            ctx,
            path.to_path_buf(),
            operation,
            operation_id.clone(),
            after_hash.clone(),
            after.len() as u64,
        )
        .await
        .ok(),
        _ => None,
    };
    let result = result.with_metadata(json!({
        "operation_id": operation_id, "file_path": actual_path,
        "mutation": mutation, "verification": "not_run", "formatter": formatter,
        "before_hash": before_hash, "after_hash": after_hash,
        "is_new": !existed, "lines": line_count, "bytes": after.len(),
    }));
    match receipt {
        Some(receipt) => result.with_mutation_receipt(receipt),
        None => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
    use std::sync::Arc;

    struct DenyPaths;
    impl PermissionHandler for DenyPaths {
        fn check_permission(&self, request: &PermissionRequest) -> PermissionDecision {
            if request.path.is_some() {
                PermissionDecision::Deny
            } else {
                PermissionDecision::Allow
            }
        }
        fn request_permission(&self, request: &PermissionRequest) -> PermissionDecision {
            self.check_permission(request)
        }
    }

    #[tokio::test]
    async fn path_denial_precedes_directory_creation() {
        let dir = tempfile::tempdir().unwrap();
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.permission_handler = Arc::new(DenyPaths);
        let result = FileWriteTool
            .execute(
                json!({
                    "file_path": "new-parent/denied.txt", "content": "never write"
                }),
                &ctx,
            )
            .await;
        assert!(
            result.is_error,
            "path-aware permission must deny this write"
        );
        assert!(!dir.path().join("new-parent").exists());
    }

    #[tokio::test]
    async fn identical_write_reports_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("same.txt");
        std::fs::write(&path, "same bytes").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = FileWriteTool
            .execute(
                json!({
                    "file_path": path, "content": "same bytes"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(result.metadata.as_ref().unwrap()["mutation"], "no_op");
        assert_eq!(ctx.file_history.lock().len(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn history_records_actual_post_formatter_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("formatted.txt");
        let mut ctx = allow_all_context(dir.path().to_path_buf());
        ctx.config.formatter.insert(
            "fixture".into(),
            claurst_core::FormatterConfig {
                command: vec![
                    "/bin/sh".into(),
                    "-c".into(),
                    "printf formatted > \"$1\"".into(),
                    "fixture".into(),
                    "$FILE".into(),
                ],
                extensions: vec![".txt".into()],
                disabled: false,
            },
        );
        let result = FileWriteTool
            .execute(
                json!({
                    "file_path": path, "content": "intended"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(std::fs::read(&path).unwrap(), b"formatted");
        let history = ctx.file_history.lock();
        assert_eq!(
            history.entries().last().unwrap().after_text.as_deref(),
            Some("formatted")
        );
        assert_eq!(
            result.metadata.as_ref().unwrap()["after_hash"],
            claurst_core::crypto_utils::sha256_hex(b"formatted")
        );
    }
}
