//! Shared authorized tool execution.
//!
//! Every caller that invokes a tool outside the query scheduler must use this
//! broker. It applies configured selectors, the central capability check, the
//! existing self-gating permission behavior, and retained tool-text evidence in
//! one place.

use crate::{evidence, tool_policy, PermissionLevel, Tool, ToolContext, ToolResult};
use claurst_core::permissions::PermissionDecision;
use serde_json::Value;

/// Execution outcome when normal completion cannot be established. A completed
/// call may still have failed; this does not certify a mutation or its checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    NotRun,
    Cancelled,
    Unknown,
}

impl ExecutionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotRun => "not_run",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }
}

fn not_run(message: impl Into<String>) -> ToolResult {
    ToolResult::error(message).with_execution_status(ExecutionStatus::NotRun)
}

fn permission_level_is_gated(level: PermissionLevel) -> bool {
    !matches!(level, PermissionLevel::None | PermissionLevel::ReadOnly)
}

fn synthesize_permission_description(name: &str, input: &Value) -> String {
    let rendered = serde_json::to_string(input).unwrap_or_default();
    let preview: String = rendered.chars().take(200).collect();
    if preview.is_empty() || preview == "{}" || preview == "null" {
        format!("Run tool '{}'", name)
    } else {
        format!("Run tool '{}' with input: {}", name, preview)
    }
}

/// Apply every deny-only check that must run before configured or plugin
/// hooks. `Ask` remains a nonterminal decision here: the normal broker keeps
/// ownership of interactive permission requests immediately before execution.
pub fn authorization_preflight(
    tool: &dyn Tool,
    input: &Value,
    ctx: &ToolContext,
) -> Result<PermissionDecision, Box<ToolResult>> {
    let name = tool.name();
    if ctx.cancel_token.is_cancelled() {
        return Err(Box::new(not_run(format!(
            "Tool '{}' was cancelled before it started; this call was not run.",
            name
        ))));
    }
    match tool_policy::tool_allowed(&ctx.config, name) {
        Ok(true) => {}
        Ok(false) => {
            tracing::warn!("Tool blocked by configured tool policy");
            return Err(Box::new(not_run(format!(
                "Tool '{}' is disallowed by configured tool policy",
                name
            ))));
        }
        Err(error) => {
            tracing::warn!("Tool blocked by invalid configured tool policy");
            return Err(Box::new(not_run(format!(
                "Tool '{}' is blocked by invalid configured tool policy: {}",
                name, error
            ))));
        }
    }

    let description = synthesize_permission_description(name, input);
    let is_read_only = matches!(
        tool.permission_level(),
        PermissionLevel::None | PermissionLevel::ReadOnly
    );
    let capability = if tool.self_gates() {
        ctx.check_self_gating_capability(name, &description, is_read_only)
    } else {
        ctx.check_capability(name, &description, is_read_only)
    };
    match capability {
        PermissionDecision::Deny | PermissionDecision::DenyPermanently => {
            tracing::warn!("Tool blocked by central capability policy");
            Err(Box::new(not_run(format!(
                "Permission denied for tool '{}'",
                name
            ))))
        }
        decision => Ok(decision),
    }
}

/// Execute one tool through the shared policy, capability, permission, and
/// evidence path.
pub async fn execute_authorized(tool: &dyn Tool, input: Value, ctx: &ToolContext) -> ToolResult {
    let name = tool.name();
    let effective;
    let ctx = if matches!(name, "EnterWorktree" | "ExitWorktree") {
        ctx
    } else {
        effective = match crate::worktree::execution_context(ctx) {
            Ok(context) => context,
            Err(error) => return not_run(error),
        };
        &effective
    };
    let description = synthesize_permission_description(name, &input);
    let is_gated = permission_level_is_gated(tool.permission_level());
    let capability = match authorization_preflight(tool, &input, ctx) {
        Ok(decision) => decision,
        Err(result) => return *result,
    };
    match capability {
        PermissionDecision::Allow | PermissionDecision::AllowPermanently => {}
        PermissionDecision::Ask { .. } if tool.self_gates() || !is_gated => {}
        PermissionDecision::Ask { .. } => {
            if let Err(error) = ctx.check_permission(name, &description, false) {
                tracing::warn!("Tool blocked by central permission backstop");
                return not_run(error.to_string());
            }
        }
        PermissionDecision::Deny | PermissionDecision::DenyPermanently => {
            unreachable!("authorization_preflight returns denied decisions as typed NotRun results")
        }
    }

    // Cancellation can arrive while an interactive permission request is
    // outstanding.  Keep that pre-start path equivalent to cancellation at
    // entry: it must not advance the mutation ledger or spawn a child.
    if ctx.cancel_token.is_cancelled() {
        return not_run(format!(
            "Tool '{}' was cancelled before it started; this call was not run.",
            name
        ));
    }

    let result = match crate::mutation::admit(ctx, tool) {
        Err(error) => {
            // A concurrent or cross-scope potential mutation is fail-closed.
            // The underlying tool is never called, and no transport/process
            // side effect can occur.
            return ToolResult::error(error).with_execution_status(ExecutionStatus::NotRun);
        }
        Ok(crate::mutation::MutationAdmission::Observation) => tool.execute(input, ctx).await,
        Ok(crate::mutation::MutationAdmission::Acquired(lease))
        | Ok(crate::mutation::MutationAdmission::Reentrant(lease)) => {
            crate::mutation::with_lease(&lease, tool.execute(input, ctx)).await
        }
    };
    if name == "EvidenceRead" {
        result
    } else {
        evidence::retain_result(ctx, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolEffect;

    struct LargeObservation {
        output: String,
    }

    #[async_trait::async_trait]
    impl Tool for LargeObservation {
        fn name(&self) -> &str {
            "GenericOutputFixture"
        }

        fn description(&self) -> &str {
            "Return a deterministic large observation"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({"type":"object","additionalProperties":false})
        }

        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }

        fn effect(&self) -> ToolEffect {
            ToolEffect::Observation
        }

        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success(self.output.clone())
        }
    }

    #[tokio::test]
    async fn generic_broker_retains_omitted_middle_and_tail_before_excerpting() {
        let workspace = tempfile::tempdir().unwrap();
        let mut ctx = crate::test_support::allow_all_context(workspace.path().to_path_buf());
        ctx.session_id = format!("generic-output-{}", uuid::Uuid::new_v4());
        let marker = "DISTINCTIVE-GENERIC-MIDDLE";
        let tail = "generic-tail-failure";
        let source = format!(
            "HEAD:{}:{marker}:{}:{tail}",
            "h".repeat(40_000),
            "t".repeat(40_000)
        );
        let tool = LargeObservation {
            output: source.clone(),
        };

        let result = execute_authorized(&tool, serde_json::json!({}), &ctx).await;
        assert!(!result.is_error, "{}", result.content);
        assert!(!result.content.contains(marker));
        assert!(result.content.contains(tail));
        let metadata = result.metadata.as_ref().expect("generic evidence metadata");
        let id = metadata["evidence"]["id"].as_str().expect("evidence id");

        let matches = crate::evidence::EvidenceReadTool
            .execute(serde_json::json!({"id": id, "query": marker}), &ctx)
            .await;
        assert!(!matches.is_error, "{}", matches.content);
        let matches: Value = serde_json::from_str(&matches.content).unwrap();
        let offset = matches["match_offsets"][0].as_u64().unwrap() as usize;
        let middle = crate::evidence::EvidenceReadTool
            .execute(
                serde_json::json!({"id": id, "offset": offset, "limit": marker.len()}),
                &ctx,
            )
            .await;
        assert_eq!(middle.content, marker);

        let (_, exact) = crate::evidence::EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(id)
            .unwrap();
        assert_eq!(exact, source);
        let tail_result = crate::evidence::EvidenceReadTool
            .execute(
                serde_json::json!({"id": id, "offset": exact.len() - tail.len()}),
                &ctx,
            )
            .await;
        assert_eq!(tail_result.content, tail);
    }
}
