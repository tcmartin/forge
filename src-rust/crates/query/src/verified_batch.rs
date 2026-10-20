//! Runtime-only retention for tool batches that may carry a host receipt.
//!
//! A `ToolResult` is model-visible, while its application receipt is private
//! host authority.  This module takes that receipt before callers move the
//! result content into a message and keeps the authority alive until all
//! outer work and hook decisions have completed.

use claurst_tools::{
    finalize_proof, ExecutionStatus, ReceiptBundle, ToolContext, ToolEffect, ToolResult,
    TrustedVerificationProof,
};

/// Why a batch did or did not produce a verification proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchFinishReason {
    Disabled,
    Cancelled,
    HooksIncomplete,
    HooksIneligible,
    PendingWork,
    ToolError,
    UnknownToolEffect,
    NotRun,
    CancelledResult,
    UnknownResult,
    MetadataDisqualified,
    NoReceipt,
    MultipleReceipts,
    FinalizationFailed,
    Eligible,
}

/// One ordered result retained until post-tool hooks and proof selection end.
pub struct RetainedToolResult {
    result: ToolResult,
    receipt: Option<ReceiptBundle>,
    effect: ToolEffect,
    metadata_disqualified: bool,
}

impl RetainedToolResult {
    /// Inspect the model-visible result while retaining its private receipt.
    pub fn result(&self) -> &ToolResult {
        &self.result
    }

    /// Return the scheduler effect captured at batch construction.
    pub fn effect(&self) -> ToolEffect {
        self.effect
    }

    /// Whether this result carried a private application receipt.
    pub fn has_receipt(&self) -> bool {
        self.receipt.is_some()
    }

    /// Consume the retained result after proof selection has finished.
    pub fn into_result(self) -> ToolResult {
        self.result
    }
}

/// The completed batch, including ordered model-visible results and an
/// optional runtime-only proof.
pub struct BatchFinish {
    ordered_results: Vec<ToolResult>,
    proof: Option<TrustedVerificationProof>,
    reason: BatchFinishReason,
}

impl BatchFinish {
    pub fn ordered_results(&self) -> &[ToolResult] {
        &self.ordered_results
    }

    pub fn proof(&self) -> Option<&TrustedVerificationProof> {
        self.proof.as_ref()
    }

    pub fn reason(&self) -> BatchFinishReason {
        self.reason
    }

    pub fn into_parts(
        self,
    ) -> (
        Vec<ToolResult>,
        Option<TrustedVerificationProof>,
        BatchFinishReason,
    ) {
        (self.ordered_results, self.proof, self.reason)
    }
}

/// A non-cloneable runtime owner for a tool batch and its private receipt
/// candidates.  It must be finished only after all outer mutation leases and
/// post-tool hooks have completed.
pub struct RetainedToolBatch {
    ordered: Vec<RetainedToolResult>,
    cancelled: bool,
    any_is_error: bool,
    unknown_tool_effect: bool,
    pending: bool,
    hooks_complete: bool,
    hooks_ineligible: bool,
    enabled: bool,
}

impl RetainedToolBatch {
    /// Capture ordered results and extract every private receipt immediately.
    ///
    /// A mismatched effect list is treated as unknown.  `enabled=false` is a
    /// fixed-base fast path and guarantees that `finish` will not perform
    /// filesystem verification.
    pub fn new(
        results: Vec<ToolResult>,
        effects: impl IntoIterator<Item = ToolEffect>,
        cancelled: bool,
        enabled: bool,
    ) -> Self {
        let effects: Vec<ToolEffect> = effects.into_iter().collect();
        let effect_alignment_unknown = effects.len() != results.len();
        let any_is_error = results.iter().any(|result| result.is_error);
        let unknown_tool_effect =
            effect_alignment_unknown || effects.contains(&ToolEffect::Unknown);
        let ordered = results
            .into_iter()
            .enumerate()
            .map(|(index, mut result)| {
                let receipt = result.take_application_receipts();
                let metadata_disqualified = metadata_disqualifies(&result);
                RetainedToolResult {
                    result,
                    receipt,
                    effect: effects.get(index).copied().unwrap_or(ToolEffect::Unknown),
                    metadata_disqualified,
                }
            })
            .collect();

        Self {
            ordered,
            cancelled,
            any_is_error,
            unknown_tool_effect,
            pending: false,
            hooks_complete: false,
            hooks_ineligible: claurst_plugins::global_hooks_present(),
            enabled,
        }
    }

    /// Inspect ordered results before invoking configured post-tool hooks.
    pub fn inspect_results_for_posthooks(&self) -> impl Iterator<Item = &ToolResult> {
        self.ordered.iter().map(RetainedToolResult::result)
    }

    /// Inspect retained entries when a hook needs scheduler effect or receipt
    /// presence in addition to the model-visible result.
    pub fn retained_results(&self) -> &[RetainedToolResult] {
        &self.ordered
    }

    pub fn any_is_error(&self) -> bool {
        self.any_is_error
    }

    pub fn has_unknown_tool_effect(&self) -> bool {
        self.unknown_tool_effect
    }

    /// Mark all awaited post-tool hooks complete.  Callers must perform their
    /// own configured/global hook accounting before invoking this method.
    pub fn mark_hooks_complete(&mut self) {
        if !self.hooks_ineligible {
            self.hooks_complete = true;
        }
    }

    /// Irreversibly revoke proof eligibility for an unowned or uncertain hook.
    pub fn mark_hooks_ineligible(&mut self) {
        self.hooks_ineligible = true;
        self.hooks_complete = false;
    }

    /// Retain a pending outer writer/child/service barrier.
    pub fn mark_pending(&mut self) {
        self.pending = true;
    }

    /// Finish the batch and attempt exactly one proof finalization.
    ///
    /// All conservative gates are checked before calling `finalize_proof`.
    /// The private receipt is never serialized or reconstructed from metadata.
    pub async fn finish(self, ctx: &ToolContext) -> BatchFinish {
        let RetainedToolBatch {
            ordered,
            cancelled,
            any_is_error,
            unknown_tool_effect,
            pending,
            hooks_complete,
            hooks_ineligible,
            enabled,
        } = self;

        let mut ordered_results = Vec::with_capacity(ordered.len());
        let mut receipt = None;
        let mut receipt_count = 0usize;
        let mut reason = None;

        for retained in ordered {
            let retained_reason = result_reason(&retained);
            let retained_receipt = retained.receipt;
            let retained_result = retained.result;
            if retained_receipt.is_some() {
                receipt_count += 1;
                if receipt.is_none() {
                    receipt = retained_receipt;
                }
            }
            if reason.is_none() {
                reason = retained_reason;
            }
            ordered_results.push(retained_result);
        }

        if !enabled {
            return without_proof(ordered_results, BatchFinishReason::Disabled);
        }
        if cancelled || ctx.cancel_token.is_cancelled() {
            return without_proof(ordered_results, BatchFinishReason::Cancelled);
        }
        if hooks_ineligible || context_hooks_present(ctx) || claurst_plugins::global_hooks_present()
        {
            return without_proof(ordered_results, BatchFinishReason::HooksIneligible);
        }
        if !hooks_complete {
            return without_proof(ordered_results, BatchFinishReason::HooksIncomplete);
        }
        if pending {
            return without_proof(ordered_results, BatchFinishReason::PendingWork);
        }
        if let Some(reason) = reason {
            return without_proof(ordered_results, reason);
        }
        if any_is_error {
            return without_proof(ordered_results, BatchFinishReason::ToolError);
        }
        if unknown_tool_effect
            || ordered_results.is_empty()
            || receipt_count == 0
            || receipt_count > 1
        {
            return without_proof(
                ordered_results,
                if unknown_tool_effect {
                    BatchFinishReason::UnknownToolEffect
                } else if receipt_count == 0 {
                    BatchFinishReason::NoReceipt
                } else if receipt_count > 1 {
                    BatchFinishReason::MultipleReceipts
                } else {
                    BatchFinishReason::NoReceipt
                },
            );
        }

        let Some(receipt) = receipt else {
            return without_proof(ordered_results, BatchFinishReason::NoReceipt);
        };
        match finalize_proof(ctx, &receipt).await {
            Ok(proof) => BatchFinish {
                ordered_results,
                proof: Some(proof),
                reason: BatchFinishReason::Eligible,
            },
            Err(_) => without_proof(ordered_results, BatchFinishReason::FinalizationFailed),
        }
    }
}

fn without_proof(results: Vec<ToolResult>, reason: BatchFinishReason) -> BatchFinish {
    BatchFinish {
        ordered_results: results,
        proof: None,
        reason,
    }
}

fn result_reason(result: &RetainedToolResult) -> Option<BatchFinishReason> {
    match result.result.execution_status() {
        Some(ExecutionStatus::NotRun) => Some(BatchFinishReason::NotRun),
        Some(ExecutionStatus::Cancelled) => Some(BatchFinishReason::CancelledResult),
        Some(ExecutionStatus::Unknown) => Some(BatchFinishReason::UnknownResult),
        None if result.metadata_disqualified => Some(BatchFinishReason::MetadataDisqualified),
        None if result.result.is_error => Some(BatchFinishReason::ToolError),
        None if result.effect == ToolEffect::Unknown => Some(BatchFinishReason::UnknownToolEffect),
        None => None,
    }
}

fn metadata_disqualifies(result: &ToolResult) -> bool {
    result
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("execution_status"))
        .is_some()
}

fn context_hooks_present(ctx: &ToolContext) -> bool {
    ctx.config.hooks.values().any(|hooks| !hooks.is_empty())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use claurst_core::config::{Config, HookEntry, HookEvent};
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
    use claurst_tools::mutation;
    use claurst_tools::{
        execute_authorized, EditCheckTool, RegistrationSpec, TrustedCheckerCatalogBuilder,
        TrustedCheckerInvocation, TrustedCheckerOperation, TrustedCheckerRegistration,
        TrustedCheckerShell,
    };
    use serde_json::json;
    use std::path::Path;
    use std::sync::{atomic::AtomicUsize, Arc};
    use std::time::Duration;

    struct AllowAll;

    impl PermissionHandler for AllowAll {
        fn check_permission(&self, _request: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }

        fn request_permission(&self, _request: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    fn context(root: &Path, command: &str) -> ToolContext {
        let session_id = format!("verified-batch-{}", uuid::Uuid::new_v4());
        let mut ctx = ToolContext {
            working_dir: root.to_path_buf(),
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: Arc::new(AllowAll),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: session_id.clone(),
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
            catalog_view: None,
            trusted_checker_catalog: Arc::new(TrustedCheckerCatalogBuilder::new().build()),
        };
        let invocation = TrustedCheckerInvocation::foreground(
            TrustedCheckerShell::Bash,
            command,
            root,
            Duration::from_secs(5),
        )
        .expect("bash is available in Unix test environments");
        let registration = TrustedCheckerRegistration::new(RegistrationSpec {
            id: "verified-batch".to_string(),
            revision: 1,
            source: claurst_tools::TrustedCheckerSource::Application,
            workspace: root.to_path_buf(),
            session_id,
            target: root.join("target"),
            operation: TrustedCheckerOperation::Write,
            invocation,
        })
        .expect("valid registration");
        let mut builder = TrustedCheckerCatalogBuilder::new();
        builder.register(registration).expect("unique registration");
        ctx.trusted_checker_catalog = Arc::new(builder.build());
        ctx
    }

    async fn authentic_result(root: &Path) -> (ToolContext, ToolResult) {
        std::fs::write(root.join("target"), b"before").expect("target");
        let command = "test \"$(cat target)\" = after; printf x >> checker-runs";
        let ctx = context(root, command);
        let result = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": "target", "content": "after"},
                "check": {"tool": "Bash", "input": {"command": command, "timeout": 5000}}
            }),
            &ctx,
        )
        .await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.execution_status().is_none());
        assert!(result
            .metadata
            .as_ref()
            .and_then(|m| m.get("verification"))
            .is_some());
        (ctx, result)
    }

    #[tokio::test]
    async fn authentic_editcheck_batch_preserves_order_and_finalizes_once() {
        let root = tempfile::tempdir().expect("tempdir");
        let (ctx, result) = authentic_result(root.path()).await;
        let mut batch = RetainedToolBatch::new(
            vec![ToolResult::success("observation"), result],
            [ToolEffect::Observation, ToolEffect::Mutation],
            false,
            true,
        );
        let inspected = batch
            .inspect_results_for_posthooks()
            .map(|result| result.content.as_str())
            .collect::<Vec<_>>();
        assert_eq!(inspected[0], "observation");
        assert!(inspected[1].contains("Verification status: passed."));
        batch.mark_hooks_complete();
        let finish = batch.finish(&ctx).await;
        assert_eq!(finish.reason(), BatchFinishReason::Eligible);
        assert!(finish.proof().is_some());
        assert_eq!(finish.ordered_results().len(), 2);
    }

    #[tokio::test]
    async fn disabled_batch_does_not_consume_receipt_and_same_bundle_can_finalize() {
        let root = tempfile::tempdir().expect("tempdir");
        let (ctx, result) = authentic_result(root.path()).await;
        let retained = result.clone();
        let mut disabled =
            RetainedToolBatch::new(vec![result], [ToolEffect::Mutation], false, false);
        disabled.mark_hooks_complete();
        assert_eq!(
            disabled.finish(&ctx).await.reason(),
            BatchFinishReason::Disabled
        );

        let mut enabled =
            RetainedToolBatch::new(vec![retained], [ToolEffect::Mutation], false, true);
        enabled.mark_hooks_complete();
        let finish = enabled.finish(&ctx).await;
        assert_eq!(finish.reason(), BatchFinishReason::Eligible);
        assert!(finish.proof().is_some());
    }

    #[tokio::test]
    async fn failed_checker_and_multiple_receipts_never_finalize() {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(root.path().join("target"), b"before").expect("target");
        let command = "false";
        let ctx = context(root.path(), command);
        let failed = execute_authorized(
            &EditCheckTool,
            json!({
                "operation": "Write",
                "edit": {"file_path": "target", "content": "after"},
                "check": {"tool": "Bash", "input": {"command": command, "timeout": 5000}}
            }),
            &ctx,
        )
        .await;
        assert!(failed.is_error);
        let mut batch = RetainedToolBatch::new(vec![failed], [ToolEffect::Mutation], false, true);
        batch.mark_hooks_complete();
        assert_eq!(
            batch.finish(&ctx).await.reason(),
            BatchFinishReason::ToolError
        );

        let root = tempfile::tempdir().expect("tempdir");
        let (ctx, result) = authentic_result(root.path()).await;
        let clone = result.clone();
        let mut multiple = RetainedToolBatch::new(
            vec![result, clone],
            [ToolEffect::Mutation, ToolEffect::Mutation],
            false,
            true,
        );
        multiple.mark_hooks_complete();
        assert_eq!(
            multiple.finish(&ctx).await.reason(),
            BatchFinishReason::MultipleReceipts
        );
    }

    #[tokio::test]
    async fn cancellation_pending_stale_foreign_and_hooks_are_fail_closed() {
        let root = tempfile::tempdir().expect("tempdir");
        let (ctx, result) = authentic_result(root.path()).await;

        let mut cancelled =
            RetainedToolBatch::new(vec![result.clone()], [ToolEffect::Mutation], true, true);
        cancelled.mark_hooks_complete();
        assert_eq!(
            cancelled.finish(&ctx).await.reason(),
            BatchFinishReason::Cancelled
        );

        let mut pending =
            RetainedToolBatch::new(vec![result.clone()], [ToolEffect::Mutation], false, true);
        pending.mark_hooks_complete();
        pending.mark_pending();
        assert_eq!(
            pending.finish(&ctx).await.reason(),
            BatchFinishReason::PendingWork
        );

        let foreign_root = tempfile::tempdir().expect("foreign root");
        let (foreign_owner, foreign_result) = authentic_result(foreign_root.path()).await;
        let foreign_ctx = context(foreign_root.path(), "true");
        let foreign_ticket = mutation::register_pending(
            mutation::scope_for_context(&foreign_ctx),
            "verified-batch foreign pending fixture",
        )
        .expect("durable foreign pending ticket");
        let mut foreign_batch =
            RetainedToolBatch::new(vec![foreign_result], [ToolEffect::Mutation], false, true);
        foreign_batch.mark_hooks_complete();
        let foreign_reason = foreign_batch.finish(&foreign_owner).await.reason();
        foreign_ticket
            .confirm_terminal()
            .expect("clear foreign pending ticket");
        assert_eq!(foreign_reason, BatchFinishReason::FinalizationFailed);

        std::fs::write(root.path().join("target"), b"foreign").expect("stale target");
        let mut stale =
            RetainedToolBatch::new(vec![result.clone()], [ToolEffect::Mutation], false, true);
        stale.mark_hooks_complete();
        assert_eq!(
            stale.finish(&ctx).await.reason(),
            BatchFinishReason::FinalizationFailed
        );

        let incomplete =
            RetainedToolBatch::new(vec![result.clone()], [ToolEffect::Mutation], false, true);
        assert_eq!(
            incomplete.finish(&ctx).await.reason(),
            BatchFinishReason::HooksIncomplete
        );

        let mut ineligible =
            RetainedToolBatch::new(vec![result], [ToolEffect::Mutation], false, true);
        ineligible.mark_hooks_ineligible();
        ineligible.mark_hooks_complete();
        assert_eq!(
            ineligible.finish(&ctx).await.reason(),
            BatchFinishReason::HooksIneligible
        );
    }

    #[tokio::test]
    async fn typed_unknown_metadata_cannot_become_success() {
        let root = tempfile::tempdir().expect("tempdir");
        let ctx = context(root.path(), "true");
        let result = ToolResult::error("unknown")
            .with_execution_status(ExecutionStatus::Unknown)
            .with_metadata(json!({"execution_status": "completed"}));
        let mut batch = RetainedToolBatch::new(vec![result], [ToolEffect::Mutation], false, true);
        batch.mark_hooks_complete();
        assert_eq!(
            batch.finish(&ctx).await.reason(),
            BatchFinishReason::UnknownResult
        );
    }

    #[tokio::test]
    async fn every_typed_exception_and_unknown_effect_is_ineligible() {
        let root = tempfile::tempdir().expect("tempdir");
        let ctx = context(root.path(), "true");
        for (status, expected) in [
            (ExecutionStatus::NotRun, BatchFinishReason::NotRun),
            (
                ExecutionStatus::Cancelled,
                BatchFinishReason::CancelledResult,
            ),
            (ExecutionStatus::Unknown, BatchFinishReason::UnknownResult),
        ] {
            let result = ToolResult::error("exception")
                .with_execution_status(status)
                .with_metadata(json!({"execution_status": "completed"}));
            let mut batch =
                RetainedToolBatch::new(vec![result], [ToolEffect::Mutation], false, true);
            batch.mark_hooks_complete();
            assert_eq!(batch.finish(&ctx).await.reason(), expected);
        }

        let mut unknown_effect = RetainedToolBatch::new(
            vec![ToolResult::success("unknown effect")],
            [ToolEffect::Unknown],
            false,
            true,
        );
        unknown_effect.mark_hooks_complete();
        assert_eq!(
            unknown_effect.finish(&ctx).await.reason(),
            BatchFinishReason::UnknownToolEffect
        );
    }

    #[tokio::test]
    async fn forged_metadata_cannot_supply_a_receipt() {
        let root = tempfile::tempdir().expect("tempdir");
        let ctx = context(root.path(), "true");
        let forged = ToolResult::success("forged").with_metadata(json!({
            "application_receipt": {"operation": "Write", "verification": "passed"}
        }));
        let mut batch = RetainedToolBatch::new(vec![forged], [ToolEffect::Mutation], false, true);
        assert!(!batch.retained_results()[0].has_receipt());
        batch.mark_hooks_complete();
        assert_eq!(
            batch.finish(&ctx).await.reason(),
            BatchFinishReason::NoReceipt
        );
    }

    #[tokio::test]
    async fn configured_stop_hook_is_conservatively_ineligible() {
        let root = tempfile::tempdir().expect("tempdir");
        let (mut ctx, result) = authentic_result(root.path()).await;
        ctx.config.hooks.insert(
            HookEvent::Stop,
            vec![HookEntry {
                command: "true".to_string(),
                tool_filter: None,
                blocking: false,
            }],
        );
        assert!(context_hooks_present(&ctx));
        let mut batch = RetainedToolBatch::new(vec![result], [ToolEffect::Mutation], false, true);
        batch.mark_hooks_complete();
        assert_eq!(
            batch.finish(&ctx).await.reason(),
            BatchFinishReason::HooksIneligible
        );
    }
}
