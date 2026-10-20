// Post-sampling / stop hooks fired around each model turn.
//
// Hook processes are supervised by claurst-core. This adapter keeps query
// policy (error injection and hard veto) separate from process lifecycle.

use std::path::{Path, PathBuf};

use tokio_util::sync::CancellationToken;

/// Result returned by `fire_post_sampling_hooks`.
#[derive(Debug, Default)]
pub struct PostSamplingHookResult {
    /// Error messages produced by hooks with non-zero exit codes.
    /// These are injected into the conversation as user messages before the
    /// next model turn so the model can react to them.
    pub blocking_errors: Vec<claurst_core::types::Message>,
    /// When `true` the query loop must not continue and should surface the
    /// error messages to the caller. Set when any hook exits with code > 1.
    pub prevent_continuation: bool,
}

fn hook_matches(entry: &claurst_core::config::HookEntry, tool_name: Option<&str>) -> bool {
    let Some(filter) = entry.tool_filter.as_deref() else {
        return true;
    };
    match tool_name {
        Some(tool) => filter.is_empty() || filter == "*" || filter == tool,
        None => true,
    }
}

/// Execute all `PostModelTurn` hooks with bounded asynchronous supervision.
///
/// Non-zero exits become user messages, preserving the query loop's existing
/// soft-error and hard-veto policy. Supervision failures are reported too,
/// because a bounded hook that could not be safely completed must not vanish
/// as a successful turn.
pub async fn fire_post_sampling_hooks(
    turn_result: &claurst_core::types::Message,
    config: &claurst_core::config::Config,
    working_dir: &Path,
    cancel: CancellationToken,
) -> PostSamplingHookResult {
    use claurst_core::config::HookEvent;
    use claurst_core::hooks::{HookContext, HookOutcome};
    use claurst_core::types::Message;

    let mut result = PostSamplingHookResult::default();
    let Some(entries) = config.hooks.get(&HookEvent::PostModelTurn) else {
        return result;
    };
    let ctx = HookContext {
        event: "PostModelTurn".to_string(),
        tool_name: None,
        tool_input: None,
        tool_output: Some(turn_result.get_all_text()),
        is_error: None,
        session_id: None,
    };

    for entry in entries {
        if !hook_matches(entry, None) {
            continue;
        }
        // PostModelTurn has always treated exit > 1 as a hard veto,
        // independently of the generic PreToolUse blocking flag.
        let mut post_entry = entry.clone();
        post_entry.blocking = true;
        match claurst_core::hooks::run_hook(&post_entry, &ctx, working_dir, &cancel).await {
            HookOutcome::Blocked(reason) => {
                result.blocking_errors.push(Message::user(format!(
                    "[Hook '{}' error]:\n{}",
                    entry.command, reason
                )));
            }
            HookOutcome::Vetoed(reason) => {
                result.blocking_errors.push(Message::user(format!(
                    "[Hook '{}' error]:\n{}",
                    entry.command, reason
                )));
                result.prevent_continuation = true;
            }
            HookOutcome::Failed(reason) => {
                result.blocking_errors.push(Message::user(format!(
                    "[Hook '{}' error]:\n{}",
                    entry.command, reason
                )));
            }
            HookOutcome::SupervisionFailed(reason) => {
                result.blocking_errors.push(Message::user(format!(
                    "[Hook '{}' error]:\n{}",
                    entry.command, reason
                )));
                if entry.blocking {
                    result.prevent_continuation = true;
                }
            }
            HookOutcome::Allowed | HookOutcome::Modified(_) => {}
        }
    }
    result
}

/// Owned scope for non-blocking `Stop` hooks.
///
/// The query loop can enqueue hooks without waiting for their result. It must
/// call `close().await` on normal shutdown and cancellation paths; Drop still
/// cancels and aborts the owned tasks as a last-resort cleanup.
pub struct StopHookScope {
    inner: claurst_core::hooks::HookTaskScope,
}

impl StopHookScope {
    pub fn new(cancel: CancellationToken) -> Self {
        Self {
            inner: claurst_core::hooks::HookTaskScope::with_cancel(cancel),
        }
    }

    pub fn spawn(
        &mut self,
        turn_result: &claurst_core::types::Message,
        config: &claurst_core::config::Config,
        working_dir: PathBuf,
    ) {
        use claurst_core::config::HookEvent;
        use claurst_core::hooks::HookContext;
        if !config.hooks.contains_key(&HookEvent::Stop) {
            return;
        }
        let ctx = HookContext {
            event: "Stop".to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: Some(turn_result.get_all_text()),
            is_error: None,
            session_id: None,
        };
        self.inner.spawn(config.hooks.clone(), ctx, working_dir);
    }

    pub async fn close(self) {
        self.inner.close().await;
    }

    pub async fn cancel_and_close(self) {
        self.inner.cancel_and_close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::hook_matches;
    use claurst_core::config::HookEntry;

    #[test]
    fn hook_filter_preserves_wildcard_and_unset_behavior() {
        let entry = HookEntry {
            command: "true".to_string(),
            tool_filter: Some("*".to_string()),
            blocking: false,
        };
        assert!(hook_matches(&entry, Some("Bash")));
        let entry = HookEntry {
            tool_filter: Some("Bash".to_string()),
            ..entry
        };
        assert!(hook_matches(&entry, Some("Bash")));
        assert!(!hook_matches(&entry, Some("Read")));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn post_model_hard_veto_does_not_depend_on_generic_blocking_flag() {
        let mut config = claurst_core::config::Config::default();
        config.hooks.insert(
            claurst_core::config::HookEvent::PostModelTurn,
            vec![HookEntry {
                command: "exec 0<&-; printf 'veto' >&2; exit 2".to_string(),
                tool_filter: None,
                blocking: false,
            }],
        );
        let result = super::fire_post_sampling_hooks(
            &claurst_core::types::Message::assistant("x".repeat(30_000)),
            &config,
            std::path::Path::new("."),
            tokio_util::sync::CancellationToken::new(),
        )
        .await;
        assert!(result.prevent_continuation);
        assert_eq!(result.blocking_errors.len(), 1);
        assert!(result.blocking_errors[0].get_all_text().contains("veto"));
    }
}
