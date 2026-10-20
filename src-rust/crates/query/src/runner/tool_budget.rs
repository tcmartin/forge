// Tool-result budgeting helpers used when trimming/compacting the transcript.
// Extracted from lib.rs (issue #232). Behavior-preserving move.

#[cfg(test)]
use crate::*;

/// Apply the outcome of a reactive-compact / context-collapse call to the live
/// message list, preserving the conversation when compaction fails.
///
/// Fixes #213: the reactive paths used to `std::mem::take(messages)` before
/// calling `compact::reactive_compact` / `compact::context_collapse`. That left
/// `*messages` empty, and on ANY failure (API error, `Cancelled`, empty
/// summary) the drained messages were never restored — silently destroying the
/// live conversation. Here we only overwrite `*messages` when compaction
/// returns `Ok`; on `Err` the original messages are left completely untouched,
/// so a failed compaction can never wipe the session.
///
/// Returns `Ok(tokens_freed)` on success, or the original error on failure.
#[cfg(test)]
pub(crate) fn apply_compact_result<E>(
    messages: &mut Vec<Message>,
    outcome: Result<compact::CompactResult, E>,
) -> Result<u64, E> {
    match outcome {
        Ok(result) => {
            let tokens_freed = result.tokens_freed;
            *messages = result.messages;
            Ok(tokens_freed)
        }
        // Failure: leave `*messages` untouched so the conversation survives.
        Err(e) => Err(e),
    }
}
