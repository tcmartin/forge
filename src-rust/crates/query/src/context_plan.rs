//! Pure request selection. Persistence and provider serialization are supplied
//! by the caller; this module never changes the canonical transcript.

use claurst_core::types::{ContentBlock, Message, MessageContent, Role};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const UNKNOWN_CONTEXT_LIMIT: u64 = 32_000;
const MAX_CONTEXT_LIMIT: u64 = 100_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetSource {
    ExplicitConfig,
    ProviderRegistry,
    ProviderReported,
    UnknownFallback,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestBudget {
    pub context_limit_tokens: u64,
    pub input_limit_tokens: u64,
    pub output_reserve_tokens: u64,
    pub wire_byte_limit: u64,
    pub source: BudgetSource,
}

impl RequestBudget {
    pub fn resolve(
        explicit: Option<u64>,
        registry: Option<u64>,
        reported: Option<u64>,
        fallback: Option<u64>,
        output_reserve: u64,
        wire_byte_limit: u64,
    ) -> Result<Self, ContextPlanError> {
        let plausible = |limit: &u64| *limit > 0 && *limit <= MAX_CONTEXT_LIMIT;
        let (limit, source) = if let Some(limit) = explicit {
            if !plausible(&limit) {
                return Err(ContextPlanError::InvalidBudget);
            }
            (limit, BudgetSource::ExplicitConfig)
        } else if let Some(limit) = registry.filter(plausible) {
            (limit, BudgetSource::ProviderRegistry)
        } else if let Some(limit) = reported.filter(plausible) {
            (limit, BudgetSource::ProviderReported)
        } else {
            let limit = fallback.unwrap_or(UNKNOWN_CONTEXT_LIMIT);
            if !plausible(&limit) {
                return Err(ContextPlanError::InvalidBudget);
            }
            (limit, BudgetSource::UnknownFallback)
        };
        if output_reserve >= limit || wire_byte_limit == 0 {
            return Err(ContextPlanError::InvalidBudget);
        }
        Ok(Self {
            context_limit_tokens: limit,
            input_limit_tokens: limit - output_reserve,
            output_reserve_tokens: output_reserve,
            wire_byte_limit,
            source,
        })
    }

    fn fits(&self, cost: &CandidateCost) -> bool {
        cost.input_tokens <= self.input_limit_tokens && cost.wire_bytes <= self.wire_byte_limit
    }

    pub(crate) fn fits_for_query(&self, cost: &CandidateCost) -> bool {
        self.fits(cost)
    }
}

/// Numeric cost for the complete candidate, including its system, active
/// ledger, omission manifest, schemas, media, and final adapter framing.
/// The serializer owns the estimate/exact distinction and final transport hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateCost {
    pub input_tokens: u64,
    pub wire_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ContextPlanError {
    #[error("context budget must be finite and leave room for input")]
    InvalidBudget,
    #[error("persist every message with a unique ID before context selection")]
    MissingOrDuplicateMessageId,
    #[error("current request ID is absent from the transcript")]
    MissingCurrentRequest,
    #[error(
        "current request ID must identify a user request, not a tool result or assistant turn"
    )]
    InvalidCurrentRequest,
    #[error("mandatory tool component {group_id} is malformed")]
    MalformedToolPair { group_id: String },
    #[error(
        "mandatory request exceeds input/wire budget: {input_tokens} tokens, {wire_bytes} bytes; budget={budget:?}. Reduce this request or configure a larger finite limit."
    )]
    CurrentRequestTooLarge {
        input_tokens: u64,
        wire_bytes: u64,
        budget: RequestBudget,
    },
    #[error("provider measurement failed: {0}")]
    Measurement(String),
    #[error("selected content violates the provider media policy: {0}")]
    MediaPolicy(#[source] claurst_api::WireBudgetError),
    #[error("request sanitizer would change the selected transcript")]
    SanitizerChangedSelection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OmissionReason {
    Budget,
    MalformedToolPair,
    UnsupportedMedia,
}

/// A selection descriptor, not yet a retrievable reference. The integration
/// layer must persist exact members and mint a scoped HistoryRef before send.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OmittedComponent {
    pub group_id: String,
    pub message_ids: Vec<String>,
    pub reason: OmissionReason,
}

#[derive(Debug)]
pub struct ContextPlan {
    pub messages: Vec<Message>,
    pub selected_groups: Vec<String>,
    /// Historical groups selected in the preview and therefore eligible for
    /// complete-group eviction when final provider preparation is larger than
    /// the preview measurement.
    pub optional_selected_groups: Vec<String>,
    pub omitted: Vec<OmittedComponent>,
    pub cost: CandidateCost,
    pub budget: RequestBudget,
    pub measurements: usize,
}

#[derive(Debug)]
struct Component {
    id: String,
    indices: Vec<usize>,
    malformed: bool,
}

pub struct ContextAssembler;

impl ContextAssembler {
    /// `measure` must cost the whole request, not only `messages`. It receives
    /// every omission descriptor, allowing the manifest to count against the
    /// same budget. The caller persists omissions and prepares the exact wire
    /// body before dispatch. Measurement cannot perform inference or tools.
    pub fn plan<F>(
        transcript: &[Message],
        current_request_id: &str,
        budget: RequestBudget,
        measure: F,
    ) -> Result<ContextPlan, ContextPlanError>
    where
        F: FnMut(&[Message], &[OmittedComponent]) -> Result<CandidateCost, ContextPlanError>,
    {
        Self::plan_with_excluded(
            transcript,
            current_request_id,
            budget,
            &BTreeSet::new(),
            measure,
        )
    }

    /// Plan a request while keeping the supplied historical groups omitted.
    ///
    /// The query loop uses this for bounded final-wire reselection: when the
    /// exact provider body is larger than the planner preview, it adds one
    /// complete optional group to this set and prepares again.  The set only
    /// grows, making retries deterministic and monotone.
    pub fn plan_with_excluded<F>(
        transcript: &[Message],
        current_request_id: &str,
        budget: RequestBudget,
        excluded_group_ids: &BTreeSet<String>,
        mut measure: F,
    ) -> Result<ContextPlan, ContextPlanError>
    where
        F: FnMut(&[Message], &[OmittedComponent]) -> Result<CandidateCost, ContextPlanError>,
    {
        if budget.context_limit_tokens == 0
            || budget.context_limit_tokens > MAX_CONTEXT_LIMIT
            || budget.output_reserve_tokens >= budget.context_limit_tokens
            || budget.input_limit_tokens
                != budget.context_limit_tokens - budget.output_reserve_tokens
            || budget.wire_byte_limit == 0
        {
            return Err(ContextPlanError::InvalidBudget);
        }
        let components = dependency_components(transcript)?;
        let current = transcript
            .iter()
            .position(|message| message.uuid.as_deref() == Some(current_request_id))
            .ok_or(ContextPlanError::MissingCurrentRequest)?;
        let request = &transcript[current];
        let has_user_content = match &request.content {
            MessageContent::Text(text) => !text.trim().is_empty(),
            MessageContent::Blocks(blocks) => blocks.iter().any(|block| match block {
                ContentBlock::Text { text } => !text.trim().is_empty(),
                ContentBlock::Image { .. }
                | ContentBlock::Document { .. }
                | ContentBlock::UserCommand { .. }
                | ContentBlock::UserMemoryInput { .. }
                | ContentBlock::UserLocalCommandOutput { .. } => true,
                _ => false,
            }),
        };
        if request.role != Role::User || !has_user_content {
            return Err(ContextPlanError::InvalidCurrentRequest);
        }
        let mut selected = BTreeSet::new();
        for (index, component) in components.iter().enumerate() {
            // Keep the current request and every subsequent in-progress round.
            if component.indices.iter().any(|member| *member >= current) {
                if component.malformed {
                    return Err(ContextPlanError::MalformedToolPair {
                        group_id: component.id.clone(),
                    });
                }
                selected.insert(index);
            }
        }

        let mut unsupported_media = BTreeSet::new();
        let (mut messages, mut omitted) =
            selection(transcript, &components, &selected, &unsupported_media);
        let mut cost = measure(&messages, &omitted)?;
        let mut measurements = 1;
        if !budget.fits(&cost) {
            return Err(ContextPlanError::CurrentRequestTooLarge {
                input_tokens: cost.input_tokens,
                wire_bytes: cost.wire_bytes,
                budget,
            });
        }

        // One measured candidate per historical component. A rejected large
        // component cannot starve an older small one; every trial is finite.
        for (index, component) in components.iter().enumerate().rev() {
            if selected.contains(&index)
                || component.malformed
                || excluded_group_ids.contains(&component.id)
            {
                continue;
            }
            selected.insert(index);
            let (candidate, candidate_omissions) =
                selection(transcript, &components, &selected, &unsupported_media);
            measurements += 1;
            match measure(&candidate, &candidate_omissions) {
                Ok(candidate_cost) if budget.fits(&candidate_cost) => {
                    messages = candidate;
                    omitted = candidate_omissions;
                    cost = candidate_cost;
                }
                Ok(_) => {
                    selected.remove(&index);
                }
                Err(ContextPlanError::MediaPolicy(_)) => {
                    // A historical component whose media cannot be accounted
                    // for is omitted as a whole.  Keep trying older groups;
                    // the accepted candidate's omission manifest must carry
                    // the durable reason and be remeasured with that change.
                    selected.remove(&index);
                    unsupported_media.insert(component.id.clone());
                    let (accepted_messages, accepted_omitted) =
                        selection(transcript, &components, &selected, &unsupported_media);
                    messages = accepted_messages;
                    omitted = accepted_omitted;
                    measurements += 1;
                    cost = measure(&messages, &omitted)?;
                    if !budget.fits(&cost) {
                        return Err(ContextPlanError::CurrentRequestTooLarge {
                            input_tokens: cost.input_tokens,
                            wire_bytes: cost.wire_bytes,
                            budget,
                        });
                    }
                }
                Err(error) => return Err(error),
            }
        }

        // Assertion only: never use repair placeholders to hide a planner bug
        // or label a model-written unavailable result as runtime evidence.
        let sanitized = crate::sanitize::sanitize_history(messages.clone());
        let original = serde_json::to_vec(&messages)
            .map_err(|error| ContextPlanError::Measurement(error.to_string()))?;
        let checked = serde_json::to_vec(&sanitized)
            .map_err(|error| ContextPlanError::Measurement(error.to_string()))?;
        if original != checked {
            return Err(ContextPlanError::SanitizerChangedSelection);
        }

        Ok(ContextPlan {
            messages,
            selected_groups: selected
                .iter()
                .map(|index| components[*index].id.clone())
                .collect(),
            optional_selected_groups: selected
                .iter()
                .filter(|index| {
                    components[**index]
                        .indices
                        .iter()
                        .all(|member| *member < current)
                })
                .map(|index| components[*index].id.clone())
                .collect(),
            omitted,
            cost,
            budget,
            measurements,
        })
    }
}

fn selection(
    transcript: &[Message],
    components: &[Component],
    selected: &BTreeSet<usize>,
    unsupported_media: &BTreeSet<String>,
) -> (Vec<Message>, Vec<OmittedComponent>) {
    let mut indices = BTreeSet::new();
    let mut omitted = Vec::new();
    for (index, component) in components.iter().enumerate() {
        if selected.contains(&index) {
            indices.extend(component.indices.iter().copied());
        } else {
            omitted.push(OmittedComponent {
                group_id: component.id.clone(),
                message_ids: component
                    .indices
                    .iter()
                    .filter_map(|index| transcript[*index].uuid.clone())
                    .collect(),
                reason: if component.malformed {
                    OmissionReason::MalformedToolPair
                } else if unsupported_media.contains(&component.id) {
                    OmissionReason::UnsupportedMedia
                } else {
                    OmissionReason::Budget
                },
            });
        }
    }
    (
        indices
            .into_iter()
            .map(|index| transcript[index].clone())
            .collect(),
        omitted,
    )
}

/// Union API-round membership and every call/result dependency, including
/// malformed duplicates, before deciding which components may be selected.
fn dependency_components(transcript: &[Message]) -> Result<Vec<Component>, ContextPlanError> {
    let mut ids = BTreeSet::new();
    for message in transcript {
        match message.uuid.as_deref() {
            Some(id) if !id.is_empty() && ids.insert(id) => {}
            _ => return Err(ContextPlanError::MissingOrDuplicateMessageId),
        }
    }
    let mut parents: Vec<usize> = (0..transcript.len()).collect();
    let mut start = 0;
    for (index, message) in transcript.iter().enumerate() {
        if message.role == Role::Assistant {
            for member in start..=index {
                union(&mut parents, start, member);
            }
            start = index + 1;
        }
    }
    if start < transcript.len() {
        for member in start..transcript.len() {
            union(&mut parents, start, member);
        }
    }

    let mut calls: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut results: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut bad = BTreeSet::new();
    for (index, message) in transcript.iter().enumerate() {
        if let MessageContent::Blocks(blocks) = &message.content {
            for block in blocks {
                match block {
                    ContentBlock::ToolUse { id, .. } => {
                        calls.entry(id).or_default().push(index);
                        if id.is_empty() || message.role != Role::Assistant {
                            bad.insert(index);
                        }
                    }
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        results.entry(tool_use_id).or_default().push(index);
                        if tool_use_id.is_empty() || message.role != Role::User {
                            bad.insert(index);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let tool_ids: BTreeSet<&str> = calls.keys().chain(results.keys()).copied().collect();
    for id in tool_ids {
        let uses = calls.get(id).map(Vec::as_slice).unwrap_or_default();
        let answers = results.get(id).map(Vec::as_slice).unwrap_or_default();
        let members: Vec<usize> = uses.iter().chain(answers).copied().collect();
        if let Some(first) = members.first() {
            for member in &members {
                union(&mut parents, *first, *member);
            }
        }
        if uses.len() != 1 || answers.len() != 1 || answers[0] != uses[0] + 1 {
            bad.extend(members);
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..transcript.len() {
        let root = find(&mut parents, index);
        groups.entry(root).or_default().push(index);
    }
    groups
        .into_values()
        .map(|indices| {
            let member_ids: Vec<&str> = indices
                .iter()
                .filter_map(|index| transcript[*index].uuid.as_deref())
                .collect();
            let framed = serde_json::to_vec(&member_ids)
                .map_err(|error| ContextPlanError::Measurement(error.to_string()))?;
            Ok(Component {
                id: format!("group-{}", claurst_core::crypto_utils::sha256_hex(&framed)),
                malformed: indices.iter().any(|index| bad.contains(index)),
                indices,
            })
        })
        .collect()
}

fn find(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn union(parents: &mut [usize], a: usize, b: usize) {
    let a = find(parents, a);
    let b = find(parents, b);
    parents[a.max(b)] = a.min(b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::types::ToolResultContent;

    const TINY_PNG: &str =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

    fn indexed(mut messages: Vec<Message>) -> Vec<Message> {
        for (index, message) in messages.iter_mut().enumerate() {
            message.uuid = Some(format!("m-{index}"));
        }
        messages
    }

    fn budget(input: u64) -> RequestBudget {
        RequestBudget::resolve(Some(input + 100), None, None, None, 100, 1_000_000).unwrap()
    }

    fn cost(
        messages: &[Message],
        omitted: &[OmittedComponent],
    ) -> Result<CandidateCost, ContextPlanError> {
        let bytes = serde_json::to_vec(&(messages, omitted)).unwrap().len() as u64;
        Ok(CandidateCost {
            input_tokens: bytes,
            wire_bytes: bytes,
        })
    }

    fn call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.into(),
            name: "Read".into(),
            input: serde_json::json!({}),
            thought_signature: None,
        }
    }

    fn answer(id: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: ToolResultContent::Text("evidence".into()),
            is_error: None,
        }
    }

    fn image() -> ContentBlock {
        ContentBlock::Image {
            source: claurst_core::types::ImageSource {
                source_type: "base64".into(),
                media_type: Some("image/png".into()),
                data: Some(TINY_PNG.into()),
                url: None,
            },
        }
    }

    fn rejects_media(
        messages: &[Message],
        omitted: &[OmittedComponent],
    ) -> Result<CandidateCost, ContextPlanError> {
        if messages.iter().any(|message| {
            matches!(
                &message.content,
                MessageContent::Blocks(blocks)
                    if blocks.iter().any(|block| {
                        matches!(block, ContentBlock::Image { .. } | ContentBlock::Document { .. })
                    })
            )
        }) {
            return Err(ContextPlanError::MediaPolicy(
                claurst_api::WireBudgetError::UnsupportedMedia,
            ));
        }
        cost(messages, omitted)
    }

    #[test]
    fn finite_fallback_and_precedence_are_visible() {
        let unknown = RequestBudget::resolve(None, None, None, None, 4096, 100_000).unwrap();
        assert_eq!(unknown.context_limit_tokens, UNKNOWN_CONTEXT_LIMIT);
        assert_eq!(unknown.source, BudgetSource::UnknownFallback);
        assert_eq!(
            RequestBudget::resolve(Some(9000), Some(8000), Some(7000), None, 100, 1000)
                .unwrap()
                .source,
            BudgetSource::ExplicitConfig
        );
        assert_eq!(
            RequestBudget::resolve(None, Some(8000), Some(7000), None, 100, 1000)
                .unwrap()
                .source,
            BudgetSource::ProviderRegistry
        );
        assert_eq!(
            RequestBudget::resolve(None, Some(0), Some(7000), None, 100, 1000)
                .unwrap()
                .source,
            BudgetSource::ProviderReported
        );
        assert!(RequestBudget::resolve(Some(0), None, None, None, 0, 1000).is_err());
        assert!(
            RequestBudget::resolve(None, None, None, None, UNKNOWN_CONTEXT_LIMIT, 1000).is_err()
        );
        assert!(RequestBudget::resolve(None, None, None, Some(u64::MAX), 0, 1000).is_err());
    }

    #[test]
    fn parallel_calls_and_cross_round_results_form_one_component() {
        let messages = indexed(vec![
            Message::user("request"),
            Message::assistant_blocks(vec![call("a"), call("b")]),
            Message::user_blocks(vec![answer("b"), answer("a")]),
            Message::assistant("done"),
            Message::user("current"),
        ]);
        let groups = dependency_components(&messages).unwrap();
        assert_eq!(groups[0].indices, vec![0, 1, 2, 3]);
        assert!(!groups[0].malformed);
        let plan = ContextAssembler::plan(&messages, "m-4", budget(10_000), cost).unwrap();
        assert_eq!(
            serde_json::to_vec(&plan.messages).unwrap(),
            serde_json::to_vec(&messages).unwrap()
        );
    }

    #[test]
    fn malformed_historical_pairs_omit_and_mandatory_pairs_refuse() {
        for blocks in [vec![call("a"), call("a")], vec![call("a")]] {
            let messages = indexed(vec![
                Message::user("old"),
                Message::assistant_blocks(blocks),
                Message::user_blocks(vec![answer("a"), answer("a")]),
                Message::assistant("old done"),
                Message::user("current"),
            ]);
            let plan = ContextAssembler::plan(&messages, "m-4", budget(10_000), cost).unwrap();
            assert_eq!(plan.messages.len(), 1);
            assert_eq!(plan.omitted[0].reason, OmissionReason::MalformedToolPair);
            assert!(matches!(
                ContextAssembler::plan(&messages, "m-0", budget(10_000), cost),
                Err(ContextPlanError::MalformedToolPair { .. })
            ));
        }
        for messages in [
            vec![
                Message::user("current"),
                Message::user_blocks(vec![answer("orphan")]),
            ],
            vec![
                Message::user("current"),
                Message::assistant_blocks(vec![call("missing")]),
            ],
        ] {
            let messages = indexed(messages);
            assert!(matches!(
                ContextAssembler::plan(&messages, "m-0", budget(10_000), cost),
                Err(ContextPlanError::MalformedToolPair { .. })
            ));
        }
    }

    #[test]
    fn budget_rejects_whole_pair_and_keeps_exact_canonical_history() {
        let messages = indexed(vec![
            Message::user("x".repeat(5000)),
            Message::assistant_blocks(vec![call("a")]),
            Message::user_blocks(vec![answer("a")]),
            Message::assistant("done"),
            Message::user("current"),
        ]);
        let before = serde_json::to_vec(&messages).unwrap();
        let plan = ContextAssembler::plan(&messages, "m-4", budget(1000), cost).unwrap();
        assert_eq!(plan.messages.len(), 1);
        assert_eq!(
            plan.omitted[0].message_ids,
            vec!["m-0", "m-1", "m-2", "m-3"]
        );
        assert_eq!(before, serde_json::to_vec(&messages).unwrap());
        assert!(plan.measurements <= dependency_components(&messages).unwrap().len() + 1);
    }

    #[test]
    fn schemas_ledger_and_manifest_are_charged_in_measurement() {
        let messages = indexed(vec![Message::user("current")]);
        let mut calls = 0;
        let error =
            ContextAssembler::plan(&messages, "m-0", budget(1000), |messages, omissions| {
                calls += 1;
                let mut measured = cost(messages, omissions)?;
                measured.input_tokens += 2000; // system/ledger/schema/framing supplied by adapter
                Ok(measured)
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ContextPlanError::CurrentRequestTooLarge { .. }
        ));
        assert_eq!(calls, 1);
        let mut wire_budget = budget(1000);
        wire_budget.wire_byte_limit = 1;
        assert!(matches!(
            ContextAssembler::plan(&messages, "m-0", wire_budget, cost),
            Err(ContextPlanError::CurrentRequestTooLarge { .. })
        ));
    }

    #[test]
    fn stable_ids_missing_ids_and_duplicate_ids_are_checked() {
        let messages = indexed(vec![
            Message::user("old"),
            Message::assistant("done"),
            Message::user("current"),
        ]);
        let first = dependency_components(&messages).unwrap();
        let reloaded: Vec<Message> =
            serde_json::from_slice(&serde_json::to_vec(&messages).unwrap()).unwrap();
        assert_eq!(first[0].id, dependency_components(&reloaded).unwrap()[0].id);
        assert!(matches!(
            dependency_components(&[Message::user("no id")]),
            Err(ContextPlanError::MissingOrDuplicateMessageId)
        ));
        let mut duplicates = messages.clone();
        duplicates[1].uuid = duplicates[0].uuid.clone();
        assert!(dependency_components(&duplicates).is_err());
        assert!(matches!(
            ContextAssembler::plan(&messages, "unknown", budget(1000), cost),
            Err(ContextPlanError::MissingCurrentRequest)
        ));
    }

    #[test]
    fn rejected_large_recent_group_does_not_starve_older_small_group() {
        let messages = indexed(vec![
            Message::user("small"),
            Message::assistant("done"),
            Message::user("x".repeat(5000)),
            Message::assistant("done"),
            Message::user("current"),
        ]);
        let plan = ContextAssembler::plan(&messages, "m-4", budget(1000), cost).unwrap();
        assert_eq!(
            plan.messages
                .iter()
                .filter_map(|message| message.uuid.as_deref())
                .collect::<Vec<_>>(),
            vec!["m-0", "m-1", "m-4"]
        );
        assert_eq!(plan.measurements, 3);
        assert_eq!(plan.omitted.len(), 1);
    }

    #[test]
    fn mandatory_unsupported_media_is_typed_and_leaves_history_unchanged() {
        let messages = indexed(vec![Message::user_blocks(vec![image()])]);
        let before = serde_json::to_vec(&messages).unwrap();
        let error =
            ContextAssembler::plan(&messages, "m-0", budget(10_000), rejects_media).unwrap_err();
        assert!(matches!(
            error,
            ContextPlanError::MediaPolicy(claurst_api::WireBudgetError::UnsupportedMedia)
        ));
        assert_eq!(before, serde_json::to_vec(&messages).unwrap());
    }

    #[test]
    fn historical_unsupported_media_omits_whole_group_and_keeps_older_history() {
        let messages = indexed(vec![
            Message::user("older small"),
            Message::assistant("older done"),
            Message::user_blocks(vec![image()]),
            Message::assistant_blocks(vec![call("media-pair")]),
            Message::user_blocks(vec![answer("media-pair")]),
            Message::assistant("media done"),
            Message::user("current"),
        ]);
        let before = serde_json::to_vec(&messages).unwrap();
        let plan = ContextAssembler::plan(&messages, "m-6", budget(10_000), rejects_media).unwrap();
        assert_eq!(
            plan.messages
                .iter()
                .filter_map(|message| message.uuid.as_deref())
                .collect::<Vec<_>>(),
            vec!["m-0", "m-1", "m-6"]
        );
        assert_eq!(plan.omitted[0].reason, OmissionReason::UnsupportedMedia);
        assert_eq!(
            plan.omitted[0].message_ids,
            vec!["m-2", "m-3", "m-4", "m-5"]
        );
        assert_eq!(before, serde_json::to_vec(&messages).unwrap());
    }

    #[test]
    fn final_reselection_excludes_one_complete_historical_group() {
        let messages = indexed(vec![
            Message::user("old request"),
            Message::assistant_blocks(vec![call("pair")]),
            Message::user_blocks(vec![answer("pair")]),
            Message::assistant("old done"),
            Message::user("current"),
        ]);
        let group = dependency_components(&messages).unwrap()[0].id.clone();
        let excluded = BTreeSet::from([group.clone()]);
        let plan =
            ContextAssembler::plan_with_excluded(&messages, "m-4", budget(10_000), &excluded, cost)
                .unwrap();
        assert_eq!(
            plan.messages
                .iter()
                .filter_map(|message| message.uuid.as_deref())
                .collect::<Vec<_>>(),
            vec!["m-4"]
        );
        assert_eq!(
            plan.omitted[0].message_ids,
            vec!["m-0", "m-1", "m-2", "m-3"]
        );

        let selected = ContextAssembler::plan(&messages, "m-4", budget(10_000), cost).unwrap();
        assert_eq!(selected.optional_selected_groups, vec![group]);
    }

    #[test]
    fn current_request_is_user_content_and_overflow_carries_budget() {
        for messages in [
            vec![Message::assistant("not a user")],
            vec![Message::user(" ")],
            vec![Message::user_blocks(vec![answer("tool-only")])],
        ] {
            let messages = indexed(messages);
            let mut measurements = 0;
            let result =
                ContextAssembler::plan(&messages, "m-0", budget(1000), |messages, omitted| {
                    measurements += 1;
                    cost(messages, omitted)
                });
            assert!(matches!(
                result,
                Err(ContextPlanError::InvalidCurrentRequest)
            ));
            assert_eq!(measurements, 0);
        }
        let messages = indexed(vec![Message::user("large".repeat(1000))]);
        let mut invalid = budget(1000);
        invalid.input_limit_tokens = u64::MAX;
        assert!(matches!(
            ContextAssembler::plan(&messages, "m-0", invalid, cost),
            Err(ContextPlanError::InvalidBudget)
        ));
        let expected = budget(1000);
        match ContextAssembler::plan(&messages, "m-0", expected.clone(), cost).unwrap_err() {
            ContextPlanError::CurrentRequestTooLarge {
                budget,
                input_tokens,
                ..
            } => {
                assert_eq!(budget, expected);
                assert!(input_tokens > budget.input_limit_tokens);
            }
            error => panic!("unexpected error: {error}"),
        }
    }
}
