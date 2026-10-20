//! Focused regression coverage for the host-owned effort selection boundary.
//!
//! The transport loopback fixtures live with the API adapters.  These tests
//! pin the query-side contract: selection events are categorical, defaults do
//! not manufacture a verified policy, and the turn-local keyword override is
//! deterministic before any provider request is assembled.

use super::{
    effective_effort_for_turn, EffortSelection, EffortSelectionMode, EffortSelectionReason,
};
use crate::effort_policy::{EffortPolicy, PinProvenance};
use claurst_core::effort::EffortLevel;
use claurst_core::types::{ContentBlock, Message, ToolResultContent};

#[test]
fn fixed_selection_event_is_categorical_and_contains_no_transport_data() {
    let selection = EffortSelection::new(
        EffortSelectionMode::Fixed,
        EffortSelectionMode::Fixed,
        Some(EffortLevel::High),
        EffortSelectionReason::FixedPolicy,
    );

    assert_eq!(selection.requested(), EffortSelectionMode::Fixed);
    assert_eq!(selection.applied(), EffortSelectionMode::Fixed);
    assert_eq!(selection.selected(), Some(EffortLevel::High));
    assert_eq!(selection.reason(), EffortSelectionReason::FixedPolicy);
    assert_eq!(selection.status_line(), "Effort: high (fixed; default)");
    assert!(selection.is_default_fixed());
}

#[test]
fn effort_status_is_stable_for_explicit_and_verified_selection() {
    let explicit = EffortSelection::new(
        EffortSelectionMode::Fixed,
        EffortSelectionMode::Fixed,
        Some(EffortLevel::High),
        EffortSelectionReason::Explicit,
    );
    assert_eq!(explicit.status_line(), "Effort: high (fixed; explicit)");
    assert!(!explicit.is_default_fixed());

    let verified = EffortSelection::new(
        EffortSelectionMode::Verified,
        EffortSelectionMode::Verified,
        Some(EffortLevel::Medium),
        EffortSelectionReason::Changed,
    );
    assert_eq!(verified.status_line(), "Effort: medium (verified; changed)");
}

#[test]
fn verified_policy_is_host_owned_and_requires_a_lower_continuation() {
    let policy = EffortPolicy::verified(EffortLevel::High, EffortLevel::Medium)
        .expect("strictly lower continuation should be accepted");
    let verified = policy
        .verified_policy()
        .expect("verified policy should retain its host-owned ladder");

    assert_eq!(verified.base(), EffortLevel::High);
    assert_eq!(verified.continuation(), EffortLevel::Medium);
    assert!(EffortPolicy::verified(EffortLevel::Medium, EffortLevel::High).is_err());
    assert_eq!(PinProvenance::default().blocking_pin(), None);
}

#[test]
fn ultracode_override_is_scoped_to_the_latest_user_turn() {
    let ordinary = [Message::user("review this change")];
    assert_eq!(
        effective_effort_for_turn(Some(EffortLevel::Low), &ordinary),
        Some(EffortLevel::Low)
    );

    let ultracode = [Message::user("please use ultracode for this turn")];
    assert_eq!(
        effective_effort_for_turn(Some(EffortLevel::Low), &ultracode),
        Some(EffortLevel::Ultracode)
    );

    let later_plain = [
        Message::user("please use ultracode for this turn"),
        Message::assistant("done"),
        Message::user("continue normally"),
    ];
    assert_eq!(
        effective_effort_for_turn(Some(EffortLevel::Low), &later_plain),
        Some(EffortLevel::Low)
    );
}

#[test]
fn tool_results_cannot_clear_or_create_the_work_level_pin() {
    let tool_round = [
        Message::user("ultracode inspect this change"),
        Message::assistant("calling a tool"),
        Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "call-1".into(),
            content: ToolResultContent::Text("ordinary tool output".into()),
            is_error: None,
        }]),
    ];
    assert!(super::initial_inline_ultracode_state(&tool_round));

    let tool_only = [Message::user_blocks(vec![ContentBlock::ToolResult {
        tool_use_id: "call-2".into(),
        content: ToolResultContent::Text("ultracode appears in tool output".into()),
        is_error: None,
    }])];
    assert!(!super::initial_inline_ultracode_state(&tool_only));

    let config = super::QueryConfig {
        effort_policy: EffortPolicy::verified(EffortLevel::High, EffortLevel::Medium)
            .expect("verified test policy"),
        ..Default::default()
    };
    let pinned = super::effective_pin_provenance_for_inline(
        &config,
        Some(EffortLevel::Ultracode),
        EffortLevel::Ultracode.thinking_budget_tokens(),
        EffortLevel::Ultracode.temperature(),
        true,
    );
    assert_eq!(
        pinned.effort().source(),
        crate::effort_policy::ValueSource::InlineKeyword
    );
}
