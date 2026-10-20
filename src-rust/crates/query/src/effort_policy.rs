//! Host-owned effort and wire-pin provenance.
//!
//! These types deliberately have no serde representation. They are assembled
//! by trusted host code before a provider request is built; model-visible
//! inputs must not be able to mint or alter them.

use claurst_core::effort::EffortLevel;
use std::fmt;

/// Where a value came from before defaults and normalization were applied.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueSource {
    Default,
    ExplicitUser,
    Settings,
    Agent,
    InlineKeyword,
    VerifiedBase,
    VerifiedContinuation,
    Unknown,
}

impl fmt::Debug for ValueSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Default => "Default",
            Self::ExplicitUser => "ExplicitUser",
            Self::Settings => "Settings",
            Self::Agent => "Agent",
            Self::InlineKeyword => "InlineKeyword",
            Self::VerifiedBase => "VerifiedBase",
            Self::VerifiedContinuation => "VerifiedContinuation",
            Self::Unknown => "Unknown",
        })
    }
}

/// A value source together with whether a raw value existed before
/// normalization populated a default or provider-specific representation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueProvenance {
    source: ValueSource,
    raw_present: bool,
}

impl Default for ValueProvenance {
    fn default() -> Self {
        Self::new(ValueSource::Default, false)
    }
}

impl ValueProvenance {
    /// Construct host-owned provenance for one normalized field.
    pub const fn new(source: ValueSource, raw_present: bool) -> Self {
        Self {
            source,
            raw_present,
        }
    }

    pub const fn source(self) -> ValueSource {
        self.source
    }

    pub const fn raw_present(self) -> bool {
        self.raw_present
    }
}

impl fmt::Debug for ValueProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValueProvenance")
            .field("source", &self.source)
            .field("raw_present", &self.raw_present)
            .finish()
    }
}

/// Why a wire value is pinned and cannot be adapted by effort policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinReason {
    Effort,
    ThinkingBudget,
    Temperature,
    AgentTemperature,
    InlineKeyword,
    UnknownWireOverride,
}

/// Provenance for effort, thinking budget, temperature, and model selection.
///
/// Model provenance is tracked separately and intentionally does not block
/// wire-level effort adaptation within that selected model.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinProvenance {
    effort: ValueProvenance,
    thinking_budget: ValueProvenance,
    temperature: ValueProvenance,
    model_source: ValueProvenance,
}

impl PinProvenance {
    /// Construct host-owned provenance before normalized request values are
    /// passed to a provider adapter.
    pub const fn new(
        effort: ValueProvenance,
        thinking_budget: ValueProvenance,
        temperature: ValueProvenance,
        model_source: ValueProvenance,
    ) -> Self {
        Self {
            effort,
            thinking_budget,
            temperature,
            model_source,
        }
    }

    pub const fn effort(self) -> ValueProvenance {
        self.effort
    }

    pub const fn thinking_budget(self) -> ValueProvenance {
        self.thinking_budget
    }

    pub const fn temperature(self) -> ValueProvenance {
        self.temperature
    }

    pub const fn model_source(self) -> ValueProvenance {
        self.model_source
    }

    /// Replace the source captured for the normalized effort field.
    pub const fn with_effort(self, provenance: ValueProvenance) -> Self {
        Self {
            effort: provenance,
            ..self
        }
    }

    /// Replace the source captured for the normalized thinking-budget field.
    pub const fn with_thinking_budget(self, provenance: ValueProvenance) -> Self {
        Self {
            thinking_budget: provenance,
            ..self
        }
    }

    /// Replace the source captured for the normalized temperature field.
    pub const fn with_temperature(self, provenance: ValueProvenance) -> Self {
        Self {
            temperature: provenance,
            ..self
        }
    }

    /// Replace the source captured for the selected model.
    pub const fn with_model_source(self, provenance: ValueProvenance) -> Self {
        Self {
            model_source: provenance,
            ..self
        }
    }

    /// Overlay any sources known at a later trusted host boundary.
    pub const fn overlay(
        self,
        effort: Option<ValueProvenance>,
        thinking_budget: Option<ValueProvenance>,
        temperature: Option<ValueProvenance>,
        model_source: Option<ValueProvenance>,
    ) -> Self {
        Self {
            effort: match effort {
                Some(provenance) => provenance,
                None => self.effort,
            },
            thinking_budget: match thinking_budget {
                Some(provenance) => provenance,
                None => self.thinking_budget,
            },
            temperature: match temperature {
                Some(provenance) => provenance,
                None => self.temperature,
            },
            model_source: match model_source {
                Some(provenance) => provenance,
                None => self.model_source,
            },
        }
    }

    /// Mark populated fields that still carry an untouched default as unknown.
    ///
    /// This is the fail-closed bridge for callers that already computed a wire
    /// value before provenance was threaded through. Verified and explicit
    /// sources remain unchanged.
    pub const fn conservative_for_present_wire(
        self,
        effort_present: bool,
        thinking_budget_present: bool,
        temperature_present: bool,
    ) -> Self {
        Self {
            effort: conservative_present(self.effort, effort_present),
            thinking_budget: conservative_present(self.thinking_budget, thinking_budget_present),
            temperature: conservative_present(self.temperature, temperature_present),
            model_source: self.model_source,
        }
    }

    /// Return the first categorical reason that blocks wire adaptation.
    pub const fn blocking_pin(self) -> Option<PinReason> {
        if let Some(reason) = blocking_field(self.effort, PinReason::Effort, false) {
            return Some(reason);
        }
        if let Some(reason) = blocking_field(self.thinking_budget, PinReason::ThinkingBudget, false)
        {
            return Some(reason);
        }
        if let Some(reason) = blocking_field(self.temperature, PinReason::Temperature, true) {
            return Some(reason);
        }
        None
    }
}

const fn conservative_present(value: ValueProvenance, present: bool) -> ValueProvenance {
    if present && matches!(value.source, ValueSource::Default) && !value.raw_present {
        ValueProvenance::new(ValueSource::Unknown, true)
    } else {
        value
    }
}

impl Default for PinProvenance {
    fn default() -> Self {
        Self::new(
            ValueProvenance::default(),
            ValueProvenance::default(),
            ValueProvenance::default(),
            ValueProvenance::default(),
        )
    }
}

const fn blocking_field(
    value: ValueProvenance,
    ordinary_reason: PinReason,
    agent_temperature: bool,
) -> Option<PinReason> {
    match value.source {
        ValueSource::ExplicitUser | ValueSource::Settings => Some(ordinary_reason),
        ValueSource::Agent if agent_temperature => Some(PinReason::AgentTemperature),
        ValueSource::Agent if value.raw_present => Some(ordinary_reason),
        ValueSource::InlineKeyword => Some(PinReason::InlineKeyword),
        ValueSource::Unknown if value.raw_present => Some(PinReason::UnknownWireOverride),
        ValueSource::Default
        | ValueSource::Agent
        | ValueSource::VerifiedBase
        | ValueSource::VerifiedContinuation
        | ValueSource::Unknown => None,
    }
}

impl fmt::Debug for PinProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PinProvenance")
            .field("effort", &self.effort)
            .field("thinking_budget", &self.thinking_budget)
            .field("temperature", &self.temperature)
            .field("model_source", &self.model_source)
            .finish()
    }
}

/// A bounded, host-verified effort ladder.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifiedEffortPolicy {
    base: EffortLevel,
    continuation: EffortLevel,
}

/// Failure to construct a strict descending verified effort ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedEffortPolicyError {
    ContinuationMustBeLower,
}

impl fmt::Display for VerifiedEffortPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("verified continuation effort must be lower than base effort")
    }
}

impl std::error::Error for VerifiedEffortPolicyError {}

impl VerifiedEffortPolicy {
    pub fn new(
        base: EffortLevel,
        continuation: EffortLevel,
    ) -> Result<Self, VerifiedEffortPolicyError> {
        if continuation >= base {
            return Err(VerifiedEffortPolicyError::ContinuationMustBeLower);
        }
        Ok(Self { base, continuation })
    }

    pub const fn base(self) -> EffortLevel {
        self.base
    }

    pub const fn continuation(self) -> EffortLevel {
        self.continuation
    }
}

impl fmt::Debug for VerifiedEffortPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VerifiedEffortPolicy(..)")
    }
}

/// Effort behavior selected by the host.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum EffortPolicy {
    #[default]
    Fixed,
    Verified(VerifiedEffortPolicy),
}

impl EffortPolicy {
    pub fn verified(
        base: EffortLevel,
        continuation: EffortLevel,
    ) -> Result<Self, VerifiedEffortPolicyError> {
        VerifiedEffortPolicy::new(base, continuation).map(Self::Verified)
    }

    pub const fn verified_policy(self) -> Option<VerifiedEffortPolicy> {
        match self {
            Self::Fixed => None,
            Self::Verified(policy) => Some(policy),
        }
    }
}

impl fmt::Debug for EffortPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed => formatter.write_str("EffortPolicy::Fixed"),
            Self::Verified(_) => formatter.write_str("EffortPolicy::Verified(..)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT: ValueProvenance = ValueProvenance::new(ValueSource::Default, false);
    const MODEL: ValueProvenance = ValueProvenance::new(ValueSource::ExplicitUser, true);

    fn provenance(
        effort: ValueProvenance,
        thinking_budget: ValueProvenance,
        temperature: ValueProvenance,
    ) -> PinProvenance {
        PinProvenance::new(effort, thinking_budget, temperature, MODEL)
    }

    #[test]
    fn source_and_raw_presence_survive_default_normalization() {
        let raw = ValueProvenance::new(ValueSource::Unknown, true);
        let normalized = ValueProvenance::new(ValueSource::Default, false);
        assert_eq!(raw.source(), ValueSource::Unknown);
        assert!(raw.raw_present());
        assert_eq!(normalized.source(), ValueSource::Default);
        assert!(!normalized.raw_present());
        assert_eq!(
            provenance(raw, DEFAULT, DEFAULT).blocking_pin(),
            Some(PinReason::UnknownWireOverride)
        );
    }

    #[test]
    fn default_provenance_is_unpinned_until_a_wire_value_appears() {
        let provenance = PinProvenance::default();
        assert_eq!(provenance.effort(), DEFAULT);
        assert_eq!(provenance.thinking_budget(), DEFAULT);
        assert_eq!(provenance.temperature(), DEFAULT);
        assert_eq!(provenance.model_source(), DEFAULT);
        assert_eq!(provenance.blocking_pin(), None);
        assert_eq!(
            provenance
                .conservative_for_present_wire(true, false, false)
                .blocking_pin(),
            Some(PinReason::UnknownWireOverride)
        );
    }

    #[test]
    fn overlays_replace_only_the_sources_the_host_knows() {
        let explicit = ValueProvenance::new(ValueSource::ExplicitUser, true);
        let model = ValueProvenance::new(ValueSource::Settings, true);
        let provenance = PinProvenance::default()
            .with_effort(explicit)
            .with_model_source(model);
        assert_eq!(provenance.effort(), explicit);
        assert_eq!(provenance.model_source(), model);
        assert_eq!(provenance.thinking_budget(), DEFAULT);
        assert_eq!(
            provenance
                .overlay(None, Some(explicit), None, None)
                .thinking_budget(),
            explicit
        );
    }

    #[test]
    fn unknown_present_wire_override_fails_closed() {
        assert_eq!(
            provenance(
                ValueProvenance::new(ValueSource::Unknown, true),
                DEFAULT,
                DEFAULT,
            )
            .blocking_pin(),
            Some(PinReason::UnknownWireOverride)
        );
        assert_eq!(
            provenance(
                ValueProvenance::new(ValueSource::Unknown, false),
                DEFAULT,
                DEFAULT,
            )
            .blocking_pin(),
            None
        );
    }

    #[test]
    fn default_and_verified_sources_do_not_pin() {
        let verified = ValueProvenance::new(ValueSource::VerifiedBase, true);
        assert_eq!(provenance(DEFAULT, DEFAULT, DEFAULT).blocking_pin(), None);
        assert_eq!(
            provenance(
                verified,
                ValueProvenance::new(ValueSource::VerifiedContinuation, true),
                DEFAULT
            )
            .blocking_pin(),
            None
        );
    }

    #[test]
    fn each_explicit_wire_source_blocks() {
        for source in [ValueSource::ExplicitUser, ValueSource::Settings] {
            assert_eq!(
                provenance(ValueProvenance::new(source, true), DEFAULT, DEFAULT).blocking_pin(),
                Some(PinReason::Effort)
            );
            assert_eq!(
                provenance(DEFAULT, ValueProvenance::new(source, true), DEFAULT).blocking_pin(),
                Some(PinReason::ThinkingBudget)
            );
            assert_eq!(
                provenance(DEFAULT, DEFAULT, ValueProvenance::new(source, true)).blocking_pin(),
                Some(PinReason::Temperature)
            );
        }
    }

    #[test]
    fn agent_temperature_and_inline_keyword_block() {
        assert_eq!(
            provenance(
                DEFAULT,
                DEFAULT,
                ValueProvenance::new(ValueSource::Agent, false)
            )
            .blocking_pin(),
            Some(PinReason::AgentTemperature)
        );
        assert_eq!(
            provenance(
                ValueProvenance::new(ValueSource::InlineKeyword, false),
                DEFAULT,
                DEFAULT,
            )
            .blocking_pin(),
            Some(PinReason::InlineKeyword)
        );
    }

    #[test]
    fn raw_agent_effort_and_budget_overrides_block() {
        let raw_agent = ValueProvenance::new(ValueSource::Agent, true);
        assert_eq!(
            provenance(raw_agent, DEFAULT, DEFAULT).blocking_pin(),
            Some(PinReason::Effort)
        );
        assert_eq!(
            provenance(DEFAULT, raw_agent, DEFAULT).blocking_pin(),
            Some(PinReason::ThinkingBudget)
        );
        assert_eq!(
            provenance(
                ValueProvenance::new(ValueSource::Agent, false),
                DEFAULT,
                DEFAULT,
            )
            .blocking_pin(),
            None
        );
    }

    #[test]
    fn explicit_model_source_is_not_a_wire_pin() {
        let model = ValueProvenance::new(ValueSource::ExplicitUser, true);
        assert_eq!(
            PinProvenance::new(DEFAULT, DEFAULT, DEFAULT, model).blocking_pin(),
            None
        );
        assert_eq!(
            PinProvenance::new(DEFAULT, DEFAULT, DEFAULT, model).model_source(),
            model
        );
    }

    #[test]
    fn verified_policy_requires_strictly_lower_continuation() {
        let policy = VerifiedEffortPolicy::new(EffortLevel::High, EffortLevel::Medium).unwrap();
        assert_eq!(policy.base(), EffortLevel::High);
        assert_eq!(policy.continuation(), EffortLevel::Medium);
        assert!(VerifiedEffortPolicy::new(EffortLevel::High, EffortLevel::High).is_err());
        assert!(VerifiedEffortPolicy::new(EffortLevel::High, EffortLevel::Max).is_err());
        assert!(VerifiedEffortPolicy::new(EffortLevel::None, EffortLevel::None).is_err());
        assert!(EffortPolicy::verified(EffortLevel::Max, EffortLevel::High).is_ok());
    }
}
