//! Bounded, content-free evidence for the exact Codex request bytes sent.
//!
//! This module deliberately keeps client-wire authority separate from native
//! response observations.  It never stores request content, credentials, or
//! requested configuration.

use claurst_core::crypto_utils::sha256_hex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider_wire::PreparedWireRequest;

const MAX_WITNESS_STRING_BYTES: usize = 256;
const MAX_WITNESS_ATTEMPTS: usize = 2;
const MAX_WITNESS_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireEffort {
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

impl WireEffort {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessBindingStatus {
    Matched,
    Mismatch,
    Missing,
    Malformed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireAttemptOutcome {
    Accepted,
    RejectedContinuation,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProviderWireAttempt {
    pub attempt_index: u8,
    pub continuation: bool,
    pub outcome: WireAttemptOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire_effort: Option<WireEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire_max_output_tokens: Option<u32>,
    pub wire_sha256: String,
    pub wire_bytes: u64,
    pub wire_fields_valid: bool,
    pub wire_sha256_valid: bool,
    pub wire_bytes_valid: bool,
    pub http_status: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProviderWireWitness {
    pub route_provider: String,
    pub native_adapter: String,
    pub wire_authority: String,
    pub server_authority: String,
    pub attempts: Vec<ProviderWireAttempt>,
    pub native_response_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire_effort: Option<WireEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_effort: Option<WireEffort>,
    pub binding_status: WitnessBindingStatus,
}

#[derive(Clone, Debug)]
pub struct PreparedWitnessAttempt {
    attempt_index: u8,
    continuation: bool,
    wire_model: Option<String>,
    wire_effort: Option<WireEffort>,
    wire_max_output_tokens: Option<u32>,
    wire_sha256: String,
    wire_bytes: u64,
    wire_fields_valid: bool,
    wire_sha256_valid: bool,
    wire_bytes_valid: bool,
}

impl PreparedWitnessAttempt {
    /// Extract only bounded fields from the immutable bytes that are handed to
    /// the transport.  A parse failure is represented in the attempt rather
    /// than changing the request's normal send behavior.
    pub fn from_prepared(
        prepared: &PreparedWireRequest,
        attempt_index: u8,
        continuation: bool,
    ) -> Self {
        let body = prepared.bytes();
        let parsed = serde_json::from_slice::<Value>(body).ok();
        let wire_model = parsed
            .as_ref()
            .and_then(|value| value.get("model"))
            .and_then(Value::as_str)
            .filter(|value| bounded_string(value))
            .map(ToOwned::to_owned);
        let wire_effort = parsed
            .as_ref()
            .and_then(|value| value.get("reasoning"))
            .and_then(|value| value.get("effort"))
            .and_then(Value::as_str)
            .and_then(WireEffort::parse);
        let wire_max_output_tokens = parsed
            .as_ref()
            .and_then(|value| value.get("max_output_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0);
        let wire_fields_valid = parsed.is_some()
            && wire_model.is_some()
            && parsed
                .as_ref()
                .and_then(|value| value.get("reasoning"))
                .and_then(|value| value.get("effort"))
                .and_then(Value::as_str)
                .is_some_and(|value| WireEffort::parse(value).is_some())
            && wire_max_output_tokens.is_some();

        Self {
            attempt_index,
            continuation,
            wire_model,
            wire_effort,
            wire_max_output_tokens,
            wire_sha256: prepared.wire_sha256.clone(),
            wire_bytes: prepared.wire_bytes,
            wire_fields_valid,
            wire_sha256_valid: is_sha256(&prepared.wire_sha256)
                && prepared.wire_sha256 == sha256_hex(body),
            wire_bytes_valid: prepared.wire_bytes == body.len() as u64,
        }
    }

    pub fn finish(self, http_status: u16, outcome: WireAttemptOutcome) -> ProviderWireAttempt {
        ProviderWireAttempt {
            attempt_index: self.attempt_index,
            continuation: self.continuation,
            outcome,
            wire_model: self.wire_model,
            wire_effort: self.wire_effort,
            wire_max_output_tokens: self.wire_max_output_tokens,
            wire_sha256: self.wire_sha256,
            wire_bytes: self.wire_bytes,
            wire_fields_valid: self.wire_fields_valid,
            wire_sha256_valid: self.wire_sha256_valid,
            wire_bytes_valid: self.wire_bytes_valid,
            http_status: Some(http_status),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WitnessSeedError {
    AttemptCount,
    AttemptIndex,
    Outcome,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderWireWitnessSeed {
    attempts: Vec<ProviderWireAttempt>,
}

impl ProviderWireWitnessSeed {
    /// Validate the complete set of physical sends.  This rejects overflow,
    /// duplicate/non-contiguous indices, contradictory outcomes, and missing
    /// accepted attempts instead of silently truncating or manufacturing one.
    pub fn try_new(mut attempts: Vec<ProviderWireAttempt>) -> Result<Self, WitnessSeedError> {
        if attempts.is_empty() || attempts.len() > MAX_WITNESS_ATTEMPTS {
            return Err(WitnessSeedError::AttemptCount);
        }
        attempts.sort_by_key(|attempt| attempt.attempt_index);
        if attempts
            .iter()
            .enumerate()
            .any(|(expected, attempt)| attempt.attempt_index as usize != expected)
        {
            return Err(WitnessSeedError::AttemptIndex);
        }
        let accepted = attempts
            .iter()
            .filter(|attempt| matches!(attempt.outcome, WireAttemptOutcome::Accepted))
            .count();
        let every_attempt_valid = attempts.iter().all(|attempt| {
            bounded_string_option(attempt.wire_model.as_deref())
                && attempt.wire_effort.is_some()
                && attempt
                    .wire_max_output_tokens
                    .is_some_and(|value| value > 0)
                && is_sha256(&attempt.wire_sha256)
                && attempt.wire_bytes > 0
                && attempt.wire_bytes <= MAX_WITNESS_BYTES
                && attempt.wire_fields_valid
                && attempt.wire_sha256_valid
                && attempt.wire_bytes_valid
                && attempt.http_status.is_some()
        });
        let exact_sequence = match attempts.as_slice() {
            [accepted] => {
                matches!(accepted.outcome, WireAttemptOutcome::Accepted)
                    && accepted
                        .http_status
                        .is_some_and(|status| (200..300).contains(&status))
            }
            [rejected, accepted] => {
                matches!(rejected.outcome, WireAttemptOutcome::RejectedContinuation)
                    && rejected.continuation
                    && rejected.http_status.is_some_and(|status| {
                        (100..600).contains(&status) && !(200..300).contains(&status)
                    })
                    && matches!(accepted.outcome, WireAttemptOutcome::Accepted)
                    && !accepted.continuation
                    && accepted
                        .http_status
                        .is_some_and(|status| (200..300).contains(&status))
                    && rejected.wire_model == accepted.wire_model
                    && rejected.wire_effort == accepted.wire_effort
                    && rejected.wire_max_output_tokens == accepted.wire_max_output_tokens
            }
            _ => false,
        };
        if accepted != 1 || !every_attempt_valid || !exact_sequence {
            return Err(WitnessSeedError::Outcome);
        }
        Ok(Self { attempts })
    }

    pub fn accepted_attempt(&self) -> Option<&ProviderWireAttempt> {
        self.attempts
            .iter()
            .find(|attempt| matches!(attempt.outcome, WireAttemptOutcome::Accepted))
    }
}

#[derive(Clone, Debug, Default)]
pub struct NativeResponseBinding {
    response_id: Option<String>,
    model: Option<String>,
    effort: Option<WireEffort>,
    completed_id: Option<String>,
    completed_model: Option<String>,
    completed: bool,
    invalid: bool,
    mismatch: bool,
}

impl NativeResponseBinding {
    pub fn observe(&mut self, response: Option<&Value>, completed: bool) {
        if completed && self.completed {
            self.mismatch = true;
        }
        let Some(response) = response.filter(|value| value.is_object()) else {
            self.invalid = true;
            if completed {
                self.completed = false;
            }
            return;
        };
        let Some(id) = response.get("id").and_then(Value::as_str) else {
            self.invalid = true;
            if completed {
                self.completed = false;
            }
            return;
        };
        let Some(model) = response.get("model").and_then(Value::as_str) else {
            self.invalid = true;
            if completed {
                self.completed = false;
            }
            return;
        };
        if !bounded_string(id) || !bounded_string(model) {
            self.invalid = true;
        }
        if record_consistent(&mut self.response_id, id) {
            self.mismatch = true;
        }
        if record_consistent(&mut self.model, model) {
            self.mismatch = true;
        }

        if let Some(raw_effort) = response
            .get("reasoning")
            .and_then(|value| value.get("effort"))
        {
            let Some(effort) = raw_effort.as_str().and_then(WireEffort::parse) else {
                self.invalid = true;
                return;
            };
            if let Some(previous) = &self.effort {
                if previous != &effort {
                    self.mismatch = true;
                }
            } else {
                self.effort = Some(effort);
            }
        }

        if completed {
            self.completed = true;
            self.completed_id = Some(id.to_owned());
            self.completed_model = Some(model.to_owned());
        }
    }

    pub fn finalize(&self, seed: &ProviderWireWitnessSeed) -> Option<ProviderWireWitness> {
        if !self.completed
            || self
                .completed_id
                .as_deref()
                .is_none_or(|value| !bounded_string(value))
            || self
                .completed_model
                .as_deref()
                .is_none_or(|value| !bounded_string(value))
        {
            return None;
        }
        let accepted = seed.accepted_attempt()?;
        let mut status = if self.invalid {
            WitnessBindingStatus::Malformed
        } else if self.mismatch {
            WitnessBindingStatus::Mismatch
        } else {
            WitnessBindingStatus::Matched
        };
        if !accepted.wire_fields_valid || !accepted.wire_sha256_valid || !accepted.wire_bytes_valid
        {
            status = WitnessBindingStatus::Malformed;
        } else if accepted.wire_model.as_ref() != self.completed_model.as_ref()
            || (self.effort.is_some() && accepted.wire_effort.as_ref() != self.effort.as_ref())
            || accepted.wire_max_output_tokens.is_none()
        {
            status = WitnessBindingStatus::Mismatch;
        }

        Some(ProviderWireWitness {
            route_provider: "codex".to_owned(),
            native_adapter: "codex_responses".to_owned(),
            wire_authority: "serialized_request".to_owned(),
            server_authority: if self.model.is_some() || self.effort.is_some() {
                "native_response_field".to_owned()
            } else {
                "unavailable".to_owned()
            },
            attempts: seed.attempts.clone(),
            native_response_id: self.completed_id.clone()?,
            wire_model: accepted.wire_model.clone(),
            wire_effort: accepted.wire_effort,
            request_max_output_tokens: accepted.wire_max_output_tokens,
            server_model: self.completed_model.clone(),
            server_effort: self.effort,
            binding_status: status,
        })
    }
}

fn bounded_string(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_WITNESS_STRING_BYTES
}

fn bounded_string_option(value: Option<&str>) -> bool {
    value.is_some_and(bounded_string)
}

fn record_consistent(slot: &mut Option<String>, value: &str) -> bool {
    if let Some(previous) = slot {
        previous != value
    } else {
        *slot = Some(value.to_owned());
        false
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(index: u8, continuation: bool, outcome: WireAttemptOutcome) -> ProviderWireAttempt {
        ProviderWireAttempt {
            attempt_index: index,
            continuation,
            outcome,
            wire_model: Some("fixture-model".to_owned()),
            wire_effort: Some(WireEffort::XHigh),
            wire_max_output_tokens: Some(512),
            wire_sha256: "0".repeat(64),
            wire_bytes: 1,
            wire_fields_valid: true,
            wire_sha256_valid: true,
            wire_bytes_valid: true,
            http_status: Some(200),
        }
    }

    #[test]
    fn accepts_all_codex_wire_efforts_including_xhigh() {
        assert_eq!(WireEffort::parse("low"), Some(WireEffort::Low));
        assert_eq!(WireEffort::parse("medium"), Some(WireEffort::Medium));
        assert_eq!(WireEffort::parse("high"), Some(WireEffort::High));
        assert_eq!(WireEffort::parse("xhigh"), Some(WireEffort::XHigh));
        assert_eq!(WireEffort::parse("max"), None);
        assert_eq!(
            serde_json::to_string(&WireEffort::XHigh).unwrap(),
            "\"xhigh\""
        );
    }

    #[test]
    fn rejects_overflow_duplicate_indices_and_contradictory_outcomes() {
        assert_eq!(
            ProviderWireWitnessSeed::try_new(vec![
                attempt(0, false, WireAttemptOutcome::Accepted),
                attempt(1, false, WireAttemptOutcome::Failed),
                attempt(2, false, WireAttemptOutcome::Failed),
            ]),
            Err(WitnessSeedError::AttemptCount)
        );
        assert_eq!(
            ProviderWireWitnessSeed::try_new(vec![
                attempt(0, false, WireAttemptOutcome::Accepted),
                attempt(0, true, WireAttemptOutcome::RejectedContinuation),
            ]),
            Err(WitnessSeedError::AttemptIndex)
        );
        assert_eq!(
            ProviderWireWitnessSeed::try_new(vec![
                attempt(0, false, WireAttemptOutcome::Accepted),
                attempt(1, true, WireAttemptOutcome::Accepted),
            ]),
            Err(WitnessSeedError::Outcome)
        );
        assert_eq!(
            ProviderWireWitnessSeed::try_new(vec![attempt(0, false, WireAttemptOutcome::Failed),]),
            Err(WitnessSeedError::Outcome)
        );
        let mut malformed_first = attempt(0, true, WireAttemptOutcome::RejectedContinuation);
        malformed_first.wire_fields_valid = false;
        assert_eq!(
            ProviderWireWitnessSeed::try_new(vec![
                malformed_first,
                attempt(1, false, WireAttemptOutcome::Accepted),
            ]),
            Err(WitnessSeedError::Outcome)
        );
        assert_eq!(
            ProviderWireWitnessSeed::try_new(vec![
                attempt(0, false, WireAttemptOutcome::RejectedContinuation),
                attempt(1, false, WireAttemptOutcome::Accepted),
            ]),
            Err(WitnessSeedError::Outcome)
        );
        for status in [0, 99, 600, u16::MAX] {
            let mut rejected = attempt(0, true, WireAttemptOutcome::RejectedContinuation);
            rejected.http_status = Some(status);
            assert_eq!(
                ProviderWireWitnessSeed::try_new(vec![
                    rejected,
                    attempt(1, false, WireAttemptOutcome::Accepted),
                ]),
                Err(WitnessSeedError::Outcome)
            );
        }
    }

    #[test]
    fn late_native_conflict_is_mismatch_and_missing_completed_identity_is_unavailable() {
        let seed =
            ProviderWireWitnessSeed::try_new(vec![attempt(0, false, WireAttemptOutcome::Accepted)])
                .expect("valid seed");
        let mut binding = NativeResponseBinding::default();
        binding.observe(
            Some(&serde_json::json!({"id":"resp-1","model":"fixture-model"})),
            false,
        );
        binding.observe(
            Some(&serde_json::json!({"id":"resp-2","model":"fixture-model"})),
            false,
        );
        binding.observe(
            Some(&serde_json::json!({"id":"resp-2","model":"fixture-model"})),
            true,
        );
        assert_eq!(
            binding.finalize(&seed).expect("witness").binding_status,
            WitnessBindingStatus::Mismatch
        );

        let mut missing = NativeResponseBinding::default();
        missing.observe(
            Some(&serde_json::json!({"id":"resp-1","model":"fixture-model"})),
            false,
        );
        missing.observe(Some(&serde_json::json!({"model":"fixture-model"})), true);
        assert!(missing.finalize(&seed).is_none());
    }

    #[test]
    fn late_server_effort_conflict_is_mismatch() {
        let seed =
            ProviderWireWitnessSeed::try_new(vec![attempt(0, false, WireAttemptOutcome::Accepted)])
                .expect("valid seed");
        let mut binding = NativeResponseBinding::default();
        binding.observe(
            Some(&serde_json::json!({
                "id":"resp-1","model":"fixture-model","reasoning":{"effort":"high"}
            })),
            false,
        );
        binding.observe(
            Some(&serde_json::json!({
                "id":"resp-1","model":"fixture-model","reasoning":{"effort":"xhigh"}
            })),
            true,
        );
        assert_eq!(
            binding.finalize(&seed).expect("witness").binding_status,
            WitnessBindingStatus::Mismatch
        );
    }

    #[test]
    fn duplicate_completed_native_identity_is_mismatch() {
        let seed =
            ProviderWireWitnessSeed::try_new(vec![attempt(0, false, WireAttemptOutcome::Accepted)])
                .expect("valid seed");
        let response = serde_json::json!({"id":"resp-1","model":"fixture-model"});
        let mut binding = NativeResponseBinding::default();
        binding.observe(Some(&response), true);
        binding.observe(Some(&response), true);
        assert_eq!(
            binding.finalize(&seed).expect("witness").binding_status,
            WitnessBindingStatus::Mismatch
        );
    }
}
