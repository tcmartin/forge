//! Immutable host declarations for individually certified dispatch routes.
//!
//! Registration declares model support; it does not discover server features
//! or mint verification authority. The first supported encoding is the
//! standard Anthropic API-key thinking budget and temperature pair.

use std::collections::BTreeMap;
use std::fmt;

use claurst_core::dispatch::{
    DispatchAttempt, DispatchAuthorization, DispatchDenied, DispatchRoute, DispatchRouteSpec,
};
use claurst_core::effort::EffortLevel;
use claurst_core::provider_id::ProviderId;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::dispatch_request::DispatchRequest;
use crate::inference_admission::AdmissionLease;
use crate::provider_wire::PreparedWireRequest;

const MAX_MODELS: usize = 64;
const MAX_BINDINGS: usize = 8;
const MAX_MODEL_BYTES: usize = 4096;
const OPENAI_CHAT_COMPLETIONS_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

fn is_canonical_openai_endpoint(endpoint: &str) -> bool {
    endpoint == OPENAI_CHAT_COMPLETIONS_ENDPOINT
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TemperatureBinding {
    ModelDefault,
    Exact(u64),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WireEncoding {
    Anthropic {
        thinking_budget: Option<u32>,
        temperature: TemperatureBinding,
    },
    OpenAi {
        reasoning_effort: &'static str,
    },
}

/// A canonical effort and its complete supported wire encoding.
#[derive(Clone, PartialEq, Eq)]
pub struct EffortWireBinding {
    effort: EffortLevel,
    encoding: WireEncoding,
}

impl EffortWireBinding {
    pub fn anthropic_thinking(effort: EffortLevel) -> Self {
        Self {
            effort,
            encoding: WireEncoding::Anthropic {
                thinking_budget: effort.thinking_budget_tokens(),
                temperature: effort
                    .temperature()
                    .map_or(TemperatureBinding::ModelDefault, |value| {
                        TemperatureBinding::Exact(f64::from(value).to_bits())
                    }),
            },
        }
    }

    pub fn openai_reasoning(effort: EffortLevel) -> Self {
        Self {
            effort,
            encoding: WireEncoding::OpenAi {
                reasoning_effort: match effort {
                    EffortLevel::None => "none",
                    EffortLevel::Minimal => "minimal",
                    EffortLevel::Low => "low",
                    EffortLevel::Medium => "medium",
                    EffortLevel::High => "high",
                    EffortLevel::XHigh => "xhigh",
                    EffortLevel::Max | EffortLevel::Ultracode => "max",
                },
            },
        }
    }

    pub fn effort(&self) -> EffortLevel {
        self.effort
    }

    fn route_value(&self) -> String {
        match self.encoding {
            WireEncoding::Anthropic {
                thinking_budget,
                temperature,
            } => {
                let budget = thinking_budget
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "off".into());
                let temperature = match temperature {
                    TemperatureBinding::ModelDefault => "default".to_owned(),
                    TemperatureBinding::Exact(bits) => f64::from_bits(bits).to_string(),
                };
                format!("{}:anthropic:{budget}:{temperature}", self.effort)
            }
            WireEncoding::OpenAi { reasoning_effort } => {
                format!("{}:openai:{reasoning_effort}", self.effort)
            }
        }
    }

    fn same_encoding(&self, other: &Self) -> bool {
        self.encoding == other.encoding
    }

    fn matches_anthropic_body(&self, body: &Value) -> bool {
        let WireEncoding::Anthropic {
            thinking_budget,
            temperature,
        } = self.encoding
        else {
            return false;
        };
        if body.get("output_config").is_some()
            || body
                .get("max_tokens")
                .and_then(Value::as_u64)
                .is_none_or(|value| value == 0)
        {
            return false;
        }
        let thinking_matches = match thinking_budget {
            None => body.get("thinking").is_none_or(Value::is_null),
            Some(budget) => {
                let expected = serde_json::json!({"type":"enabled", "budget_tokens":budget});
                body.get("thinking") == Some(&expected)
                    && body
                        .get("max_tokens")
                        .and_then(Value::as_u64)
                        .is_some_and(|max_tokens| max_tokens > u64::from(budget))
            }
        };
        let temperature_matches = match temperature {
            TemperatureBinding::ModelDefault => body.get("temperature").is_none_or(Value::is_null),
            TemperatureBinding::Exact(bits) => {
                body.get("temperature").and_then(Value::as_f64) == Some(f64::from_bits(bits))
            }
        };
        thinking_matches && temperature_matches
    }

    fn openai_reasoning_effort(&self) -> Option<&'static str> {
        match self.encoding {
            WireEncoding::OpenAi { reasoning_effort } => Some(reasoning_effort),
            WireEncoding::Anthropic { .. } => None,
        }
    }

    fn matches_openai_body(&self, body: &Value, output_tokens: u32) -> bool {
        let Some(reasoning_effort) = self.openai_reasoning_effort() else {
            return false;
        };
        body.get("reasoning_effort").and_then(Value::as_str) == Some(reasoning_effort)
            && body
                .get("max_completion_tokens")
                .and_then(Value::as_u64)
                .is_some_and(|value| value == u64::from(output_tokens) && value > 0)
            && body.get("max_tokens").is_none()
    }
}

impl fmt::Debug for EffortWireBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("EffortWireBinding(..)")
    }
}

/// Explicit host input for an exact model, never a model-name heuristic.
#[derive(Clone)]
pub struct DispatchModelSpec {
    pub model_id: String,
    pub efforts: Vec<EffortWireBinding>,
}

impl fmt::Debug for DispatchModelSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchModelSpec(..)")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DispatchPolicyError {
    #[error("dispatch policy revision must be non-zero")]
    InvalidRevision,
    #[error("dispatch policy must contain between one and 64 models")]
    InvalidModelCount,
    #[error("dispatch model identifier is empty, too long, or contains controls")]
    InvalidModel,
    #[error("dispatch policy contains a duplicate model")]
    DuplicateModel,
    #[error("dispatch model must contain between one and eight effort bindings")]
    InvalidBindingCount,
    #[error("dispatch model contains a duplicate effort")]
    DuplicateEffort,
    #[error("dispatch efforts have identical wire encodings")]
    EquivalentEncoding,
}

/// Immutable, bounded host declarations. Absent policy means unsupported.
pub struct DispatchPolicy {
    revision: u64,
    models: BTreeMap<String, Vec<EffortWireBinding>>,
}

impl fmt::Debug for DispatchPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchPolicy(..)")
    }
}

impl DispatchPolicy {
    pub fn new(revision: u64, specs: Vec<DispatchModelSpec>) -> Result<Self, DispatchPolicyError> {
        if revision == 0 {
            return Err(DispatchPolicyError::InvalidRevision);
        }
        if specs.is_empty() || specs.len() > MAX_MODELS {
            return Err(DispatchPolicyError::InvalidModelCount);
        }
        let mut models = BTreeMap::new();
        for spec in specs {
            if spec.model_id.trim().is_empty()
                || spec.model_id.len() > MAX_MODEL_BYTES
                || spec.model_id.chars().any(char::is_control)
            {
                return Err(DispatchPolicyError::InvalidModel);
            }
            if spec.efforts.is_empty() || spec.efforts.len() > MAX_BINDINGS {
                return Err(DispatchPolicyError::InvalidBindingCount);
            }
            for (index, binding) in spec.efforts.iter().enumerate() {
                for previous in &spec.efforts[..index] {
                    if previous.effort == binding.effort {
                        return Err(DispatchPolicyError::DuplicateEffort);
                    }
                    if previous.same_encoding(binding) {
                        return Err(DispatchPolicyError::EquivalentEncoding);
                    }
                }
            }
            if models.insert(spec.model_id, spec.efforts).is_some() {
                return Err(DispatchPolicyError::DuplicateModel);
            }
        }
        Ok(Self { revision, models })
    }

    pub(crate) fn anthropic_route(
        &self,
        identity: &DispatchIdentity,
        endpoint: &str,
        model: &str,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, DispatchDenied> {
        let binding = self
            .models
            .get(model)
            .and_then(|bindings| bindings.iter().find(|binding| binding.effort == effort))
            .ok_or(DispatchDenied::UnsupportedRoute)?;
        self.route_for_binding(identity, endpoint, model, binding)
    }

    fn route_for_binding(
        &self,
        identity: &DispatchIdentity,
        endpoint: &str,
        model: &str,
        binding: &EffortWireBinding,
    ) -> Result<DispatchRoute, DispatchDenied> {
        self.route_for_binding_with_provider(
            identity,
            ProviderId::ANTHROPIC,
            "/v1/messages",
            endpoint,
            model,
            binding,
        )
    }

    fn route_for_binding_with_provider(
        &self,
        identity: &DispatchIdentity,
        provider_id: &str,
        expected_path: &str,
        endpoint: &str,
        model: &str,
        binding: &EffortWireBinding,
    ) -> Result<DispatchRoute, DispatchDenied> {
        let canonical = url::Url::parse(endpoint).map_err(|_| DispatchDenied::InvalidAttempt)?;
        if !matches!(canonical.scheme(), "http" | "https")
            || canonical.host_str().is_none()
            || !canonical.username().is_empty()
            || canonical.password().is_some()
            || canonical.query().is_some()
            || canonical.fragment().is_some()
            || canonical.path() != expected_path
            || canonical.as_str() != endpoint
        {
            return Err(DispatchDenied::InvalidAttempt);
        }
        DispatchRoute::new(DispatchRouteSpec {
            provider_id: provider_id.into(),
            provider_instance_id: identity.0.clone(),
            account_identity: format!("credential-slot:{}", identity.0),
            model_id: model.into(),
            endpoint_identity: canonical.to_string(),
            effort: binding.route_value(),
            policy_revision: self.revision,
        })
        .map_err(|_| DispatchDenied::InvalidAttempt)
    }

    pub(crate) fn openai_route(
        &self,
        identity: &DispatchIdentity,
        endpoint: &str,
        model: &str,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, DispatchDenied> {
        if !is_canonical_openai_endpoint(endpoint) {
            return Err(DispatchDenied::UnsupportedRoute);
        }
        let binding = self
            .models
            .get(model)
            .and_then(|bindings| bindings.iter().find(|binding| binding.effort == effort))
            .ok_or(DispatchDenied::UnsupportedRoute)?;
        if binding.openai_reasoning_effort().is_none() {
            return Err(DispatchDenied::UnsupportedRoute);
        }
        self.route_for_binding_with_provider(
            identity,
            ProviderId::OPENAI,
            "/v1/chat/completions",
            endpoint,
            model,
            binding,
        )
    }

    pub(crate) fn validate_anthropic_envelope(
        &self,
        identity: &DispatchIdentity,
        endpoint: &str,
        dispatch: &DispatchRequest,
    ) -> Result<(), DispatchDenied> {
        self.binding_for_envelope(identity, endpoint, dispatch)?;
        Ok(())
    }

    pub(crate) fn openai_reasoning_for_envelope(
        &self,
        identity: &DispatchIdentity,
        endpoint: &str,
        dispatch: &DispatchRequest,
    ) -> Result<&'static str, DispatchDenied> {
        if !is_canonical_openai_endpoint(endpoint) {
            return Err(DispatchDenied::UnsupportedRoute);
        }
        self.binding_for_envelope_with_provider(
            identity,
            ProviderId::OPENAI,
            "/v1/chat/completions",
            endpoint,
            dispatch,
        )?
        .openai_reasoning_effort()
        .ok_or(DispatchDenied::RouteMismatch)
    }

    fn binding_for_envelope(
        &self,
        identity: &DispatchIdentity,
        endpoint: &str,
        dispatch: &DispatchRequest,
    ) -> Result<&EffortWireBinding, DispatchDenied> {
        self.binding_for_envelope_with_provider(
            identity,
            ProviderId::ANTHROPIC,
            "/v1/messages",
            endpoint,
            dispatch,
        )
    }

    fn binding_for_envelope_with_provider(
        &self,
        identity: &DispatchIdentity,
        provider_id: &str,
        expected_path: &str,
        endpoint: &str,
        dispatch: &DispatchRequest,
    ) -> Result<&EffortWireBinding, DispatchDenied> {
        let expected = dispatch.expected_route();
        if expected.provider_id() != provider_id {
            return Err(DispatchDenied::RouteMismatch);
        }
        let binding = self
            .models
            .get(expected.model_id())
            .and_then(|bindings| {
                bindings
                    .iter()
                    .find(|binding| binding.route_value() == expected.effort())
            })
            .ok_or(DispatchDenied::RouteMismatch)?;
        let route = self.route_for_binding_with_provider(
            identity,
            provider_id,
            expected_path,
            endpoint,
            expected.model_id(),
            binding,
        )?;
        if &route != expected {
            return Err(DispatchDenied::RouteMismatch);
        }
        Ok(binding)
    }
}

/// Separate from session metadata. Immutable client configuration owns this
/// credential slot; replacing its policy creates a fresh identity.
pub(crate) struct DispatchIdentity(String);

impl DispatchIdentity {
    pub(crate) fn fresh() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl fmt::Debug for DispatchIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchIdentity(..)")
    }
}

/// Both guards outlive the response transport/body. Callers drop a retry's
/// authorization separately before retaining admission for the next attempt.
pub(crate) struct AuthorizedInferenceLease {
    _authorization: Option<DispatchAuthorization>,
    _admission: AdmissionLease,
}

impl AuthorizedInferenceLease {
    pub(crate) fn new(
        admission: AdmissionLease,
        authorization: Option<DispatchAuthorization>,
    ) -> Self {
        Self {
            _authorization: authorization,
            _admission: admission,
        }
    }
}

/// Inspect the exact final wire body, reconstruct its route independently of
/// caller labels, then ask the opaque host authority for its owned guard.
pub(crate) async fn authorize_anthropic_dispatch(
    policy: Option<&DispatchPolicy>,
    identity: &DispatchIdentity,
    endpoint: &str,
    prepared: &PreparedWireRequest,
    dispatch: Option<&DispatchRequest>,
) -> Result<Option<DispatchAuthorization>, DispatchDenied> {
    let Some(dispatch) = dispatch else {
        return Ok(None);
    };
    let policy = policy.ok_or(DispatchDenied::UnsupportedRoute)?;
    let binding = policy.binding_for_envelope(identity, endpoint, dispatch)?;
    let body: Value =
        serde_json::from_slice(prepared.bytes()).map_err(|_| DispatchDenied::InvalidAttempt)?;
    if prepared.provider != ProviderId::ANTHROPIC
        || prepared.endpoint != endpoint
        || body.get("stream").and_then(Value::as_bool) != Some(prepared.stream)
        || u64::try_from(prepared.bytes().len()).ok() != Some(prepared.wire_bytes)
    {
        return Err(DispatchDenied::InvalidAttempt);
    }
    if body.get("model").and_then(Value::as_str) != Some(dispatch.expected_route().model_id())
        || !binding.matches_anthropic_body(&body)
    {
        return Err(DispatchDenied::RouteMismatch);
    }
    let route = policy.route_for_binding(
        identity,
        endpoint,
        dispatch.expected_route().model_id(),
        binding,
    )?;
    let body_sha256 = hex::encode(Sha256::digest(prepared.bytes()));
    if body_sha256 != prepared.wire_sha256 {
        return Err(DispatchDenied::InvalidAttempt);
    }
    let attempt = DispatchAttempt::new(route, dispatch.logical_request_id().clone(), body_sha256)
        .map_err(|_| DispatchDenied::InvalidAttempt)?;
    dispatch.permit().authorize(&attempt).await.map(Some)
}

pub(crate) async fn authorize_openai_dispatch(
    policy: Option<&DispatchPolicy>,
    identity: &DispatchIdentity,
    endpoint: &str,
    prepared: &PreparedWireRequest,
    dispatch: Option<&DispatchRequest>,
    output_tokens: u32,
) -> Result<Option<DispatchAuthorization>, DispatchDenied> {
    let Some(dispatch) = dispatch else {
        return Ok(None);
    };
    let policy = policy.ok_or(DispatchDenied::UnsupportedRoute)?;
    let binding = policy.binding_for_envelope_with_provider(
        identity,
        ProviderId::OPENAI,
        "/v1/chat/completions",
        endpoint,
        dispatch,
    )?;
    let body: Value =
        serde_json::from_slice(prepared.bytes()).map_err(|_| DispatchDenied::InvalidAttempt)?;
    if prepared.provider != ProviderId::OPENAI
        || prepared.endpoint != endpoint
        || body.get("stream").and_then(Value::as_bool) != Some(prepared.stream)
        || u64::try_from(prepared.bytes().len()).ok() != Some(prepared.wire_bytes)
    {
        return Err(DispatchDenied::InvalidAttempt);
    }
    if body.get("model").and_then(Value::as_str) != Some(dispatch.expected_route().model_id())
        || !binding.matches_openai_body(&body, output_tokens)
    {
        return Err(DispatchDenied::RouteMismatch);
    }
    let route = policy.route_for_binding_with_provider(
        identity,
        ProviderId::OPENAI,
        "/v1/chat/completions",
        endpoint,
        dispatch.expected_route().model_id(),
        binding,
    )?;
    let body_sha256 = hex::encode(Sha256::digest(prepared.bytes()));
    if body_sha256 != prepared.wire_sha256 {
        return Err(DispatchDenied::InvalidAttempt);
    }
    let attempt = DispatchAttempt::new(route, dispatch.logical_request_id().clone(), body_sha256)
        .map_err(|_| DispatchDenied::InvalidAttempt)?;
    dispatch.permit().authorize(&attempt).await.map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_wire::{prepare_json_checked, MediaAccounting, WireBudget};
    use claurst_core::dispatch::{
        DispatchAuthorizer, DispatchDecision, DispatchPermit, DispatchReservation, LogicalRequestId,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn policy(efforts: &[EffortLevel]) -> Result<DispatchPolicy, DispatchPolicyError> {
        DispatchPolicy::new(
            1,
            vec![DispatchModelSpec {
                model_id: "fixture-model".into(),
                efforts: efforts
                    .iter()
                    .map(|effort| EffortWireBinding::anthropic_thinking(*effort))
                    .collect(),
            }],
        )
    }

    #[test]
    fn registration_is_bounded_opaque_and_rejects_semantic_aliases() {
        assert!(matches!(
            policy(&[EffortLevel::Max, EffortLevel::Ultracode]),
            Err(DispatchPolicyError::EquivalentEncoding)
        ));
        assert!(matches!(
            policy(&[EffortLevel::High, EffortLevel::High]),
            Err(DispatchPolicyError::DuplicateEffort)
        ));
        assert!(matches!(
            policy(&[]),
            Err(DispatchPolicyError::InvalidBindingCount)
        ));
        assert!(matches!(
            DispatchPolicy::new(0, vec![]),
            Err(DispatchPolicyError::InvalidRevision)
        ));
        assert!(matches!(
            DispatchPolicy::new(1, vec![]),
            Err(DispatchPolicyError::InvalidModelCount)
        ));
        let specs: Vec<_> = (0..65)
            .map(|index| DispatchModelSpec {
                model_id: format!("model-{index}"),
                efforts: vec![EffortWireBinding::anthropic_thinking(EffortLevel::High)],
            })
            .collect();
        assert!(matches!(
            DispatchPolicy::new(1, specs),
            Err(DispatchPolicyError::InvalidModelCount)
        ));
        let policy = policy(&[EffortLevel::None, EffortLevel::Low]).unwrap();
        assert_eq!(format!("{policy:?}"), "DispatchPolicy(..)");
        let identity = DispatchIdentity::fresh();
        let none = policy
            .anthropic_route(
                &identity,
                "http://localhost:9/v1/messages",
                "fixture-model",
                EffortLevel::None,
            )
            .unwrap();
        let low = policy
            .anthropic_route(
                &identity,
                "http://localhost:9/v1/messages",
                "fixture-model",
                EffortLevel::Low,
            )
            .unwrap();
        assert_ne!(none, low);
        assert!(policy
            .anthropic_route(
                &identity,
                "http://localhost:9/v1/messages",
                "unknown-model",
                EffortLevel::None
            )
            .is_err());
    }

    #[test]
    fn certified_endpoints_exclude_credentials_queries_fragments_and_other_paths() {
        let policy = policy(&[EffortLevel::High]).unwrap();
        let identity = DispatchIdentity::fresh();
        for endpoint in [
            "http://user:password@localhost:9/v1/messages",
            "http://localhost:9/v1/messages?key=secret",
            "http://localhost:9/v1/messages#fragment",
            "http://localhost:9/prefix/v1/messages",
            "http://localhost:9//v1/messages",
            "http://localhost:9/v1/../v1/messages",
            "http://LOCALHOST:9/v1/messages",
            "file:///v1/messages",
        ] {
            assert!(
                policy
                    .anthropic_route(&identity, endpoint, "fixture-model", EffortLevel::High)
                    .is_err(),
                "accepted endpoint: {endpoint}"
            );
        }
    }

    #[test]
    fn openai_adaptive_route_accepts_only_the_production_host() {
        let policy = DispatchPolicy::new(
            1,
            vec![DispatchModelSpec {
                model_id: "gpt-5.5".into(),
                efforts: vec![EffortWireBinding::openai_reasoning(EffortLevel::Medium)],
            }],
        )
        .unwrap();
        let identity = DispatchIdentity::fresh();
        assert!(policy
            .openai_route(
                &identity,
                "https://api.openai.com/v1/chat/completions",
                "gpt-5.5",
                EffortLevel::Medium,
            )
            .is_ok());
        for endpoint in [
            "http://127.0.0.1:9/v1/chat/completions",
            "https://example.com/v1/chat/completions",
            "https://api.openai.com:443/v1/chat/completions",
        ] {
            assert_eq!(
                policy.openai_route(&identity, endpoint, "gpt-5.5", EffortLevel::Medium),
                Err(DispatchDenied::UnsupportedRoute),
                "accepted non-production OpenAI endpoint: {endpoint}"
            );
        }
    }

    #[test]
    fn final_effort_encoding_rejects_overrides_and_boundary_budgets() {
        let high = EffortWireBinding::anthropic_thinking(EffortLevel::High);
        let body = serde_json::json!({"model":"fixture-model", "max_tokens":10001, "thinking":{"type":"enabled","budget_tokens":10000}});
        assert!(high.matches_anthropic_body(&body));
        for (field, value) in [
            ("temperature", serde_json::json!(0.0)),
            ("max_tokens", serde_json::json!(10000)),
            ("thinking", serde_json::json!({"type":"adaptive"})),
            (
                "thinking",
                serde_json::json!({"type":"enabled","budget_tokens":9999}),
            ),
            ("output_config", serde_json::json!({"effort":"low"})),
        ] {
            let mut changed = body.clone();
            changed[field] = value;
            assert!(
                !high.matches_anthropic_body(&changed),
                "accepted override: {field}"
            );
        }
        let low = EffortWireBinding::anthropic_thinking(EffortLevel::Low);
        assert!(low.matches_anthropic_body(&serde_json::json!({"max_tokens":32,"temperature":0.0})));
        assert!(!low.matches_anthropic_body(&serde_json::json!({"max_tokens":32})));
        let none = EffortWireBinding::anthropic_thinking(EffortLevel::None);
        assert!(none.matches_anthropic_body(&serde_json::json!({"max_tokens":32})));
        assert!(
            !none.matches_anthropic_body(&serde_json::json!({"max_tokens":32,"temperature":0.0}))
        );
    }

    struct CountingAuthorizer(AtomicUsize);
    struct Reservation;
    impl DispatchReservation for Reservation {}
    impl DispatchAuthorizer for CountingAuthorizer {
        fn authorize<'a>(
            &'a self,
            _: &'a DispatchAttempt,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<DispatchAuthorization, DispatchDenied>>
                    + Send
                    + 'a,
            >,
        > {
            self.0.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Ok(DispatchAuthorization::new(
                    DispatchDecision::AuthorizedFresh,
                    Box::new(Reservation),
                ))
            })
        }
    }

    #[tokio::test]
    async fn exact_final_bytes_and_identity_are_checked_before_authority() {
        let policy = policy(&[EffortLevel::Low]).unwrap();
        let identity = DispatchIdentity::fresh();
        let endpoint = "http://localhost:9/v1/messages";
        let route = policy
            .anthropic_route(&identity, endpoint, "fixture-model", EffortLevel::Low)
            .unwrap();
        let authorizer = Arc::new(CountingAuthorizer(AtomicUsize::new(0)));
        let dispatch = DispatchRequest::new(
            DispatchPermit::new(authorizer.clone()),
            LogicalRequestId::fresh(),
            route,
        );
        let body = serde_json::json!({"model":"fixture-model", "max_tokens":32, "temperature":0.0, "stream":false});
        let budget =
            WireBudget::finite(8192, 7680, 512, 1024 * 1024, MediaAccounting::None).unwrap();
        let prepared =
            prepare_json_checked("anthropic", endpoint, false, &body, Some(&budget)).unwrap();
        assert!(authorize_anthropic_dispatch(
            Some(&policy),
            &identity,
            endpoint,
            &prepared,
            Some(&dispatch)
        )
        .await
        .is_ok());
        assert_eq!(authorizer.0.load(Ordering::SeqCst), 1);
        for field in ["provider", "endpoint", "stream", "bytes"] {
            let mut changed = prepared.clone();
            match field {
                "provider" => changed.provider = "other".into(),
                "endpoint" => changed.endpoint = "http://localhost:9/other".into(),
                "stream" => changed.stream = true,
                _ => changed.wire_bytes += 1,
            }
            assert!(matches!(
                authorize_anthropic_dispatch(
                    Some(&policy),
                    &identity,
                    endpoint,
                    &changed,
                    Some(&dispatch)
                )
                .await,
                Err(DispatchDenied::InvalidAttempt)
            ));
        }
        let foreign_identity = DispatchIdentity::fresh();
        assert!(matches!(
            authorize_anthropic_dispatch(
                Some(&policy),
                &foreign_identity,
                endpoint,
                &prepared,
                Some(&dispatch)
            )
            .await,
            Err(DispatchDenied::RouteMismatch)
        ));
        let mut corrupt = prepared.clone();
        corrupt.wire_sha256 = "f".repeat(64);
        assert!(matches!(
            authorize_anthropic_dispatch(
                Some(&policy),
                &identity,
                endpoint,
                &corrupt,
                Some(&dispatch)
            )
            .await,
            Err(DispatchDenied::InvalidAttempt)
        ));
        assert!(
            authorize_anthropic_dispatch(None, &foreign_identity, endpoint, &prepared, None)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(authorizer.0.load(Ordering::SeqCst), 1);
    }
}
