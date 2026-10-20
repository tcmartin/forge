//! Typed, dependency-neutral authorization inputs for inference dispatch.
//!
//! This module deliberately has no serialization surface. Callers construct
//! validated route and attempt values in Rust, then hand an opaque permit to a
//! host-owned authorizer. Route identity and body identity therefore cannot be
//! supplied through model or JSON metadata.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

const MAX_ROUTE_FIELD_BYTES: usize = 4096;
const MAX_EFFORT_BYTES: usize = 64;
const SHA256_HEX_BYTES: usize = 64;

/// A host-created identifier that remains stable across retry attempts for
/// one logical request.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LogicalRequestId(String);

impl LogicalRequestId {
    /// Create a fresh logical request identifier.
    pub fn fresh() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Return the opaque identifier as its host-facing string form.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for LogicalRequestId {
    fn default() -> Self {
        Self::fresh()
    }
}

impl fmt::Debug for LogicalRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LogicalRequestId(..)")
    }
}

/// Typed host input used to construct a dispatch route.
#[derive(Clone, PartialEq, Eq)]
pub struct DispatchRouteSpec {
    pub provider_id: String,
    pub provider_instance_id: String,
    pub account_identity: String,
    pub model_id: String,
    pub endpoint_identity: String,
    pub effort: String,
    pub policy_revision: u64,
}

impl fmt::Debug for DispatchRouteSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DispatchRouteSpec")
            .field("provider_id", &"<redacted>")
            .field("provider_instance_id", &"<redacted>")
            .field("account_identity", &"<redacted>")
            .field("model_id", &"<redacted>")
            .field("endpoint_identity", &"<redacted>")
            .field("effort", &"<redacted>")
            .field("policy_revision", &self.policy_revision)
            .finish()
    }
}

/// Errors from constructing a typed dispatch value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchInputError {
    EmptyField,
    FieldTooLong,
    EffortTooLong,
    ControlCharacter,
    InvalidBodyHash,
    InvalidPolicyRevision,
}

impl fmt::Display for DispatchInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyField => "dispatch input contains an empty field",
            Self::FieldTooLong => "dispatch input field exceeds its size limit",
            Self::EffortTooLong => "dispatch effort exceeds its size limit",
            Self::ControlCharacter => "dispatch input contains a control character",
            Self::InvalidBodyHash => "dispatch body hash must be 64 ASCII hexadecimal characters",
            Self::InvalidPolicyRevision => "dispatch policy revision must be non-zero",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DispatchInputError {}

/// The validated route identity used by an inference attempt.
///
/// The fields are private by design. Construct a route through the public
/// constructor so host input receives bounded validation everywhere.
///
/// ```compile_fail
/// use claurst_core::dispatch::DispatchRoute;
///
/// let _route = DispatchRoute {
///     provider_id: String::new(),
///     provider_instance_id: String::new(),
///     account_identity: String::new(),
///     model_id: String::new(),
///     endpoint_identity: String::new(),
///     effort: String::new(),
///     policy_revision: 1,
/// };
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DispatchRoute {
    provider_id: String,
    provider_instance_id: String,
    account_identity: String,
    model_id: String,
    endpoint_identity: String,
    effort: String,
    policy_revision: u64,
}

impl DispatchRoute {
    /// Validate and construct a route from host-provided typed fields.
    pub fn new(spec: DispatchRouteSpec) -> Result<Self, DispatchInputError> {
        if spec.policy_revision == 0 {
            return Err(DispatchInputError::InvalidPolicyRevision);
        }
        validate_route_field(&spec.provider_id)?;
        validate_route_field(&spec.provider_instance_id)?;
        validate_route_field(&spec.account_identity)?;
        validate_route_field(&spec.model_id)?;
        validate_route_field(&spec.endpoint_identity)?;
        validate_effort(&spec.effort)?;

        Ok(Self {
            provider_id: spec.provider_id,
            provider_instance_id: spec.provider_instance_id,
            account_identity: spec.account_identity,
            model_id: spec.model_id,
            endpoint_identity: spec.endpoint_identity,
            effort: spec.effort,
            policy_revision: spec.policy_revision,
        })
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn provider_instance_id(&self) -> &str {
        &self.provider_instance_id
    }

    pub fn account_identity(&self) -> &str {
        &self.account_identity
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn endpoint_identity(&self) -> &str {
        &self.endpoint_identity
    }

    pub fn effort(&self) -> &str {
        &self.effort
    }

    pub fn policy_revision(&self) -> u64 {
        self.policy_revision
    }
}

impl fmt::Debug for DispatchRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchRoute(..)")
    }
}

/// A body-bound attempt for a logical request.
#[derive(Clone, PartialEq, Eq)]
pub struct DispatchAttempt {
    route: DispatchRoute,
    logical_request_id: LogicalRequestId,
    body_sha256: String,
}

impl DispatchAttempt {
    /// Validate and construct an attempt bound to one exact request body hash.
    pub fn new(
        route: DispatchRoute,
        logical_request_id: LogicalRequestId,
        body_sha256: impl Into<String>,
    ) -> Result<Self, DispatchInputError> {
        let body_sha256 = body_sha256.into();
        if body_sha256.len() != SHA256_HEX_BYTES
            || !body_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(DispatchInputError::InvalidBodyHash);
        }

        Ok(Self {
            route,
            logical_request_id,
            body_sha256,
        })
    }

    pub fn route(&self) -> &DispatchRoute {
        &self.route
    }

    pub fn logical_request_id(&self) -> &LogicalRequestId {
        &self.logical_request_id
    }

    pub fn body_sha256(&self) -> &str {
        &self.body_sha256
    }
}

impl fmt::Debug for DispatchAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchAttempt(..)")
    }
}

/// The result of a host authorization decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchDecision {
    AuthorizedFresh,
    AuthorizedRetry,
}

/// Categorical failures from the host dispatch authorizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchDenied {
    Cancelled,
    InvalidAttempt,
    RouteMismatch,
    ProofInvalidated,
    AlreadyConsumed,
    Timeout,
    UnsupportedRoute,
    InFlight,
}

impl fmt::Display for DispatchDenied {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let category = match self {
            Self::Cancelled => "cancelled",
            Self::InvalidAttempt => "invalid attempt",
            Self::RouteMismatch => "route mismatch",
            Self::ProofInvalidated => "proof invalidated",
            Self::AlreadyConsumed => "already consumed",
            Self::Timeout => "timeout",
            Self::UnsupportedRoute => "unsupported route",
            Self::InFlight => "in flight",
        };
        write!(formatter, "dispatch denied: {category}")
    }
}

impl std::error::Error for DispatchDenied {}

/// Host-owned authorization state for typed dispatch attempts.
pub trait DispatchAuthorizer: Send + Sync {
    fn authorize<'a>(
        &'a self,
        attempt: &'a DispatchAttempt,
    ) -> Pin<Box<dyn Future<Output = Result<DispatchAuthorization, DispatchDenied>> + Send + 'a>>;
}

/// A host-owned lifetime guard held through the actual inference send.
pub trait DispatchReservation: Send + Sync {}

/// A one-shot authorization result that retains its host reservation guard.
pub struct DispatchAuthorization {
    decision: DispatchDecision,
    _reservation: Box<dyn DispatchReservation>,
}

impl DispatchAuthorization {
    pub fn new(decision: DispatchDecision, reservation: Box<dyn DispatchReservation>) -> Self {
        Self {
            decision,
            _reservation: reservation,
        }
    }

    pub fn decision(&self) -> DispatchDecision {
        self.decision
    }
}

impl fmt::Debug for DispatchAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchAuthorization(..)")
    }
}

/// Opaque, cloneable handle to a host dispatch authorizer.
#[derive(Clone)]
pub struct DispatchPermit {
    authorizer: Arc<dyn DispatchAuthorizer>,
}

impl DispatchPermit {
    /// Create a permit backed by host-owned authorization state.
    pub fn new(authorizer: Arc<dyn DispatchAuthorizer>) -> Self {
        Self { authorizer }
    }

    /// Ask the host authorizer to authorize one typed attempt.
    pub async fn authorize(
        &self,
        attempt: &DispatchAttempt,
    ) -> Result<DispatchAuthorization, DispatchDenied> {
        self.authorizer.authorize(attempt).await
    }
}

impl fmt::Debug for DispatchPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchPermit(..)")
    }
}

fn validate_route_field(value: &str) -> Result<(), DispatchInputError> {
    if value.trim().is_empty() {
        return Err(DispatchInputError::EmptyField);
    }
    if value.len() > MAX_ROUTE_FIELD_BYTES {
        return Err(DispatchInputError::FieldTooLong);
    }
    if value.chars().any(char::is_control) {
        return Err(DispatchInputError::ControlCharacter);
    }
    Ok(())
}

fn validate_effort(value: &str) -> Result<(), DispatchInputError> {
    validate_route_field(value).map_err(|error| match error {
        DispatchInputError::FieldTooLong => DispatchInputError::EffortTooLong,
        other => other,
    })?;
    if value.len() > MAX_EFFORT_BYTES {
        return Err(DispatchInputError::EffortTooLong);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;

    fn route_spec() -> DispatchRouteSpec {
        DispatchRouteSpec {
            provider_id: "openai".into(),
            provider_instance_id: "openai-primary".into(),
            account_identity: "account-redacted-by-host".into(),
            model_id: "gpt-test".into(),
            endpoint_identity: "responses".into(),
            effort: "medium".into(),
            policy_revision: 1,
        }
    }

    fn route() -> DispatchRoute {
        DispatchRoute::new(route_spec()).unwrap()
    }

    fn attempt() -> DispatchAttempt {
        DispatchAttempt::new(route(), LogicalRequestId::fresh(), "a".repeat(64)).unwrap()
    }

    #[test]
    fn route_validates_bounds_and_controls() {
        let mut spec = route_spec();
        spec.provider_id.clear();
        assert_eq!(
            DispatchRoute::new(spec),
            Err(DispatchInputError::EmptyField)
        );

        let mut spec = route_spec();
        spec.model_id = "x".repeat(MAX_ROUTE_FIELD_BYTES + 1);
        assert_eq!(
            DispatchRoute::new(spec),
            Err(DispatchInputError::FieldTooLong)
        );

        let mut spec = route_spec();
        spec.policy_revision = 0;
        assert_eq!(
            DispatchRoute::new(spec),
            Err(DispatchInputError::InvalidPolicyRevision)
        );

        let mut spec = route_spec();
        spec.provider_id = "   ".into();
        assert_eq!(
            DispatchRoute::new(spec),
            Err(DispatchInputError::EmptyField)
        );

        let mut spec = route_spec();
        spec.endpoint_identity = "endpoint\n".into();
        assert_eq!(
            DispatchRoute::new(spec),
            Err(DispatchInputError::ControlCharacter)
        );

        let mut spec = route_spec();
        spec.effort = "x".repeat(MAX_EFFORT_BYTES + 1);
        assert_eq!(
            DispatchRoute::new(spec),
            Err(DispatchInputError::EffortTooLong)
        );
    }

    #[test]
    fn route_is_opaque_but_cloneable_and_hashable() {
        let first = route();
        let second = first.clone();
        assert_eq!(first, second);
        let mut routes = HashSet::new();
        routes.insert(first);
        assert!(routes.contains(&second));

        let debug = format!("{second:?}");
        assert_eq!(debug, "DispatchRoute(..)");
        assert!(!debug.contains("account-redacted-by-host"));

        let mut changed_spec = route_spec();
        changed_spec.policy_revision = 2;
        let changed = DispatchRoute::new(changed_spec).unwrap();
        assert_ne!(second, changed);
        assert!(!routes.contains(&changed));

        let spec_debug = format!("{:?}", route_spec());
        assert!(!spec_debug.contains("account-redacted-by-host"));
        assert!(spec_debug.contains("policy_revision"));
    }

    #[test]
    fn attempt_requires_exact_ascii_hex_sha256() {
        let route = route();
        let id = LogicalRequestId::default();
        assert!(DispatchAttempt::new(route.clone(), id.clone(), "f".repeat(64)).is_ok());
        assert!(DispatchAttempt::new(route.clone(), id.clone(), "F".repeat(64)).is_ok());
        assert_eq!(
            DispatchAttempt::new(route.clone(), id.clone(), "f".repeat(63)),
            Err(DispatchInputError::InvalidBodyHash)
        );
        assert_eq!(
            DispatchAttempt::new(route, id, format!("{}g", "f".repeat(63))),
            Err(DispatchInputError::InvalidBodyHash)
        );
    }

    #[test]
    fn attempt_debug_redacts_route_and_body_identity() {
        let attempt = attempt();
        assert_eq!(format!("{attempt:?}"), "DispatchAttempt(..)");
        assert!(!format!("{attempt:?}").contains(&"a".repeat(64)));
    }

    struct TestAuthorizer {
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl DispatchAuthorizer for TestAuthorizer {
        fn authorize<'a>(
            &'a self,
            attempt: &'a DispatchAttempt,
        ) -> Pin<Box<dyn Future<Output = Result<DispatchAuthorization, DispatchDenied>> + Send + 'a>>
        {
            self.calls
                .lock()
                .expect("test authorizer mutex")
                .push(attempt.logical_request_id().as_str().to_owned());
            Box::pin(async {
                Ok(DispatchAuthorization::new(
                    DispatchDecision::AuthorizedFresh,
                    Box::new(NoopReservation),
                ))
            })
        }
    }

    struct NoopReservation;
    impl DispatchReservation for NoopReservation {}

    #[tokio::test]
    async fn permit_forwards_authorization_opaquely() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let permit = DispatchPermit::new(Arc::new(TestAuthorizer {
            calls: calls.clone(),
        }));
        assert_eq!(format!("{permit:?}"), "DispatchPermit(..)");
        let authorization = permit.authorize(&attempt()).await.unwrap();
        assert_eq!(authorization.decision(), DispatchDecision::AuthorizedFresh);
        assert_eq!(format!("{authorization:?}"), "DispatchAuthorization(..)");
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn authorization_drop_releases_host_reservation() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct DropReservation(Arc<AtomicBool>);
        impl DispatchReservation for DropReservation {}
        impl Drop for DropReservation {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let authorization = DispatchAuthorization::new(
            DispatchDecision::AuthorizedFresh,
            Box::new(DropReservation(dropped.clone())),
        );
        assert!(!dropped.load(Ordering::Acquire));
        drop(authorization);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn denied_display_is_categorical() {
        for denied in [
            DispatchDenied::Cancelled,
            DispatchDenied::InvalidAttempt,
            DispatchDenied::RouteMismatch,
            DispatchDenied::ProofInvalidated,
            DispatchDenied::AlreadyConsumed,
            DispatchDenied::Timeout,
            DispatchDenied::UnsupportedRoute,
            DispatchDenied::InFlight,
        ] {
            let display = denied.to_string();
            assert!(display.starts_with("dispatch denied: "));
            assert!(!display.contains("openai"));
            assert!(!display.contains("gpt-test"));
        }
    }
}
