//! Host-owned construction of Anthropic clients.
//!
//! The factory carries only the immutable dispatch policy. Credentials,
//! endpoint configuration, and the opaque client identity are supplied by
//! each `create` call; rebuilding a client therefore rotates its identity.

use std::fmt;
use std::sync::Arc;

use crate::client::{AnthropicClient, ClientConfig};
use crate::dispatch_policy::DispatchPolicy;

/// Immutable host factory for policy-aware Anthropic clients.
///
/// A factory without a policy preserves the existing fixed/unsupported
/// behavior. The policy is shared by clones, while every created client gets
/// a fresh dispatch identity from `AnthropicClient::new`.
#[derive(Clone, Default)]
pub struct AnthropicClientFactory {
    dispatch_policy: Option<Arc<DispatchPolicy>>,
}

impl fmt::Debug for AnthropicClientFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AnthropicClientFactory(..)")
    }
}

impl AnthropicClientFactory {
    /// Construct a fixed factory with no verified dispatch policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a factory carrying an immutable host dispatch policy.
    pub fn with_dispatch_policy(policy: Arc<DispatchPolicy>) -> Self {
        Self {
            dispatch_policy: Some(policy),
        }
    }

    /// Construct a client from fresh credentials and endpoint settings.
    pub fn create(&self, config: ClientConfig) -> anyhow::Result<AnthropicClient> {
        let client = AnthropicClient::new(config)?;
        Ok(match &self.dispatch_policy {
            Some(policy) => client.with_dispatch_policy(Arc::clone(policy)),
            None => client,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::dispatch::DispatchDenied;
    use claurst_core::effort::EffortLevel;

    const MODEL: &str = "fixture-model";

    fn policy() -> Arc<DispatchPolicy> {
        Arc::new(
            DispatchPolicy::new(
                17,
                vec![crate::dispatch_policy::DispatchModelSpec {
                    model_id: MODEL.to_string(),
                    efforts: vec![
                        crate::dispatch_policy::EffortWireBinding::anthropic_thinking(
                            EffortLevel::High,
                        ),
                    ],
                }],
            )
            .expect("valid fixture policy"),
        )
    }

    fn config(api_key: &str, api_base: &str) -> ClientConfig {
        ClientConfig {
            api_key: api_key.to_string(),
            api_base: api_base.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn clones_share_policy_but_created_clients_rotate_identity() {
        let factory = AnthropicClientFactory::with_dispatch_policy(policy());
        let first = factory
            .create(config("key-one", "http://localhost:9001"))
            .expect("first client");
        let second = factory
            .clone()
            .create(config("key-one", "http://localhost:9001"))
            .expect("second client");

        let first_route = first
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("first route");
        let second_route = second
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("second route");
        assert_eq!(first_route.policy_revision(), 17);
        assert_eq!(second_route.policy_revision(), 17);
        assert_ne!(
            first_route.provider_instance_id(),
            second_route.provider_instance_id()
        );
        assert_ne!(
            first_route.account_identity(),
            second_route.account_identity()
        );
    }

    #[test]
    fn rebuilt_credentials_and_endpoint_receive_fresh_routes() {
        let factory = AnthropicClientFactory::with_dispatch_policy(policy());
        let first = factory
            .create(config("first-secret", "http://localhost:9001"))
            .expect("first client")
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("first route");
        let second = factory
            .create(config("second-secret", "http://localhost:9002"))
            .expect("second client")
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("second route");

        assert_ne!(first.provider_instance_id(), second.provider_instance_id());
        assert_ne!(first.endpoint_identity(), second.endpoint_identity());
        assert_ne!(first.account_identity(), second.account_identity());
    }

    #[test]
    fn unsupported_routes_fail_closed_without_policy_or_credentials() {
        let fixed = AnthropicClientFactory::new()
            .create(config("key", "http://localhost:9001"))
            .expect("fixed client");
        assert_eq!(
            fixed.dispatch_route(MODEL, EffortLevel::High),
            Err(DispatchDenied::UnsupportedRoute)
        );

        let policy_factory = AnthropicClientFactory::with_dispatch_policy(policy());
        for client in [
            policy_factory
                .create(config("", "http://localhost:9001"))
                .expect("missing-key client"),
            policy_factory
                .create(ClientConfig {
                    api_key: "key".into(),
                    api_base: "http://localhost:9001".into(),
                    use_bearer_auth: true,
                    ..Default::default()
                })
                .expect("oauth client"),
            policy_factory
                .create(ClientConfig {
                    api_key: "key".into(),
                    api_base: "http://localhost:9001".into(),
                    provider: crate::client::Provider::Codex,
                    ..Default::default()
                })
                .expect("codex client"),
        ] {
            assert_eq!(
                client.dispatch_route(MODEL, EffortLevel::High),
                Err(DispatchDenied::UnsupportedRoute)
            );
        }
    }

    #[test]
    fn unregistered_model_and_effort_are_unsupported() {
        let client = AnthropicClientFactory::with_dispatch_policy(policy())
            .create(config("key", "http://localhost:9001"))
            .expect("client");
        assert_eq!(
            client.dispatch_route("other-model", EffortLevel::High),
            Err(DispatchDenied::UnsupportedRoute)
        );
        assert_eq!(
            client.dispatch_route(MODEL, EffortLevel::Low),
            Err(DispatchDenied::UnsupportedRoute)
        );
    }

    #[test]
    fn debug_is_redacted() {
        let factory = AnthropicClientFactory::with_dispatch_policy(policy());
        assert_eq!(format!("{factory:?}"), "AnthropicClientFactory(..)");
        assert!(!format!("{factory:?}").contains("fixture-model"));
        assert!(!format!("{factory:?}").contains("key"));
    }
}
