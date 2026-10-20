use std::sync::Arc;

use claurst_api::{LlmProvider, ModelRegistry, ProviderCapabilities};
use claurst_core::dispatch::{DispatchDenied, DispatchRoute};
use claurst_core::effort::EffortLevel;
use claurst_tools::ToolContext;

use crate::startup_policy::ProviderSelectionSource;
use crate::QueryConfig;

const KNOWN_PROVIDERS: &[&str] = &[
    "anthropic",
    "openai",
    "google",
    "azure",
    "amazon-bedrock",
    "github-copilot",
    "codex",
    "openai-codex",
    "cohere",
    "minimax",
    "ollama",
    "lmstudio",
    "lm-studio",
    "llamacpp",
    "llama-cpp",
    "llama-server",
    "groq",
    "mistral",
    "deepseek",
    "xai",
    "perplexity",
    "cerebras",
    "openrouter",
    "togetherai",
    "together-ai",
    "deepinfra",
    "venice",
    "huggingface",
    "nvidia",
    "fireworks",
    "sambanova",
    "qwen",
    "alibaba",
    "siliconflow",
    "moonshot",
    "moonshotai",
    "zhipu",
    "zhipuai",
    "zai",
    "nebius",
    "novita",
    "ovhcloud",
    "scaleway",
    "vultr",
    "vultr-ai",
    "baseten",
    "friendli",
    "upstage",
    "stepfun",
];

struct ProviderSelection {
    provider: Arc<dyn LlmProvider>,
    source: ProviderSelectionSource,
    registry_owner: Option<Arc<dyn LlmProvider>>,
    cacheable: bool,
}

/// The provider/model route selected for one active query model.
///
/// This is deliberately runtime-only. It retains the actual provider handle
/// when one is available, while representing a missing or unavailable handle
/// as the configured legacy client route. The query loop keeps one value
/// across local context reselection and replaces it only when the effective
/// model changes.
pub(crate) struct ResolvedTarget<'a> {
    requested_model: String,
    provider_id: String,
    model_id: String,
    provider: Option<Arc<dyn LlmProvider>>,
    /// Fixed fallback remains available when startup binding fails.
    verified_dispatch_bound: bool,
    /// The exact legacy client retained for this query lifetime.  Keeping a
    /// reference here prevents local context reselection from silently
    /// rebuilding a different client or credentials between attempts.
    legacy_client: Option<&'a claurst_api::AnthropicClient>,
    capabilities: Option<ProviderCapabilities>,
    catalog_context_limit_tokens: Option<u64>,
    /// Route identities are frozen when this target is resolved. Re-reading a
    /// mutable provider through both the old and new target would make an
    /// account or endpoint switch compare equal to itself.
    dispatch_route_identity: Vec<Option<DispatchRoute>>,
    response_continuation_route_identity: Option<String>,
}

impl<'a> ResolvedTarget<'a> {
    pub(crate) fn resolve(
        config: &QueryConfig,
        tool_ctx: &ToolContext,
        effective_model: &str,
        client: &'a claurst_api::AnthropicClient,
    ) -> Result<Self, String> {
        let model_registry = config.model_registry.clone().unwrap_or_else(|| {
            let mut registry = ModelRegistry::new();
            if let Some(cache_dir) = dirs::cache_dir() {
                let cache_path = cache_dir.join("claurst").join("models_dev.json");
                registry.load_cache(&cache_path);
            }
            Arc::new(registry)
        });

        let (selected_provider_id, selected_model_id) =
            select_requested_route(tool_ctx, &model_registry, effective_model);
        if tool_ctx
            .config
            .provider_configs
            .get(&selected_provider_id)
            .is_some_and(|provider| !provider.enabled)
        {
            if let Some(binding) = config.verified_startup_binding.as_ref() {
                binding.clear_bound_provider_cache();
            }
            return Err("configured startup provider is disabled".to_string());
        }
        // An API-key Anthropic client already carries the startup policy.
        // Keep its existing query path, including configured tool hooks.
        let provider_dispatch_requested =
            selected_provider_id != "anthropic" || client.api_key_is_empty();

        let provider_selection = if provider_dispatch_requested {
            match resolve_provider(config, tool_ctx, &selected_provider_id) {
                Ok(selection) => selection,
                Err(error) => {
                    if let Some(binding) = config.verified_startup_binding.as_ref() {
                        binding.clear_bound_provider_cache();
                    }
                    return Err(error);
                }
            }
        } else {
            if let Some(binding) = config.verified_startup_binding.as_ref() {
                binding.clear_bound_provider_cache();
            }
            None
        };

        if provider_selection.is_none() {
            if let Some(binding) = config.verified_startup_binding.as_ref() {
                binding.clear_bound_provider_cache();
            }
        }

        let mut verified_dispatch_bound = config
            .verified_startup_binding
            .as_ref()
            .is_none_or(|binding| binding.provider_id() == selected_provider_id);
        let provider = if let Some(selection) = provider_selection {
            if let Some(binding) = config
                .verified_startup_binding
                .as_ref()
                .filter(|binding| binding.provider_id() == selected_provider_id)
            {
                if selection.cacheable {
                    let raw_provider = Arc::clone(&selection.provider);
                    Some(
                        match binding.try_bind_provider_for_selection(
                            selection.provider,
                            selection.source,
                            selection.registry_owner,
                        ) {
                            Ok(provider) => provider,
                            Err(_) => {
                                verified_dispatch_bound = false;
                                raw_provider
                            }
                        },
                    )
                } else {
                    binding.clear_bound_provider_cache();
                    verified_dispatch_bound = false;
                    Some(selection.provider)
                }
            } else {
                if let Some(binding) = config.verified_startup_binding.as_ref() {
                    binding.clear_bound_provider_cache();
                }
                Some(selection.provider)
            }
        } else {
            None
        };

        if config.anthropic_client_factory.is_some()
            && selected_provider_id == "anthropic"
            && provider_dispatch_requested
            && provider.is_none()
        {
            return Err("configured startup Anthropic route unavailable".to_string());
        }

        // A missing selected provider is an intentional legacy fallback. The
        // wire budget and eventual request must describe the route that can
        // actually run, rather than the unavailable requested provider.
        let (provider_id, model_id, provider) = if let Some(provider) = provider {
            (provider.id().to_string(), selected_model_id, Some(provider))
        } else {
            let legacy_id = legacy_provider_id(client.provider_kind());
            let model_id = if !provider_dispatch_requested && legacy_id == "anthropic" {
                selected_model_id
            } else {
                effective_model.to_string()
            };
            (legacy_id.to_string(), model_id, None)
        };

        if config
            .verified_startup_binding
            .as_ref()
            .is_some_and(|binding| binding.provider_id() != provider_id)
        {
            verified_dispatch_bound = false;
        }

        let catalog_model_id = model_id
            .strip_prefix(&format!("{provider_id}/"))
            .unwrap_or(&model_id);
        let catalog_entry = model_registry.get(&provider_id, catalog_model_id);
        let catalog_context_limit_tokens = catalog_entry
            .map(|entry| entry.info.context_window as u64)
            .filter(|limit| *limit > 0);

        let capabilities = provider.as_ref().map(|provider| {
            let mut capabilities = provider.capabilities();
            if let Some(entry) = catalog_entry {
                capabilities.image_input = entry.vision();
                capabilities.tool_calling = entry.tool_calling;
                capabilities.thinking = entry.reasoning;
            }
            capabilities
        });

        let has_provider = provider.is_some();
        let response_continuation_route_identity = provider
            .as_ref()
            .and_then(|provider| provider.response_continuation_route_identity(&model_id));
        let mut target = Self {
            requested_model: effective_model.to_string(),
            provider_id,
            model_id,
            provider,
            verified_dispatch_bound,
            legacy_client: if has_provider { None } else { Some(client) },
            capabilities,
            catalog_context_limit_tokens,
            dispatch_route_identity: Vec::new(),
            response_continuation_route_identity,
        };
        target.dispatch_route_identity = target.snapshot_dispatch_route_identity();
        Ok(target)
    }

    pub(crate) fn requested_model(&self) -> &str {
        &self.requested_model
    }

    pub(crate) fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub(crate) fn model_id(&self) -> &str {
        &self.model_id
    }

    pub(crate) fn provider(&self) -> Option<&Arc<dyn LlmProvider>> {
        self.provider.as_ref()
    }

    pub(crate) fn legacy_client(&self) -> Option<&'a claurst_api::AnthropicClient> {
        self.legacy_client
    }

    /// Resolve a dispatch route through this exact retained transport.  This
    /// is pure host lookup: adapters return a categorical unsupported result
    /// and no auth/profile/network work is performed here.
    pub(crate) fn dispatch_route(
        &self,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, DispatchDenied> {
        if !self.verified_dispatch_bound {
            return Err(DispatchDenied::UnsupportedRoute);
        }
        if let Some(provider) = self.provider.as_ref() {
            provider.dispatch_route(self.model_id(), effort)
        } else if let Some(client) = self.legacy_client {
            client.dispatch_route(self.model_id(), effort)
        } else {
            Err(DispatchDenied::UnsupportedRoute)
        }
    }

    /// Build the one lower-effort route pair used by verified selection.  The
    /// route identity must remain exact; only the registered effort binding
    /// may differ between the pair.
    pub(crate) fn verified_route_pair(
        &self,
        base: EffortLevel,
        lower: EffortLevel,
    ) -> Result<(), DispatchDenied> {
        let base_route = self.dispatch_route(base)?;
        let lower_route = self.dispatch_route(lower)?;
        if !same_route_identity(&base_route, &lower_route)
            || base_route.effort() == lower_route.effort()
        {
            return Err(DispatchDenied::RouteMismatch);
        }
        Ok(())
    }

    /// Compare the retained transport identity, including every route binding
    /// exposed by the installed policy.  A refreshed client/provider must not
    /// inherit proof produced by its predecessor, even when the model string
    /// and request bytes happen to be unchanged.
    pub(crate) fn same_transport_identity(&self, other: &Self) -> bool {
        if self.requested_model != other.requested_model
            || self.provider_id != other.provider_id
            || self.model_id != other.model_id
            || self.verified_dispatch_bound != other.verified_dispatch_bound
        {
            return false;
        }
        match (&self.provider, &other.provider) {
            (Some(left), Some(right)) if Arc::ptr_eq(left, right) => {}
            (None, None) => match (self.legacy_client, other.legacy_client) {
                (Some(left), Some(right)) if std::ptr::eq(left, right) => {}
                _ => return false,
            },
            _ => return false,
        }

        self.dispatch_route_identity == other.dispatch_route_identity
            && self.response_continuation_route_identity
                == other.response_continuation_route_identity
    }

    fn snapshot_dispatch_route_identity(&self) -> Vec<Option<DispatchRoute>> {
        [
            EffortLevel::None,
            EffortLevel::Minimal,
            EffortLevel::Low,
            EffortLevel::Medium,
            EffortLevel::High,
            EffortLevel::XHigh,
            EffortLevel::Max,
            EffortLevel::Ultracode,
        ]
        .into_iter()
        .map(|effort| self.dispatch_route(effort).ok())
        .collect()
    }

    pub(crate) fn capabilities(&self) -> Option<&ProviderCapabilities> {
        self.capabilities.as_ref()
    }

    pub(crate) fn catalog_context_limit_tokens(&self) -> Option<u64> {
        self.catalog_context_limit_tokens
    }
}

fn same_route_identity(left: &DispatchRoute, right: &DispatchRoute) -> bool {
    left.provider_id() == right.provider_id()
        && left.provider_instance_id() == right.provider_instance_id()
        && left.account_identity() == right.account_identity()
        && left.model_id() == right.model_id()
        && left.endpoint_identity() == right.endpoint_identity()
        && left.policy_revision() == right.policy_revision()
}

fn legacy_provider_id(provider: claurst_api::client::Provider) -> &'static str {
    match provider {
        claurst_api::client::Provider::Anthropic => "anthropic",
        claurst_api::client::Provider::Codex => "codex",
    }
}

fn select_requested_route(
    tool_ctx: &ToolContext,
    model_registry: &ModelRegistry,
    effective_model: &str,
) -> (String, String) {
    if let Some(provider) = tool_ctx
        .config
        .provider
        .as_deref()
        .filter(|provider| *provider != "anthropic")
    {
        let provider_prefix = format!("{provider}/");
        let model_id = effective_model
            .strip_prefix(&provider_prefix)
            .unwrap_or(effective_model)
            .to_string();
        return (provider.to_string(), model_id);
    }

    if let Some((provider, model)) = effective_model.split_once('/') {
        if KNOWN_PROVIDERS.contains(&provider) {
            return (provider.to_string(), model.to_string());
        }
        let fallback_provider = tool_ctx.config.provider.as_deref().unwrap_or("anthropic");
        return (fallback_provider.to_string(), effective_model.to_string());
    }

    if let Some(provider) = model_registry.find_provider_for_model(effective_model) {
        return (provider.to_string(), effective_model.to_string());
    }

    let fallback_provider = tool_ctx.config.provider.as_deref().unwrap_or("anthropic");
    (fallback_provider.to_string(), effective_model.to_string())
}

fn resolve_provider(
    config: &QueryConfig,
    tool_ctx: &ToolContext,
    provider_id: &str,
) -> Result<Option<ProviderSelection>, String> {
    if let Some(factory) = config.anthropic_client_factory.as_ref() {
        if provider_id == "anthropic" {
            if tool_ctx
                .config
                .provider_configs
                .get(provider_id)
                .is_some_and(|provider| !provider.enabled)
            {
                return Ok(None);
            }
            if let Some(provider) = config.provider_registry.as_ref().and_then(|registry| {
                registry
                    .get(&claurst_core::provider_id::ProviderId::new(provider_id))
                    .cloned()
            }) {
                return Ok(Some(ProviderSelection {
                    provider: Arc::clone(&provider),
                    source: ProviderSelectionSource::Registry,
                    registry_owner: Some(provider),
                    cacheable: true,
                }));
            }
            if let Some(provider) = claurst_api::registry::provider_from_config_with_factory(
                &tool_ctx.config,
                provider_id,
                factory,
            )
            .map_err(|_| {
                "configured startup Anthropic provider initialization failed".to_string()
            })? {
                return Ok(Some(ProviderSelection {
                    provider,
                    source: ProviderSelectionSource::Configured,
                    registry_owner: None,
                    cacheable: true,
                }));
            }
            return Ok(claurst_api::registry::runtime_provider_for_with_factory(
                provider_id,
                factory,
            )
            .map_err(|_| "configured startup Anthropic provider initialization failed".to_string())?
            .map(|provider| ProviderSelection {
                provider,
                source: ProviderSelectionSource::Runtime,
                registry_owner: None,
                cacheable: true,
            }));
        }
    }

    // A user-selected endpoint (including an environment override) is the
    // current route and account selection. Rebuild that provider first;
    // startup policy authority belongs only to the canonical retained route.
    let explicit_api_base = tool_ctx
        .config
        .provider_configs
        .get(provider_id)
        .and_then(|provider| provider.api_base.as_deref())
        .is_some_and(|base| !base.is_empty())
        || claurst_core::config::api_base_env_var_for_provider(provider_id)
            .and_then(|name| std::env::var(name).ok())
            .is_some_and(|base| !base.is_empty());
    let has_current_credentials = tool_ctx
        .config
        .resolve_provider_api_key(provider_id)
        .is_some_and(|key| !key.trim().is_empty());
    if explicit_api_base && !has_current_credentials {
        // A selected custom destination without a current credential cannot
        // safely fall back to an older registry account or endpoint.
        return Ok(None);
    }
    let configured_provider = has_current_credentials
        .then(|| claurst_api::registry::provider_from_config(&tool_ctx.config, provider_id))
        .flatten();
    let registry_provider = config.provider_registry.as_ref().and_then(|registry| {
        registry
            .get(&claurst_core::provider_id::ProviderId::new(provider_id))
            .cloned()
    });
    // Keep current configured credentials and auth-store credentials ahead of
    // an older registry handle. A registry provider is only a fallback for
    // fixtures or callers that have no current credential source.
    if let Some(provider) = configured_provider {
        return Ok(Some(ProviderSelection {
            provider,
            source: ProviderSelectionSource::Configured,
            registry_owner: None,
            cacheable: !explicit_api_base,
        }));
    }
    if let Some(provider) = claurst_api::registry::runtime_provider_for(provider_id) {
        return Ok(Some(ProviderSelection {
            provider,
            source: ProviderSelectionSource::Runtime,
            registry_owner: None,
            cacheable: !explicit_api_base,
        }));
    }
    Ok(registry_provider.map(|provider| ProviderSelection {
        provider: Arc::clone(&provider),
        source: ProviderSelectionSource::Registry,
        registry_owner: Some(provider),
        cacheable: !explicit_api_base,
    }))
}

#[cfg(test)]
mod tests {
    use super::{legacy_provider_id, ResolvedTarget};
    use claurst_api::{
        LlmProvider, ModelRegistry, ProviderCapabilities, ProviderError, ProviderRequest,
        ProviderResponse, ProviderStatus, ResponseContinuationSupport, StreamEvent,
        SystemPromptStyle,
    };
    use claurst_core::dispatch::{DispatchDenied, DispatchRoute, DispatchRouteSpec};
    use claurst_core::effort::EffortLevel;
    use claurst_core::provider_id::ProviderId;
    use futures::Stream;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct MutableRouteProvider {
        id: ProviderId,
        generation: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LlmProvider for MutableRouteProvider {
        fn id(&self) -> &ProviderId {
            &self.id
        }

        fn name(&self) -> &str {
            "mutable-route-test"
        }

        async fn create_message(
            &self,
            _request: ProviderRequest,
        ) -> Result<ProviderResponse, ProviderError> {
            panic!("route identity fixture does not dispatch inference")
        }

        async fn create_message_stream(
            &self,
            _request: ProviderRequest,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
            ProviderError,
        > {
            panic!("route identity fixture does not dispatch inference")
        }

        async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
            Ok(ProviderStatus::Healthy)
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                streaming: true,
                tool_calling: true,
                thinking: true,
                image_input: false,
                pdf_input: false,
                audio_input: false,
                video_input: false,
                caching: false,
                structured_output: false,
                system_prompt_style: SystemPromptStyle::TopLevel,
            }
        }

        fn response_continuation_support(&self) -> ResponseContinuationSupport {
            ResponseContinuationSupport::FullContextFallback
        }

        fn response_continuation_route_identity(&self, model: &str) -> Option<String> {
            Some(format!(
                "route:{}:{model}",
                self.generation.load(Ordering::SeqCst)
            ))
        }

        fn dispatch_route(
            &self,
            model: &str,
            effort: EffortLevel,
        ) -> Result<DispatchRoute, DispatchDenied> {
            let generation = self.generation.load(Ordering::SeqCst);
            DispatchRoute::new(DispatchRouteSpec {
                provider_id: self.id.to_string(),
                provider_instance_id: "mutable-instance".into(),
                account_identity: format!("account-{generation}"),
                model_id: model.into(),
                endpoint_identity: format!("endpoint-{generation}"),
                effort: effort.as_str().into(),
                policy_revision: 1,
            })
            .map_err(|_| DispatchDenied::UnsupportedRoute)
        }
    }

    fn snapshot_target(provider: Arc<dyn LlmProvider>) -> ResolvedTarget<'static> {
        let response_continuation_route_identity =
            provider.response_continuation_route_identity("model-a");
        let mut target = ResolvedTarget {
            requested_model: "model-a".into(),
            provider_id: provider.id().to_string(),
            model_id: "model-a".into(),
            provider: Some(provider),
            verified_dispatch_bound: true,
            legacy_client: None,
            capabilities: None,
            catalog_context_limit_tokens: None,
            dispatch_route_identity: Vec::new(),
            response_continuation_route_identity,
        };
        target.dispatch_route_identity = target.snapshot_dispatch_route_identity();
        target
    }

    #[test]
    fn transport_identity_is_frozen_across_account_and_endpoint_mutation() {
        let mutable = Arc::new(MutableRouteProvider {
            id: ProviderId::new("mutable-route"),
            generation: AtomicUsize::new(1),
        });
        let provider: Arc<dyn LlmProvider> = mutable.clone();
        let first = snapshot_target(provider.clone());
        let unchanged = snapshot_target(provider.clone());
        assert!(first.same_transport_identity(&unchanged));

        mutable.generation.store(2, Ordering::SeqCst);
        let changed = snapshot_target(provider);
        assert!(!first.same_transport_identity(&changed));
    }

    #[test]
    fn legacy_codex_route_keeps_codex_identity_and_does_not_use_anthropic_budget() {
        let registry = ModelRegistry::new();
        let model = "claude-sonnet-4-6";

        assert_eq!(
            legacy_provider_id(claurst_api::client::Provider::Codex),
            "codex"
        );
        assert!(registry.get("codex", model).is_none());
        assert!(registry.get("anthropic", model).is_some());
        assert_eq!(
            registry
                .get(
                    legacy_provider_id(claurst_api::client::Provider::Codex),
                    model,
                )
                .map(|entry| entry.info.context_window),
            None
        );
    }

    #[test]
    fn legacy_anthropic_route_uses_anthropic_catalog_identity() {
        let registry = ModelRegistry::new();
        let model = "claude-sonnet-4-6";

        assert_eq!(
            registry
                .get(
                    legacy_provider_id(claurst_api::client::Provider::Anthropic),
                    model,
                )
                .map(|entry| entry.info.context_window),
            Some(1_000_000)
        );
    }
}
