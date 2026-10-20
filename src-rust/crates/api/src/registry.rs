// registry.rs — Registry of all available LLM providers.
//
// Holds an `Arc<dyn LlmProvider>` for each registered provider and exposes
// lookup, health-check, and default-provider helpers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use claurst_core::dispatch::DispatchDenied;
use claurst_core::ProviderId;

use crate::client::{AnthropicClient, ClientConfig};
use crate::client_factory::AnthropicClientFactory;
use crate::dispatch_policy::DispatchPolicy;
use crate::provider::LlmProvider;
use crate::provider_types::ProviderStatus;
use crate::providers::{
    AnthropicProvider, AzureProvider, BedrockProvider, CodexProvider, CohereProvider,
    CopilotProvider, FreeEntry, FreeProvider, GoogleProvider, MinimaxProvider, OpenAiProvider,
    FREE_CATALOG,
};

fn normalize_openai_compat_base(override_base: &str) -> String {
    let trimmed = override_base.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{}/v1", trimmed)
    }
}

fn normalize_openai_base(override_base: &str) -> String {
    let trimmed = override_base.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.trim_end_matches("/v1").to_string()
    } else {
        trimmed.to_string()
    }
}

fn canonical_local_provider_id(provider_id: &str) -> &str {
    match provider_id {
        "lmstudio" => ProviderId::LM_STUDIO,
        "llamacpp" | "llama-server" => ProviderId::LLAMA_CPP,
        _ => provider_id,
    }
}

pub fn resolve_provider_api_base(
    config: &claurst_core::config::Config,
    provider_id: &str,
) -> Option<String> {
    let base = config.resolve_provider_api_base(provider_id)?;
    if provider_id == "openai" {
        Some(normalize_openai_base(&base))
    } else if crate::providers::openai_compat_providers::provider_for_id(provider_id).is_some() {
        Some(normalize_openai_compat_base(&base))
    } else {
        Some(base)
    }
}

/// Registry of all available LLM providers.
/// Holds `Arc<dyn LlmProvider>` for each registered provider.
pub struct ProviderRegistry {
    providers: HashMap<ProviderId, Arc<dyn LlmProvider>>,
    default_provider_id: ProviderId,
}

fn provider_from_key(provider_id: &str, key: String) -> Option<Arc<dyn LlmProvider>> {
    use crate::providers::openai_compat_providers as p;

    if let Some(provider) = p::provider_for_id(provider_id) {
        return Some(Arc::new(provider.with_api_key(key)));
    }

    match provider_id {
        "anthropic" => Some(Arc::new(AnthropicProvider::from_config(ClientConfig {
            api_key: key,
            ..Default::default()
        }))),
        "minimax" => Some(Arc::new(MinimaxProvider::new(key))),
        "openai" => Some(Arc::new(OpenAiProvider::new(key))),
        "google" => Some(Arc::new(GoogleProvider::new(key))),
        "github-copilot" => Some(Arc::new(CopilotProvider::new(key))),
        "codex" | "openai-codex" => {
            // The Codex provider is OAuth-based; the `key` field is not used.
            // Load from the stored token file instead.
            CodexProvider::from_stored().map(|p| Arc::new(p) as Arc<dyn LlmProvider>)
        }
        "cohere" => Some(Arc::new(CohereProvider::new(key))),
        "custom-openai" => Some(Arc::new(p::custom_openai().with_api_key(key))),
        // "free" needs two keys (Zen + OpenRouter) — single-key path doesn't
        // apply.  The auth-store-aware path `runtime_provider_for` handles it.
        "free" => build_free_provider(),
        _ => None,
    }
}

/// Build a [`FreeProvider`] by walking [`FREE_CATALOG`] and pulling any keys
/// the user has stored in the auth store. Each catalog entry whose upstream
/// has a key becomes one link in the fallback chain.
///
/// Returns `None` only if *no* catalog entry has a configured key — a single
/// key is enough to run, and more is better.
pub fn build_free_provider() -> Option<Arc<dyn LlmProvider>> {
    let auth_store = claurst_core::AuthStore::load();
    let mut chain: Vec<FreeEntry> = Vec::new();

    for upstream in FREE_CATALOG {
        let key = match upstream.id {
            // OpenCode Zen and Go share `OPENCODE_API_KEY`; accept either slot.
            "opencode-zen" => auth_store
                .api_key_for(claurst_core::ProviderId::OPENCODE_ZEN)
                .or_else(|| auth_store.api_key_for(claurst_core::ProviderId::OPENCODE_GO)),
            other => auth_store.api_key_for(other),
        }
        .filter(|k| !k.trim().is_empty());

        let Some(key) = key else {
            continue;
        };

        let provider: Option<Arc<dyn LlmProvider>> = match upstream.id {
            "google" => Some(Arc::new(GoogleProvider::new(key))),
            "cohere" => Some(Arc::new(CohereProvider::new(key))),
            id => crate::providers::openai_compat_providers::provider_for_id(id)
                .map(|p| Arc::new(p.with_api_key(key)) as Arc<dyn LlmProvider>),
        };

        if let Some(provider) = provider {
            chain.push(FreeEntry {
                upstream: *upstream,
                provider,
            });
        }
    }

    if chain.is_empty() {
        return None;
    }
    Some(Arc::new(FreeProvider::new(chain)) as Arc<dyn LlmProvider>)
}

pub fn provider_from_config(
    config: &claurst_core::config::Config,
    provider_id: &str,
) -> Option<Arc<dyn LlmProvider>> {
    let provider_cfg = config.provider_configs.get(provider_id);
    if provider_cfg.is_some_and(|provider| !provider.enabled) {
        return None;
    }

    let api_key = config.resolve_provider_api_key(provider_id);
    let api_base = resolve_provider_api_base(config, provider_id).filter(|base| !base.is_empty());

    use crate::providers;

    match provider_id {
        "anthropic" => None,
        // Composite "Free" provider — two keys are pulled internally from the
        // auth store; the `api_key` resolved above is ignored.
        "free" => build_free_provider(),
        "openai" => {
            let mut provider = OpenAiProvider::new(api_key.unwrap_or_default());
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            // The loopback transport is available only in the explicit A016
            // test feature; normal builds cannot select this option.
            #[cfg(feature = "a016-openai-loopback")]
            if let Some(loopback_base) = provider_cfg
                .and_then(|config| config.options.get("a016_test_loopback_base_url"))
                .and_then(|value| value.as_str())
                .filter(|base| !base.is_empty())
            {
                provider = provider.with_test_loopback_base_url(loopback_base.to_string());
            }
            Some(Arc::new(provider))
        }
        "google" => api_key.map(|key| {
            let mut provider = GoogleProvider::new(key);
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Arc::new(provider) as Arc<dyn LlmProvider>
        }),
        "minimax" => api_key.map(|key| {
            let mut provider = MinimaxProvider::new(key);
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            if let Some(service_tier) = provider_cfg
                .and_then(|config| config.options.get("service_tier"))
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
            {
                provider = provider.with_service_tier(service_tier);
            }
            Arc::new(provider) as Arc<dyn LlmProvider>
        }),
        "azure" => {
            let resource_name = provider_cfg
                .and_then(|provider| provider.options.get("resource_name"))
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    std::env::var("AZURE_RESOURCE_NAME")
                        .ok()
                        .filter(|value| !value.is_empty())
                });

            match (resource_name, api_key) {
                (Some(resource_name), Some(key)) => {
                    Some(Arc::new(AzureProvider::new(resource_name, key)) as Arc<dyn LlmProvider>)
                }
                _ => None,
            }
        }
        "ollama" => {
            let mut provider = providers::ollama();
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "lmstudio" | "lm-studio" => {
            let mut provider = providers::lm_studio();
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "llamacpp" | "llama-cpp" | "llama-server" => {
            let mut provider = providers::llama_cpp();
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "deepseek" => {
            let mut provider = providers::deepseek();
            if let Some(key) = api_key {
                provider = provider.with_api_key(key);
            }
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "groq" => {
            let mut provider = providers::groq();
            if let Some(key) = api_key {
                provider = provider.with_api_key(key);
            }
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "xai" => {
            let mut provider = providers::xai();
            if let Some(key) = api_key {
                provider = provider.with_api_key(key);
            }
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "openrouter" => {
            let mut provider = providers::openrouter();
            if let Some(key) = api_key {
                provider = provider.with_api_key(key);
            }
            if let Some(base) = api_base {
                provider = provider.with_base_url(base);
            }
            Some(Arc::new(provider))
        }
        "cohere" => api_key.map(|key| Arc::new(CohereProvider::new(key)) as Arc<dyn LlmProvider>),
        "github-copilot" => {
            api_key.map(|key| Arc::new(CopilotProvider::new(key)) as Arc<dyn LlmProvider>)
        }
        "codex" | "openai-codex" => {
            CodexProvider::from_stored().map(|provider| Arc::new(provider) as Arc<dyn LlmProvider>)
        }
        _ => api_key.and_then(|key| provider_from_key(provider_id, key)),
    }
}

pub fn runtime_provider_for(provider_id: &str) -> Option<Arc<dyn LlmProvider>> {
    use crate::providers::openai_compat_providers as p;

    // Local providers never require an API key — build them directly so that
    // the auth-store bypass below doesn't silently drop them.
    // Accept both the hyphenated canonical IDs ("llama-cpp", "lm-studio") and
    // the non-hyphenated aliases ("llamacpp", "lmstudio") used throughout the
    // TUI / connect dialog.
    match provider_id {
        "ollama" => return Some(Arc::new(p::ollama())),
        "lmstudio" | "lm-studio" => return Some(Arc::new(p::lm_studio())),
        // "llama-server" is the binary name for the modern llama.cpp server.
        "llamacpp" | "llama-cpp" | "llama-server" => return Some(Arc::new(p::llama_cpp())),
        "codex" | "openai-codex" => {
            return CodexProvider::from_stored().map(|p| Arc::new(p) as Arc<dyn LlmProvider>);
        }
        // "free" pulls two keys (Zen + OpenRouter) from the auth store and
        // wraps them in a fallback composite — handled here so the generic
        // single-key path below doesn't short-circuit on a missing key.
        "free" => return build_free_provider(),
        _ => {}
    }

    let auth_store = claurst_core::AuthStore::load();
    let key = auth_store.api_key_for(provider_id)?;
    if key.is_empty() {
        return None;
    }
    provider_from_key(provider_id, key)
}

/// Build a runtime provider through the host-owned Anthropic factory.
///
/// Anthropic credentials are read once from the auth store for this call;
/// other providers retain the existing runtime construction path.
pub fn runtime_provider_for_with_factory(
    provider_id: &str,
    factory: &AnthropicClientFactory,
) -> anyhow::Result<Option<Arc<dyn LlmProvider>>> {
    if provider_id != ProviderId::ANTHROPIC {
        return Ok(runtime_provider_for(provider_id));
    }

    let auth_store = claurst_core::AuthStore::load();
    let Some(api_key) = auth_store
        .api_key_for(provider_id)
        .filter(|key| !key.trim().is_empty())
    else {
        return Ok(None);
    };

    let client = factory.create(ClientConfig {
        api_key,
        ..Default::default()
    })?;
    Ok(Some(Arc::new(AnthropicProvider::new(Arc::new(client)))))
}

/// Build a configured provider through the host-owned Anthropic factory.
///
/// Non-Anthropic providers delegate to the established configuration path so
/// their key, alias, endpoint, and option precedence remain unchanged.
pub fn provider_from_config_with_factory(
    config: &claurst_core::config::Config,
    provider_id: &str,
    factory: &AnthropicClientFactory,
) -> anyhow::Result<Option<Arc<dyn LlmProvider>>> {
    if provider_id != ProviderId::ANTHROPIC {
        return Ok(provider_from_config(config, provider_id));
    }

    if config
        .provider_configs
        .get(provider_id)
        .is_some_and(|provider| !provider.enabled)
    {
        return Ok(None);
    }

    let Some(api_key) = config.resolve_provider_api_key(provider_id) else {
        return Ok(None);
    };
    let defaults = ClientConfig::default();
    let client_config = ClientConfig {
        api_key,
        api_base: resolve_provider_api_base(config, provider_id)
            .filter(|base| !base.is_empty())
            .unwrap_or(defaults.api_base),
        request_timeout: Duration::from_secs(config.resolve_request_timeout_secs(provider_id)),
        ..defaults
    };

    let client = factory.create(client_config)?;
    Ok(Some(Arc::new(AnthropicProvider::new(Arc::new(client)))))
}

impl ProviderRegistry {
    /// Create an empty registry with Anthropic as the default provider ID.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider_id: ProviderId::new(ProviderId::ANTHROPIC),
        }
    }

    /// Register a provider. Returns `&mut self` for builder chaining.
    pub fn register(&mut self, provider: Arc<dyn LlmProvider>) -> &mut Self {
        let id = provider.id().clone();
        self.providers.insert(id, provider);
        self
    }

    /// Set the default provider by ID.
    ///
    /// # Panics
    /// Panics if no provider with that ID has been registered.
    pub fn set_default(&mut self, id: ProviderId) -> &mut Self {
        let canonical_id = ProviderId::new(canonical_local_provider_id(&id));
        assert!(
            self.providers.contains_key(&canonical_id),
            "set_default: provider '{}' is not registered",
            id,
        );
        self.default_provider_id = canonical_id;
        self
    }

    /// Get a provider by ID.
    pub fn get(&self, id: &ProviderId) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(id).or_else(|| {
            let canonical_id = canonical_local_provider_id(id);
            (canonical_id != &**id)
                .then(|| self.providers.get(&ProviderId::new(canonical_id)))
                .flatten()
        })
    }

    /// Clone the registry while attaching immutable host dispatch policy to
    /// one retained provider instance. Providers that do not explicitly
    /// support policy binding remain fixed/base-only.
    pub fn with_dispatch_policy(
        &self,
        provider_id: &str,
        policy: Arc<DispatchPolicy>,
    ) -> Result<Self, DispatchDenied> {
        let requested = ProviderId::new(provider_id);
        let provider = self
            .get(&requested)
            .ok_or(DispatchDenied::UnsupportedRoute)?;
        let bound = provider.bind_dispatch_policy(policy)?;
        let mut providers = self.providers.clone();
        providers.insert(bound.id().clone(), bound);
        Ok(Self {
            providers,
            default_provider_id: self.default_provider_id.clone(),
        })
    }

    /// Get the default provider.
    pub fn default_provider(&self) -> Option<&Arc<dyn LlmProvider>> {
        self.providers.get(&self.default_provider_id)
    }

    /// Get the default provider ID.
    pub fn default_provider_id(&self) -> &ProviderId {
        &self.default_provider_id
    }

    /// List all registered provider IDs.
    pub fn provider_ids(&self) -> Vec<&ProviderId> {
        self.providers.keys().collect()
    }

    /// Check health of all providers sequentially.
    /// Returns `(provider_id, status)` pairs.
    pub async fn check_all_health(&self) -> Vec<(ProviderId, ProviderStatus)> {
        let mut results = Vec::new();
        for (id, provider) in &self.providers {
            let status = provider
                .health_check()
                .await
                .unwrap_or(ProviderStatus::Unavailable {
                    reason: "health check failed".to_string(),
                });
            results.push((id.clone(), status));
        }
        results
    }

    /// Convenience: build a registry with just Anthropic registered as the
    /// default provider.  Takes the same [`ClientConfig`] that
    /// [`AnthropicClient`] takes.
    ///
    /// [`AnthropicClient`]: crate::client::AnthropicClient
    pub fn with_anthropic(config: ClientConfig) -> Self {
        let mut registry = Self::new();
        let provider = Arc::new(AnthropicProvider::from_config(config));
        registry.register(provider);
        registry
    }

    fn register_environment_providers(&mut self) {
        self.with_openai_if_key_set()
            .with_google_if_key_set()
            .with_azure_if_configured()
            .with_bedrock_if_configured()
            .with_copilot_if_configured()
            .with_codex_if_configured()
            .with_cohere_if_key_set()
            .with_available_providers();
    }

    fn register_auth_store_providers(&mut self) {
        let auth_store = claurst_core::AuthStore::load();

        for provider_id in auth_store.credentials.keys() {
            let pid = claurst_core::ProviderId::new(provider_id.as_str());
            // Skip if already registered from env vars or a host-supplied
            // client.
            if self.get(&pid).is_some() {
                continue;
            }
            let Some(key) = auth_store
                .api_key_for(provider_id)
                .filter(|key| !key.is_empty())
            else {
                continue;
            };
            if let Some(provider) = provider_from_key(provider_id, key) {
                self.register(provider);
            }
        }
    }

    pub fn from_config(
        config: &claurst_core::config::Config,
        anthropic_config: ClientConfig,
    ) -> Self {
        // Apply the user-configured request timeout (issue #175) before any
        // provider HTTP clients are built, so they all honour it. Uses the
        // active provider's resolved value (per-provider override or global).
        crate::set_request_timeout_secs(
            config.resolve_request_timeout_secs(config.selected_provider_id()),
        );
        let mut registry = Self::from_environment_with_auth_store(anthropic_config);
        let active_provider = config.selected_provider_id();

        let mut configured_provider_ids: Vec<String> =
            config.provider_configs.keys().cloned().collect();
        if configured_provider_ids
            .iter()
            .all(|id| id != active_provider)
        {
            configured_provider_ids.push(active_provider.to_string());
        }

        for provider_id in configured_provider_ids {
            if let Some(provider) = provider_from_config(config, &provider_id) {
                registry.register(provider);
            }
        }

        let default_provider_id = ProviderId::new(active_provider);
        if registry.get(&default_provider_id).is_some() {
            registry.set_default(default_provider_id);
        }

        registry
    }

    /// Build a configured registry while retaining the exact host-created
    /// Anthropic client and its dispatch identity.
    pub fn from_config_with_anthropic_client(
        config: &claurst_core::config::Config,
        anthropic_client: Arc<AnthropicClient>,
    ) -> Self {
        crate::set_request_timeout_secs(
            config.resolve_request_timeout_secs(config.selected_provider_id()),
        );
        let mut registry = Self::new();
        registry.register(Arc::new(AnthropicProvider::new(anthropic_client)));
        registry.register_environment_providers();
        registry.register_auth_store_providers();

        let active_provider = config.selected_provider_id();
        let mut configured_provider_ids: Vec<String> =
            config.provider_configs.keys().cloned().collect();
        if configured_provider_ids
            .iter()
            .all(|id| id != active_provider)
        {
            configured_provider_ids.push(active_provider.to_string());
        }
        for provider_id in configured_provider_ids {
            if provider_id == ProviderId::ANTHROPIC {
                continue;
            }
            if let Some(provider) = provider_from_config(config, &provider_id) {
                registry.register(provider);
            }
        }

        let default_provider_id = ProviderId::new(active_provider);
        if registry.get(&default_provider_id).is_some() {
            registry.set_default(default_provider_id);
        }
        registry
    }

    /// Register [`GoogleProvider`] if `GOOGLE_API_KEY` or
    /// `GOOGLE_GENERATIVE_AI_API_KEY` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_google_if_key_set(&mut self) -> &mut Self {
        let key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_GENERATIVE_AI_API_KEY"));
        if let Ok(key) = key {
            let provider = Arc::new(GoogleProvider::new(key));
            self.register(provider);
        }
        self
    }

    /// Register [`OpenAiProvider`] if `OPENAI_API_KEY` is set in the
    /// environment.  Returns `&mut self` for builder chaining.
    pub fn with_openai_if_key_set(&mut self) -> &mut Self {
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let provider = Arc::new(OpenAiProvider::new(key));
            self.register(provider);
        }
        self
    }

    /// Register [`AzureProvider`] if `AZURE_API_KEY` and `AZURE_RESOURCE_NAME`
    /// are set in the environment.  Returns `&mut self` for builder chaining.
    pub fn with_azure_if_configured(&mut self) -> &mut Self {
        if let Some(p) = AzureProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`BedrockProvider`] if AWS credentials are available in the
    /// environment (`AWS_ACCESS_KEY_ID`+`AWS_SECRET_ACCESS_KEY` or
    /// `AWS_BEARER_TOKEN_BEDROCK`).  Returns `&mut self` for builder chaining.
    pub fn with_bedrock_if_configured(&mut self) -> &mut Self {
        if let Some(p) = BedrockProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CopilotProvider`] if `GITHUB_TOKEN` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_copilot_if_configured(&mut self) -> &mut Self {
        if let Some(p) = CopilotProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CodexProvider`] if stored Codex OAuth tokens are available in
    /// `~/.claurst/codex_tokens.json`.  Returns `&mut self` for builder chaining.
    pub fn with_codex_if_configured(&mut self) -> &mut Self {
        if let Some(p) = CodexProvider::from_stored() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Register [`CohereProvider`] if `COHERE_API_KEY` is set in the environment.
    /// Returns `&mut self` for builder chaining.
    pub fn with_cohere_if_key_set(&mut self) -> &mut Self {
        if let Some(p) = CohereProvider::from_env() {
            self.register(Arc::new(p));
        }
        self
    }

    /// Build a registry with **all** providers that have credentials configured
    /// in the environment.  Anthropic is always the default provider.
    ///
    /// This is the recommended constructor for production use.
    pub fn from_environment(anthropic_config: ClientConfig) -> Self {
        let mut registry = Self::with_anthropic(anthropic_config);
        registry.register_environment_providers();
        registry
    }

    /// Build a registry that checks **both** environment variables and the
    /// persistent [`AuthStore`] (`~/.claurst/auth.json`) for credentials.
    ///
    /// This ensures that API keys stored via `/connect` or `claurst auth` are
    /// picked up at startup, not just env vars.  Falls back to
    /// `from_environment` for providers that only support env-var config, and
    /// adds any extra providers that have keys in the auth store.
    ///
    /// [`AuthStore`]: claurst_core::AuthStore
    pub fn from_environment_with_auth_store(anthropic_config: ClientConfig) -> Self {
        // Start with env-based registration.
        let mut registry = Self::from_environment(anthropic_config);
        registry.register_auth_store_providers();
        registry
    }

    /// Register all providers that have environment variable credentials set.
    ///
    /// Local providers (Ollama, LM Studio, llama.cpp) are always registered
    /// regardless of credentials — `health_check()` will report them as
    /// unavailable if the server is not running.
    ///
    /// Remote API-key providers are only registered when their respective
    /// environment variables are set (non-empty).
    ///
    /// Returns `&mut self` for builder chaining.
    pub fn with_available_providers(&mut self) -> &mut Self {
        use crate::providers::openai_compat_providers as p;

        // Local providers — always try to register.
        self.register(Arc::new(p::ollama()));
        self.register(Arc::new(p::lm_studio()));
        self.register(Arc::new(p::llama_cpp()));

        // Remote providers — only register when an API key is present.
        if std::env::var("DEEPSEEK_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::deepseek()));
        }
        if std::env::var("GROQ_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::groq()));
        }
        if std::env::var("XAI_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::xai()));
        }
        if std::env::var("OPENROUTER_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::openrouter()));
        }
        if std::env::var("TOGETHER_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::together_ai()));
        }
        if std::env::var("PERPLEXITY_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::perplexity()));
        }
        if std::env::var("CEREBRAS_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::cerebras()));
        }
        if std::env::var("DEEPINFRA_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::deepinfra()));
        }
        if std::env::var("VENICE_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::venice()));
        }
        if std::env::var("DASHSCOPE_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::qwen()));
        }
        if std::env::var("MISTRAL_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::mistral()));
        }
        if std::env::var("SAMBANOVA_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::sambanova()));
        }
        if std::env::var("HF_TOKEN")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::huggingface()));
        }
        if std::env::var("MINIMAX_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            let key = std::env::var("MINIMAX_API_KEY").unwrap_or_default();
            self.register(Arc::new(MinimaxProvider::new(key)));
        }
        if std::env::var("NVIDIA_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::nvidia()));
        }
        if std::env::var("SILICONFLOW_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::siliconflow()));
        }
        if std::env::var("MOONSHOT_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::moonshot()));
        }
        if std::env::var("ZHIPU_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::zhipu()));
        }
        if std::env::var("ZAI_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::zai()));
        }
        if std::env::var("NEBIUS_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::nebius()));
        }
        if std::env::var("NOVITA_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::novita()));
        }
        if std::env::var("OVHCLOUD_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::ovhcloud()));
        }
        if std::env::var("SCALEWAY_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::scaleway()));
        }
        if std::env::var("VULTR_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::vultr_ai()));
        }
        if std::env::var("BASETEN_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::baseten()));
        }
        if std::env::var("FRIENDLI_TOKEN")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::friendli()));
        }
        if std::env::var("UPSTAGE_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::upstage()));
        }
        if std::env::var("STEPFUN_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::stepfun()));
        }
        if std::env::var("FIREWORKS_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::fireworks()));
        }
        if std::env::var("OPENCODE_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            self.register(Arc::new(p::opencode_go()));
        }
        self
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_policy::{DispatchModelSpec, DispatchPolicy, EffortWireBinding};
    use crate::providers;
    use claurst_core::config::{Config, ProviderConfig};
    use claurst_core::effort::EffortLevel;

    const MODEL: &str = "registry-factory-fixture";

    fn factory(revision: u64) -> AnthropicClientFactory {
        AnthropicClientFactory::with_dispatch_policy(Arc::new(
            DispatchPolicy::new(
                revision,
                vec![DispatchModelSpec {
                    model_id: MODEL.to_string(),
                    efforts: vec![EffortWireBinding::anthropic_thinking(EffortLevel::High)],
                }],
            )
            .expect("fixture policy"),
        ))
    }

    fn client_config(api_key: &str, api_base: &str) -> ClientConfig {
        ClientConfig {
            api_key: api_key.to_string(),
            api_base: api_base.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn local_provider_aliases_resolve_to_canonical_registrations() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(providers::lm_studio()));
        registry.register(Arc::new(providers::llama_cpp()));

        let lm_studio = registry
            .get(&ProviderId::new("lmstudio"))
            .expect("lmstudio alias should resolve");
        let llama_cpp = registry
            .get(&ProviderId::new("llamacpp"))
            .expect("llamacpp alias should resolve");

        assert_eq!(&**lm_studio.id(), ProviderId::LM_STUDIO);
        assert_eq!(&**llama_cpp.id(), ProviderId::LLAMA_CPP);
    }

    #[test]
    fn alias_can_select_canonical_default_provider() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(providers::lm_studio()));
        registry.set_default(ProviderId::new("lmstudio"));

        assert_eq!(&**registry.default_provider_id(), ProviderId::LM_STUDIO);
    }

    #[test]
    fn configured_registry_reuses_host_anthropic_client_identity_and_policy() {
        let factory = factory(31);
        let host_client = Arc::new(
            factory
                .create(client_config("host-key", "http://host.example"))
                .expect("host client"),
        );
        let expected = host_client
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("host route");
        let config = Config {
            provider: Some(ProviderId::ANTHROPIC.to_string()),
            api_key: Some("different-config-key".to_string()),
            ..Default::default()
        };

        let registry =
            ProviderRegistry::from_config_with_anthropic_client(&config, Arc::clone(&host_client));
        let provider = registry
            .get(&ProviderId::new(ProviderId::ANTHROPIC))
            .expect("registered Anthropic provider");
        let actual = provider
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("registered route");

        assert_eq!(
            actual.provider_instance_id(),
            expected.provider_instance_id()
        );
        assert_eq!(actual.account_identity(), expected.account_identity());
        assert_eq!(actual.endpoint_identity(), expected.endpoint_identity());
        assert_eq!(actual.policy_revision(), 31);
    }

    #[test]
    fn configured_factory_path_preserves_key_base_and_rebuilds_identity() {
        let mut config = Config {
            provider: Some(ProviderId::ANTHROPIC.to_string()),
            api_key: Some("configured-key".to_string()),
            ..Default::default()
        };
        config.provider_configs.insert(
            ProviderId::ANTHROPIC.to_string(),
            ProviderConfig {
                api_base: Some("http://configured.example".to_string()),
                ..Default::default()
            },
        );
        let factory = factory(37);
        let first = provider_from_config_with_factory(&config, "anthropic", &factory)
            .expect("first factory provider")
            .expect("configured provider")
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("first route");
        let second = provider_from_config_with_factory(&config, "anthropic", &factory)
            .expect("second factory provider")
            .expect("configured provider")
            .dispatch_route(MODEL, EffortLevel::High)
            .expect("second route");

        assert_eq!(first.policy_revision(), 37);
        assert_eq!(
            first.endpoint_identity(),
            "http://configured.example/v1/messages"
        );
        assert_ne!(first.provider_instance_id(), second.provider_instance_id());
        assert_ne!(first.account_identity(), second.account_identity());

        config
            .provider_configs
            .get_mut("anthropic")
            .unwrap()
            .enabled = false;
        assert!(
            provider_from_config_with_factory(&config, "anthropic", &factory)
                .expect("disabled factory path")
                .is_none()
        );

        config
            .provider_configs
            .get_mut("anthropic")
            .unwrap()
            .enabled = true;
        config.api_key = None;
        assert!(
            provider_from_config_with_factory(&config, "anthropic", &factory)
                .expect("missing-key factory path")
                .is_none()
        );
    }

    #[test]
    fn factory_path_delegates_non_anthropic_aliases() {
        let config = Config::default();
        let provider = provider_from_config_with_factory(&config, "ollama", &factory(41))
            .expect("local provider factory path")
            .expect("local provider");
        assert_eq!(&**provider.id(), ProviderId::OLLAMA);
    }
}
