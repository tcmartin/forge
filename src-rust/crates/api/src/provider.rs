// provider.rs — Core trait definitions for the provider abstraction layer.
//
// Every LLM provider adapter must implement `LlmProvider`.  The trait is
// intentionally minimal: only what is needed to send messages, list models,
// and report capabilities.  Auth concerns live in `auth.rs`.

use async_trait::async_trait;
use claurst_core::dispatch::{DispatchDenied, DispatchRoute};
use claurst_core::effort::EffortLevel;
use claurst_core::provider_id::{ModelId, ProviderId};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

use crate::dispatch_policy::DispatchPolicy;
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus,
    ResponseContinuationSupport, StreamEvent,
};

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------

/// Static metadata about a model available through a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The model's unique identifier (e.g. `"claude-opus-4-5"`).
    pub id: ModelId,

    /// The provider that hosts this model.
    pub provider_id: ProviderId,

    /// Human-readable display name (e.g. `"Claude Opus 4.5"`).
    pub name: String,

    /// Total context window size in tokens.
    pub context_window: u32,

    /// Maximum number of tokens the model can emit in a single response.
    pub max_output_tokens: u32,

    /// First public availability (ISO 8601 date), when known.  Catalog-backed
    /// providers populate this from the models.dev snapshot; live-discovery
    /// providers may leave it `None`.  Drives date-DESC listing in pickers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,

    /// Lifecycle status string (`"active"`, `"beta"`, …), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl Default for ModelInfo {
    /// Empty placeholder used with struct-update syntax
    /// (`ModelInfo { id, .., ..Default::default() }`) so the optional metadata
    /// fields don't have to be repeated at every construction site.
    fn default() -> Self {
        Self {
            id: ModelId::new(""),
            provider_id: ProviderId::new(""),
            name: String::new(),
            context_window: 0,
            max_output_tokens: 0,
            release_date: None,
            status: None,
        }
    }
}

// ---------------------------------------------------------------------------
// LlmProvider
// ---------------------------------------------------------------------------

/// The core trait every LLM provider adapter must implement.
///
/// Implementors are required to be `Send + Sync` so they can be held behind an
/// `Arc<dyn LlmProvider>` and shared across async tasks.
// async-trait emits a bare `#[must_use]` on its boxed futures;
// Rust 1.99 Clippy sees the boxed future's own marker and reports a duplicate.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Unique machine-readable identifier, e.g. `"anthropic"`, `"openai"`.
    fn id(&self) -> &ProviderId;

    /// Human-readable display name, e.g. `"Anthropic"`, `"OpenAI"`.
    fn name(&self) -> &str;

    /// Send a message and receive a complete (non-streaming) response.
    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError>;

    /// Send a message and receive a streaming response as a pinned `Stream` of
    /// provider-agnostic `StreamEvent`s.
    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>;

    /// Discover models exposed by a *live* endpoint (e.g. `GET /v1/models` for
    /// a local Ollama/LM Studio server, or a Copilot entitlement query).
    ///
    /// Catalog-backed providers (Anthropic, OpenAI, Google, …) do **not**
    /// override this: their model list is a read-only projection of the
    /// models.dev catalog held in [`crate::ModelRegistry`], so the picker never
    /// turns a provider return value into the displayed list.  The default impl
    /// therefore returns an empty vector — only providers whose models cannot be
    /// known ahead of time (local runtimes, dynamic gateways) implement it.
    async fn discover_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(Vec::new())
    }

    /// Check whether the provider is authenticated and reachable.
    ///
    /// Typically involves a lightweight API call (e.g. listing models or
    /// fetching account info).  Should not be called on the hot path.
    async fn health_check(&self) -> Result<ProviderStatus, ProviderError>;

    /// Return the static capabilities of this provider.
    ///
    /// This must not make a network call — it describes the provider's known
    /// feature set as compiled in.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Declare whether this exact adapter route accepts provider-managed
    /// response-ID continuation. The conservative default is full context.
    fn response_continuation_support(&self) -> ResponseContinuationSupport {
        ResponseContinuationSupport::Unsupported
    }

    /// Return an opaque, host-only identity for the exact account and endpoint
    /// that would own a response-continuation anchor for `model`.
    ///
    /// Providers that enable response-ID continuation must override this and
    /// change the value whenever the account, endpoint, or compatible route
    /// changes. The query layer snapshots it before dispatch; it is never sent
    /// to the provider.
    fn response_continuation_route_identity(&self, model: &str) -> Option<String> {
        let _ = model;
        None
    }

    /// Return an opaque host-only fingerprint for the current provider
    /// configuration, when this adapter can identify its account and endpoint
    /// without exposing either value. The query layer uses this only as a
    /// continuity-cache hint; dispatch authority remains bound to the
    /// returned provider and its verified route.
    fn host_configuration_fingerprint(&self) -> Option<String> {
        None
    }

    /// Return a typed dispatch route for this provider when it supports
    /// host-authorized inference dispatch. Unsupported providers fail closed.
    fn dispatch_route(
        &self,
        model: &str,
        effort: EffortLevel,
    ) -> Result<DispatchRoute, DispatchDenied> {
        let _ = (model, effort);
        Err(DispatchDenied::UnsupportedRoute)
    }

    /// Return a retained provider instance carrying host-verified dispatch
    /// policy. Providers that cannot bind policy remain fixed/base-only.
    fn bind_dispatch_policy(
        &self,
        _policy: Arc<DispatchPolicy>,
    ) -> Result<Arc<dyn LlmProvider>, DispatchDenied> {
        Err(DispatchDenied::UnsupportedRoute)
    }
}
