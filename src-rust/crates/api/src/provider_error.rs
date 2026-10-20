// provider_error.rs — Unified error type for all provider adapters.
//
// Every provider implementation maps its own error representation onto
// `ProviderError` so that the application-layer code can handle errors
// generically without knowing which provider was involved.

use crate::provider_wire::WireBudgetError;
use claurst_core::dispatch::DispatchDenied;
use claurst_core::error::ClaudeError;
use claurst_core::provider_id::ProviderId;
use std::fmt;

// ---------------------------------------------------------------------------
// ProviderError
// ---------------------------------------------------------------------------

/// A structured error produced by any provider adapter.
#[derive(Clone)]
pub enum ProviderError {
    /// A host dispatch envelope was supplied to a route that is not yet
    /// certified to consume it. No inference transport was opened.
    DispatchRejected {
        provider: ProviderId,
        error: DispatchDenied,
    },
    /// Local admission rejected this call before an inference transport opened.
    AdmissionRejected {
        provider: ProviderId,
        error: crate::inference_admission::AdmissionError,
    },
    /// Local final-wire preflight rejected the request before any inference
    /// transport was opened. The structured error retains measured costs and
    /// limits for deterministic context eviction and retry decisions.
    BudgetExceeded {
        provider: ProviderId,
        error: Box<WireBudgetError>,
    },
    /// The request exceeded the model's context window.
    ContextOverflow {
        provider: ProviderId,
        message: String,
        /// The provider's advertised context limit in tokens, if known.
        max_tokens: Option<u64>,
    },

    /// The provider returned HTTP 429 or an equivalent rate-limit signal.
    RateLimited {
        provider: ProviderId,
        /// How long to wait before retrying, in seconds (if provided).
        retry_after: Option<u64>,
    },

    /// The API key or credentials were rejected by the provider.
    AuthFailed {
        provider: ProviderId,
        message: String,
    },

    /// The account's usage quota has been exhausted.
    QuotaExceeded {
        provider: ProviderId,
        message: String,
    },

    /// The requested model does not exist or is not accessible.
    ModelNotFound {
        provider: ProviderId,
        model: String,
        /// Alternative model IDs the caller might try instead.
        suggestions: Vec<String>,
    },

    /// The provider returned a 5xx or equivalent server-side error.
    ServerError {
        provider: ProviderId,
        /// HTTP status code, if applicable.
        status: Option<u16>,
        message: String,
        /// Whether the caller should retry this request.
        is_retryable: bool,
    },

    /// The request itself was malformed or contained invalid parameters.
    InvalidRequest {
        provider: ProviderId,
        message: String,
    },

    /// The response was blocked by the provider's content-safety system.
    ContentFiltered {
        provider: ProviderId,
        message: String,
    },

    /// An error occurred during streaming after the response had already begun.
    StreamError {
        provider: ProviderId,
        message: String,
        /// Any content blocks that had been received before the error, if any.
        partial_response: Option<String>,
    },

    /// A catch-all variant for errors that do not fit any of the above.
    Other {
        provider: ProviderId,
        message: String,
        /// HTTP status code, if applicable.
        status: Option<u16>,
        /// Raw response body, if available.
        body: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// impl ProviderError
// ---------------------------------------------------------------------------

impl ProviderError {
    pub fn budget(provider: ProviderId, error: WireBudgetError) -> Self {
        Self::BudgetExceeded {
            provider,
            error: Box::new(error),
        }
    }

    /// Returns `true` if the caller should retry the request after a delay.
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::DispatchRejected { .. } => false,
            ProviderError::BudgetExceeded { .. } => false,
            ProviderError::RateLimited { .. } => true,
            ProviderError::ServerError { is_retryable, .. } => *is_retryable,
            ProviderError::StreamError { .. } => true,
            _ => false,
        }
    }

    /// Returns the `ProviderId` of the provider that produced this error.
    pub fn provider_id(&self) -> &ProviderId {
        match self {
            ProviderError::DispatchRejected { provider, .. } => provider,
            ProviderError::AdmissionRejected { provider, .. } => provider,
            ProviderError::BudgetExceeded { provider, .. } => provider,
            ProviderError::ContextOverflow { provider, .. } => provider,
            ProviderError::RateLimited { provider, .. } => provider,
            ProviderError::AuthFailed { provider, .. } => provider,
            ProviderError::QuotaExceeded { provider, .. } => provider,
            ProviderError::ModelNotFound { provider, .. } => provider,
            ProviderError::ServerError { provider, .. } => provider,
            ProviderError::InvalidRequest { provider, .. } => provider,
            ProviderError::ContentFiltered { provider, .. } => provider,
            ProviderError::StreamError { provider, .. } => provider,
            ProviderError::Other { provider, .. } => provider,
        }
    }

    /// A stable category for diagnostics that never includes provider text,
    /// response bodies, request URLs, credentials, or model input.
    pub fn diagnostic_kind(&self) -> &'static str {
        match self {
            ProviderError::DispatchRejected { .. } => "dispatch_rejected",
            ProviderError::AdmissionRejected { .. } => "admission_rejected",
            ProviderError::BudgetExceeded { .. } => "budget_exceeded",
            ProviderError::ContextOverflow { .. } => "context_overflow",
            ProviderError::RateLimited { .. } => "rate_limited",
            ProviderError::AuthFailed { .. } => "auth_failed",
            ProviderError::QuotaExceeded { .. } => "quota_exceeded",
            ProviderError::ModelNotFound { .. } => "model_not_found",
            ProviderError::ServerError { .. } => "server_error",
            ProviderError::InvalidRequest { .. } => "invalid_request",
            ProviderError::ContentFiltered { .. } => "content_filtered",
            ProviderError::StreamError { .. } => "stream_error",
            ProviderError::Other { .. } => "other",
        }
    }

    /// The HTTP status attached to an error, when the structured variant has
    /// one. This is safe to record alongside [`Self::diagnostic_kind`].
    pub fn diagnostic_status(&self) -> Option<u16> {
        match self {
            ProviderError::ServerError { status, .. } | ProviderError::Other { status, .. } => {
                *status
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Privacy-safe formatting
// ---------------------------------------------------------------------------

impl fmt::Debug for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderError")
            .field("provider", self.provider_id())
            .field("kind", &self.diagnostic_kind())
            .field("status", &self.diagnostic_status())
            .field("retryable", &self.is_retryable())
            .finish()
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::DispatchRejected { provider, .. } => {
                write!(f, "[{}] Dispatch rejected before inference", provider)
            }
            ProviderError::AdmissionRejected { provider, .. } => {
                write!(f, "[{}] Inference admission rejected request", provider)
            }
            ProviderError::BudgetExceeded { provider, .. } => {
                write!(f, "[{}] Wire budget rejected request", provider)
            }
            ProviderError::ContextOverflow {
                provider,
                max_tokens,
                ..
            } => {
                write!(f, "[{}] Context overflow", provider)?;
                if let Some(max) = max_tokens {
                    write!(f, " (max {} tokens)", max)?;
                }
                Ok(())
            }
            ProviderError::RateLimited {
                provider,
                retry_after,
            } => {
                write!(f, "[{}] Rate limited", provider)?;
                if let Some(secs) = retry_after {
                    write!(f, "; retry after {}s", secs)?;
                }
                Ok(())
            }
            ProviderError::AuthFailed { provider, .. } => {
                write!(f, "[{}] Authentication failed", provider)
            }
            ProviderError::QuotaExceeded { provider, .. } => {
                write!(f, "[{}] Quota exceeded", provider)
            }
            ProviderError::ModelNotFound { provider, .. } => {
                write!(f, "[{}] Requested model is unavailable", provider)
            }
            ProviderError::ServerError {
                provider, status, ..
            } => match status {
                Some(s) => write!(f, "[{}] Server error {}", provider, s),
                None => write!(f, "[{}] Server error", provider),
            },
            ProviderError::InvalidRequest { provider, .. } => {
                write!(f, "[{}] Invalid request", provider)
            }
            ProviderError::ContentFiltered { provider, .. } => {
                write!(f, "[{}] Content filtered", provider)
            }
            ProviderError::StreamError { provider, .. } => {
                write!(f, "[{}] Stream error", provider)
            }
            ProviderError::Other {
                provider, status, ..
            } => match status {
                Some(s) => write!(f, "[{}] Provider request failed ({})", provider, s),
                None => write!(f, "[{}] Provider request failed", provider),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// std::error::Error
// ---------------------------------------------------------------------------

impl std::error::Error for ProviderError {}

// ---------------------------------------------------------------------------
// From<ProviderError> for ClaudeError
// ---------------------------------------------------------------------------

impl From<ProviderError> for ClaudeError {
    fn from(err: ProviderError) -> Self {
        match err {
            ProviderError::DispatchRejected { error, .. } => {
                ClaudeError::InferenceDispatchDenied(error)
            }
            ProviderError::AdmissionRejected { provider, error } => ClaudeError::Api(format!(
                "[{}] Inference admission rejected request: {}",
                provider, error
            )),
            ProviderError::BudgetExceeded { provider: _, error } => {
                if let Some((
                    input_tokens,
                    input_limit_tokens,
                    wire_bytes,
                    wire_byte_limit,
                    context_limit_tokens,
                    output_reserve_tokens,
                    budget_source,
                )) = error.request_budget_fields()
                {
                    ClaudeError::RequestBudgetExceeded {
                        input_tokens,
                        input_limit_tokens,
                        wire_bytes,
                        wire_byte_limit,
                        context_limit_tokens,
                        output_reserve_tokens,
                        budget_source: budget_source.to_string(),
                    }
                } else {
                    ClaudeError::Api("Wire budget rejected request before inference".to_string())
                }
            }
            ProviderError::ContextOverflow { .. } => ClaudeError::ContextWindowExceeded,
            ProviderError::RateLimited { .. } => ClaudeError::RateLimit,
            ProviderError::AuthFailed { provider, .. } => ClaudeError::Auth(format!(
                "[{}] Provider rejected the configured credentials",
                provider
            )),
            ProviderError::QuotaExceeded { provider, .. } => {
                ClaudeError::Api(format!("[{}] Provider quota exceeded", provider))
            }
            ProviderError::ModelNotFound { provider, .. } => ClaudeError::Api(format!(
                "[{}] Requested model was not found or is unavailable",
                provider
            )),
            ProviderError::ServerError {
                provider, status, ..
            } => match status {
                Some(status) => ClaudeError::ApiStatus {
                    status,
                    message: format!("[{}] Provider server error", provider),
                },
                None => ClaudeError::Api(format!("[{}] Provider server error", provider)),
            },
            ProviderError::InvalidRequest { provider, .. } => {
                ClaudeError::Api(format!("[{}] Provider rejected the request", provider))
            }
            ProviderError::ContentFiltered { provider, .. } => {
                ClaudeError::Api(format!("[{}] Provider blocked the response", provider))
            }
            ProviderError::StreamError { provider, .. } => {
                ClaudeError::Api(format!("[{}] Provider stream failed", provider))
            }
            ProviderError::Other {
                provider, status, ..
            } => match status {
                Some(status) => ClaudeError::ApiStatus {
                    status,
                    message: format!("[{}] Provider request failed", provider),
                },
                None => ClaudeError::Api(format!("[{}] Provider request failed", provider)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_metadata_excludes_provider_content() {
        const PRIVATE: &str = "PRIVATE_PROVIDER_RESPONSE_AND_CREDENTIAL";
        let error = ProviderError::Other {
            provider: ProviderId::new("fixture"),
            message: format!("request failed at https://example.test/?key={PRIVATE}"),
            status: Some(418),
            body: Some(PRIVATE.to_string()),
        };

        let diagnostic = format!(
            "kind={} status={:?}",
            error.diagnostic_kind(),
            error.diagnostic_status()
        );
        assert_eq!(diagnostic, "kind=other status=Some(418)");
        assert!(!diagnostic.contains(PRIVATE));
        assert!(!error.to_string().contains(PRIVATE));
        assert!(!format!("{error:?}").contains(PRIVATE));
    }

    #[test]
    fn user_facing_conversion_excludes_provider_content() {
        const PRIVATE: &str = "PRIVATE_PROVIDER_RESPONSE_AND_CREDENTIAL";
        let error = ProviderError::Other {
            provider: ProviderId::new("fixture"),
            message: PRIVATE.to_string(),
            status: Some(418),
            body: Some(PRIVATE.to_string()),
        };

        let presented = ClaudeError::from(error).to_string();
        assert!(presented.contains("fixture"));
        assert!(presented.contains("418"));
        assert!(presented.contains("Provider request failed"));
        assert!(!presented.contains(PRIVATE));
    }
}
