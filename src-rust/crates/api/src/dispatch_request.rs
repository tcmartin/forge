//! Host-owned dispatch metadata carried alongside a provider request.
//!
//! The envelope is intentionally not serializable. Provider adapters must
//! reject a present envelope until a route-specific authorization stage exists;
//! silently dropping it would turn an unauthorized request into a normal send.

use claurst_core::dispatch::{DispatchDenied, DispatchPermit, DispatchRoute, LogicalRequestId};
use claurst_core::provider_id::ProviderId;
use std::fmt;

use crate::provider_error::ProviderError;

/// Typed host authorization metadata attached to one logical provider request.
#[derive(Clone)]
pub struct DispatchRequest {
    permit: DispatchPermit,
    logical_request_id: LogicalRequestId,
    expected_route: DispatchRoute,
}

impl DispatchRequest {
    pub fn new(
        permit: DispatchPermit,
        logical_request_id: LogicalRequestId,
        expected_route: DispatchRoute,
    ) -> Self {
        Self {
            permit,
            logical_request_id,
            expected_route,
        }
    }

    pub fn permit(&self) -> &DispatchPermit {
        &self.permit
    }

    pub fn logical_request_id(&self) -> &LogicalRequestId {
        &self.logical_request_id
    }

    pub fn expected_route(&self) -> &DispatchRoute {
        &self.expected_route
    }
}

impl fmt::Debug for DispatchRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DispatchRequest(..)")
    }
}

/// Reject an envelope while provider-specific dispatch authorization is not
/// yet wired. This helper is shared by every concrete provider and wrapper.
pub(crate) fn reject_unsupported_dispatch(
    provider: &ProviderId,
    request: Option<&DispatchRequest>,
) -> Result<(), ProviderError> {
    if request.is_some() {
        return Err(ProviderError::DispatchRejected {
            provider: provider.clone(),
            error: DispatchDenied::UnsupportedRoute,
        });
    }
    Ok(())
}
