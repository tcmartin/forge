//! Runtime-only state for one logical inference continuation.
//!
//! The state is intentionally private to the query host.  It is the boundary
//! that keeps a local, typed wire-budget reselection on the same logical
//! request while preventing a completed response or an uncertain stream from
//! silently changing its route, body settings, or proof authority.

use std::sync::Arc;

use claurst_api::DispatchRequest;
use claurst_core::dispatch::LogicalRequestId;
use claurst_core::effort::EffortLevel;

use crate::resolved_target::ResolvedTarget;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContinuationPhase {
    PreSend,
    Attempting,
    Streaming,
    Completed,
    Aborted,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SelectedSettings {
    pub(crate) effort: Option<EffortLevel>,
    pub(crate) thinking_budget: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContinuationTransitionError {
    WrongPhase,
    FallbackAlreadyUsed,
}

/// One logical request's host-owned continuation state.
pub(crate) struct ContinuationState<'a> {
    logical_request_id: LogicalRequestId,
    target: Arc<ResolvedTarget<'a>>,
    settings: SelectedSettings,
    dispatch: Option<DispatchRequest>,
    fallback_used: bool,
    phase: ContinuationPhase,
}

impl<'a> ContinuationState<'a> {
    pub(crate) fn new(target: Arc<ResolvedTarget<'a>>, settings: SelectedSettings) -> Self {
        Self {
            logical_request_id: LogicalRequestId::fresh(),
            target,
            settings,
            dispatch: None,
            fallback_used: false,
            phase: ContinuationPhase::PreSend,
        }
    }

    pub(crate) fn logical_request_id(&self) -> &LogicalRequestId {
        &self.logical_request_id
    }

    pub(crate) fn target(&self) -> &Arc<ResolvedTarget<'a>> {
        &self.target
    }

    pub(crate) fn settings(&self) -> SelectedSettings {
        self.settings
    }

    pub(crate) fn dispatch(&self) -> Option<&DispatchRequest> {
        self.dispatch.as_ref()
    }

    pub(crate) fn set_dispatch(
        &mut self,
        dispatch: DispatchRequest,
    ) -> Result<(), ContinuationTransitionError> {
        if self.phase != ContinuationPhase::PreSend {
            return Err(ContinuationTransitionError::WrongPhase);
        }
        self.dispatch = Some(dispatch);
        Ok(())
    }

    pub(crate) fn begin_attempt(&mut self) -> Result<(), ContinuationTransitionError> {
        if self.phase != ContinuationPhase::PreSend {
            return Err(ContinuationTransitionError::WrongPhase);
        }
        self.phase = ContinuationPhase::Attempting;
        Ok(())
    }

    pub(crate) fn mark_streaming(&mut self) -> Result<(), ContinuationTransitionError> {
        if self.phase != ContinuationPhase::Attempting {
            return Err(ContinuationTransitionError::WrongPhase);
        }
        self.phase = ContinuationPhase::Streaming;
        Ok(())
    }

    pub(crate) fn mark_completed(&mut self) -> Result<(), ContinuationTransitionError> {
        if self.phase != ContinuationPhase::Streaming {
            return Err(ContinuationTransitionError::WrongPhase);
        }
        self.phase = ContinuationPhase::Completed;
        self.dispatch = None;
        Ok(())
    }

    pub(crate) fn abort(&mut self) {
        self.phase = ContinuationPhase::Aborted;
        self.dispatch = None;
    }

    /// Return to pre-send planning after a typed local wire-budget failure.
    /// The logical ID, exact target, settings, and any unused authority remain
    /// unchanged; no tool or stream replay is implied by this transition.
    pub(crate) fn reselection_after_local_budget(
        &mut self,
    ) -> Result<(), ContinuationTransitionError> {
        if self.phase != ContinuationPhase::Attempting {
            return Err(ContinuationTransitionError::WrongPhase);
        }
        self.phase = ContinuationPhase::PreSend;
        Ok(())
    }

    /// Permit at most one fresh fixed/base fallback after a typed proof
    /// invalidation.  This transition is separate from local budget
    /// reselection because it must discard the old authority and identity.
    pub(crate) fn use_proof_fallback(&mut self) -> Result<(), ContinuationTransitionError> {
        if self.fallback_used {
            return Err(ContinuationTransitionError::FallbackAlreadyUsed);
        }
        self.fallback_used = true;
        self.abort();
        Ok(())
    }
}
