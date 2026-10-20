//! Per-call query event delivery with bounded, owned retention.

use crate::QueryEvent;
use std::io;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EventDeliveryFailure {
    CountLimit = 1,
    ByteLimit = 2,
    EventSizeLimit = 3,
    ReceiverClosed = 4,
    ForwardingFailed = 5,
}

impl EventDeliveryFailure {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CountLimit => "event_count_limit",
            Self::ByteLimit => "event_byte_limit",
            Self::EventSizeLimit => "event_size_limit",
            Self::ReceiverClosed => "event_receiver_closed",
            Self::ForwardingFailed => "event_forwarding_failed",
        }
    }
}

#[derive(Clone)]
pub struct EventDeliveryState {
    failure: Arc<AtomicU8>,
    cancel: CancellationToken,
}

impl EventDeliveryState {
    pub fn failure(&self) -> Option<EventDeliveryFailure> {
        match self.failure.load(Ordering::Acquire) {
            1 => Some(EventDeliveryFailure::CountLimit),
            2 => Some(EventDeliveryFailure::ByteLimit),
            3 => Some(EventDeliveryFailure::EventSizeLimit),
            4 => Some(EventDeliveryFailure::ReceiverClosed),
            5 => Some(EventDeliveryFailure::ForwardingFailed),
            _ => None,
        }
    }

    pub fn forwarding_failed(&self) {
        self.fail(EventDeliveryFailure::ForwardingFailed);
    }

    fn fail(&self, failure: EventDeliveryFailure) {
        let _ =
            self.failure
                .compare_exchange(0, failure as u8, Ordering::AcqRel, Ordering::Acquire);
        self.cancel.cancel();
    }
}

pub struct QueryEventLease {
    event: QueryEvent,
    _slot: OwnedSemaphorePermit,
    _bytes: OwnedSemaphorePermit,
}

impl QueryEventLease {
    /// Borrow through forwarding; consuming the lease would release its charge early.
    pub fn event(&self) -> &QueryEvent {
        &self.event
    }
}

#[derive(Clone)]
struct BoundedSender {
    tx: mpsc::Sender<QueryEventLease>,
    slots: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    max_event_bytes: usize,
    charge: fn(&QueryEvent) -> Option<usize>,
    state: EventDeliveryState,
}

#[derive(Clone)]
enum SenderInner {
    Unbounded(mpsc::UnboundedSender<QueryEvent>),
    Bounded(BoundedSender),
}

#[derive(Clone)]
pub struct QueryEventSender {
    inner: SenderInner,
}

#[derive(Debug)]
pub struct EventSendError;

const MAX_DIAGNOSTIC_CHARGE_BYTES: usize = 1024 * 1024;
const DIAGNOSTIC_ENVELOPE_OVERHEAD_BYTES: usize = 512;

struct DiagnosticChargeWriter {
    bytes: usize,
    limit: usize,
}

impl io::Write for DiagnosticChargeWriter {
    fn write(&mut self, chunk: &[u8]) -> io::Result<usize> {
        let next = self.bytes.checked_add(chunk.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "diagnostic size overflow")
        })?;
        if next > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "diagnostic event exceeds bounded charge",
            ));
        }
        self.bytes = next;
        Ok(chunk.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn bounded_diagnostic_charge<T: serde::Serialize>(value: &T) -> Option<usize> {
    let mut writer = DiagnosticChargeWriter {
        bytes: 0,
        limit: MAX_DIAGNOSTIC_CHARGE_BYTES.saturating_sub(DIAGNOSTIC_ENVELOPE_OVERHEAD_BYTES),
    };
    serde_json::to_writer(&mut writer, value).ok()?;
    writer.bytes.checked_add(DIAGNOSTIC_ENVELOPE_OVERHEAD_BYTES)
}

/// Production callers use this charge for additive MCP diagnostics. It bounds
/// the complete serialized event envelope, including base64 and identity
/// arrays, rather than charging only individual fields or a test helper.
pub fn diagnostic_event_charge(event: &QueryEvent) -> Option<usize> {
    match event {
        QueryEvent::McpDiscovery(witness) => {
            bounded_diagnostic_charge(witness).or(Some(usize::MAX))
        }
        QueryEvent::McpInvocation {
            tool_id,
            identity,
            operation_context,
        } => {
            bounded_diagnostic_charge(&(tool_id, identity, operation_context)).or(Some(usize::MAX))
        }
        _ => None,
    }
}

/// Charge actual retained text capacity and bounded serialized structured data.
/// Serialization counts into a finite writer; it does not allocate a second body.
pub fn stream_event_charge(event: &QueryEvent) -> Option<usize> {
    use claurst_api::streaming::ContentDelta;
    use claurst_api::AnthropicStreamEvent as Stream;
    const OVERHEAD: usize = 1024;
    let strings = |values: &[&String]| {
        values
            .iter()
            .try_fold(OVERHEAD, |n, s| n.checked_add(s.capacity()))
    };
    let structured = |bytes: Option<usize>| {
        bytes
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_add(OVERHEAD))
    };
    let charge = match event {
        QueryEvent::McpDiscovery(_) | QueryEvent::McpInvocation { .. } => {
            diagnostic_event_charge(event)
        }
        QueryEvent::ProviderWireWitness(w) => structured(bounded_diagnostic_charge(w)),
        QueryEvent::Status(s) | QueryEvent::Error(s) => strings(&[s]),
        QueryEvent::ToolStart {
            tool_name,
            tool_id,
            input_json,
        } => strings(&[tool_name, tool_id, input_json]),
        QueryEvent::ToolEnd {
            tool_name,
            tool_id,
            result,
            native_operation_receipt,
            execution_witness,
            ..
        } => strings(&[tool_name, tool_id, result]).and_then(|n| {
            structured(bounded_diagnostic_charge(execution_witness))
                .and_then(|w| n.checked_add(w))
                .and_then(|n| {
                    structured(bounded_diagnostic_charge(native_operation_receipt))
                        .and_then(|receipt| n.checked_add(receipt))
                })
        }),
        QueryEvent::TurnComplete { stop_reason, .. } => strings(&[stop_reason]),
        QueryEvent::Stream(stream) => match stream {
            Stream::MessageStart { id, model, .. } => strings(&[id, model]),
            Stream::ContentBlockStart { content_block, .. } => {
                structured(bounded_diagnostic_charge(content_block))
            }
            Stream::ContentBlockDelta { delta, .. } => match delta {
                ContentDelta::TextDelta { text } => strings(&[text]),
                ContentDelta::InputJsonDelta { partial_json } => strings(&[partial_json]),
                ContentDelta::ThinkingDelta { thinking } => strings(&[thinking]),
                ContentDelta::SignatureDelta { signature } => strings(&[signature]),
            },
            Stream::MessageDelta { stop_reason, .. } => {
                Some(OVERHEAD.saturating_add(stop_reason.as_ref().map_or(0, String::capacity)))
            }
            Stream::Error {
                error_type,
                message,
            } => strings(&[error_type, message]),
            Stream::ContentBlockStop { .. } | Stream::MessageStop | Stream::Ping => Some(OVERHEAD),
        },
        QueryEvent::TokenWarning { .. } | QueryEvent::EffortSelection(_) => Some(OVERHEAD),
    };
    Some(charge.unwrap_or(usize::MAX))
}

impl From<mpsc::UnboundedSender<QueryEvent>> for QueryEventSender {
    fn from(sender: mpsc::UnboundedSender<QueryEvent>) -> Self {
        Self {
            inner: SenderInner::Unbounded(sender),
        }
    }
}

impl QueryEventSender {
    pub fn send(&self, event: QueryEvent) -> Result<(), EventSendError> {
        let sender = match &self.inner {
            SenderInner::Unbounded(sender) => {
                return sender.send(event).map_err(|_| EventSendError)
            }
            SenderInner::Bounded(sender) => sender,
        };
        if sender.state.failure().is_some() {
            return Err(EventSendError);
        }
        let Some(charge) = (sender.charge)(&event) else {
            return Ok(());
        };
        let charge = charge.max(std::mem::size_of::<QueryEventLease>());
        if charge > sender.max_event_bytes {
            sender.state.fail(EventDeliveryFailure::EventSizeLimit);
            return Err(EventSendError);
        }
        let slot = match sender.slots.clone().try_acquire_owned() {
            Ok(slot) => slot,
            Err(_) => {
                sender.state.fail(EventDeliveryFailure::CountLimit);
                return Err(EventSendError);
            }
        };
        // Constructor bounds make this conversion finite on every supported target.
        let bytes = match sender.bytes.clone().try_acquire_many_owned(charge as u32) {
            Ok(bytes) => bytes,
            Err(_) => {
                sender.state.fail(EventDeliveryFailure::ByteLimit);
                return Err(EventSendError);
            }
        };
        let lease = QueryEventLease {
            event,
            _slot: slot,
            _bytes: bytes,
        };
        match sender.tx.try_send(lease) {
            Ok(()) => Ok(()),
            Err(error) => {
                let failure = match error {
                    mpsc::error::TrySendError::Full(_) => EventDeliveryFailure::CountLimit,
                    mpsc::error::TrySendError::Closed(_) => EventDeliveryFailure::ReceiverClosed,
                };
                sender.state.fail(failure);
                Err(EventSendError)
            }
        }
    }

    pub(crate) fn delivery_state(&self) -> Option<EventDeliveryState> {
        match &self.inner {
            SenderInner::Unbounded(_) => None,
            SenderInner::Bounded(sender) => Some(sender.state.clone()),
        }
    }
}

pub struct QueryEventReceiver {
    rx: mpsc::Receiver<QueryEventLease>,
    state: EventDeliveryState,
    complete: bool,
}

impl QueryEventReceiver {
    pub async fn recv(&mut self) -> Option<QueryEventLease> {
        let result = self.rx.recv().await;
        if result.is_none() {
            self.complete = true;
        }
        result
    }
}

impl Drop for QueryEventReceiver {
    fn drop(&mut self) {
        if !self.complete {
            self.state.fail(EventDeliveryFailure::ReceiverClosed);
        }
    }
}

#[derive(Debug)]
pub struct InvalidEventLimits;

/// `charge` is host-owned: None filters an event; Some includes retained payload
/// and projection allocation. Limits cover both queued and actively forwarded items.
pub fn bounded_events(
    cancel: CancellationToken,
    capacity: usize,
    bytes: usize,
    max_event_bytes: usize,
    charge: fn(&QueryEvent) -> Option<usize>,
) -> Result<(QueryEventSender, QueryEventReceiver, EventDeliveryState), InvalidEventLimits> {
    if capacity == 0
        || capacity > 4096
        || bytes == 0
        || bytes > 64 * 1024 * 1024
        || max_event_bytes == 0
        || max_event_bytes > bytes
    {
        return Err(InvalidEventLimits);
    }
    let state = EventDeliveryState {
        failure: Arc::new(AtomicU8::new(0)),
        cancel,
    };
    let (tx, rx) = mpsc::channel(capacity);
    let sender = QueryEventSender {
        inner: SenderInner::Bounded(BoundedSender {
            tx,
            slots: Arc::new(Semaphore::new(capacity)),
            bytes: Arc::new(Semaphore::new(bytes)),
            max_event_bytes,
            charge,
            state: state.clone(),
        }),
    };
    Ok((
        sender,
        QueryEventReceiver {
            rx,
            state: state.clone(),
            complete: false,
        },
        state,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn charge(event: &QueryEvent) -> Option<usize> {
        match event {
            QueryEvent::Status(_) => None,
            QueryEvent::Error(text) => Some(512 + text.capacity()),
            QueryEvent::McpDiscovery(witness) => {
                diagnostic_event_charge(&QueryEvent::McpDiscovery(witness.clone()))
            }
            QueryEvent::McpInvocation {
                tool_id,
                identity,
                operation_context,
            } => diagnostic_event_charge(&QueryEvent::McpInvocation {
                tool_id: tool_id.clone(),
                identity: identity.clone(),
                operation_context: operation_context.clone(),
            }),
            _ => Some(512),
        }
    }

    #[tokio::test]
    async fn active_item_keeps_count_reservation_and_preserves_fifo() {
        let cancel = CancellationToken::new();
        let (tx, mut rx, state) = bounded_events(cancel.clone(), 2, 4096, 2048, charge).unwrap();
        tx.send(QueryEvent::Error("one".to_string())).unwrap();
        tx.send(QueryEvent::Error("two".to_string())).unwrap();
        let active = rx.recv().await.unwrap();
        assert!(matches!(active.event(), QueryEvent::Error(text) if text == "one"));
        assert!(tx.send(QueryEvent::Error("three".to_string())).is_err());
        assert_eq!(state.failure(), Some(EventDeliveryFailure::CountLimit));
        assert!(cancel.is_cancelled());
        drop(active);
        assert!(
            matches!(rx.recv().await.unwrap().event(), QueryEvent::Error(text) if text == "two")
        );
    }

    #[tokio::test]
    async fn active_item_keeps_byte_reservation() {
        let (tx, mut rx, state) =
            bounded_events(CancellationToken::new(), 16, 1600, 1000, charge).unwrap();
        tx.send(QueryEvent::Error("a".repeat(200))).unwrap();
        tx.send(QueryEvent::Error("b".repeat(200))).unwrap();
        let _active = rx.recv().await.unwrap();
        assert!(tx.send(QueryEvent::Error("c".repeat(200))).is_err());
        assert_eq!(state.failure(), Some(EventDeliveryFailure::ByteLimit));
    }

    #[test]
    fn excessive_capacity_is_charged_even_with_short_text() {
        let (tx, _rx, state) =
            bounded_events(CancellationToken::new(), 16, 4096, 2048, charge).unwrap();
        let mut text = String::with_capacity(4096);
        text.push('x');
        assert!(tx.send(QueryEvent::Error(text)).is_err());
        assert_eq!(state.failure(), Some(EventDeliveryFailure::EventSizeLimit));
    }

    #[tokio::test]
    async fn ignored_events_do_not_fill_the_queue_and_normal_eof_does_not_cancel() {
        let cancel = CancellationToken::new();
        let (tx, mut rx, state) = bounded_events(cancel.clone(), 1, 4096, 2048, charge).unwrap();
        for _ in 0..1000 {
            tx.send(QueryEvent::Status("ignored".to_string())).unwrap();
        }
        tx.send(QueryEvent::Error("visible".to_string())).unwrap();
        drop(rx.recv().await.unwrap());
        drop(tx);
        assert!(rx.recv().await.is_none());
        drop(rx);
        assert!(state.failure().is_none());
        assert!(!cancel.is_cancelled());
    }

    #[test]
    fn receiver_loss_cancels_immediately_and_first_failure_is_stable() {
        let cancel = CancellationToken::new();
        let (_tx, rx, state) = bounded_events(cancel.clone(), 1, 4096, 2048, charge).unwrap();
        state.forwarding_failed();
        drop(rx);
        assert_eq!(
            state.failure(),
            Some(EventDeliveryFailure::ForwardingFailed)
        );
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn receiver_drop_cancels_only_its_own_query() {
        let first = CancellationToken::new();
        let next = CancellationToken::new();
        let (_tx, rx, state) = bounded_events(first.clone(), 1, 4096, 2048, charge).unwrap();
        let (next_tx, _next_rx, next_state) =
            bounded_events(next.clone(), 1, 4096, 2048, charge).unwrap();
        drop(rx);
        assert_eq!(state.failure(), Some(EventDeliveryFailure::ReceiverClosed));
        assert!(first.is_cancelled());
        assert!(!next.is_cancelled());
        assert!(next_state.failure().is_none());
        next_tx
            .send(QueryEvent::Error("fresh query".to_string()))
            .unwrap();
    }

    #[tokio::test]
    async fn legacy_sender_keeps_existing_delivery_and_drop_behavior() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let tx = QueryEventSender::from(tx);
        tx.send(QueryEvent::Status("legacy".to_string())).unwrap();
        assert!(matches!(rx.recv().await.unwrap(), QueryEvent::Status(text) if text == "legacy"));
        assert!(tx.delivery_state().is_none());
        drop(rx);
        assert!(tx.send(QueryEvent::Status("closed".to_string())).is_err());
    }

    #[test]
    fn invalid_limits_are_rejected_before_channel_allocation() {
        assert!(bounded_events(CancellationToken::new(), 0, 4096, 2048, charge).is_err());
        assert!(bounded_events(CancellationToken::new(), 1, 4096, 8192, charge).is_err());
    }
}

#[cfg(test)]
mod stream_charge_tests {
    use super::*;
    #[test]
    fn ordinary_stream_burst_fits_actual_byte_budget() {
        let (tx, _rx, state) = bounded_events(
            CancellationToken::new(),
            256,
            8 * 1024 * 1024,
            1024 * 1024,
            stream_event_charge,
        )
        .unwrap();
        for _ in 0..64 {
            tx.send(QueryEvent::Status("small event".into())).unwrap();
        }
        assert!(state.failure().is_none());
    }
    #[test]
    fn oversized_retained_payload_is_rejected_before_enqueue() {
        let (tx, _rx, state) = bounded_events(
            CancellationToken::new(),
            256,
            8 * 1024 * 1024,
            1024 * 1024,
            stream_event_charge,
        )
        .unwrap();
        let mut text = String::with_capacity(2 * 1024 * 1024);
        text.push('x');
        assert!(tx.send(QueryEvent::Error(text)).is_err());
        assert_eq!(state.failure(), Some(EventDeliveryFailure::EventSizeLimit));
    }
    #[tokio::test]
    async fn headless_lease_remains_charged_through_forwarding() {
        let (tx, mut rx, state) = bounded_events(
            CancellationToken::new(),
            1,
            2 * 1024 * 1024,
            1024 * 1024,
            stream_event_charge,
        )
        .unwrap();
        tx.send(QueryEvent::Status("first".into())).unwrap();
        let lease = rx.recv().await.unwrap();
        let borrowed = lease.event();
        assert!(matches!(borrowed,QueryEvent::Status(s) if s=="first"));
        assert!(tx.send(QueryEvent::Status("second".into())).is_err());
        assert_eq!(state.failure(), Some(EventDeliveryFailure::CountLimit));
        drop(lease);
    }
}
