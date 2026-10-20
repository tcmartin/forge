//! Finite, process-wide admission at inference transport leaves.
//!
//! Wrappers and metadata requests do not acquire. A leaf retains its lease
//! until its response body or stream has been consumed or dropped.

use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::provider_error::ProviderError;
use crate::provider_types::StreamEvent;

pub const DEFAULT_INFERENCE_CAPACITY: usize = 4;
pub const MAX_INFERENCE_CAPACITY: usize = 64;
pub const DEFAULT_QUEUE_TIMEOUT: Duration = Duration::from_secs(30);
pub const MAX_QUEUE_TIMEOUT: Duration = Duration::from_secs(300);

static PROCESS_ADMISSION: OnceLock<Arc<InferenceAdmission>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("inference capacity must be between 1 and 64")]
    InvalidCapacity,
    #[error("inference queue timeout must be greater than zero and at most 300 seconds")]
    InvalidQueueTimeout,
    #[error("process inference capacity is already configured as {capacity}")]
    AlreadyConfigured { capacity: usize },
    #[error("inference queue deadline exceeded before transport")]
    QueueTimeout,
    #[error("inference admission is closed")]
    Closed,
}

/// Host-owned queue options. These values are not part of provider JSON.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionOptions {
    queue_timeout: Duration,
}

impl Default for AdmissionOptions {
    fn default() -> Self {
        Self {
            queue_timeout: DEFAULT_QUEUE_TIMEOUT,
        }
    }
}

impl AdmissionOptions {
    pub fn with_queue_timeout(queue_timeout: Duration) -> Result<Self, AdmissionError> {
        if queue_timeout.is_zero() || queue_timeout > MAX_QUEUE_TIMEOUT {
            return Err(AdmissionError::InvalidQueueTimeout);
        }
        Ok(Self { queue_timeout })
    }
}

#[derive(Debug)]
pub struct InferenceAdmission {
    capacity: usize,
    semaphore: Arc<Semaphore>,
}

/// An owned, non-cloneable lease. Dropping it releases the slot.
pub struct AdmissionLease {
    _permit: OwnedSemaphorePermit,
}

impl std::fmt::Debug for AdmissionLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AdmissionLease")
    }
}

impl InferenceAdmission {
    pub fn new(capacity: usize) -> Result<Arc<Self>, AdmissionError> {
        if !(1..=MAX_INFERENCE_CAPACITY).contains(&capacity) {
            return Err(AdmissionError::InvalidCapacity);
        }
        Ok(Arc::new(Self {
            capacity,
            semaphore: Arc::new(Semaphore::new(capacity)),
        }))
    }

    /// Library callers receive the same finite default service. CLI startup
    /// configures the service before constructing any provider.
    pub fn process() -> Arc<Self> {
        Arc::clone(PROCESS_ADMISSION.get_or_init(|| {
            Arc::new(Self {
                capacity: DEFAULT_INFERENCE_CAPACITY,
                semaphore: Arc::new(Semaphore::new(DEFAULT_INFERENCE_CAPACITY)),
            })
        }))
    }

    /// Configuration is immutable after the first provider has initialized it.
    pub fn configure_process(capacity: usize) -> Result<Arc<Self>, AdmissionError> {
        Self::configure(&PROCESS_ADMISSION, capacity)
    }

    fn configure(cell: &OnceLock<Arc<Self>>, capacity: usize) -> Result<Arc<Self>, AdmissionError> {
        // Validate even when the cell has already been initialized.
        let candidate = Self::new(capacity)?;
        let current = cell.get_or_init(|| candidate);
        if current.capacity != capacity {
            return Err(AdmissionError::AlreadyConfigured {
                capacity: current.capacity,
            });
        }
        Ok(Arc::clone(current))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub async fn acquire(&self) -> Result<AdmissionLease, AdmissionError> {
        self.acquire_with_options(AdmissionOptions::default()).await
    }

    /// Dropping this future cancels its queued acquisition. No transport or
    /// adaptive authorization may precede successful admission.
    pub async fn acquire_with_options(
        &self,
        options: AdmissionOptions,
    ) -> Result<AdmissionLease, AdmissionError> {
        match tokio::time::timeout(
            options.queue_timeout,
            Arc::clone(&self.semaphore).acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => Ok(AdmissionLease { _permit: permit }),
            Ok(Err(_)) => Err(AdmissionError::Closed),
            Err(_) => Err(AdmissionError::QueueTimeout),
        }
    }
}

type ProviderStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>;

struct LeasedStream<G> {
    inner: Option<ProviderStream>,
    lease: Option<G>,
}

impl<G: Unpin> Stream for LeasedStream<G> {
    type Item = Result<StreamEvent, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        let result = inner.as_mut().poll_next(cx);
        if matches!(
            &result,
            Poll::Ready(None | Some(Err(_) | Ok(StreamEvent::Error { .. })))
        ) {
            // Drop the transport before admitting a replacement request.
            drop(this.inner.take());
            drop(this.lease.take());
        }
        result
    }
}

/// Transfer a leaf's lease into its lazy response stream. EOF, a terminal
/// error, or consumer drop releases both transport and lease promptly.
pub fn hold_stream<S>(stream: S, lease: AdmissionLease) -> ProviderStream
where
    S: Stream<Item = Result<StreamEvent, ProviderError>> + Send + 'static,
{
    hold_stream_guard(stream, lease)
}

/// Apply the same terminal cleanup to a leaf with multiple owned guards.
pub(crate) fn hold_stream_guard<S, G>(stream: S, guard: G) -> ProviderStream
where
    S: Stream<Item = Result<StreamEvent, ProviderError>> + Send + 'static,
    G: Send + Unpin + 'static,
{
    Box::pin(LeasedStream {
        inner: Some(Box::pin(stream)),
        lease: Some(guard),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::ProviderId;
    use futures::StreamExt;

    #[test]
    fn configuration_is_bounded_immutable_and_race_safe() {
        assert!(matches!(
            InferenceAdmission::new(0),
            Err(AdmissionError::InvalidCapacity)
        ));
        assert!(matches!(
            InferenceAdmission::new(65),
            Err(AdmissionError::InvalidCapacity)
        ));
        let cell = Arc::new(OnceLock::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cell = Arc::clone(&cell);
                std::thread::spawn(move || InferenceAdmission::configure(&cell, 3).unwrap())
            })
            .collect();
        let services: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(services.iter().all(|s| Arc::ptr_eq(s, &services[0])));
        assert!(matches!(
            InferenceAdmission::configure(&cell, 2),
            Err(AdmissionError::AlreadyConfigured { capacity: 3 })
        ));
        assert!(matches!(
            InferenceAdmission::configure(&cell, 0),
            Err(AdmissionError::InvalidCapacity)
        ));
        assert!(AdmissionOptions::with_queue_timeout(Duration::ZERO).is_err());
        assert!(AdmissionOptions::with_queue_timeout(Duration::MAX).is_err());
    }

    #[test]
    fn process_construction_shares_one_finite_service() {
        let first = InferenceAdmission::process();
        let second = InferenceAdmission::process();
        assert!(Arc::ptr_eq(&first, &second));
        assert!((1..=MAX_INFERENCE_CAPACITY).contains(&first.capacity()));
    }

    #[tokio::test]
    async fn queued_deadline_and_future_drop_do_not_leak_slots() {
        let service = InferenceAdmission::new(1).unwrap();
        let active = service.acquire().await.unwrap();
        let options = AdmissionOptions::with_queue_timeout(Duration::from_millis(5)).unwrap();
        assert!(matches!(
            service.acquire_with_options(options).await,
            Err(AdmissionError::QueueTimeout)
        ));
        let mut queued = Box::pin(service.acquire());
        assert!(futures::poll!(&mut queued).is_pending());
        drop(queued);
        drop(active);
        let replacement = service.acquire_with_options(options).await.unwrap();
        assert_eq!(service.semaphore.available_permits(), 0);
        drop(replacement);
        assert_eq!(service.semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn unpolled_or_pending_stream_drop_releases_slot() {
        let service = InferenceAdmission::new(1).unwrap();
        let lease = service.acquire().await.unwrap();
        let mut stream = hold_stream(futures::stream::pending(), lease);
        assert_eq!(service.semaphore.available_permits(), 0);
        assert!(futures::poll!(stream.next()).is_pending());
        drop(stream);
        assert_eq!(service.semaphore.available_permits(), 1);
        let lease = service.acquire().await.unwrap();
        let stream = hold_stream(futures::stream::pending(), lease);
        drop(stream);
        assert_eq!(service.semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn streaming_retains_slot_until_eof_and_releases_before_error_is_returned() {
        let service = InferenceAdmission::new(1).unwrap();
        let lease = service.acquire().await.unwrap();
        let mut stream = hold_stream(futures::stream::iter([Ok(StreamEvent::MessageStop)]), lease);
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::MessageStop))
        ));
        assert_eq!(service.semaphore.available_permits(), 0);
        assert!(stream.next().await.is_none());
        assert_eq!(service.semaphore.available_permits(), 1);
        let lease = service.acquire().await.unwrap();
        let error = ProviderError::InvalidRequest {
            provider: ProviderId::new("fixture"),
            message: "fixture".into(),
        };
        let mut stream = hold_stream(
            futures::stream::iter([Err(error), Ok(StreamEvent::MessageStop)]),
            lease,
        );
        assert!(matches!(stream.next().await, Some(Err(_))));
        assert_eq!(service.semaphore.available_permits(), 1);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn transport_is_dropped_before_slot_is_released() {
        struct PendingTransport(Arc<InferenceAdmission>, Arc<std::sync::atomic::AtomicBool>);
        impl Stream for PendingTransport {
            type Item = Result<StreamEvent, ProviderError>;
            fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                Poll::Pending
            }
        }
        impl Drop for PendingTransport {
            fn drop(&mut self) {
                assert_eq!(self.0.semaphore.available_permits(), 0);
                self.1.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let service = InferenceAdmission::new(1).unwrap();
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let lease = service.acquire().await.unwrap();
        let stream = hold_stream(
            PendingTransport(Arc::clone(&service), Arc::clone(&dropped)),
            lease,
        );
        drop(stream);
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(service.semaphore.available_permits(), 1);
    }
}
