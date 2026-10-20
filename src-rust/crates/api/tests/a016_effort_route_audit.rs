use std::time::Duration;

use claurst_api::provider::LlmProvider;
use claurst_api::providers::{GoogleProvider, OpenAiCompatProvider, OpenAiProvider};
use claurst_core::dispatch::DispatchDenied;
use claurst_core::effort::EffortLevel;
use tokio::net::TcpListener;

/// Adaptive authority is currently certified only for the legacy Anthropic
/// route. These distinct provider adapters inherit the trait's fail-closed
/// route default; this regression also proves route selection opens no socket.
#[tokio::test]
async fn additional_provider_routes_remain_fixed_without_transport_traffic() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind zero-traffic sentinel");
    let endpoint = format!(
        "http://{}",
        listener.local_addr().expect("sentinel address")
    );
    let providers: [Box<dyn LlmProvider>; 3] = [
        Box::new(OpenAiProvider::new("fixture-key".into()).with_base_url(endpoint.clone())),
        Box::new(GoogleProvider::new("fixture-key".into()).with_base_url(endpoint.clone())),
        Box::new(
            OpenAiCompatProvider::new("fixture-compat", "fixture", endpoint)
                .with_api_key("fixture-key".into()),
        ),
    ];

    for provider in providers {
        assert_eq!(
            provider.dispatch_route("fixture-model", EffortLevel::Medium),
            Err(DispatchDenied::UnsupportedRoute),
            "{} must retain fixed/base behavior until an actual route proof exists",
            provider.id()
        );
    }

    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "route audit must not open an inference transport"
    );
}
