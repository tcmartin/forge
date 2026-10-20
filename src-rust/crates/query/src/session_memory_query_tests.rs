// Included inside the query loop's test module to reuse its isolated host fixtures.
struct MemoryFixtureProvider {
    id: claurst_core::provider_id::ProviderId,
    requests: StdMutex<Vec<claurst_api::ProviderRequest>>,
    pending: AtomicBool,
    future_dropped: Arc<AtomicBool>,
    entered: tokio::sync::Notify,
}

struct MemoryFutureDrop(Arc<AtomicBool>);
impl Drop for MemoryFutureDrop {
    fn drop(&mut self) {
        self.0.store(true, AtomicOrdering::SeqCst);
    }
}

#[async_trait::async_trait]
impl claurst_api::LlmProvider for MemoryFixtureProvider {
    fn id(&self) -> &claurst_core::provider_id::ProviderId {
        &self.id
    }
    fn name(&self) -> &str {
        "memory custom route"
    }
    async fn create_message(
        &self,
        request: claurst_api::ProviderRequest,
    ) -> Result<claurst_api::ProviderResponse, claurst_api::ProviderError> {
        self.requests.lock().unwrap().push(request);
        let _owner = MemoryFutureDrop(Arc::clone(&self.future_dropped));
        self.entered.notify_one();
        if self.pending.load(AtomicOrdering::SeqCst) {
            std::future::pending::<()>().await;
        }
        Ok(claurst_api::ProviderResponse {
            id: "memory-response".into(),
            model: "memory-model".into(),
            content: vec![ContentBlock::Text {
                text: "MEMORY: constraint | 9 | Use the selected provider".into(),
            }],
            stop_reason: claurst_api::StopReason::EndTurn,
            usage: UsageInfo {
                input_tokens: 7,
                output_tokens: 3,
                ..Default::default()
            },
        })
    }
    async fn create_message_stream(
        &self,
        _: claurst_api::ProviderRequest,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<claurst_api::StreamEvent, claurst_api::ProviderError>,
                    > + Send,
            >,
        >,
        claurst_api::ProviderError,
    > {
        panic!("memory extraction must use the selected bounded nonstream request");
    }
    async fn health_check(
        &self,
    ) -> Result<claurst_api::ProviderStatus, claurst_api::ProviderError> {
        Ok(claurst_api::ProviderStatus::Healthy)
    }
    fn capabilities(&self) -> claurst_api::ProviderCapabilities {
        claurst_api::ProviderCapabilities {
            streaming: true,
            tool_calling: false,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: claurst_api::SystemPromptStyle::TopLevel,
        }
    }
}

struct MemoryQueryFixture {
    _directory: tempfile::TempDir,
    config: QueryConfig,
    ctx: ToolContext,
    client: claurst_api::AnthropicClient,
    provider: Arc<MemoryFixtureProvider>,
    messages: Vec<Message>,
}

impl MemoryQueryFixture {
    fn new(enabled: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryFixtureProvider {
            id: claurst_core::provider_id::ProviderId::new("custom-memory"),
            requests: StdMutex::new(Vec::new()),
            pending: AtomicBool::new(false),
            future_dropped: Arc::new(AtomicBool::new(false)),
            entered: tokio::sync::Notify::new(),
        });
        let mut registry = claurst_api::ProviderRegistry::new();
        registry.register(provider.clone());
        let config = QueryConfig {
            model: "memory-model".into(),
            provider_registry: Some(Arc::new(registry)),
            session_memory_enabled: enabled,
            ..Default::default()
        };
        let mut ctx = deny_all_context();
        ctx.working_dir = directory.path().to_path_buf();
        ctx.config.provider = Some("custom-memory".into());
        let client = claurst_api::AnthropicClient::new(claurst_api::client::ClientConfig {
            api_key: "foreign-anthropic-key-must-not-be-used".into(),
            ..Default::default()
        })
        .unwrap();
        let messages = (0..20)
            .map(|index| {
                let mut message = if index % 2 == 0 {
                    Message::user("session input")
                } else {
                    Message::assistant("session output")
                };
                message.uuid = Some(format!("memory-{index}"));
                message
            })
            .collect();
        Self {
            _directory: directory,
            config,
            ctx,
            client,
            provider,
            messages,
        }
    }
    async fn run(&self, deadline: std::time::Duration) -> bool {
        let target = resolved_target::ResolvedTarget::resolve(
            &self.config,
            &self.ctx,
            &self.config.model,
            &self.client,
        )
        .unwrap();
        assert_eq!(target.provider_id(), "custom-memory");
        session_memory::extract_for_query(
            &self.config,
            &target,
            &self.ctx,
            &self.messages,
            &self.ctx.cost_tracker,
            deadline,
        )
        .await
    }
}

#[tokio::test]
async fn session_memory_default_off_and_child_never_extract() {
    let mut fixture = MemoryQueryFixture::new(false);
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    fixture.config.session_memory_enabled = true;
    fixture.config.child_execution = true;
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert!(fixture.provider.requests.lock().unwrap().is_empty());
    assert!(!fixture.ctx.working_dir.join(".claurst/AGENTS.md").exists());
}

#[tokio::test]
async fn session_memory_selected_route_budget_and_session_cursor() {
    let mut fixture = MemoryQueryFixture::new(true);
    for _ in 0..2 {
        assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    }
    {
        let requests = fixture.provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "memory-model");
        assert_eq!(requests[0].max_tokens, 1024);
        assert!(requests[0].wire_budget.is_some());
        assert!(requests[0].dispatch.is_none());
        assert!(requests[0].tools.is_empty());
        let prompt = requests[0].messages[0].get_all_text();
        assert!(!prompt.contains(&fixture.ctx.working_dir.display().to_string()));
    }
    assert_eq!(fixture.ctx.cost_tracker.total_tokens(), 10);
    fixture.ctx.session_id = "different-session".into();
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert_eq!(fixture.provider.requests.lock().unwrap().len(), 2);
    let saved = tokio::fs::read_to_string(fixture.ctx.working_dir.join(".claurst/AGENTS.md"))
        .await
        .unwrap();
    assert_eq!(saved.matches("Use the selected provider").count(), 1);
}

#[tokio::test]
async fn session_memory_timeout_drops_provider_future_and_allows_retry() {
    let fixture = MemoryQueryFixture::new(true);
    fixture.provider.pending.store(true, AtomicOrdering::SeqCst);
    assert!(!fixture.run(std::time::Duration::from_millis(10)).await);
    assert!(fixture.provider.future_dropped.load(AtomicOrdering::SeqCst));
    assert!(!fixture.config.session_memory_state.lock().initialized);
    assert!(!fixture.ctx.working_dir.join(".claurst/AGENTS.md").exists());
    fixture
        .provider
        .pending
        .store(false, AtomicOrdering::SeqCst);
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert_eq!(fixture.provider.requests.lock().unwrap().len(), 2);
    assert!(fixture.config.session_memory_state.lock().initialized);
}

#[tokio::test]
async fn session_memory_cancellation_drops_provider_without_late_persistence() {
    let fixture = MemoryQueryFixture::new(true);
    fixture.provider.pending.store(true, AtomicOrdering::SeqCst);
    let mut run = Box::pin(fixture.run(std::time::Duration::from_secs(1)));
    tokio::select! {
        _ = fixture.provider.entered.notified() => {},
        _ = &mut run => panic!("provider must stay pending"),
    }
    fixture.ctx.cancel_token.cancel();
    assert!(run.await);
    assert!(fixture.provider.future_dropped.load(AtomicOrdering::SeqCst));
    assert!(!fixture.config.session_memory_state.lock().initialized);
    assert!(!fixture.ctx.working_dir.join(".claurst/AGENTS.md").exists());
}

#[tokio::test]
async fn session_memory_owner_drop_releases_reservation_and_provider() {
    let fixture = MemoryQueryFixture::new(true);
    fixture.provider.pending.store(true, AtomicOrdering::SeqCst);
    let mut run = Box::pin(fixture.run(std::time::Duration::from_secs(1)));
    tokio::select! {
        _ = fixture.provider.entered.notified() => {},
        _ = &mut run => panic!("provider must stay pending"),
    }
    drop(run);
    assert!(fixture.provider.future_dropped.load(AtomicOrdering::SeqCst));
    fixture
        .provider
        .pending
        .store(false, AtomicOrdering::SeqCst);
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert_eq!(fixture.provider.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn session_memory_exhausted_budget_prevents_auxiliary_request() {
    let mut fixture = MemoryQueryFixture::new(true);
    fixture.config.max_budget_usd = Some(0.0);
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert!(fixture.provider.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn session_memory_pruned_history_gets_one_fresh_extraction() {
    let mut fixture = MemoryQueryFixture::new(true);
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    for (index, message) in fixture.messages.iter_mut().enumerate() {
        message.uuid = Some(format!("pruned-history-{index}"));
    }
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert_eq!(fixture.provider.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn session_memory_oversized_persistence_preserves_existing_instructions() {
    let fixture = MemoryQueryFixture::new(true);
    let path = fixture.ctx.working_dir.join(".claurst/AGENTS.md");
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    let original = "x".repeat(256 * 1024 + 1);
    tokio::fs::write(&path, &original).await.unwrap();
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), original);
    assert!(!fixture.config.session_memory_state.lock().initialized);
    let mut entries = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    entries.sort();
    assert_eq!(entries, [".AGENTS.md.lock", "AGENTS.md"]);
}

#[tokio::test]
async fn session_memory_failed_persistence_does_not_repeat_paid_extraction() {
    let fixture = MemoryQueryFixture::new(true);
    let path = fixture.ctx.working_dir.join(".claurst/AGENTS.md");
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&path, "x".repeat(256 * 1024 + 1))
        .await
        .unwrap();
    for _ in 0..2 {
        assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    }
    assert_eq!(fixture.provider.requests.lock().unwrap().len(), 1);
    assert_eq!(fixture.ctx.cost_tracker.total_tokens(), 10);
    assert!(!fixture.config.session_memory_state.lock().initialized);
    tokio::fs::remove_file(&path).await.unwrap();
    assert!(!fixture.run(std::time::Duration::from_secs(1)).await);
    assert_eq!(fixture.provider.requests.lock().unwrap().len(), 1);
    assert!(fixture.config.session_memory_state.lock().initialized);
    assert!(tokio::fs::read_to_string(&path)
        .await
        .unwrap()
        .contains("Use the selected provider"));
}
