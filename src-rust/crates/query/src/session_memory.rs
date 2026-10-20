// Session Memory Extraction for cc-query.
//
// Extracts key facts after an explicitly enabled session and persists them to
// AGENTS.md through the host-owned query task.
//
// This mirrors TypeScript services/SessionMemory/sessionMemory.ts and
// services/extractMemories/extractMemories.ts.
//
// Strategy:
//   1. After sessions with 20+ messages (or on compact), the host may call the API with a
//      structured extraction prompt.
//   2. Parse the response into typed ExtractedMemory entries.
//   3. Append entries under "## Auto-extracted memories" in AGENTS.md
//      (creating the file if it doesn't exist).
//   4. Track state so we don't re-extract from already-processed messages.

use claurst_api::{
    AnthropicStreamEvent, ApiMessage, CreateMessageRequest, LlmProvider, ProviderRequest,
    StreamAccumulator, StreamHandler, SystemPrompt,
};
use claurst_core::types::{ContentBlock, Message, MessageContent, Role, UsageInfo};
use serde_json::Value;
use std::fs::OpenOptions as StdOpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Threshold constants (mirrors TypeScript sessionMemoryUtils.ts defaults)
// ---------------------------------------------------------------------------

/// Minimum messages before extraction is attempted.
const MIN_MESSAGES_TO_EXTRACT: usize = 20;

/// Minimum tool calls since last extraction before we run again.
const MIN_TOOL_CALLS_BETWEEN_EXTRACTIONS: usize = 3;

/// Extraction is auxiliary work. Keep both the conversation excerpt and its
/// result bounded even when a host retains a very large transcript.
pub const MAX_TRANSCRIPT_CHARS: usize = 24_000;
const MAX_MESSAGE_CHARS: usize = 2_000;
const MAX_WORKSPACE_CHARS: usize = 512;
const MAX_MEMORY_CONTENT_CHARS: usize = 1_000;
const MAX_MEMORIES_PER_EXTRACTION: usize = 32;
const MAX_EXTRACTION_OUTPUT_TOKENS: u32 = 1_024;
const MAX_MEMORY_FILE_BYTES: usize = 256 * 1024;
const MEMORY_LOCK_WAIT: Duration = Duration::from_millis(10);
const MEMORY_LOCK_DEADLINE: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Category of an extracted memory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryCategory {
    UserPreference,
    ProjectFact,
    CodePattern,
    Decision,
    Constraint,
}

impl MemoryCategory {
    /// Parse from the string tag produced by the model.
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "user_preference" | "userpreference" | "preference" => Self::UserPreference,
            "project_fact" | "projectfact" | "fact" => Self::ProjectFact,
            "code_pattern" | "codepattern" | "pattern" => Self::CodePattern,
            "decision" => Self::Decision,
            "constraint" => Self::Constraint,
            _ => Self::ProjectFact, // default
        }
    }

    /// Display label used in the persisted markdown.
    fn label(&self) -> &'static str {
        match self {
            Self::UserPreference => "user-preference",
            Self::ProjectFact => "project-fact",
            Self::CodePattern => "code-pattern",
            Self::Decision => "decision",
            Self::Constraint => "constraint",
        }
    }
}

/// A single fact extracted from the conversation.
#[derive(Debug, Clone)]
pub struct ExtractedMemory {
    /// The fact to remember, as a markdown bullet point or sentence.
    pub content: String,
    /// Semantic category for the fact.
    pub category: MemoryCategory,
    /// Model confidence, 0.0–1.0.
    pub confidence: f32,
}

fn memory_lock_path(target_path: &Path) -> PathBuf {
    let file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("AGENTS.md");
    target_path.with_file_name(format!(".{file_name}.lock"))
}

async fn acquire_memory_lock(target_path: &Path) -> anyhow::Result<std::fs::File> {
    let lock_path = memory_lock_path(target_path);
    let mut options = StdOpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let file = options.open(&lock_path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = file.metadata()?.permissions();
        permissions.set_mode(0o600);
        file.set_permissions(permissions)?;
    }

    let deadline = Instant::now() + MEMORY_LOCK_DEADLINE;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    anyhow::bail!(
                        "timed out acquiring session-memory lock {}",
                        lock_path.display()
                    );
                }
                tokio::time::sleep(MEMORY_LOCK_WAIT).await;
            }
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "failed to acquire session-memory lock {}: {error}",
                    lock_path.display()
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Session state (tracks what has already been extracted)
// ---------------------------------------------------------------------------

/// Mutable per-session state for the memory extractor.
#[derive(Debug, Default)]
pub struct SessionMemoryState {
    /// UUID of the last message that was fully extracted.
    pub last_extracted_message_uuid: Option<String>,
    /// Fallback cursor for hosts whose messages do not carry UUIDs.
    pub last_extracted_message_count: usize,
    /// Fallback tool cursor for hosts whose messages do not carry UUIDs.
    pub tool_calls_at_last_extraction: usize,
    /// Token count at the time of the last extraction.
    pub tokens_at_last_extraction: u64,
    /// Whether session memory has been initialised for this session.
    pub initialized: bool,
    /// Workspace identity for the cursor. A cursor is never reused across
    /// workspaces because that can suppress extraction for an unrelated
    /// session with colliding message IDs.
    pub workspace: Option<PathBuf>,
    pub session_id: Option<String>,
    in_progress: bool,
    pending: Option<Arc<PendingExtraction>>,
}

impl SessionMemoryState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return `true` if the last extracted message UUID is still present in
    /// `messages`, which tells us new messages have been added since then.
    pub fn has_new_messages_since_last_extraction(&self, messages: &[Message]) -> bool {
        match &self.last_extracted_message_uuid {
            None if !self.initialized => true,
            None => messages.len() > self.last_extracted_message_count,
            Some(uuid) => messages.last().and_then(|m| m.uuid.as_deref()) != Some(uuid.as_str()),
        }
    }

    /// Bind this state to a workspace. Returns `true` when the state was
    /// already bound to this workspace or was freshly initialized for it.
    /// A changed workspace clears the extraction cursor before returning.
    pub fn bind_workspace(&mut self, workspace: &Path) -> bool {
        let workspace = workspace.to_path_buf();
        match &self.workspace {
            Some(previous) if previous != &workspace => {
                self.last_extracted_message_uuid = None;
                self.last_extracted_message_count = 0;
                self.tool_calls_at_last_extraction = 0;
                self.tokens_at_last_extraction = 0;
                self.initialized = false;
                self.pending = None;
                self.workspace = Some(workspace);
                false
            }
            Some(_) => true,
            None => {
                self.workspace = Some(workspace);
                true
            }
        }
    }

    /// Advance the cursor to the last message in `messages`.
    pub fn advance_cursor(&mut self, messages: &[Message]) {
        MemoryCursor::capture(messages).apply(self);
    }
}

#[derive(Debug)]
struct MemoryCursor {
    uuid: Option<String>,
    message_count: usize,
    tool_calls: usize,
}

impl MemoryCursor {
    fn capture(messages: &[Message]) -> Self {
        Self {
            uuid: messages.last().and_then(|message| message.uuid.clone()),
            message_count: messages.len(),
            tool_calls: messages
                .iter()
                .filter(|message| message.role == Role::Assistant)
                .map(|message| message.get_tool_use_blocks().len())
                .sum(),
        }
    }

    fn apply(&self, state: &mut SessionMemoryState) {
        state.last_extracted_message_uuid = self.uuid.clone();
        state.last_extracted_message_count = self.message_count;
        state.tool_calls_at_last_extraction = self.tool_calls;
        state.initialized = true;
    }
}

#[derive(Debug)]
struct PendingExtraction {
    memories: Vec<ExtractedMemory>,
    cursor: MemoryCursor,
}

// ---------------------------------------------------------------------------
// SessionMemoryExtractor
// ---------------------------------------------------------------------------

/// Extracts and persists key memories from a conversation.
pub struct SessionMemoryExtractor {
    pub model: String,
    pub max_tokens: u32,
}

impl SessionMemoryExtractor {
    /// Create a new extractor using the given model.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            max_tokens: 4096,
        }
    }

    /// Return `true` if we have enough messages to warrant extraction.
    ///
    /// Mirrors `shouldExtractMemory` from TypeScript sessionMemory.ts:
    /// - At least `MIN_MESSAGES_TO_EXTRACT` messages total
    /// - The last assistant turn must not have pending tool calls (safe extraction point)
    pub fn should_extract(messages: &[Message]) -> bool {
        let model_visible = messages
            .iter()
            .filter(|m| m.role == Role::User || m.role == Role::Assistant)
            .count();

        if model_visible < MIN_MESSAGES_TO_EXTRACT {
            return false;
        }

        // Don't extract mid-tool-call chain — wait for a clean end-turn
        let last_assistant = messages.iter().rev().find(|m| m.role == Role::Assistant);
        if let Some(last) = last_assistant {
            if last.has_tool_use() {
                return false; // still in a tool chain
            }
        }

        true
    }

    /// Count tool calls since the last extracted message UUID.
    fn count_tool_calls_since(messages: &[Message], since_uuid: Option<&str>) -> usize {
        let mut found_start = since_uuid.is_none();
        let mut count = 0usize;

        for msg in messages {
            if !found_start {
                if msg.uuid.as_deref() == since_uuid {
                    found_start = true;
                }
                continue;
            }
            if msg.role == Role::Assistant {
                count += msg.get_tool_use_blocks().len();
            }
        }
        count
    }

    fn count_total_tool_calls(messages: &[Message]) -> usize {
        messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .map(|message| message.get_tool_use_blocks().len())
            .sum()
    }

    /// Check whether extraction should run given the current session state.
    pub fn should_extract_with_state(messages: &[Message], state: &SessionMemoryState) -> bool {
        if !Self::should_extract(messages) {
            return false;
        }

        // Require minimum tool calls between updates (mirrors TS toolCallsBetweenUpdates)
        let tool_calls_since = match state.last_extracted_message_uuid.as_deref() {
            Some(cursor) => Self::count_tool_calls_since(messages, Some(cursor)),
            None => Self::count_total_tool_calls(messages)
                .saturating_sub(state.tool_calls_at_last_extraction),
        };

        if !state.has_new_messages_since_last_extraction(messages) {
            return false;
        }

        // The first extraction is allowed once the normal threshold is met.
        // Subsequent extractions require enough new tool activity to avoid
        // repeatedly sending the same conversation to the provider.
        !state.initialized
            || state
                .last_extracted_message_uuid
                .as_deref()
                .is_some_and(|cursor| {
                    !messages
                        .iter()
                        .any(|message| message.uuid.as_deref() == Some(cursor))
                })
            || tool_calls_since >= MIN_TOOL_CALLS_BETWEEN_EXTRACTIONS
    }

    /// Extract key memories from a conversation.
    ///
    /// Calls the API with a structured extraction prompt and parses the
    /// response into `ExtractedMemory` entries.
    pub async fn extract(
        &self,
        messages: &[Message],
        working_dir: &Path,
        api_client: &claurst_api::AnthropicClient,
    ) -> anyhow::Result<(Vec<ExtractedMemory>, UsageInfo)> {
        let model_visible: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role == Role::User || m.role == Role::Assistant)
            .collect();

        if model_visible.is_empty() {
            return Ok((vec![], UsageInfo::default()));
        }

        let transcript = bounded_transcript(&model_visible);

        let _ = working_dir;
        let prompt = build_extraction_prompt(&transcript, "current workspace");

        let api_msgs = vec![ApiMessage {
            role: "user".to_string(),
            content: Value::String(prompt),
        }];

        let wire_budget = crate::default_wire_budget(u64::from(
            self.max_tokens.min(MAX_EXTRACTION_OUTPUT_TOKENS),
        ))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let request = CreateMessageRequest::builder(
            &self.model,
            self.max_tokens.min(MAX_EXTRACTION_OUTPUT_TOKENS),
        )
        .messages(api_msgs)
        .system(SystemPrompt::Text(EXTRACTION_SYSTEM_PROMPT.to_string()))
        .wire_budget(wire_budget)
        .build();

        let handler: Arc<dyn StreamHandler> = Arc::new(claurst_api::streaming::NullStreamHandler);
        let mut rx = api_client
            .create_message_stream(request, handler)
            .await
            .map_err(|_| anyhow::anyhow!("API request failed"))?;

        let mut acc = StreamAccumulator::new();
        while let Some(evt) = rx.recv().await {
            acc.on_event(&evt);
            if matches!(evt, AnthropicStreamEvent::MessageStop) {
                break;
            }
        }

        let (response_msg, usage, _stop) = acc.finish();
        let response_text = response_msg.get_all_text();

        if response_text.is_empty() {
            debug!("Session memory extraction produced empty response");
            return Ok((vec![], usage));
        }

        let memories = parse_extraction_response(&response_text);
        info!(count = memories.len(), "Session memory extraction complete");

        Ok((memories, usage))
    }

    /// Extract using the exact provider selected for the active query route.
    /// The provider is supplied by the host; this method never reads
    /// credentials or constructs a replacement client.
    pub async fn extract_with_provider(
        &self,
        messages: &[Message],
        working_dir: &Path,
        provider: &dyn LlmProvider,
    ) -> anyhow::Result<(Vec<ExtractedMemory>, UsageInfo)> {
        let model_visible: Vec<&Message> = messages
            .iter()
            .filter(|m| m.role == Role::User || m.role == Role::Assistant)
            .collect();
        if model_visible.is_empty() {
            return Ok((vec![], UsageInfo::default()));
        }

        let transcript = bounded_transcript(&model_visible);
        let _ = working_dir;
        let prompt = build_extraction_prompt(&transcript, "current workspace");
        let wire_budget = crate::default_wire_budget(u64::from(
            self.max_tokens.min(MAX_EXTRACTION_OUTPUT_TOKENS),
        ))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let request = ProviderRequest {
            model: self.model.clone(),
            messages: vec![Message::user(prompt)],
            system_prompt: Some(SystemPrompt::Text(EXTRACTION_SYSTEM_PROMPT.to_string())),
            tools: Vec::new(),
            max_tokens: self.max_tokens.min(MAX_EXTRACTION_OUTPUT_TOKENS),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            thinking: None,
            provider_options: Value::Object(serde_json::Map::new()),
            cache_scope: None,
            response_continuation: None,
            wire_budget: Some(wire_budget),
            // Auxiliary memory extraction must not inherit adaptive authority
            // from the primary query route.
            dispatch: None,
        };

        let response = provider
            .create_message(request)
            .await
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let usage = response.usage;
        let response_text = Message::assistant_blocks(response.content).get_all_text();
        if response_text.is_empty() {
            debug!(provider = %provider.id(), "Session memory extraction produced empty response");
            return Ok((vec![], usage));
        }
        let memories = parse_extraction_response(&response_text);
        info!(provider = %provider.id(), count = memories.len(), "Session memory extraction complete");
        Ok((memories, usage))
    }

    /// Persist extracted memories to `target_path` (creates directories and
    /// the file if they don't exist).  Appends under `## Auto-extracted memories`.
    pub async fn persist(memories: &[ExtractedMemory], target_path: &Path) -> anyhow::Result<()> {
        if memories.is_empty() {
            return Ok(());
        }

        // Ensure parent directory exists
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // The lock is a stable sidecar inode. Keep it from before the read
        // through the atomic rename so independent sessions/processes cannot
        // lose one another's merge. The inode is intentionally never removed.
        let _lock = acquire_memory_lock(target_path).await?;

        // Read existing content (or start fresh)
        let existing = match fs::File::open(target_path).await {
            Ok(file) => {
                let mut content = String::new();
                file.take((MAX_MEMORY_FILE_BYTES + 1) as u64)
                    .read_to_string(&mut content)
                    .await?;
                anyhow::ensure!(
                    content.len() <= MAX_MEMORY_FILE_BYTES,
                    "memory file exceeds byte limit"
                );
                content
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e.into()),
        };

        // Build the new entries block
        let date_str = chrono::Local::now().format("%Y-%m-%d").to_string();
        let mut new_entries = Vec::new();
        for memory in memories {
            let content = truncate_chars(memory.content.trim(), MAX_MEMORY_CONTENT_CHARS);
            if content.is_empty() {
                continue;
            }
            let entry = format!(
                "- **[{}]** {} *(confidence: {:.0}%)*\n",
                memory.category.label(),
                content,
                memory.confidence.clamp(0.0, 1.0) * 100.0
            );
            let same_content = |line: &str| {
                line.strip_prefix("- **[")
                    .and_then(|rest| rest.split_once("]** "))
                    .and_then(|(_, body)| body.rsplit_once(" *(confidence: "))
                    .is_some_and(|(saved, _)| saved == content)
            };
            if !existing.lines().any(same_content)
                && !new_entries
                    .iter()
                    .any(|prior: &String| same_content(prior.trim_end()))
            {
                new_entries.push(entry);
            }
        }
        if new_entries.is_empty() {
            debug!(path = %target_path.display(), "No new session memories to persist");
            return Ok(());
        }
        let mut new_block = format!("\n### Session memories — {}\n\n", date_str);
        for entry in new_entries {
            new_block.push_str(&entry);
        }

        // Insert under the auto-extracted memories section header (or append it)
        const SECTION_HEADER: &str = "## Auto-extracted memories";

        let updated = if existing.contains(SECTION_HEADER) {
            // Find the section and append to it
            if let Some(section_pos) = existing.find(SECTION_HEADER) {
                // Find the end of the section (next ## or end of file)
                let after_header = &existing[section_pos + SECTION_HEADER.len()..];
                let section_end = after_header
                    .find("\n## ")
                    .map(|p| p + section_pos + SECTION_HEADER.len())
                    .unwrap_or(existing.len());

                let mut result = existing[..section_end].to_string();
                result.push_str(&new_block);
                result.push_str(&existing[section_end..]);
                result
            } else {
                existing
            }
        } else {
            // Append the section at the end
            let mut result = existing;
            if !result.is_empty() && !result.ends_with('\n') {
                result.push('\n');
            }
            result.push_str(&format!("\n{}\n", SECTION_HEADER));
            result.push_str(&new_block);
            result
        };

        anyhow::ensure!(
            updated.len() <= MAX_MEMORY_FILE_BYTES,
            "updated memory file exceeds byte limit"
        );
        let pending = PendingMemoryWrite {
            path: target_path.with_file_name(format!(".memory-{}.tmp", uuid::Uuid::new_v4())),
        };
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(&pending.path).await?;
        file.write_all(updated.as_bytes()).await?;
        file.flush().await?;
        drop(file);
        // The only destination mutation is this atomic commit. Cancellation
        // while preparing bytes drops the scratch guard and preserves AGENTS.md.
        std::fs::rename(&pending.path, target_path)?;
        info!(path = %target_path.display(), count = memories.len(), "Memories persisted");
        Ok(())
    }
}

struct PendingMemoryWrite {
    path: PathBuf,
}

impl Drop for PendingMemoryWrite {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct ExtractionReservation {
    state: Arc<parking_lot::Mutex<SessionMemoryState>>,
}

impl Drop for ExtractionReservation {
    fn drop(&mut self) {
        self.state.lock().in_progress = false;
    }
}

/// Return true only when the owning query was cancelled. All provider work is
/// awaited here and uses the exact already-resolved transport and model.
pub(crate) async fn extract_for_query(
    config: &crate::QueryConfig,
    target: &crate::resolved_target::ResolvedTarget<'_>,
    ctx: &claurst_tools::ToolContext,
    messages: &[Message],
    cost: &claurst_core::cost::CostTracker,
    deadline: std::time::Duration,
) -> bool {
    if !config.session_memory_enabled || config.child_execution {
        return false;
    }
    if ctx.cancel_token.is_cancelled() {
        return true;
    }
    {
        let mut state = config.session_memory_state.lock();
        if state.in_progress {
            return false;
        }
        if state.session_id.as_deref() != Some(ctx.session_id.as_str()) {
            *state = SessionMemoryState::new();
            state.session_id = Some(ctx.session_id.clone());
        }
        state.bind_workspace(&ctx.working_dir);
        if state.pending.is_none()
            && !SessionMemoryExtractor::should_extract_with_state(messages, &state)
        {
            return false;
        }
        state.in_progress = true;
    }
    let _reservation = ExtractionReservation {
        state: Arc::clone(&config.session_memory_state),
    };
    let extractor = SessionMemoryExtractor::new(target.model_id());
    let operation = async {
        let pending = config.session_memory_state.lock().pending.clone();
        let pending = if let Some(pending) = pending {
            pending
        } else {
            if config
                .max_budget_usd
                .is_some_and(|limit| cost.total_cost_usd() >= limit)
            {
                return Ok(());
            }
            let (memories, usage) = if let Some(provider) = target.provider() {
                extractor
                    .extract_with_provider(messages, &ctx.working_dir, provider.as_ref())
                    .await?
            } else if let Some(client) = target.legacy_client() {
                extractor
                    .extract(messages, &ctx.working_dir, client)
                    .await?
            } else {
                anyhow::bail!("selected memory extraction route is unavailable")
            };
            cost.add_usage(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
            );
            // Cache the bounded result until its atomic write succeeds. A disk
            // error must not cause another paid extraction of the same history.
            let pending = Arc::new(PendingExtraction {
                memories,
                cursor: MemoryCursor::capture(messages),
            });
            config.session_memory_state.lock().pending = Some(Arc::clone(&pending));
            pending
        };
        let path = ctx.working_dir.join(".claurst").join("AGENTS.md");
        SessionMemoryExtractor::persist(&pending.memories, &path).await?;
        let mut state = config.session_memory_state.lock();
        if state
            .pending
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &pending))
        {
            pending.cursor.apply(&mut state);
            state.pending = None;
        }
        Ok::<_, anyhow::Error>(())
    };
    tokio::select! {
        biased;
        _ = ctx.cancel_token.cancelled() => return true,
        result = tokio::time::timeout(deadline, operation) => match result {
            Ok(Ok(())) => {},
            Ok(Err(_)) => warn!("Session memory extraction or persistence failed"),
            Err(_) => warn!("Session memory extraction or persistence deadline exceeded"),
        },
    }

    false
}

// ---------------------------------------------------------------------------
// Extraction prompt
// ---------------------------------------------------------------------------

const EXTRACTION_SYSTEM_PROMPT: &str = "You are a memory extraction assistant. Your job is to \
identify key facts, preferences, patterns, and decisions from a conversation that would be \
useful to remember for future interactions. Be precise, concise, and only extract genuinely \
useful information. Do not extract trivial or transient details.";

fn build_extraction_prompt(transcript: &str, working_dir: &str) -> String {
    let working_dir = truncate_chars(working_dir, MAX_WORKSPACE_CHARS);
    format!(
        "Please analyze the following conversation transcript from a coding session in \
directory `{}` and extract key memories that would be useful to remember for future \
interactions.\n\
\n\
For each memory, output a line in this exact format:\n\
MEMORY: <category> | <confidence 0-10> | <concise fact>\n\
\n\
Where <category> is one of:\n\
- user_preference: how the user likes to work, communication style, tool preferences\n\
- project_fact: facts about the codebase, architecture, languages, frameworks\n\
- code_pattern: coding patterns, idioms, or styles used in this project\n\
- decision: key decisions made during the session\n\
- constraint: constraints, requirements, or limitations discovered\n\
\n\
Only output MEMORY: lines — no other text.  If there are no useful memories, output nothing.\n\
\n\
<conversation>\n\
{}\n\
</conversation>",
        working_dir,
        truncate_chars(transcript, MAX_TRANSCRIPT_CHARS)
    )
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut truncated: String = value.chars().take(max_chars).collect();
    if value.chars().count() > max_chars {
        truncated.push_str("…[truncated]");
    }
    truncated
}

fn bounded_transcript(messages: &[&Message]) -> String {
    let mut entries = Vec::new();
    let mut remaining = MAX_TRANSCRIPT_CHARS;
    for msg in messages.iter().rev() {
        let role_label = match msg.role {
            Role::User => "Human",
            Role::Assistant => "Assistant",
        };
        let text: String = match &msg.content {
            MessageContent::Text(text) => text.chars().take(MAX_MESSAGE_CHARS).collect(),
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .flat_map(str::chars)
                .take(MAX_MESSAGE_CHARS)
                .collect(),
        };
        if text.is_empty() {
            continue;
        }
        let entry: String = format!("{role_label}: {text}\n\n")
            .chars()
            .take(remaining)
            .collect();
        remaining -= entry.chars().count();
        entries.push(entry);
        if remaining == 0 {
            break;
        }
    }
    entries.reverse();
    entries.concat()
}

/// Parse the model's response into `ExtractedMemory` entries.
/// Expected line format: `MEMORY: <category> | <confidence 0-10> | <fact>`
fn parse_extraction_response(response: &str) -> Vec<ExtractedMemory> {
    let mut memories = Vec::new();

    for line in response.lines() {
        let line = line.trim();
        if !line.starts_with("MEMORY:") {
            continue;
        }

        let rest = line["MEMORY:".len()..].trim();
        let parts: Vec<&str> = rest.splitn(3, '|').collect();
        if parts.len() < 3 {
            warn!("Skipping malformed memory line: {}", line);
            continue;
        }

        let category = MemoryCategory::from_str(parts[0].trim());
        let confidence_raw: f32 = parts[1].trim().parse().unwrap_or(5.0);
        // Normalize confidence from 0-10 scale to 0.0-1.0
        let confidence = if confidence_raw.is_finite() {
            (confidence_raw / 10.0).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let content = truncate_chars(parts[2].trim(), MAX_MEMORY_CONTENT_CHARS);

        if content.is_empty() {
            continue;
        }

        memories.push(ExtractedMemory {
            content,
            category,
            confidence,
        });
        if memories.len() >= MAX_MEMORIES_PER_EXTRACTION {
            break;
        }
    }

    memories
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::types::Message;

    fn make_user(text: &str) -> Message {
        Message::user(text)
    }

    fn make_assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    fn make_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    make_user(&format!("user message {}", i))
                } else {
                    make_assistant(&format!("assistant reply {}", i))
                }
            })
            .collect()
    }

    // ---- should_extract ------------------------------------------------

    #[test]
    fn test_should_not_extract_too_few_messages() {
        let msgs = make_messages(5);
        assert!(!SessionMemoryExtractor::should_extract(&msgs));
    }

    #[test]
    fn test_should_extract_enough_messages() {
        let msgs = make_messages(MIN_MESSAGES_TO_EXTRACT);
        // All messages are simple text (no tool use), so the last assistant
        // doesn't have pending tool calls — should be ok to extract.
        assert!(SessionMemoryExtractor::should_extract(&msgs));
    }

    #[test]
    fn test_should_not_extract_mid_tool_chain() {
        use claurst_core::types::ContentBlock;
        let mut msgs = make_messages(MIN_MESSAGES_TO_EXTRACT);
        // Replace the last assistant message with one that has a tool_use block
        let last = msgs.last_mut().unwrap();
        *last = Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "tool-123".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
            thought_signature: None,
        }]);
        // Last assistant has tool_use → extraction should be deferred
        assert!(!SessionMemoryExtractor::should_extract(&msgs));
    }

    // ---- parse_extraction_response -------------------------------------

    #[test]
    fn test_parse_empty_response() {
        let memories = parse_extraction_response("");
        assert!(memories.is_empty());
    }

    #[test]
    fn test_parse_single_memory() {
        let response = "MEMORY: project_fact | 8 | The project uses Rust 2021 edition";
        let memories = parse_extraction_response(response);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].category, MemoryCategory::ProjectFact);
        assert!((memories[0].confidence - 0.8).abs() < 0.01);
        assert!(memories[0].content.contains("Rust 2021"));
    }

    #[test]
    fn test_parse_multiple_memories() {
        let response = "\
MEMORY: user_preference | 9 | User prefers verbose error messages\n\
MEMORY: decision | 7 | Decided to use tokio async runtime\n\
MEMORY: constraint | 6 | Must support Windows paths";
        let memories = parse_extraction_response(response);
        assert_eq!(memories.len(), 3);
        assert_eq!(memories[0].category, MemoryCategory::UserPreference);
        assert_eq!(memories[1].category, MemoryCategory::Decision);
        assert_eq!(memories[2].category, MemoryCategory::Constraint);
    }

    #[test]
    fn test_parse_ignores_non_memory_lines() {
        let response = "Here are the memories:\n\
MEMORY: project_fact | 8 | Uses serde for JSON\n\
This is some extra text.\n\
MEMORY: code_pattern | 7 | Uses builder pattern";
        let memories = parse_extraction_response(response);
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_parse_malformed_line_skipped() {
        let response = "MEMORY: only_two_parts | no_confidence";
        let memories = parse_extraction_response(response);
        assert!(memories.is_empty());
    }

    #[test]
    fn test_parse_confidence_normalization() {
        let response = "MEMORY: decision | 10 | High confidence fact";
        let memories = parse_extraction_response(response);
        assert!((memories[0].confidence - 1.0).abs() < 0.01);
    }

    // ---- MemoryCategory parsing ----------------------------------------

    #[test]
    fn test_category_from_str_variants() {
        assert_eq!(
            MemoryCategory::from_str("user_preference"),
            MemoryCategory::UserPreference
        );
        assert_eq!(
            MemoryCategory::from_str("project_fact"),
            MemoryCategory::ProjectFact
        );
        assert_eq!(
            MemoryCategory::from_str("code_pattern"),
            MemoryCategory::CodePattern
        );
        assert_eq!(
            MemoryCategory::from_str("decision"),
            MemoryCategory::Decision
        );
        assert_eq!(
            MemoryCategory::from_str("constraint"),
            MemoryCategory::Constraint
        );
    }

    #[test]
    fn test_category_unknown_defaults_to_project_fact() {
        assert_eq!(
            MemoryCategory::from_str("totally_unknown"),
            MemoryCategory::ProjectFact
        );
    }

    // ---- persist (integration-ish with tempfile) -----------------------

    #[tokio::test]
    async fn test_persist_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join(".claurst").join("AGENTS.md");

        let memories = vec![ExtractedMemory {
            content: "Uses async Rust".to_string(),
            category: MemoryCategory::ProjectFact,
            confidence: 0.9,
        }];

        SessionMemoryExtractor::persist(&memories, &target)
            .await
            .unwrap();

        let content = fs::read_to_string(&target).await.unwrap();
        assert!(content.contains("Auto-extracted memories"));
        assert!(content.contains("Uses async Rust"));
        assert!(content.contains("project-fact"));
    }

    #[tokio::test]
    async fn test_persist_appends_to_existing() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");

        // Write initial content
        fs::write(&target, "# My Project\n\nExisting content.\n")
            .await
            .unwrap();

        let memories = vec![ExtractedMemory {
            content: "Prefers explicit error handling".to_string(),
            category: MemoryCategory::UserPreference,
            confidence: 0.8,
        }];

        SessionMemoryExtractor::persist(&memories, &target)
            .await
            .unwrap();

        let content = fs::read_to_string(&target).await.unwrap();
        assert!(content.contains("Existing content."));
        assert!(content.contains("Auto-extracted memories"));
        assert!(content.contains("Prefers explicit error handling"));
    }

    #[tokio::test]
    async fn test_persist_appends_under_existing_section() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");

        // Pre-populate the auto-extracted section
        let initial = "# Notes\n\n## Auto-extracted memories\n\n### Old memories\n- old fact\n";
        fs::write(&target, initial).await.unwrap();

        let memories = vec![ExtractedMemory {
            content: "New fact discovered".to_string(),
            category: MemoryCategory::ProjectFact,
            confidence: 0.7,
        }];

        SessionMemoryExtractor::persist(&memories, &target)
            .await
            .unwrap();

        let content = fs::read_to_string(&target).await.unwrap();
        // Should have both old and new facts
        assert!(content.contains("old fact"));
        assert!(content.contains("New fact discovered"));
        // Section header should appear only once
        assert_eq!(content.matches("## Auto-extracted memories").count(), 1);
    }

    #[tokio::test]
    async fn test_persist_no_op_for_empty_memories() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");

        SessionMemoryExtractor::persist(&[], &target).await.unwrap();

        // File should NOT be created when there are no memories to persist
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn test_persist_deduplicates_identical_entries() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");
        let memories = vec![ExtractedMemory {
            content: "The project uses bounded memory extraction".to_string(),
            category: MemoryCategory::Constraint,
            confidence: 0.9,
        }];

        SessionMemoryExtractor::persist(&memories, &target)
            .await
            .unwrap();
        SessionMemoryExtractor::persist(&memories, &target)
            .await
            .unwrap();

        let content = fs::read_to_string(&target).await.unwrap();
        assert_eq!(
            content
                .matches("The project uses bounded memory extraction")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn test_persist_serializes_contenders_and_keeps_stable_lock_inode() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");
        let lock_path = memory_lock_path(&target);
        fs::create_dir_all(target.parent().unwrap()).await.unwrap();
        let held = acquire_memory_lock(&target).await.unwrap();
        assert!(lock_path.is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&lock_path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        let waiting_target = target.clone();
        let waiting = tokio::spawn(async move {
            SessionMemoryExtractor::persist(
                &[ExtractedMemory {
                    content: "waited for the persistent lock".to_string(),
                    category: MemoryCategory::Constraint,
                    confidence: 1.0,
                }],
                &waiting_target,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !waiting.is_finished(),
            "contender bypassed the sidecar lock"
        );
        drop(held);
        waiting.await.unwrap().unwrap();
        assert!(lock_path.is_file(), "lock inode must never be unlinked");
        assert!(fs::read_to_string(&target)
            .await
            .unwrap()
            .contains("waited for the persistent lock"));
    }

    #[tokio::test]
    async fn test_concurrent_persist_calls_merge_before_atomic_commit() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");
        let first_target = target.clone();
        let second_target = target.clone();
        let first_memories = [ExtractedMemory {
            content: "first independent session memory".to_string(),
            category: MemoryCategory::ProjectFact,
            confidence: 0.8,
        }];
        let second_memories = [ExtractedMemory {
            content: "second independent session memory".to_string(),
            category: MemoryCategory::Decision,
            confidence: 0.9,
        }];
        let (first, second) = tokio::join!(
            SessionMemoryExtractor::persist(&first_memories, &first_target),
            SessionMemoryExtractor::persist(&second_memories, &second_target),
        );
        first.unwrap();
        second.unwrap();
        let content = fs::read_to_string(&target).await.unwrap();
        assert!(content.contains("first independent session memory"));
        assert!(content.contains("second independent session memory"));
    }

    #[tokio::test]
    async fn test_cancelled_lock_wait_recovers_after_guard_drop() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("AGENTS.md");
        fs::create_dir_all(target.parent().unwrap()).await.unwrap();
        let held = acquire_memory_lock(&target).await.unwrap();
        let waiting_target = target.clone();
        let waiting = tokio::spawn(async move {
            SessionMemoryExtractor::persist(
                &[ExtractedMemory {
                    content: "cancelled contender".to_string(),
                    category: MemoryCategory::ProjectFact,
                    confidence: 0.5,
                }],
                &waiting_target,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        waiting.abort();
        let _ = waiting.await;
        drop(held);
        SessionMemoryExtractor::persist(
            &[ExtractedMemory {
                content: "recovered after cancellation".to_string(),
                category: MemoryCategory::ProjectFact,
                confidence: 1.0,
            }],
            &target,
        )
        .await
        .unwrap();
        assert!(fs::read_to_string(&target)
            .await
            .unwrap()
            .contains("recovered after cancellation"));
    }

    // ---- SessionMemoryState --------------------------------------------

    #[test]
    fn test_state_has_new_messages_no_cursor() {
        let state = SessionMemoryState::new();
        let msgs = vec![make_user("Hello")];
        // No cursor → always has new messages
        assert!(state.has_new_messages_since_last_extraction(&msgs));
    }

    #[test]
    fn test_state_advance_cursor() {
        let mut state = SessionMemoryState::new();
        let mut msg = make_user("hello");
        msg.uuid = Some("uuid-1".to_string());
        let msgs = vec![msg];
        state.advance_cursor(&msgs);
        assert_eq!(state.last_extracted_message_uuid.as_deref(), Some("uuid-1"));
        assert!(!state.has_new_messages_since_last_extraction(&msgs));
    }

    #[test]
    fn test_state_count_cursor_handles_messages_without_uuid() {
        let mut state = SessionMemoryState::new();
        let msgs = make_messages(2);
        state.advance_cursor(&msgs);
        assert!(!state.has_new_messages_since_last_extraction(&msgs));
        let mut more = msgs.clone();
        more.push(make_user("new"));
        assert!(state.has_new_messages_since_last_extraction(&more));
    }

    #[test]
    fn test_state_rebinds_cursor_when_workspace_changes() {
        let mut state = SessionMemoryState::new();
        let msgs = make_messages(2);
        state.advance_cursor(&msgs);
        assert!(state.bind_workspace(Path::new("/tmp/project-a")));
        assert!(!state.bind_workspace(Path::new("/tmp/project-b")));
        assert!(state.last_extracted_message_uuid.is_none());
        assert_eq!(state.last_extracted_message_count, 0);
        assert_eq!(state.tool_calls_at_last_extraction, 0);
    }

    #[test]
    fn test_bounded_transcript_and_parsed_memories() {
        let mut message = make_user(&"x".repeat(MAX_MESSAGE_CHARS * 2));
        message.uuid = Some("large".into());
        let transcript = bounded_transcript(&[&message]);
        assert!(transcript.chars().count() <= MAX_TRANSCRIPT_CHARS);

        let response = (0..(MAX_MEMORIES_PER_EXTRACTION + 5))
            .map(|i| format!("MEMORY: project_fact | 8 | fact-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            parse_extraction_response(&response).len(),
            MAX_MEMORIES_PER_EXTRACTION
        );
    }

    #[test]
    fn test_state_gate_rejects_repeat_without_new_messages() {
        let mut state = SessionMemoryState::new();
        let mut messages = make_messages(MIN_MESSAGES_TO_EXTRACT);
        messages.last_mut().expect("fixture has messages").uuid = Some("last".into());
        state.advance_cursor(&messages);
        assert!(!SessionMemoryExtractor::should_extract_with_state(
            &messages, &state
        ));
    }
}
