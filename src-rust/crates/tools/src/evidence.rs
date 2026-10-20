//! Exact local observations. References are authorized by workspace and session,
//! not by a caller-supplied filesystem path. Paging uses UTF-8 byte offsets.

use crate::{
    EvidenceSelection, EvidenceState, PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult,
    TypedProvenance,
};
use async_trait::async_trait;
use claurst_core::media::{MediaLimits, MediaManifest};
use claurst_core::types::{ContentBlock, ToolResultContent};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::NamedTempFile;
use uuid::Uuid;

static STORE_WRITE: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
const DEFAULT_QUOTA: u64 = 64 * 1024 * 1024;
const DEFAULT_TTL: u64 = 24 * 60 * 60;
const MAX_PAGE: usize = 32 * 1024;
const MAX_OBSERVATIONS: usize = 1024;
// The core policy bounds encoded image data. Typed evidence also preserves
// text/nesting and JSON structure, which are independently capped here before
// any persisted artifact is allocated or parsed.
const MAX_TYPED_TEXT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_TYPED_STRUCTURE_BYTES: u64 = 256 * 1024;
const MAX_EVIDENCE_METADATA_BYTES: u64 = 128 * 1024;
const MAX_EVIDENCE_TEXT_BYTES: u64 = DEFAULT_QUOTA;
const MAX_EVIDENCE_ID_BYTES: usize = 256;
const MAX_EVIDENCE_ID_CHARS: usize = 256;
const MAX_EVIDENCE_QUERY_BYTES: usize = 4096;
const MAX_EVIDENCE_QUERY_CHARS: usize = 4096;
const MAX_MODEL_TEXT_BYTES: usize = 32 * 1024;
const MAX_MODEL_PROVENANCE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub id: String,
    pub bytes: u64,
    pub partial: bool,
    pub created_at: u64,
    pub expires_at: u64,
    content_hash: u64,
    workspace: PathBuf,
    session: String,
    #[serde(default)]
    pub typed_media: Option<MediaManifest>,
    #[serde(default)]
    pub typed_media_limits: Option<MediaLimits>,
    #[serde(default)]
    pub typed_bytes: u64,
    #[serde(default)]
    pub typed_encoded_bytes: u64,
    #[serde(default)]
    typed_content_hash: u64,
    #[serde(default)]
    typed_projection_bytes: u64,
    #[serde(default)]
    typed_projection_hash: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TypedEvidenceFrame<B> {
    blocks: B,
    #[serde(default)]
    provenance: TypedProvenance,
}

#[derive(Debug)]
pub(crate) struct RestoredTypedContent {
    pub(crate) blocks: Vec<ContentBlock>,
    pub(crate) provenance: TypedProvenance,
}

pub struct EvidenceStore {
    directory: PathBuf,
    workspace: PathBuf,
    session: String,
    quota: u64,
    ttl: u64,
}

fn scope_key(value: impl Hash) -> String {
    let mut hash = DefaultHasher::new();
    value.hash(&mut hash);
    format!("{:016x}", hash.finish())
}

fn hash_bytes(value: &[u8]) -> u64 {
    let mut hash = DefaultHasher::new();
    value.hash(&mut hash);
    hash.finish()
}

fn exceeds_text_limit(value: &str, max_bytes: usize, max_chars: usize) -> bool {
    value.len() > max_bytes || value.chars().count() > max_chars
}

fn checked_text_bytes(text: &str) -> io::Result<u64> {
    let bytes = u64::try_from(text.len())
        .map_err(|_| io::Error::other("Evidence text size cannot be represented"))?;
    if bytes > MAX_EVIDENCE_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Evidence text exceeds hard size bound; exact observation was not retained",
        ));
    }
    Ok(bytes)
}

fn max_typed_artifact_bytes() -> u64 {
    let encoded = u64::try_from(MediaLimits::default().max_total_encoded_bytes).unwrap_or(u64::MAX);
    encoded
        .saturating_mul(2)
        // JSON string escaping can expand adversarial control-heavy text by
        // up to six bytes per source byte before it reaches the capped writer.
        .saturating_add(MAX_TYPED_TEXT_BYTES.saturating_mul(6))
        .saturating_add(MAX_TYPED_STRUCTURE_BYTES)
}

fn canonical_text_projection(
    blocks: &[ContentBlock],
    provenance: &TypedProvenance,
) -> io::Result<Option<String>> {
    provenance
        .validate(blocks)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let provenance_indices: std::collections::HashSet<usize> =
        provenance.resource_text_indices.iter().copied().collect();
    let mut payload = Vec::new();
    collect_payload_text(blocks, &provenance_indices, &mut payload);
    if payload.is_empty() {
        return Ok(None);
    }
    let mut total = 0usize;
    for (index, text) in payload.iter().enumerate() {
        total = total
            .checked_add(text.len())
            .and_then(|value| value.checked_add(usize::from(index > 0)))
            .ok_or_else(|| io::Error::other("canonical evidence projection size overflow"))?;
    }
    if total > usize::try_from(MAX_TYPED_TEXT_BYTES).unwrap_or(usize::MAX) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "canonical typed text projection exceeds hard size bound",
        ));
    }
    let mut projection = String::with_capacity(total);
    for (index, text) in payload.into_iter().enumerate() {
        if index > 0 {
            projection.push('\n');
        }
        projection.push_str(text);
    }
    Ok(Some(projection))
}

fn collect_payload_text<'a>(
    blocks: &'a [ContentBlock],
    provenance_indices: &std::collections::HashSet<usize>,
    payload: &mut Vec<&'a str>,
) {
    for (index, block) in blocks.iter().enumerate() {
        match block {
            ContentBlock::Text { text } if !provenance_indices.contains(&index) => {
                payload.push(text)
            }
            ContentBlock::ToolResult {
                content: ToolResultContent::Blocks(inner),
                ..
            } => collect_payload_text(inner, &std::collections::HashSet::new(), payload),
            ContentBlock::ToolResult {
                content: ToolResultContent::Text(text),
                ..
            } => payload.push(text),
            _ => {}
        }
    }
}

/// Consume a tool result into the bounded provider-facing typed content.
/// Complete source blocks remain in EvidenceStore; only the model projection
/// is budgeted here.  Host evidence notices come from private state rather than
/// model-visible metadata.
pub fn take_model_content(result: &mut ToolResult) -> ToolResultContent {
    let Some((blocks, _, _, provenance)) = result.take_validated_typed_content() else {
        let content = std::mem::take(&mut result.content);
        let Some(state) = result.take_evidence_state() else {
            return ToolResultContent::Text(content);
        };
        let page = match (state.next_offset, state.done) {
            (Some(next), Some(done)) => format!("; next_offset={next}; done={done}"),
            _ => String::new(),
        };
        if matches!(
            state.selection,
            EvidenceSelection::Page { .. } | EvidenceSelection::Query
        ) {
            return ToolResultContent::Text(format!(
                "{content}\n[EvidenceRead id={} retrieves {} retained UTF-8 bytes of tool text; partial={}{}]",
                state.id, state.bytes, state.partial, page
            ));
        }
        let notice = format!(
            "[EvidenceRead id={} retrieves {} retained UTF-8 bytes of tool text; partial={}{}]",
            state.id, state.bytes, state.partial, page
        );
        let fallback = state.fallback.unwrap_or(content);
        let budget = MAX_MODEL_TEXT_BYTES.saturating_sub(notice.len() + 2);
        return ToolResultContent::Text(format!(
            "{}\n\n{notice}",
            bounded_for_budget(&fallback, budget)
        ));
    };
    let evidence_state = result.take_evidence_state();
    let selection = evidence_state.as_ref().map(|state| &state.selection);
    let notice = evidence_state.as_ref().map(|state| {
        let page = match (state.next_offset, state.done) {
            (Some(next), Some(done)) => format!("; next_offset={next}; done={done}"),
            _ => String::new(),
        };
        format!(
            "\n[EvidenceRead id={} retrieves {} retained UTF-8 bytes of tool text; partial={}{}]",
            state.id, state.bytes, state.partial, page
        )
    });
    let notice_bytes = notice.as_ref().map_or(0, String::len);
    let payload_budget = MAX_MODEL_TEXT_BYTES
        .saturating_sub(notice_bytes.saturating_add(usize::from(notice.is_some()) * 2));
    let mut model_blocks = Vec::with_capacity(blocks.len() + 2);
    let provenance_indices: std::collections::HashSet<usize> =
        provenance.resource_text_indices.iter().copied().collect();
    let provenance_bytes: usize = provenance_indices
        .iter()
        .filter_map(|index| match blocks.get(*index) {
            Some(ContentBlock::Text { text }) => Some(text.len().saturating_add(2)),
            _ => None,
        })
        .sum();
    if provenance_bytes > MAX_MODEL_PROVENANCE_BYTES {
        result.is_error = true;
        return ToolResultContent::Text(
            "MCP content rejected locally: resource provenance exceeds the model budget"
                .to_string(),
        );
    }
    let page_range = selection.and_then(|selection| match selection {
        EvidenceSelection::Page { offset, end } => Some((*offset, *end)),
        _ => None,
    });
    // Initial previews keep payload plus the private notice within the model
    // budget. Exact pages retain the existing MAX_PAGE payload contract and
    // add the small host notice separately.
    let remaining = if page_range.is_some() {
        MAX_PAGE
    } else {
        payload_budget
    };
    let mut assembler = ModelAssembler {
        selection,
        page_range,
        cursor: 0,
        seen_payload: false,
        remaining,
    };
    assembler.walk(blocks, &provenance_indices, &mut model_blocks);
    let has_payload =
        assembler.seen_payload && !matches!(selection, Some(EvidenceSelection::Query));
    if !has_payload {
        let fallback = match selection {
            Some(EvidenceSelection::InitialPreview) => evidence_state
                .as_ref()
                .and_then(|state| state.fallback.clone()),
            _ => None,
        }
        .unwrap_or_else(|| std::mem::take(&mut result.content));
        if !fallback.is_empty() {
            model_blocks.push(ContentBlock::Text {
                text: bounded_for_budget(&fallback, assembler.remaining),
            });
        }
    }
    if let Some(notice) = notice {
        model_blocks.push(ContentBlock::Text { text: notice });
    }
    if model_blocks.is_empty() {
        ToolResultContent::Text(std::mem::take(&mut result.content))
    } else {
        ToolResultContent::Blocks(model_blocks)
    }
}

struct ModelAssembler<'a> {
    selection: Option<&'a EvidenceSelection>,
    page_range: Option<(usize, usize)>,
    cursor: usize,
    seen_payload: bool,
    remaining: usize,
}

impl<'a> ModelAssembler<'a> {
    fn walk(
        &mut self,
        blocks: Vec<ContentBlock>,
        provenance_indices: &std::collections::HashSet<usize>,
        output: &mut Vec<ContentBlock>,
    ) {
        for (index, block) in blocks.into_iter().enumerate() {
            match block {
                ContentBlock::Text { text } if provenance_indices.contains(&index) => {
                    output.push(ContentBlock::Text {
                        text: format!("\n{text}\n"),
                    });
                }
                ContentBlock::Text { text } => self.payload_slot(text, output),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let content = match content {
                        ToolResultContent::Text(text) => {
                            let mut nested = Vec::new();
                            self.walk(
                                vec![ContentBlock::Text { text }],
                                &std::collections::HashSet::new(),
                                &mut nested,
                            );
                            let text = nested
                                .into_iter()
                                .filter_map(|block| match block {
                                    ContentBlock::Text { text } => Some(text),
                                    _ => None,
                                })
                                .collect();
                            ToolResultContent::Text(text)
                        }
                        ToolResultContent::Blocks(inner) => {
                            let mut nested = Vec::new();
                            self.walk(inner, &std::collections::HashSet::new(), &mut nested);
                            ToolResultContent::Blocks(nested)
                        }
                    };
                    output.push(ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    });
                }
                block => output.push(block),
            }
        }
    }

    fn payload_slot(&mut self, text: String, output: &mut Vec<ContentBlock>) {
        if self.seen_payload {
            self.emit_separator(output);
        }
        let start = self.cursor;
        self.cursor = self.cursor.saturating_add(text.len());
        self.seen_payload = true;
        if let Some(selected) = self.select(&text, start) {
            output.push(ContentBlock::Text { text: selected });
        }
    }

    fn emit_separator(&mut self, output: &mut Vec<ContentBlock>) {
        let separator_start = self.cursor;
        self.cursor = self.cursor.saturating_add(1);
        if let Some(separator) = self.select("\n", separator_start) {
            output.push(ContentBlock::Text { text: separator });
        }
    }

    fn select(&mut self, text: &str, start: usize) -> Option<String> {
        match self.selection {
            Some(EvidenceSelection::Query) => None,
            Some(EvidenceSelection::Page { .. }) => self.page_range.and_then(|(from, to)| {
                let left = from.max(start);
                let right = to.min(start.saturating_add(text.len()));
                (left < right).then(|| text[left - start..right - start].to_string())
            }),
            _ => {
                let selected = bounded_for_budget(text, self.remaining);
                self.remaining = self.remaining.saturating_sub(selected.len());
                Some(selected)
            }
        }
    }
}

fn open_regular_artifact(path: &Path, label: &str) -> io::Result<File> {
    if fs::symlink_metadata(path)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{label} unavailable: artifact is missing"),
            )
        })?
        .file_type()
        .is_symlink()
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{label} unavailable: artifact is a symlink"),
        ));
    }
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("{label} unavailable: artifact is missing"),
                )
            })?
    };
    #[cfg(not(unix))]
    let file = File::open(path).map_err(|_| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("{label} unavailable: artifact is missing"),
        )
    })?;
    if !file.metadata()?.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} rejected: artifact is not a regular file"),
        ));
    }
    Ok(file)
}

fn read_bounded_regular_artifact(path: &Path, max_bytes: u64, label: &str) -> io::Result<Vec<u8>> {
    let file = open_regular_artifact(path, label)?;
    let length = file.metadata()?.len();
    if length > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} rejected: artifact exceeds hard size bound"),
        ));
    }
    let capacity = usize::try_from(length)
        .map_err(|_| io::Error::other(format!("{label} size cannot be represented")))?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut reader = file.take(max_bytes.saturating_add(1));
    reader.read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())
        .ok()
        .is_none_or(|size| size > max_bytes)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} rejected: artifact exceeds hard size bound"),
        ));
    }
    Ok(bytes)
}

fn regular_artifact_len(path: &Path, label: &str) -> io::Result<u64> {
    Ok(open_regular_artifact(path, label)?.metadata()?.len())
}

fn read_typed_artifact(path: &Path, max_bytes: u64) -> io::Result<Vec<u8>> {
    read_bounded_regular_artifact(path, max_bytes, "Typed media evidence")
}

struct CappedVec {
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
}

impl CappedVec {
    fn new(limit: u64) -> io::Result<Self> {
        let limit = usize::try_from(limit)
            .map_err(|_| io::Error::other("typed media evidence size cannot be represented"))?;
        Ok(Self {
            bytes: Vec::new(),
            limit,
            overflowed: false,
        })
    }
}

impl Write for CappedVec {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let remaining = self.limit.saturating_sub(self.bytes.len());
        if bytes.len() > remaining {
            self.overflowed = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "typed media evidence exceeds hard size bound",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn now() -> io::Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(io::Error::other)
}

fn unavailable() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        "Evidence unavailable: missing, expired, or evicted",
    )
}

fn unavailable_if_missing(error: io::Error) -> io::Error {
    if error.kind() == io::ErrorKind::NotFound {
        unavailable()
    } else {
        error
    }
}

impl EvidenceStore {
    pub fn for_context(ctx: &ToolContext) -> io::Result<Self> {
        Self::new(
            claurst_core::config::Settings::config_dir().join("artifacts"),
            &crate::worktree::workspace_root(ctx),
            &ctx.session_id,
            DEFAULT_QUOTA,
            DEFAULT_TTL,
        )
    }

    pub fn new(
        root: PathBuf,
        workspace: &Path,
        session: &str,
        quota: u64,
        ttl: u64,
    ) -> io::Result<Self> {
        let workspace = workspace.canonicalize()?;
        let directory = root.join(scope_key(&workspace)).join(scope_key(session));
        Ok(Self {
            directory,
            workspace,
            session: session.into(),
            quota,
            ttl,
        })
    }

    /// Capture the entire supplied observation before creating any excerpt.
    /// An over-quota observation is rejected explicitly, never silently clipped.
    pub fn capture(&self, text: &str, partial: bool) -> io::Result<EvidenceRef> {
        checked_text_bytes(text)?;
        self.capture_inner(text, partial, None)
    }

    /// Capture text plus validated typed blocks in a separate scoped artifact.
    /// The persisted manifest is metadata only; reads revalidate the blocks.
    pub(crate) fn capture_typed(
        &self,
        display_text: &str,
        partial: bool,
        blocks: &[ContentBlock],
        provenance: &TypedProvenance,
        limits: &MediaLimits,
    ) -> io::Result<EvidenceRef> {
        checked_text_bytes(display_text)?;
        if partial {
            // Partial output cannot certify complete typed content. Retain
            // only the exact text observation and omit the typed artifact.
            return self.capture(display_text, true);
        }
        let projection = canonical_text_projection(blocks, provenance)?;
        let text = projection.as_deref().unwrap_or(display_text);
        let manifest = MediaManifest::from_blocks(blocks, limits).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("typed media rejected: {error}"),
            )
        })?;
        let max_typed_bytes = max_typed_artifact_bytes();
        let mut writer = CappedVec::new(max_typed_bytes)?;
        let frame = TypedEvidenceFrame {
            blocks,
            provenance: provenance.clone(),
        };
        serde_json::to_writer(&mut writer, &frame).map_err(|error| {
            if writer.overflowed {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Typed media evidence rejected: artifact exceeds hard size bound",
                )
            } else {
                io::Error::other(error)
            }
        })?;
        let typed = writer.bytes;
        self.capture_inner(
            text,
            partial,
            Some((&typed, &manifest, limits, projection.is_some())),
        )
    }

    fn capture_inner(
        &self,
        text: &str,
        partial: bool,
        typed: Option<(&[u8], &MediaManifest, &MediaLimits, bool)>,
    ) -> io::Result<EvidenceRef> {
        let bytes = checked_text_bytes(text)?;
        let _guard = STORE_WRITE.lock();
        let (typed_bytes, typed_media, typed_media_limits, typed_encoded_bytes) = match typed {
            Some((payload, manifest, limits, _has_projection)) => (
                u64::try_from(payload.len())
                    .map_err(|_| io::Error::other("typed media evidence size overflow"))?,
                Some(manifest.clone()),
                Some(*limits),
                u64::try_from(manifest.total_encoded_bytes())
                    .map_err(|_| io::Error::other("typed media byte count overflow"))?,
            ),
            None => (0, None, None, 0),
        };
        let timestamp = now()?;
        let reference = EvidenceRef {
            id: Uuid::new_v4().to_string(),
            bytes,
            partial,
            created_at: timestamp,
            expires_at: timestamp.saturating_add(self.ttl),
            content_hash: {
                let mut hash = DefaultHasher::new();
                text.hash(&mut hash);
                hash.finish()
            },
            workspace: self.workspace.clone(),
            session: self.session.clone(),
            typed_media,
            typed_media_limits,
            typed_bytes,
            typed_encoded_bytes,
            typed_content_hash: typed
                .map(|(payload, _, _, _)| hash_bytes(payload))
                .unwrap_or(0),
            typed_projection_bytes: typed
                .filter(|(_, _, _, has_projection)| *has_projection)
                .map(|_| bytes)
                .unwrap_or(0),
            typed_projection_hash: typed
                .filter(|(_, _, _, has_projection)| *has_projection)
                .map(|_| hash_bytes(text.as_bytes()))
                .unwrap_or(0),
        };
        let metadata = serde_json::to_vec(&reference)?;
        // `typed_bytes` already contains encoded image strings inside the
        // serialized block artifact. Charge `typed_encoded_bytes` separately
        // as an explicit conservative media quota component; this deliberate
        // double charge prevents structural overhead from hiding media cost.
        let charge = bytes
            .saturating_add(metadata.len() as u64)
            .saturating_add(typed_bytes)
            .saturating_add(typed_encoded_bytes);
        if charge > self.quota {
            return Err(io::Error::other(
                "Evidence quota exceeded; exact observation was not retained",
            ));
        }
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&self.directory)?;
        let mut observations = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let path = entry?.path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            let metadata_bytes = read_bounded_regular_artifact(
                &path,
                MAX_EVIDENCE_METADATA_BYTES,
                "Evidence metadata",
            )?;
            let reference: EvidenceRef = serde_json::from_slice(&metadata_bytes)?;
            if reference.workspace != self.workspace || reference.session != self.session {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Evidence scope mismatch",
                ));
            }
            if reference.expires_at <= timestamp {
                self.remove(&reference.id)?;
            } else {
                let mut stored_bytes = reference
                    .bytes
                    .saturating_add(u64::try_from(metadata_bytes.len()).unwrap_or(u64::MAX));
                stored_bytes = stored_bytes.saturating_add(reference.typed_encoded_bytes);
                if reference.typed_bytes > 0 {
                    stored_bytes = stored_bytes.saturating_add(regular_artifact_len(
                        &self.path(&reference.id, "typed")?,
                        "Typed media evidence",
                    )?);
                }
                observations.push((reference, stored_bytes));
            }
        }
        observations.sort_by(|(left, _), (right, _)| {
            (left.created_at, &left.id).cmp(&(right.created_at, &right.id))
        });
        let mut used: u64 = observations
            .iter()
            .map(|(_, stored_bytes)| stored_bytes)
            .sum();
        let mut count = observations.len();
        for (reference, stored_bytes) in observations {
            if used.saturating_add(charge) <= self.quota && count < MAX_OBSERVATIONS {
                break;
            }
            self.remove(&reference.id)?;
            used = used.saturating_sub(stored_bytes);
            count -= 1;
        }
        self.atomic_write(&reference.id, "txt", text.as_bytes())?;
        if let Some((typed, _, _, _)) = typed {
            if let Err(error) = self.atomic_write(&reference.id, "typed", typed) {
                let _ = self.remove(&reference.id);
                return Err(error);
            }
        }
        if let Err(error) = self.atomic_write(&reference.id, "json", &metadata) {
            self.remove(&reference.id)?;
            return Err(error);
        }
        Ok(reference)
    }

    fn path(&self, id: &str, extension: &str) -> io::Result<PathBuf> {
        let uuid = Uuid::parse_str(id).map_err(|_| unavailable())?;
        if uuid.to_string() != id {
            return Err(unavailable());
        }
        Ok(self.directory.join(format!("{id}.{extension}")))
    }

    fn atomic_write(&self, id: &str, extension: &str, content: &[u8]) -> io::Result<()> {
        let mut file = NamedTempFile::new_in(&self.directory)?;
        file.write_all(content)?;
        file.as_file().sync_all()?;
        file.persist(self.path(id, extension)?)
            .map_err(|error| error.error)?;
        Ok(())
    }

    fn remove(&self, id: &str) -> io::Result<()> {
        for extension in ["json", "txt", "typed"] {
            match fs::remove_file(self.path(id, extension)?) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    pub fn read(&self, id: &str) -> io::Result<(EvidenceRef, String)> {
        let _guard = STORE_WRITE.lock();
        self.read_locked(id)
            .map(|(reference, text, _)| (reference, text))
    }

    pub(crate) fn read_with_typed(
        &self,
        id: &str,
    ) -> io::Result<(EvidenceRef, String, Option<RestoredTypedContent>)> {
        let _guard = STORE_WRITE.lock();
        self.read_locked(id)
    }

    fn read_locked(
        &self,
        id: &str,
    ) -> io::Result<(EvidenceRef, String, Option<RestoredTypedContent>)> {
        let path = self.path(id, "json")?;
        if fs::symlink_metadata(&path)
            .map_err(|_| unavailable())?
            .file_type()
            .is_symlink()
        {
            return Err(unavailable());
        }
        let metadata_bytes =
            read_bounded_regular_artifact(&path, MAX_EVIDENCE_METADATA_BYTES, "Evidence metadata")
                .map_err(unavailable_if_missing)?;
        let reference: EvidenceRef = serde_json::from_slice(&metadata_bytes)?;
        if reference.id != id
            || reference.workspace != self.workspace
            || reference.session != self.session
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Evidence scope mismatch",
            ));
        }
        if reference.expires_at <= now()? {
            return Err(unavailable());
        }
        let text_path = self.path(id, "txt")?;
        if fs::symlink_metadata(&text_path)
            .map_err(|_| unavailable())?
            .file_type()
            .is_symlink()
        {
            return Err(unavailable());
        }
        let text_bytes = read_bounded_regular_artifact(
            &text_path,
            self.quota.min(MAX_EVIDENCE_TEXT_BYTES),
            "Evidence text",
        )
        .map_err(unavailable_if_missing)?;
        let text = String::from_utf8(text_bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Evidence text is not valid UTF-8",
            )
        })?;
        let mut hash = DefaultHasher::new();
        text.hash(&mut hash);
        if text.len() as u64 != reference.bytes || hash.finish() != reference.content_hash {
            return Err(io::Error::other(
                "Evidence integrity mismatch; stored observation is damaged",
            ));
        }
        let typed_blocks = if reference.typed_bytes == 0 {
            if reference.typed_media.is_some()
                || reference.typed_media_limits.is_some()
                || reference.typed_encoded_bytes != 0
                || reference.typed_content_hash != 0
            {
                return Err(io::Error::other(
                    "Typed media evidence metadata is inconsistent",
                ));
            }
            None
        } else {
            let typed_path = self.path(id, "typed")?;
            let max_typed_bytes = max_typed_artifact_bytes();
            if reference.typed_bytes > max_typed_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Typed media evidence rejected: persisted size exceeds hard bound",
                ));
            }
            let max_encoded_bytes =
                u64::try_from(MediaLimits::default().max_total_encoded_bytes).unwrap_or(u64::MAX);
            if reference.typed_encoded_bytes > max_encoded_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Typed media evidence rejected: persisted encoded size exceeds hard bound",
                ));
            }
            match fs::symlink_metadata(&typed_path) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "Typed media evidence unavailable: artifact is a symlink",
                    ));
                }
                Ok(_) => {}
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        "Typed media evidence unavailable: artifact is missing",
                    ));
                }
            }
            let typed_bytes = read_typed_artifact(&typed_path, max_typed_bytes)?;
            if u64::try_from(typed_bytes.len()).ok() != Some(reference.typed_bytes) {
                return Err(io::Error::other(
                    "Typed media evidence integrity mismatch; artifact size changed",
                ));
            }
            if hash_bytes(&typed_bytes) != reference.typed_content_hash {
                return Err(io::Error::other(
                    "Typed media evidence integrity mismatch; artifact contents changed",
                ));
            }
            let frame: TypedEvidenceFrame<Vec<ContentBlock>> = serde_json::from_slice(&typed_bytes)
                .map_err(|_| {
                    io::Error::other("Typed media evidence integrity mismatch; artifact is invalid")
                })?;
            frame.provenance.validate(&frame.blocks).map_err(|error| {
                io::Error::other(format!("Typed media evidence rejected on read: {error}"))
            })?;
            let limits = reference
                .typed_media_limits
                .as_ref()
                .copied()
                .ok_or_else(|| {
                    io::Error::other("Typed media evidence metadata is missing its policy")
                })?;
            let manifest = MediaManifest::from_blocks(&frame.blocks, &limits).map_err(|error| {
                io::Error::other(format!("Typed media evidence rejected on read: {error}"))
            })?;
            if reference.typed_media.as_ref() != Some(&manifest)
                || u64::try_from(manifest.total_encoded_bytes()).ok()
                    != Some(reference.typed_encoded_bytes)
            {
                return Err(io::Error::other(
                    "Typed media evidence integrity mismatch; manifest changed",
                ));
            }
            let projection = canonical_text_projection(&frame.blocks, &frame.provenance)?;
            match projection {
                Some(projection) => {
                    if reference.typed_projection_bytes
                        != u64::try_from(projection.len()).unwrap_or(u64::MAX)
                        || reference.typed_projection_hash != hash_bytes(projection.as_bytes())
                        || text != projection
                    {
                        return Err(io::Error::other(
                            "Typed media evidence integrity mismatch; canonical projection changed",
                        ));
                    }
                }
                None if reference.typed_projection_bytes != 0
                    || reference.typed_projection_hash != 0 =>
                {
                    return Err(io::Error::other(
                        "Typed media evidence integrity mismatch; canonical projection missing",
                    ));
                }
                None => {}
            }
            Some(RestoredTypedContent {
                blocks: frame.blocks,
                provenance: frame.provenance,
            })
        };
        Ok((reference, text, typed_blocks))
    }
}

/// Keep display shortening deterministic and Unicode-safe; original bytes stay
/// in the artifact. An artifact retention failure is visible in the result.
pub fn retain_output(ctx: &ToolContext, text: String, is_error: bool, partial: bool) -> ToolResult {
    retain_output_inner(ctx, text, is_error, partial, None)
}

pub(crate) fn retain_typed_output(
    ctx: &ToolContext,
    text: String,
    is_error: bool,
    partial: bool,
    blocks: Vec<ContentBlock>,
    provenance: TypedProvenance,
    limits: &MediaLimits,
) -> ToolResult {
    retain_output_inner(
        ctx,
        text,
        is_error,
        partial,
        Some((blocks, provenance, limits)),
    )
}

fn bounded_display(text: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_string();
    }
    let marker = format!("\n\n[{} bytes omitted]\n\n", text.len());
    if budget <= marker.len() {
        return bounded_for_budget(&marker, budget);
    }
    let payload_budget = budget - marker.len();
    let mut head = payload_budget / 2;
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len() - (payload_budget - payload_budget / 2);
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n\n[{} bytes omitted]\n\n{}",
        &text[..head],
        tail - head,
        &text[tail..]
    )
}

fn bounded_for_budget(text: &str, budget: usize) -> String {
    if budget == 0 || budget <= "…".len() && text.len() > budget {
        String::new()
    } else if text.len() <= budget {
        text.to_string()
    } else {
        truncate_utf8(text, budget)
    }
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    if max_bytes <= "…".len() {
        return String::new();
    }
    let mut end = max_bytes - "…".len();
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

fn retain_output_inner(
    ctx: &ToolContext,
    text: String,
    is_error: bool,
    partial: bool,
    typed: Option<(Vec<ContentBlock>, TypedProvenance, &MediaLimits)>,
) -> ToolResult {
    // Partial output cannot certify that its typed blocks are complete. Keep
    // the exact text artifact, but never expose a sendable typed payload.
    let typed = if partial { None } else { typed };
    let reference = EvidenceStore::for_context(ctx).and_then(|store| match typed.as_ref() {
        Some((blocks, provenance, limits)) => {
            store.capture_typed(&text, partial, blocks, provenance, limits)
        }
        None => store.capture(&text, partial),
    });
    let notice = match &reference {
        Ok(reference) => format!(
            "EvidenceRead id={} retrieves {} retained UTF-8 bytes of tool text; partial={}",
            reference.id, reference.bytes, reference.partial
        ),
        Err(error) => format!("Exact evidence was not retained: {error}"),
    };
    let excerpt = if text.len() > MAX_PAGE {
        let mut head = MAX_PAGE / 2;
        while !text.is_char_boundary(head) {
            head -= 1;
        }
        let mut tail = text.len() - MAX_PAGE / 2;
        while !text.is_char_boundary(tail) {
            tail += 1;
        }
        format!(
            "{}\n\n[{} bytes omitted. {}]\n\n{}",
            &text[..head],
            tail - head,
            notice,
            &text[tail..]
        )
    } else {
        format!("{text}\n\n[{notice}]")
    };
    let result = if is_error {
        ToolResult::error(excerpt)
    } else {
        ToolResult::success(excerpt)
    };
    let captured = reference.is_ok();
    let mut retained = match reference {
        Ok(reference) => {
            let state = EvidenceState {
                id: reference.id.clone(),
                bytes: reference.bytes,
                partial: reference.partial,
                workspace: Some(reference.workspace.clone()),
                session: Some(reference.session.clone()),
                selection: EvidenceSelection::InitialPreview,
                fallback: None,
                next_offset: None,
                done: None,
            };
            result
                .with_metadata(json!({"evidence": reference}))
                .with_evidence_state(state)
        }
        Err(error) => {
            let mut metadata = serde_json::Map::new();
            metadata.insert("evidence_unavailable".into(), Value::Bool(true));
            if typed.is_some() {
                metadata.insert("typed_media_unavailable".into(), Value::Bool(true));
                metadata.insert("typed_media_error".into(), Value::String(error.to_string()));
            }
            result.with_metadata(Value::Object(metadata))
        }
    };
    if let Some(state) = retained.take_evidence_state() {
        // Reserve the larger typed notice/layout overhead before choosing the
        // head and tail. Prefix truncation afterward can erase Read continuations.
        let fallback_budget = MAX_MODEL_TEXT_BYTES.saturating_sub(notice.len() + 5);
        retained = retained.with_evidence_state(EvidenceState {
            fallback: Some(bounded_display(&text, fallback_budget)),
            ..state
        });
    }
    match typed {
        Some((blocks, provenance, limits)) if captured => retained
            .with_mcp_typed_content(blocks, provenance, limits)
            .unwrap_or_else(|error| {
                ToolResult::error(format!("Typed media evidence unavailable: {error}"))
            }),
        _ => retained,
    }
}

/// Preserve existing structured tool metadata while adding evidence ownership.
pub fn retain_result(ctx: &ToolContext, mut result: ToolResult) -> ToolResult {
    if result.evidence_state().is_some() && result.evidence_scope_matches(ctx) {
        if let Some(status) = result.execution_status() {
            if matches!(
                status,
                crate::ExecutionStatus::NotRun
                    | crate::ExecutionStatus::Cancelled
                    | crate::ExecutionStatus::Unknown
            ) {
                return result.with_execution_status(status);
            }
        }
        return result;
    }
    let execution_status = result.execution_status();
    let receipts = result.take_application_receipts();
    let discovery_receipt = result.take_discovery_receipt();
    let selected_evidence = result.take_evidence_state().map(|state| state.selection);
    let original_metadata = result.metadata.take();
    let typed = if execution_status.is_some_and(|status| {
        matches!(
            status,
            crate::ExecutionStatus::NotRun
                | crate::ExecutionStatus::Cancelled
                | crate::ExecutionStatus::Unknown
        )
    }) {
        result.take_validated_typed_content();
        None
    } else {
        result.take_validated_typed_content()
    };
    let mut retained = match typed {
        Some((blocks, _, limits, provenance)) => retain_typed_output(
            ctx,
            result.content,
            result.is_error,
            false,
            blocks,
            provenance,
            &limits,
        ),
        None => retain_output(ctx, result.content, result.is_error, false),
    };
    if let Some(selection) = selected_evidence {
        if !matches!(selection, EvidenceSelection::InitialPreview) {
            if let Some(state) = retained.take_evidence_state() {
                let valid = match selection {
                    EvidenceSelection::Page { end, .. } => end <= state.bytes as usize,
                    EvidenceSelection::Query => true,
                    EvidenceSelection::InitialPreview => false,
                };
                if valid {
                    let (next_offset, done) = match selection {
                        EvidenceSelection::Page { end, .. } => {
                            (Some(end), Some(end >= state.bytes as usize))
                        }
                        EvidenceSelection::Query => (None, None),
                        EvidenceSelection::InitialPreview => (None, None),
                    };
                    retained = retained.with_evidence_state(EvidenceState {
                        selection,
                        fallback: None,
                        next_offset,
                        done,
                        ..state
                    });
                } else {
                    retained = retained.with_evidence_state(state);
                }
            }
        }
    }
    if let Some(original) = original_metadata {
        let mut metadata = match original {
            Value::Object(metadata) => metadata,
            other => serde_json::Map::from_iter([("tool_metadata".into(), other)]),
        };
        metadata.remove("evidence");
        metadata.remove("evidence_unavailable");
        if let Some(Value::Object(evidence)) = retained.metadata.take() {
            metadata.extend(evidence);
        }
        retained.metadata = Some(Value::Object(metadata));
    }
    if let Some(receipts) = receipts {
        retained = retained.with_application_receipts(receipts);
    }
    if let Some(discovery_receipt) = discovery_receipt {
        retained = retained.with_discovery_receipt(discovery_receipt);
    }
    if let Some(status) = execution_status {
        retained = retained.with_execution_status(status);
    }
    retained
}

pub struct EvidenceReadTool;

#[derive(Deserialize)]
struct EvidenceInput {
    id: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_page")]
    limit: usize,
    query: Option<String>,
}

fn default_page() -> usize {
    16 * 1024
}

#[async_trait]
impl Tool for EvidenceReadTool {
    fn name(&self) -> &str {
        "EvidenceRead"
    }
    fn description(&self) -> &str {
        "Retrieve exact retained output from this workspace and session. UTF-8 byte offsets; use next_offset for the next page. Optional literal query returns up to 20 matching byte offsets. Missing or expired evidence is reported explicitly."
    }
    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }
    fn effect(&self) -> ToolEffect {
        ToolEffect::Observation
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object", "properties":{
            "id":{"type":"string","maxLength":MAX_EVIDENCE_ID_CHARS}, "offset":{"type":"integer","minimum":0},
            "limit":{"type":"integer","minimum":1,"maximum":MAX_PAGE},
            "query":{"type":"string","minLength":1,"maxLength":MAX_EVIDENCE_QUERY_CHARS}
        }, "required":["id"], "additionalProperties":false})
    }
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        retrieve(EvidenceStore::for_context(ctx), input)
    }
}

fn retrieve(store: io::Result<EvidenceStore>, input: Value) -> ToolResult {
    let input: EvidenceInput = match serde_json::from_value(input) {
        Ok(input) => input,
        Err(error) => return ToolResult::error(format!("Invalid evidence request: {error}")),
    };
    if exceeds_text_limit(&input.id, MAX_EVIDENCE_ID_BYTES, MAX_EVIDENCE_ID_CHARS) {
        return ToolResult::error("Evidence id exceeds the maximum allowed length");
    }
    if let Some(query) = input.query.as_ref() {
        if exceeds_text_limit(query, MAX_EVIDENCE_QUERY_BYTES, MAX_EVIDENCE_QUERY_CHARS) {
            return ToolResult::error("Evidence search query exceeds the maximum allowed length");
        }
        if query.is_empty() {
            return ToolResult::error("Evidence search query must not be empty");
        }
    }
    let (reference, text, typed_content) =
        match store.and_then(|store| store.read_with_typed(&input.id)) {
            Ok(value) => value,
            Err(error) => return ToolResult::error(error.to_string()),
        };
    let typed_limits = reference
        .typed_media_limits
        .as_ref()
        .copied()
        .unwrap_or_default();
    if let Some(query) = input.query {
        let offsets: Vec<usize> = text
            .match_indices(&query)
            .take(21)
            .map(|(offset, _)| offset)
            .collect();
        let result = ToolResult::success(json!({"id":reference.id, "bytes":reference.bytes,
                "partial":reference.partial, "match_offsets":offsets.iter().take(20).collect::<Vec<_>>(),
                "more_matches":offsets.len()>20}).to_string());
        let result = result.with_evidence_state(EvidenceState {
            id: reference.id.clone(),
            bytes: reference.bytes,
            partial: reference.partial,
            workspace: Some(reference.workspace.clone()),
            session: Some(reference.session.clone()),
            selection: EvidenceSelection::Query,
            fallback: None,
            next_offset: None,
            done: None,
        });
        return attach_typed_blocks(result, typed_content, &typed_limits);
    }
    if input.offset > text.len() || !text.is_char_boundary(input.offset) || input.limit == 0 {
        return ToolResult::error(
            "Invalid evidence byte range; offset must be a UTF-8 boundary and limit positive",
        );
    }
    let mut end = input
        .offset
        .saturating_add(input.limit.min(MAX_PAGE))
        .min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    if end == input.offset && end < text.len() {
        return ToolResult::error("Evidence limit too small for the next UTF-8 character");
    }
    let result = ToolResult::success(text[input.offset..end].to_string())
        .with_metadata(json!({
            "evidence":reference, "offset":input.offset, "next_offset":end, "done":end==text.len()
        }))
        .with_evidence_state(EvidenceState {
            id: reference.id.clone(),
            bytes: reference.bytes,
            partial: reference.partial,
            workspace: Some(reference.workspace.clone()),
            session: Some(reference.session.clone()),
            selection: EvidenceSelection::Page {
                offset: input.offset,
                end,
            },
            fallback: None,
            next_offset: Some(end),
            done: Some(end == text.len()),
        });
    attach_typed_blocks(result, typed_content, &typed_limits)
}

fn attach_typed_blocks(
    result: ToolResult,
    typed: Option<RestoredTypedContent>,
    limits: &MediaLimits,
) -> ToolResult {
    match typed {
        Some(typed) => {
            match result.with_mcp_typed_content(typed.blocks, typed.provenance, limits) {
                Ok(result) => result,
                Err(error) => {
                    ToolResult::error(format!("Typed media evidence unavailable: {error}"))
                }
            }
        }
        None => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_preview_preserves_read_continuation_within_model_budget() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = crate::test_support::allow_all_context(workspace.path().to_path_buf());
        let continuation = "\n... (has_more=true next_offset=1 next_byte_offset=65535)\n";
        let original = format!("1\t{}{continuation}", "雪".repeat(21_845));
        let mut result = retain_result(&ctx, ToolResult::success(original.clone()));
        let id = result.evidence_state().unwrap().id.clone();
        assert_eq!(
            EvidenceStore::for_context(&ctx)
                .unwrap()
                .read(&id)
                .unwrap()
                .1,
            original
        );
        let ToolResultContent::Text(model) = take_model_content(&mut result) else {
            panic!("plain file observation must remain text")
        };
        assert!(model.len() <= MAX_MODEL_TEXT_BYTES);
        assert!(model.starts_with("1\t雪"));
        assert!(
            model.contains(continuation),
            "Read continuation footer was lost"
        );
        assert_eq!(model.matches("EvidenceRead id=").count(), 1);
    }

    #[test]
    fn exact_unicode_and_scoped_references() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let other_workspace = tempfile::tempdir().unwrap();
        let make = |path: &Path, session: &str| {
            EvidenceStore::new(root.path().into(), path, session, 1024, 60).unwrap()
        };
        let store = make(workspace.path(), "one");
        let text = "head\n🦀 middle é\nerror at end\n";
        let reference = store.capture(text, false).unwrap();
        assert_eq!(store.read(&reference.id).unwrap().1, text);
        assert!(make(workspace.path(), "two").read(&reference.id).is_err());
        assert!(make(other_workspace.path(), "one")
            .read(&reference.id)
            .is_err());
        assert!(store.read("../../secret").is_err());
    }

    #[test]
    fn eviction_expiry_partial_and_over_quota_are_explicit() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store =
            EvidenceStore::new(root.path().into(), workspace.path(), "quota", 1400, 60).unwrap();
        let first = store.capture(&"x".repeat(800), false).unwrap();
        let second = store.capture(&"y".repeat(800), true).unwrap();
        assert!(store
            .read(&first.id)
            .unwrap_err()
            .to_string()
            .contains("unavailable"));
        assert!(store.read(&second.id).unwrap().0.partial);
        assert!(store.capture(&"z".repeat(1401), false).is_err());
        let expired =
            EvidenceStore::new(root.path().into(), workspace.path(), "expired", 1024, 0).unwrap();
        let reference = expired.capture("expires immediately", false).unwrap();
        assert!(expired.read(&reference.id).is_err());
    }

    #[test]
    fn hard_text_bound_precedes_custom_quota_without_artifact_side_effects() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let quota = MAX_EVIDENCE_TEXT_BYTES.saturating_add(1024 * 1024);
        let exact_store = EvidenceStore::new(
            root.path().join("exact"),
            workspace.path(),
            "exact",
            quota,
            60,
        )
        .unwrap();
        let exact = "x".repeat(usize::try_from(MAX_EVIDENCE_TEXT_BYTES).unwrap());
        let reference = exact_store.capture(&exact, false).unwrap();
        assert_eq!(reference.bytes, MAX_EVIDENCE_TEXT_BYTES);

        let oversized_store = EvidenceStore::new(
            root.path().join("oversized"),
            workspace.path(),
            "oversized",
            quota,
            60,
        )
        .unwrap();
        let oversized = "x".repeat(usize::try_from(MAX_EVIDENCE_TEXT_BYTES).unwrap() + 1);
        let error = oversized_store.capture(&oversized, true).unwrap_err();
        assert!(error.to_string().contains("hard size bound"));
        assert!(!oversized_store.directory.exists());
    }

    #[test]
    fn oversized_evidence_inputs_are_rejected_before_store_access() {
        let unavailable_store = Err(io::Error::new(
            io::ErrorKind::NotFound,
            "store resolution must not run",
        ));
        let oversized_id = retrieve(
            unavailable_store,
            json!({"id": "i".repeat(MAX_EVIDENCE_ID_BYTES + 1)}),
        );
        assert_eq!(
            oversized_id.content,
            "Evidence id exceeds the maximum allowed length"
        );

        let unavailable_store = Err(io::Error::new(
            io::ErrorKind::NotFound,
            "store resolution must not run",
        ));
        let oversized_query = retrieve(
            unavailable_store,
            json!({
                "id": "short",
                "query": "q".repeat(MAX_EVIDENCE_QUERY_BYTES + 1)
            }),
        );
        assert_eq!(
            oversized_query.content,
            "Evidence search query exceeds the maximum allowed length"
        );

        let schema = EvidenceReadTool.input_schema();
        assert_eq!(
            schema["properties"]["id"]["maxLength"],
            MAX_EVIDENCE_ID_CHARS
        );
        assert_eq!(
            schema["properties"]["query"]["maxLength"],
            MAX_EVIDENCE_QUERY_CHARS
        );
    }

    #[test]
    fn same_size_damage_is_not_returned_as_exact() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = EvidenceStore::new(root.path().into(), workspace.path(), "integrity", 1024, 60)
            .unwrap();
        let reference = store.capture("original", false).unwrap();
        fs::write(store.path(&reference.id, "txt").unwrap(), "tampered").unwrap();
        assert!(store
            .read(&reference.id)
            .unwrap_err()
            .to_string()
            .contains("integrity"));
    }

    #[test]
    fn paging_and_search_recover_omitted_middle_without_splitting_unicode() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let make_store =
            || EvidenceStore::new(root.path().into(), workspace.path(), "pages", 200_000, 60);
        let text = format!(
            "{}🦀 distinctive-middle é{}tail-failure",
            "h".repeat(40_000),
            "t".repeat(40_000)
        );
        let reference = make_store().unwrap().capture(&text, false).unwrap();
        let matches = retrieve(
            make_store(),
            json!({"id":reference.id,"query":"distinctive-middle"}),
        );
        assert!(!matches.is_error);
        let matches: Value = serde_json::from_str(&matches.content).unwrap();
        assert_eq!(matches["match_offsets"][0], 40_005);
        let crab = retrieve(
            make_store(),
            json!({"id":reference.id,"offset":40_000,"limit":5}),
        );
        assert!(!crab.is_error);
        assert_eq!(crab.content, "🦀 ");
        assert_eq!(crab.metadata.unwrap()["next_offset"], 40_005);
        let invalid = retrieve(make_store(), json!({"id":reference.id,"offset":40_001}));
        assert!(invalid.is_error);
        let tiny = retrieve(
            make_store(),
            json!({"id":reference.id,"offset":40_000,"limit":1}),
        );
        assert!(tiny.is_error);
        let middle = retrieve(
            make_store(),
            json!({"id":reference.id,"offset":40_005,"limit":24}),
        );
        assert!(middle.content.starts_with("distinctive-middle é"));
        let tail = retrieve(
            make_store(),
            json!({"id":reference.id,"offset":text.len()-12}),
        );
        assert_eq!(tail.content, "tail-failure");
    }

    #[test]
    fn typed_search_returns_offsets_without_source_payload() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let make_store = || {
            EvidenceStore::new(
                root.path().into(),
                workspace.path(),
                "typed-search",
                1024 * 1024,
                60,
            )
        };
        let text = format!(
            "SECRET_SOURCE{}needle{}",
            "x".repeat(40_000),
            "z".repeat(40_000)
        );
        let reference = make_store()
            .unwrap()
            .capture_typed(
                "display",
                false,
                &[
                    ContentBlock::Text { text: text.clone() },
                    typed_image_block(),
                ],
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        let mut result = retrieve(make_store(), json!({"id": reference.id, "query": "needle"}));
        let ToolResultContent::Blocks(blocks) = take_model_content(&mut result) else {
            panic!("typed search")
        };
        let texts = blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(texts.contains("match_offsets"));
        assert!(texts.contains("40013"));
        assert!(!texts.contains("SECRET_SOURCE"));
        assert_eq!(texts.matches("EvidenceRead id=").count(), 1);
        assert!(blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Image { .. })));
    }

    #[test]
    fn model_projection_keeps_provenance_media_and_payload_slots_in_order() {
        let blocks = vec![
            ContentBlock::Text {
                text: "[Resource: urn:one]".to_string(),
            },
            ContentBlock::Text {
                text: "before".to_string(),
            },
            typed_image_block(),
            ContentBlock::Text {
                text: "[Resource: urn:two]".to_string(),
            },
            ContentBlock::Text {
                text: "after".to_string(),
            },
        ];
        let result = ToolResult::success("bounded")
            .with_mcp_typed_content(
                blocks,
                TypedProvenance {
                    resource_text_indices: vec![0, 3],
                },
                &MediaLimits::default(),
            )
            .unwrap()
            .with_evidence_state(EvidenceState {
                id: "evidence-order".to_string(),
                bytes: 11,
                partial: false,
                workspace: None,
                session: None,
                selection: EvidenceSelection::InitialPreview,
                fallback: Some("bounded".to_string()),
                next_offset: None,
                done: None,
            });
        let mut result = result;
        let content = take_model_content(&mut result);
        let ToolResultContent::Blocks(content) = content else {
            panic!("typed content should preserve block order")
        };
        assert!(matches!(
            &content[0],
            ContentBlock::Text { text } if text.trim() == "[Resource: urn:one]"
        ));
        assert!(matches!(
            &content[1],
            ContentBlock::Text { text } if text == "before"
        ));
        assert!(matches!(&content[2], ContentBlock::Image { .. }));
        assert!(matches!(
            &content[3],
            ContentBlock::Text { text } if text.trim() == "[Resource: urn:two]"
        ));
        assert!(matches!(
            &content[4],
            ContentBlock::Text { text } if text == "\n"
        ));
        assert!(matches!(
            &content[5],
            ContentBlock::Text { text } if text == "after"
        ));
        assert!(matches!(
            content.last(),
            Some(ContentBlock::Text { text }) if text.contains("EvidenceRead id=evidence-order")
        ));
        assert_eq!(
            content
                .iter()
                .filter(|block| matches!(block, ContentBlock::Text { text } if text.contains("EvidenceRead id=")))
                .count(),
            1
        );
    }

    #[test]
    fn model_projection_accounts_for_empty_slots_at_utf8_page_boundaries() {
        let blocks = vec![
            ContentBlock::Text {
                text: "[Resource: urn:empty]".to_string(),
            },
            ContentBlock::Text {
                text: String::new(),
            },
            ContentBlock::Text {
                text: "a🦀b".to_string(),
            },
        ];
        let mut result = ToolResult::success("display")
            .with_mcp_typed_content(
                blocks,
                TypedProvenance {
                    resource_text_indices: vec![0],
                },
                &MediaLimits::default(),
            )
            .unwrap()
            .with_evidence_state(EvidenceState {
                id: "evidence-page".to_string(),
                bytes: 7,
                partial: false,
                workspace: None,
                session: None,
                selection: EvidenceSelection::Page { offset: 1, end: 6 },
                fallback: None,
                next_offset: Some(6),
                done: Some(false),
            });
        let content = take_model_content(&mut result);
        let ToolResultContent::Blocks(content) = content else {
            panic!("paged typed content should preserve blocks")
        };
        assert!(matches!(
            &content[0],
            ContentBlock::Text { text } if text.trim() == "[Resource: urn:empty]"
        ));
        assert!(matches!(
            &content[1],
            ContentBlock::Text { text } if text == "a🦀"
        ));
        assert!(matches!(
            content.last(),
            Some(ContentBlock::Text { text }) if text.contains("next_offset=6; done=false")
        ));
    }

    #[test]
    fn exact_max_page_is_kept_before_private_notice() {
        let payload = "x".repeat(MAX_PAGE);
        let mut result = ToolResult::success("display")
            .with_mcp_typed_content(
                vec![ContentBlock::Text {
                    text: payload.clone(),
                }],
                TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap()
            .with_evidence_state(EvidenceState {
                id: "evidence-max-page".to_string(),
                bytes: payload.len() as u64,
                partial: false,
                workspace: None,
                session: None,
                selection: EvidenceSelection::Page {
                    offset: 0,
                    end: payload.len(),
                },
                fallback: None,
                next_offset: Some(payload.len()),
                done: Some(true),
            });
        let content = take_model_content(&mut result);
        let ToolResultContent::Blocks(content) = content else {
            panic!("paged typed content should preserve blocks")
        };
        assert!(matches!(
            &content[0],
            ContentBlock::Text { text } if text == &payload
        ));
        assert!(matches!(
            content.last(),
            Some(ContentBlock::Text { text }) if text.contains("next_offset=32768; done=true")
        ));
    }

    #[test]
    fn page_can_return_canonical_separator_from_empty_text_slots() {
        let blocks = vec![
            ContentBlock::Text {
                text: String::new(),
            },
            ContentBlock::Text {
                text: "é".to_string(),
            },
            ContentBlock::Text {
                text: String::new(),
            },
        ];
        let mut result = ToolResult::success("display")
            .with_mcp_typed_content(blocks, TypedProvenance::default(), &MediaLimits::default())
            .unwrap()
            .with_evidence_state(EvidenceState {
                id: "separator-page".to_string(),
                bytes: 4,
                partial: false,
                workspace: None,
                session: None,
                selection: EvidenceSelection::Page { offset: 0, end: 1 },
                fallback: None,
                next_offset: Some(1),
                done: Some(false),
            });
        let content = take_model_content(&mut result);
        let ToolResultContent::Blocks(content) = content else {
            panic!("separator page should remain structured")
        };
        assert!(matches!(
            &content[0],
            ContentBlock::Text { text } if text == "\n"
        ));
    }

    #[test]
    fn nested_tool_result_text_and_blocks_follow_budget_and_page_selection() {
        let blocks = vec![
            ContentBlock::ToolResult {
                tool_use_id: "text-child".to_string(),
                content: ToolResultContent::Text("a".to_string()),
                is_error: None,
            },
            ContentBlock::ToolResult {
                tool_use_id: "blocks-child".to_string(),
                content: ToolResultContent::Blocks(vec![ContentBlock::Text {
                    text: "b".to_string(),
                }]),
                is_error: None,
            },
        ];
        let mut initial = ToolResult::success("display")
            .with_mcp_typed_content(
                blocks.clone(),
                TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap()
            .with_evidence_state(EvidenceState {
                id: "nested-initial".to_string(),
                bytes: 3,
                partial: false,
                workspace: None,
                session: None,
                selection: EvidenceSelection::InitialPreview,
                fallback: None,
                next_offset: None,
                done: None,
            });
        let initial = take_model_content(&mut initial);
        let ToolResultContent::Blocks(initial) = initial else {
            panic!("nested initial content should remain structured")
        };
        assert!(initial.iter().all(|block| match block {
            ContentBlock::ToolResult {
                content: ToolResultContent::Text(text),
                ..
            } => text.len() <= MAX_MODEL_TEXT_BYTES,
            ContentBlock::ToolResult {
                content: ToolResultContent::Blocks(inner),
                ..
            } => inner.iter().all(|block| {
                matches!(block, ContentBlock::Text { text } if text.len() <= MAX_MODEL_TEXT_BYTES)
            }),
            ContentBlock::Text { text } => text.len() <= MAX_MODEL_TEXT_BYTES,
            _ => true,
        }));
        let mut result = ToolResult::success("display")
            .with_mcp_typed_content(blocks, TypedProvenance::default(), &MediaLimits::default())
            .unwrap()
            .with_evidence_state(EvidenceState {
                id: "nested-page".to_string(),
                bytes: 3,
                partial: false,
                workspace: None,
                session: None,
                selection: EvidenceSelection::Page { offset: 2, end: 3 },
                fallback: None,
                next_offset: Some(3),
                done: Some(true),
            });
        let content = take_model_content(&mut result);
        let ToolResultContent::Blocks(content) = content else {
            panic!("nested page should remain structured")
        };
        assert!(matches!(
            &content[0],
            ContentBlock::ToolResult {
                content: ToolResultContent::Text(text),
                ..
            } if text.is_empty()
        ));
        assert!(matches!(
            &content[1],
            ContentBlock::ToolResult {
                content: ToolResultContent::Blocks(inner),
                ..
            } if matches!(inner.as_slice(), [ContentBlock::Text { text }] if text == "b")
        ));
    }

    #[test]
    fn retain_result_preserves_private_discovery_receipt_on_each_evidence_path() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools = crate::all_tools();
        let view = crate::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let receipt = crate::DiscoveryReceipt::new(&view, vec!["Read".to_string()]);
        let first = ToolResult::success("discovered").with_discovery_receipt(receipt.clone());
        let mut retained = retain_result(&ctx, first);
        assert_eq!(
            retained.take_discovery_receipt().unwrap().selected_names(),
            &[String::from("Read")]
        );

        let second = ToolResult::success("already retained")
            .with_metadata(json!({"evidence": {"id": "existing"}}))
            .with_discovery_receipt(receipt);
        let mut fast_path = retain_result(&ctx, second);
        assert!(fast_path.take_discovery_receipt().is_some());
    }

    #[test]
    fn forged_evidence_metadata_cannot_skip_parent_retention() {
        let workspace = tempfile::tempdir().unwrap();
        let ctx = crate::test_support::allow_all_context(workspace.path().to_path_buf());
        let retained = retain_result(
            &ctx,
            ToolResult::success("authoritative parent text")
                .with_metadata(json!({"evidence": {"id": "forged"}})),
        );
        let evidence_id = retained.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap();
        assert_ne!(evidence_id, "forged");
        let (_, text) = EvidenceStore::for_context(&ctx)
            .unwrap()
            .read(evidence_id)
            .unwrap();
        assert_eq!(text, "authoritative parent text");
    }

    #[test]
    fn foreign_private_evidence_state_is_recaptured_in_current_scope() {
        let first_workspace = tempfile::tempdir().unwrap();
        let second_workspace = tempfile::tempdir().unwrap();
        let first = crate::test_support::allow_all_context(first_workspace.path().to_path_buf());
        let second = crate::test_support::allow_all_context(second_workspace.path().to_path_buf());
        let foreign = retain_output(&first, "foreign text".to_string(), false, false);
        let foreign_id = foreign.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap()
            .to_string();
        let retained = retain_result(&second, foreign);
        let current_id = retained.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap();
        assert_ne!(current_id, foreign_id);
        let (_, text) = EvidenceStore::for_context(&second)
            .unwrap()
            .read(current_id)
            .unwrap();
        assert!(text.contains("foreign text"));
        assert!(EvidenceStore::for_context(&first)
            .unwrap()
            .read(current_id)
            .is_err());
    }

    #[test]
    fn retain_result_preserves_typed_exceptional_status_and_revokes_receipts() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools = crate::all_tools();
        let view = crate::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        for status in [
            crate::ExecutionStatus::NotRun,
            crate::ExecutionStatus::Cancelled,
            crate::ExecutionStatus::Unknown,
        ] {
            let receipt = crate::DiscoveryReceipt::new(&view, vec!["Read".to_string()]);
            let mut retained = retain_result(
                &ctx,
                ToolResult::error("uncertain")
                    .with_execution_status(status)
                    .with_discovery_receipt(receipt),
            );
            assert_eq!(retained.execution_status(), Some(status));
            assert!(retained.take_discovery_receipt().is_none());
        }
    }

    #[test]
    fn retain_result_fast_path_preserves_typed_exceptional_status() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools = crate::all_tools();
        let view = crate::ToolCatalogView::from_tools(&ctx, &tools).unwrap();
        let receipt = crate::DiscoveryReceipt::new(&view, vec!["Read".to_string()]);
        let mut retained = retain_result(
            &ctx,
            ToolResult::error("cancelled")
                .with_metadata(json!({"evidence": {"id": "existing"}}))
                .with_execution_status(crate::ExecutionStatus::Cancelled)
                .with_discovery_receipt(receipt),
        );
        assert_eq!(
            retained.execution_status(),
            Some(crate::ExecutionStatus::Cancelled)
        );
        assert!(retained.take_application_receipts().is_none());
    }

    fn typed_image_block() -> ContentBlock {
        ContentBlock::Image {
            source: claurst_core::types::ImageSource {
                source_type: "base64".to_string(),
                media_type: Some("image/png".to_string()),
                data: Some(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
                        .to_string(),
                ),
                url: None,
            },
        }
    }

    #[test]
    fn typed_evidence_round_trip_revalidates_and_restores_blocks() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store =
            EvidenceStore::new(root.path().into(), workspace.path(), "typed", 16 * 1024, 60)
                .unwrap();
        let blocks = vec![
            ContentBlock::Text {
                text: "original text".to_string(),
            },
            typed_image_block(),
        ];
        let reference = store
            .capture_typed(
                "display summary",
                false,
                &blocks,
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        assert!(reference.typed_bytes > 0);
        assert!(reference.typed_encoded_bytes > 0);
        let (_, text, restored) = store.read_with_typed(&reference.id).unwrap();
        assert_eq!(text, "original text");
        assert_eq!(
            serde_json::to_vec(&restored.unwrap().blocks).unwrap(),
            serde_json::to_vec(&blocks).unwrap()
        );
    }

    #[test]
    fn typed_evidence_quota_and_missing_artifact_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let blocks = vec![typed_image_block()];
        let small =
            EvidenceStore::new(root.path().into(), workspace.path(), "small", 128, 60).unwrap();
        let error = small
            .capture_typed(
                "summary",
                false,
                &blocks,
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("quota"));

        let store = EvidenceStore::new(
            root.path().into(),
            workspace.path(),
            "missing",
            16 * 1024,
            60,
        )
        .unwrap();
        let reference = store
            .capture_typed(
                "summary",
                false,
                &blocks,
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        fs::remove_file(store.path(&reference.id, "typed").unwrap()).unwrap();
        let error = store.read_with_typed(&reference.id).unwrap_err();
        assert!(error
            .to_string()
            .contains("Typed media evidence unavailable"));
    }

    #[test]
    fn typed_evidence_same_size_damage_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = EvidenceStore::new(
            root.path().into(),
            workspace.path(),
            "typed-integrity",
            16 * 1024,
            60,
        )
        .unwrap();
        let blocks = vec![
            ContentBlock::Text {
                text: "original".to_string(),
            },
            typed_image_block(),
        ];
        let reference = store
            .capture_typed(
                "summary",
                false,
                &blocks,
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        let tampered = vec![
            ContentBlock::Text {
                text: "tampered".to_string(),
            },
            typed_image_block(),
        ];
        let bytes = serde_json::to_vec(&TypedEvidenceFrame {
            blocks: tampered,
            provenance: TypedProvenance::default(),
        })
        .unwrap();
        assert_eq!(bytes.len(), reference.typed_bytes as usize);
        fs::write(store.path(&reference.id, "typed").unwrap(), bytes).unwrap();
        let error = store.read_with_typed(&reference.id).unwrap_err();
        assert!(error
            .to_string()
            .contains("Typed media evidence integrity mismatch"));
    }

    #[test]
    fn typed_evidence_bound_rejects_forged_size_before_read() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = EvidenceStore::new(
            root.path().into(),
            workspace.path(),
            "typed-bound",
            16 * 1024,
            60,
        )
        .unwrap();
        let reference = store
            .capture_typed(
                "summary",
                false,
                &[typed_image_block()],
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        let metadata_path = store.path(&reference.id, "json").unwrap();
        let mut metadata: Value =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        metadata["typed_bytes"] = Value::from(max_typed_artifact_bytes().saturating_add(1));
        fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        let error = store.read_with_typed(&reference.id).unwrap_err();
        assert!(error
            .to_string()
            .contains("persisted size exceeds hard bound"));
    }

    #[test]
    fn partial_typed_retention_keeps_text_only_evidence() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let result = retain_typed_output(
            &ctx,
            "partial display".to_string(),
            false,
            true,
            vec![typed_image_block()],
            TypedProvenance::default(),
            &MediaLimits::default(),
        );
        assert!(result.typed_blocks().is_none());
        let evidence_id = result.metadata.as_ref().unwrap()["evidence"]["id"]
            .as_str()
            .unwrap();
        let (_, _, restored) = EvidenceStore::for_context(&ctx)
            .unwrap()
            .read_with_typed(evidence_id)
            .unwrap();
        assert!(restored.is_none());

        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = EvidenceStore::new(
            root.path().into(),
            workspace.path(),
            "partial",
            16 * 1024,
            60,
        )
        .unwrap();
        let reference = store
            .capture_typed(
                "partial direct capture",
                true,
                &[typed_image_block()],
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        assert_eq!(reference.typed_bytes, 0);
        assert!(store.read_with_typed(&reference.id).unwrap().2.is_none());
    }

    #[test]
    fn capped_typed_serializer_writer_rejects_overflow() {
        let mut writer = CappedVec::new(4).unwrap();
        let error = writer.write_all(b"12345").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(writer.overflowed);
        assert!(writer.bytes.is_empty());
    }

    #[test]
    fn oversized_metadata_is_rejected_before_json_parse() {
        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = EvidenceStore::new(
            root.path().into(),
            workspace.path(),
            "metadata-bound",
            16 * 1024,
            60,
        )
        .unwrap();
        let reference = store.capture("summary", false).unwrap();
        let metadata_path = store.path(&reference.id, "json").unwrap();
        fs::write(
            &metadata_path,
            vec![b' '; usize::try_from(MAX_EVIDENCE_METADATA_BYTES).unwrap() + 1],
        )
        .unwrap();
        let error = store.read(&reference.id).unwrap_err();
        assert!(error
            .to_string()
            .contains("Evidence metadata rejected: artifact exceeds hard size bound"));
    }

    #[cfg(unix)]
    #[test]
    fn fifo_typed_artifact_is_rejected_without_blocking() {
        use std::ffi::CString;

        let root = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let store = EvidenceStore::new(
            root.path().into(),
            workspace.path(),
            "typed-fifo",
            16 * 1024,
            60,
        )
        .unwrap();
        let reference = store
            .capture_typed(
                "summary",
                false,
                &[typed_image_block()],
                &TypedProvenance::default(),
                &MediaLimits::default(),
            )
            .unwrap();
        let typed_path = store.path(&reference.id, "typed").unwrap();
        fs::remove_file(&typed_path).unwrap();
        let path = CString::new(typed_path.to_str().unwrap()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        let error = store.read_with_typed(&reference.id).unwrap_err();
        assert!(error
            .to_string()
            .contains("Typed media evidence rejected: artifact is not a regular file"));
    }
}
