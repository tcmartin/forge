//! Durable, deterministic history and requirement projection.
//!
//! This module deliberately keeps the transcript separate from the bounded
//! active requirement view.  A provider request may omit old material, while
//! this journal keeps the exact source and its provenance available for a
//! later, explicit retrieval.

use claurst_core::crypto_utils::{generate_uuid, sha256_hex};
use claurst_core::types::Message;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const JOURNAL_FILE: &str = "journal.ndjson";
const GROUP_DIR: &str = "groups";
const CHECKPOINT_FILE: &str = "projection.snapshot";
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Return the stable scope identifier used by the CLI history directory.
///
/// The filesystem path is canonicalized before hashing, and only the first
/// eight digest bytes are exposed in the journal scope.  Readers must retain
/// the canonical path separately: this identifier is intentionally not
/// reversible.
pub fn workspace_scope_id(path: &Path) -> String {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let digest = sha256_hex(canonical.to_string_lossy().as_bytes());
    format!("workspace-{}", &digest[..16])
}

/// Limits for the active requirement projection.  The full journal is not
/// subject to these limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveLedgerLimits {
    pub max_full_records: usize,
    pub max_full_bytes: usize,
    pub max_full_tokens: u64,
    pub max_references: usize,
    pub max_reference_bytes: usize,
}

impl Default for ActiveLedgerLimits {
    fn default() -> Self {
        Self {
            max_full_records: 16,
            max_full_bytes: 8 * 1024,
            max_full_tokens: 2_048,
            max_references: 128,
            max_reference_bytes: 8 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalScope {
    pub workspace: String,
    pub session: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRange {
    pub message_id: String,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryAuthority {
    Active,
    Historical,
    Superseded,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OmitReason {
    Budget,
    Relevance,
    ToolResultBudget,
    Summary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryRef {
    pub id: String,
    pub group_id: String,
    pub sequence: u64,
    pub source_range: SourceRange,
    /// Every canonical source message in the omitted component, in order.
    #[serde(default)]
    pub source_message_ids: Vec<String>,
    pub reason: OmitReason,
    pub authority: HistoryAuthority,
}

/// The exact canonical message component retained for later retrieval.
/// Retrieval returns these messages as stored and never re-runs a tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoricalGroup {
    pub history_ref: HistoryRef,
    pub messages: Vec<Message>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementStatus {
    Open,
    Satisfied,
    Blocked,
    Superseded,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementSubject {
    ExplicitAnchor(String),
    Turn(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionScope {
    pub anchor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementRecord {
    pub id: String,
    pub subject: RequirementSubject,
    pub source_turn: String,
    pub text: String,
    pub status: RequirementStatus,
    /// Message that authoritatively produced the current status.
    pub status_source_message_id: Option<String>,
    pub supersedes: Option<String>,
    pub superseded_by: Option<String>,
    pub correction_scope: Option<CorrectionScope>,
    pub aliases: Vec<String>,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementRef {
    pub id: String,
    pub status: RequirementStatus,
    pub source_message_id: String,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveLedgerView {
    pub full: Vec<RequirementRecord>,
    pub references: Vec<RequirementRef>,
    pub full_bytes: usize,
    pub reference_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalRecord {
    pub record: RequirementRecord,
    pub authority: HistoryAuthority,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoOpDiagnostic {
    pub raw: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReport {
    pub message_id: String,
    pub revision: u64,
    pub created: Vec<String>,
    pub diagnostics: Vec<NoOpDiagnostic>,
    pub duplicate: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryStatus {
    pub torn_tail: bool,
    pub ignored_bytes: usize,
    pub recovered_revision: u64,
}

#[derive(Debug, Error)]
pub enum HistoryError {
    #[error("history I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("history serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("history journal is corrupt: {0}")]
    Corrupt(String),
    #[error("history quota exceeded: {0}")]
    Quota(String),
    #[error("history scope is already open for writing")]
    Locked,
    #[error("active requirement overflow: {0}")]
    ActiveOverflow(String),
    #[error("unknown requirement ID: {0}")]
    UnknownRequirement(String),
    #[error("requirement anchor does not match: {0}")]
    AnchorMismatch(String),
    #[error("invalid directive: {0}")]
    InvalidDirective(String),
    #[error("status event requires operation and evidence IDs")]
    MissingVerificationProvenance,
    #[error("history group contains a message without a stable ID")]
    MissingMessageId,
    #[error("history group is invalid: {0}")]
    InvalidHistoryGroup(String),
    #[error("history reference is unavailable: {0}")]
    HistoryUnavailable(String),
    #[error("history reference committed but checkpoint persistence failed: {0}")]
    Committed(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventActor {
    UserDirective,
    ToolVerifier,
    SystemRecovery,
}

/// Origin for durable runtime turns. These records remain in the canonical
/// journal for provenance but never create user-authoritative requirements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMessageOrigin {
    MaxTokensRecovery,
    MaxStepsDegradation,
    Continuation,
    ToolCancellation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum EventPayload {
    Message {
        message_id: String,
        text: String,
    },
    RuntimeMessage {
        message_id: String,
        text: String,
        origin: RuntimeMessageOrigin,
    },
    RequirementCreated(RequirementRecord),
    RequirementCorrected {
        old_id: String,
        replacement: RequirementRecord,
    },
    RequirementRenamed {
        id: String,
        alias: String,
    },
    RequirementStatus {
        id: String,
        previous: RequirementStatus,
        new: RequirementStatus,
        actor: EventActor,
        source_message_id: String,
        operation_id: Option<String>,
        evidence_ids: Vec<String>,
        reason: String,
    },
    NoOp(NoOpDiagnostic),
    Recovery(RecoveryStatus),
    HistoryGroupStored {
        history_ref: HistoryRef,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Transaction {
    events: Vec<EventPayload>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Frame {
    len: usize,
    revision: u64,
    event_id: String,
    previous_hash: String,
    transaction: Transaction,
    hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GroupBlob {
    version: u8,
    history_ref: HistoryRef,
    messages: Vec<Message>,
    content_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProjectionSnapshot {
    version: u8,
    revision: u64,
    journal_hash: String,
    refs: Vec<HistoryRef>,
    content_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointStatus {
    pub revision: u64,
    pub journal_hash: String,
    pub rebuilt: bool,
    pub stale: bool,
}

#[derive(Clone, Debug)]
struct JournalState {
    revision: u64,
    hash: String,
    transactions: Vec<(u64, String, Transaction)>,
}

/// A private, scoped append-only journal.
#[derive(Debug)]
pub struct HistoryJournal {
    root: PathBuf,
    path: PathBuf,
    _lock: File,
    scope: JournalScope,
    state: JournalState,
    recovery: Option<RecoveryStatus>,
    max_bytes: usize,
}

impl HistoryJournal {
    pub fn open(
        config_root: impl AsRef<Path>,
        workspace: &str,
        session: &str,
    ) -> Result<Self, HistoryError> {
        Self::open_with_quota(config_root, workspace, session, 32 * 1024 * 1024)
    }

    pub fn open_with_quota(
        config_root: impl AsRef<Path>,
        workspace: &str,
        session: &str,
        max_bytes: usize,
    ) -> Result<Self, HistoryError> {
        if workspace.is_empty() || session.is_empty() || max_bytes == 0 {
            return Err(HistoryError::Quota(
                "scope and quota must be non-empty".into(),
            ));
        }
        let root = config_root.as_ref().join("history");
        let scope_dir = root.join(scope_key(workspace)).join(scope_key(session));
        fs::create_dir_all(&scope_dir)?;
        set_private(&root)?;
        set_private(&scope_dir)?;
        sync_directory(&root)?;
        sync_directory(&scope_dir)?;
        if let Some(parent) = scope_dir.parent() {
            sync_directory(parent)?;
        }
        if let Some(parent) = root.parent() {
            sync_directory(parent)?;
        }
        let path = scope_dir.join(JOURNAL_FILE);
        let lock_path = scope_dir.join(".journal.lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(HistoryError::Io)?;
        set_private_file(&lock_path)?;
        sync_directory(&scope_dir)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Err(HistoryError::Locked),
            Err(std::fs::TryLockError::Error(error)) => return Err(HistoryError::Io(error)),
        }
        let (state, recovery) = replay(&path)?;
        if path.exists() {
            set_private_file(&path)?;
        }
        let mut journal = Self {
            root,
            path,
            _lock: lock,
            scope: JournalScope {
                workspace: workspace.into(),
                session: session.into(),
            },
            state,
            recovery,
            max_bytes,
        };
        if let Some(recovery) = journal.recovery.clone() {
            let verified_len = fs::metadata(&journal.path)
                .map(|metadata| metadata.len() as usize)
                .unwrap_or(0)
                .saturating_sub(recovery.ignored_bytes);
            let file = OpenOptions::new().write(true).open(&journal.path)?;
            file.set_len(verified_len as u64)?;
            file.sync_data()?;
            // The recovery event is itself durable and visible.  It is added
            // only after replay has verified every complete preceding frame.
            journal.append_transaction(vec![EventPayload::Recovery(recovery)])?;
        }
        Ok(journal)
    }

    pub fn scope(&self) -> &JournalScope {
        &self.scope
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn revision(&self) -> u64 {
        self.state.revision
    }
    pub fn last_hash(&self) -> &str {
        &self.state.hash
    }
    pub fn recovery(&self) -> Option<&RecoveryStatus> {
        self.recovery.as_ref()
    }

    pub(crate) fn transactions(&self) -> impl Iterator<Item = &(u64, String, Transaction)> {
        self.state.transactions.iter()
    }

    pub(crate) fn append_transaction(
        &mut self,
        events: Vec<EventPayload>,
    ) -> Result<u64, HistoryError> {
        if events.is_empty() {
            return Ok(self.state.revision);
        }
        self.append_transaction_with_id(events, generate_uuid())
    }

    fn append_transaction_with_id(
        &mut self,
        events: Vec<EventPayload>,
        event_id: String,
    ) -> Result<u64, HistoryError> {
        let transaction = Transaction { events };
        let revision = self
            .state
            .revision
            .checked_add(1)
            .ok_or_else(|| HistoryError::Quota("revision exhausted".into()))?;
        let body = serde_json::to_vec(&(revision, &event_id, &self.state.hash, &transaction))?;
        if body.len() > MAX_FRAME_BYTES {
            return Err(HistoryError::Quota(
                "single journal frame is too large".into(),
            ));
        }
        let hash = sha256_hex(&[self.state.hash.as_bytes(), &body].concat());
        let frame = Frame {
            len: body.len(),
            revision,
            event_id: event_id.clone(),
            previous_hash: self.state.hash.clone(),
            transaction: transaction.clone(),
            hash: hash.clone(),
        };
        let encoded = serde_json::to_vec(&frame)?;
        if scope_usage(&self.path)?
            .saturating_add(encoded.len())
            .saturating_add(1)
            > self.max_bytes
        {
            return Err(HistoryError::Quota(format!(
                "journal limit {} bytes",
                self.max_bytes
            )));
        }
        atomic_append(&self.path, &encoded)?;
        self.state.revision = revision;
        self.state.hash = hash;
        self.state
            .transactions
            .push((revision, event_id, transaction));
        Ok(revision)
    }

    fn prepare_append(
        &self,
        events: &[EventPayload],
        event_id: &str,
    ) -> Result<(u64, String, usize), HistoryError> {
        let transaction = Transaction {
            events: events.to_vec(),
        };
        let revision = self
            .state
            .revision
            .checked_add(1)
            .ok_or_else(|| HistoryError::Quota("revision exhausted".into()))?;
        let body = serde_json::to_vec(&(revision, event_id, &self.state.hash, &transaction))?;
        if body.len() > MAX_FRAME_BYTES {
            return Err(HistoryError::Quota(
                "single journal frame is too large".into(),
            ));
        }
        let hash = sha256_hex(&[self.state.hash.as_bytes(), &body].concat());
        let frame = Frame {
            len: body.len(),
            revision,
            event_id: event_id.into(),
            previous_hash: self.state.hash.clone(),
            transaction,
            hash: hash.clone(),
        };
        Ok((revision, hash, serde_json::to_vec(&frame)?.len() + 1))
    }
}

/// The durable transcript and deterministic requirement projection.
#[derive(Debug)]
pub struct RequirementLedger {
    journal: HistoryJournal,
    records: BTreeMap<String, RequirementRecord>,
    messages: BTreeMap<String, u64>,
    runtime_messages: BTreeSet<String>,
    limits: ActiveLedgerLimits,
}

impl RequirementLedger {
    pub fn open(
        config_root: impl AsRef<Path>,
        workspace: &str,
        session: &str,
    ) -> Result<Self, HistoryError> {
        Self::open_with_limits(
            config_root,
            workspace,
            session,
            ActiveLedgerLimits::default(),
        )
    }

    pub fn open_with_limits(
        config_root: impl AsRef<Path>,
        workspace: &str,
        session: &str,
        limits: ActiveLedgerLimits,
    ) -> Result<Self, HistoryError> {
        let journal = HistoryJournal::open(config_root, workspace, session)?;
        Self::from_journal(journal, limits)
    }

    pub fn from_journal(
        journal: HistoryJournal,
        limits: ActiveLedgerLimits,
    ) -> Result<Self, HistoryError> {
        let mut ledger = Self {
            journal,
            records: BTreeMap::new(),
            messages: BTreeMap::new(),
            runtime_messages: BTreeSet::new(),
            limits,
        };
        let transactions: Vec<(u64, Transaction)> = ledger
            .journal
            .transactions()
            .map(|(revision, _, tx)| (*revision, tx.clone()))
            .collect();
        for (revision, transaction) in transactions {
            for event in transaction.events {
                ledger.apply_event_with_revision(&event, revision)?;
            }
        }
        Ok(ledger)
    }

    pub fn journal(&self) -> &HistoryJournal {
        &self.journal
    }
    pub fn journal_mut(&mut self) -> &mut HistoryJournal {
        &mut self.journal
    }
    pub fn limits(&self) -> ActiveLedgerLimits {
        self.limits
    }
    pub fn records(&self) -> impl Iterator<Item = &RequirementRecord> {
        self.records.values()
    }
    pub fn record(&self, id: &str) -> Option<&RequirementRecord> {
        self.records.get(id)
    }

    /// Ingest one complete user message.  `message_id` may be omitted; the
    /// generated ID is returned and is persisted in the same transaction.
    pub fn ingest(
        &mut self,
        message_id: Option<&str>,
        text: &str,
    ) -> Result<IngestReport, HistoryError> {
        let message_id = message_id.map(str::to_owned).unwrap_or_else(generate_uuid);
        if let Some(revision) = self.messages.get(&message_id).copied() {
            return Ok(IngestReport {
                message_id,
                revision,
                created: Vec::new(),
                diagnostics: Vec::new(),
                duplicate: true,
            });
        }
        let directives = parse_directives(text);
        let mut diagnostics = directives.diagnostics;
        let mut events = vec![EventPayload::Message {
            message_id: message_id.clone(),
            text: text.to_owned(),
        }];
        let mut created = Vec::new();
        let mut directive_events = Vec::new();
        directive_events.extend(diagnostics.iter().cloned().map(EventPayload::NoOp));
        let semantic_error = if diagnostics.is_empty() {
            self.validate_directives(&directives.valid)
        } else {
            Some(HistoryError::InvalidDirective(
                "directive batch contains malformed syntax".into(),
            ))
        };
        // Malformed or semantically invalid directives are no-ops, but the
        // surrounding ordinary prose remains one durable requirement.
        let prose = directives.prose.join("\n").trim().to_owned();
        if !prose.is_empty() {
            let id = requirement_id(&message_id, 0);
            let record = RequirementRecord {
                id: id.clone(),
                subject: RequirementSubject::Turn(message_id.clone()),
                source_turn: message_id.clone(),
                text: prose,
                status: RequirementStatus::Open,
                status_source_message_id: None,
                supersedes: None,
                superseded_by: None,
                correction_scope: None,
                aliases: Vec::new(),
                revision: 0,
            };
            directive_events.push(EventPayload::RequirementCreated(record));
            created.push(id);
        }
        if let Some(error) = semantic_error {
            if diagnostics.is_empty() {
                diagnostics.push(NoOpDiagnostic {
                    raw: directives.raw_directives.join("\n"),
                    reason: error.to_string(),
                });
                directive_events.push(EventPayload::NoOp(diagnostics.last().cloned().unwrap()));
            }
        } else {
            for (index, directive) in directives.valid.iter().enumerate() {
                match directive {
                    Directive::Correct {
                        id,
                        anchor,
                        replacement,
                    } => {
                        let mut replacement_record = RequirementRecord {
                            id: requirement_id(&message_id, index + 1),
                            subject: RequirementSubject::Turn(message_id.clone()),
                            source_turn: message_id.clone(),
                            text: replacement.clone(),
                            status: RequirementStatus::Open,
                            status_source_message_id: None,
                            supersedes: Some(id.clone()),
                            superseded_by: None,
                            correction_scope: Some(CorrectionScope {
                                anchor: anchor.clone(),
                            }),
                            aliases: Vec::new(),
                            revision: 0,
                        };
                        replacement_record.subject = self
                            .records
                            .get(id)
                            .map(|r| r.subject.clone())
                            .unwrap_or(replacement_record.subject);
                        let replacement_id = replacement_record.id.clone();
                        directive_events.push(EventPayload::RequirementCorrected {
                            old_id: id.clone(),
                            replacement: replacement_record,
                        });
                        created.push(replacement_id);
                    }
                    Directive::Rename { id, alias } => {
                        directive_events.push(EventPayload::RequirementRenamed {
                            id: id.clone(),
                            alias: alias.clone(),
                        })
                    }
                    Directive::Status { id, state, reason } => {
                        let previous = self.records[id].status.clone();
                        directive_events.push(EventPayload::RequirementStatus {
                            id: id.clone(),
                            previous,
                            new: state.clone(),
                            actor: EventActor::UserDirective,
                            source_message_id: message_id.clone(),
                            operation_id: None,
                            evidence_ids: Vec::new(),
                            reason: reason.clone(),
                        });
                    }
                }
            }
        }
        events.extend(directive_events);
        self.validate_events(&events)?;
        let revision = self.journal.append_transaction(events)?;
        let tx = self
            .journal
            .transactions()
            .last()
            .map(|(_, _, tx)| tx.clone())
            .ok_or_else(|| HistoryError::Corrupt("missing committed transaction".into()))?;
        for event in &tx.events {
            self.apply_event_with_revision(event, revision)?;
        }
        Ok(IngestReport {
            message_id,
            revision,
            created,
            diagnostics,
            duplicate: false,
        })
    }

    pub fn set_status_verified(
        &mut self,
        id: &str,
        source_message_id: &str,
        state: RequirementStatus,
        operation_id: &str,
        evidence_ids: &[String],
        reason: &str,
    ) -> Result<u64, HistoryError> {
        if source_message_id.is_empty() || operation_id.is_empty() || evidence_ids.is_empty() {
            return Err(HistoryError::MissingVerificationProvenance);
        }
        let previous = self
            .records
            .get(id)
            .ok_or_else(|| HistoryError::UnknownRequirement(id.into()))?
            .status
            .clone();
        let event = EventPayload::RequirementStatus {
            id: id.into(),
            previous,
            new: state,
            actor: EventActor::ToolVerifier,
            source_message_id: source_message_id.into(),
            operation_id: Some(operation_id.into()),
            evidence_ids: evidence_ids.to_vec(),
            reason: reason.into(),
        };
        self.validate_events(std::slice::from_ref(&event))?;
        let revision = self.journal.append_transaction(vec![event.clone()])?;
        self.apply_event_with_revision(&event, revision)?;
        Ok(revision)
    }

    pub fn retrieve(&self, id: &str) -> Result<HistoricalRecord, HistoryError> {
        let record = self
            .records
            .get(id)
            .ok_or_else(|| HistoryError::UnknownRequirement(id.into()))?
            .clone();
        let authority =
            if record.status == RequirementStatus::Superseded || record.superseded_by.is_some() {
                HistoryAuthority::Superseded
            } else {
                HistoryAuthority::Historical
            };
        Ok(HistoricalRecord { record, authority })
    }

    pub fn active_view(&self) -> Result<ActiveLedgerView, HistoryError> {
        let mut active: Vec<&RequirementRecord> = self
            .records
            .values()
            .filter(|r| {
                matches!(
                    r.status,
                    RequirementStatus::Open | RequirementStatus::Blocked
                )
            })
            .collect();
        active.sort_by_key(|record| std::cmp::Reverse(record.revision));
        let active_count = active.len();
        let mut full = Vec::new();
        let mut references = Vec::new();
        let mut full_bytes: usize = 0;
        let mut reference_bytes: usize = 0;
        for record in active {
            let bytes = record.text.len();
            let tokens = conservative_tokens(&record.text);
            let can_full = full.len() < self.limits.max_full_records
                && full_bytes.saturating_add(bytes) <= self.limits.max_full_bytes
                && full
                    .iter()
                    .map(|r: &RequirementRecord| conservative_tokens(&r.text))
                    .sum::<u64>()
                    .saturating_add(tokens)
                    <= self.limits.max_full_tokens;
            if can_full {
                full_bytes += bytes;
                full.push(record.clone());
            } else {
                let reference = RequirementRef {
                    id: record.id.clone(),
                    status: record.status.clone(),
                    source_message_id: record.source_turn.clone(),
                    revision: record.revision,
                };
                let ref_bytes = serde_json::to_vec(&reference)?.len();
                if references.len() >= self.limits.max_references
                    || reference_bytes.saturating_add(ref_bytes) > self.limits.max_reference_bytes
                {
                    return Err(HistoryError::ActiveOverflow(format!(
                        "{} active requirements exceed bounded projection; {} omitted requirements need local HistoryRead retrieval",
                        active_count,
                        active_count.saturating_sub(full.len()).saturating_sub(references.len())
                    )));
                }
                reference_bytes += ref_bytes;
                references.push(reference);
            }
        }
        Ok(ActiveLedgerView {
            full,
            references,
            full_bytes,
            reference_bytes,
        })
    }

    fn validate_directives(&self, directives: &[Directive]) -> Option<HistoryError> {
        for directive in directives {
            let id = match directive {
                Directive::Correct { id, .. }
                | Directive::Rename { id, .. }
                | Directive::Status { id, .. } => id,
            };
            let record = match self.records.get(id) {
                Some(record) => record,
                None => return Some(HistoryError::UnknownRequirement(id.clone())),
            };
            if let Directive::Correct {
                anchor: Some(anchor),
                ..
            } = directive
            {
                let text_matches = record.text.contains(anchor)
                    || record.aliases.iter().any(|alias| alias == anchor);
                if !text_matches {
                    return Some(HistoryError::AnchorMismatch(anchor.clone()));
                }
            }
        }
        None
    }

    fn validate_events(&self, events: &[EventPayload]) -> Result<(), HistoryError> {
        let mut records = self.records.clone();
        let mut messages = self.messages.clone();
        for event in events {
            apply_event_projection(&mut records, &mut messages, event, 0)?;
        }
        Ok(())
    }

    fn apply_event_with_revision(
        &mut self,
        event: &EventPayload,
        revision: u64,
    ) -> Result<(), HistoryError> {
        apply_event_projection(&mut self.records, &mut self.messages, event, revision)?;
        if let EventPayload::RuntimeMessage { message_id, .. } = event {
            self.runtime_messages.insert(message_id.clone());
        }
        Ok(())
    }

    pub fn is_runtime_message(&self, message_id: &str) -> bool {
        self.runtime_messages.contains(message_id)
    }

    pub fn runtime_message_ids(&self) -> impl Iterator<Item = &String> {
        self.runtime_messages.iter()
    }
}

fn apply_event_projection(
    records: &mut BTreeMap<String, RequirementRecord>,
    messages: &mut BTreeMap<String, u64>,
    event: &EventPayload,
    revision: u64,
) -> Result<(), HistoryError> {
    match event {
        EventPayload::Message { message_id, .. } => {
            if messages.contains_key(message_id) {
                return Err(HistoryError::Corrupt(format!(
                    "duplicate source message ID {message_id}"
                )));
            }
            messages.insert(message_id.clone(), revision);
        }
        EventPayload::RuntimeMessage { message_id, .. } => {
            if messages.contains_key(message_id) {
                return Err(HistoryError::Corrupt(format!(
                    "duplicate source message ID {message_id}"
                )));
            }
            messages.insert(message_id.clone(), revision);
        }
        EventPayload::RequirementCreated(record) => {
            if records.contains_key(&record.id) {
                return Err(HistoryError::Corrupt(format!(
                    "duplicate requirement ID {}",
                    record.id
                )));
            }
            let mut record = record.clone();
            if revision != 0 {
                record.revision = revision;
            }
            records.insert(record.id.clone(), record);
        }
        EventPayload::RequirementCorrected {
            old_id,
            replacement,
        } => {
            if records.contains_key(&replacement.id) {
                return Err(HistoryError::Corrupt(format!(
                    "duplicate replacement requirement ID {}",
                    replacement.id
                )));
            }
            let old = records.get_mut(old_id).ok_or_else(|| {
                HistoryError::Corrupt(format!(
                    "correction references unknown requirement {old_id}"
                ))
            })?;
            if old.status == RequirementStatus::Superseded || old.superseded_by.is_some() {
                return Err(HistoryError::Corrupt(format!(
                    "requirement {old_id} was corrected more than once"
                )));
            }
            if replacement.supersedes.as_deref() != Some(old_id.as_str()) {
                return Err(HistoryError::Corrupt(format!(
                    "replacement {} has contradictory supersedes link",
                    replacement.id
                )));
            }
            old.status = RequirementStatus::Superseded;
            old.superseded_by = Some(replacement.id.clone());
            let mut replacement = replacement.clone();
            if revision != 0 {
                replacement.revision = revision;
            }
            records.insert(replacement.id.clone(), replacement);
        }
        EventPayload::RequirementRenamed { id, alias } => {
            let record = records.get_mut(id).ok_or_else(|| {
                HistoryError::Corrupt(format!("rename references unknown requirement {id}"))
            })?;
            if alias.is_empty() || record.aliases.contains(alias) {
                return Err(HistoryError::Corrupt(format!(
                    "contradictory alias update for requirement {id}"
                )));
            }
            record.aliases.push(alias.clone());
        }
        EventPayload::RequirementStatus {
            id,
            previous,
            new,
            actor,
            source_message_id,
            operation_id,
            evidence_ids,
            ..
        } => {
            let record = records.get_mut(id).ok_or_else(|| {
                HistoryError::Corrupt(format!("status references unknown requirement {id}"))
            })?;
            if &record.status != previous {
                return Err(HistoryError::Corrupt(format!(
                    "status previous value does not match requirement {id}"
                )));
            }
            if source_message_id.is_empty()
                || (*actor == EventActor::ToolVerifier
                    && (operation_id.as_deref().unwrap_or_default().is_empty()
                        || evidence_ids.is_empty()))
                || *new == RequirementStatus::Superseded
                || record.status == RequirementStatus::Superseded
            {
                return Err(HistoryError::Corrupt(format!(
                    "contradictory status provenance for requirement {id}"
                )));
            }
            record.status = new.clone();
            record.status_source_message_id = Some(source_message_id.clone());
            if revision != 0 {
                record.revision = revision;
            }
        }
        EventPayload::NoOp(_)
        | EventPayload::Recovery(_)
        | EventPayload::HistoryGroupStored { .. } => {}
    }
    Ok(())
}

/// Public compatibility wrapper used by the future request assembler.
#[derive(Debug)]
pub struct HistoryIndex {
    ledger: RequirementLedger,
    groups: BTreeMap<String, HistoryRef>,
    checkpoint: CheckpointStatus,
}

impl HistoryIndex {
    pub fn open(
        config_root: impl AsRef<Path>,
        workspace: &str,
        session: &str,
    ) -> Result<Self, HistoryError> {
        Self::open_with_quota(config_root, workspace, session, 32 * 1024 * 1024)
    }
    pub fn open_with_quota(
        config_root: impl AsRef<Path>,
        workspace: &str,
        session: &str,
        max_bytes: usize,
    ) -> Result<Self, HistoryError> {
        let journal = HistoryJournal::open_with_quota(config_root, workspace, session, max_bytes)?;
        Self::from_ledger(RequirementLedger::from_journal(
            journal,
            ActiveLedgerLimits::default(),
        )?)
    }
    fn from_ledger(ledger: RequirementLedger) -> Result<Self, HistoryError> {
        let mut groups = BTreeMap::new();
        for (_, _, transaction) in ledger.journal().transactions() {
            for event in &transaction.events {
                if let EventPayload::HistoryGroupStored { history_ref } = event {
                    if groups
                        .insert(history_ref.id.clone(), history_ref.clone())
                        .is_some()
                    {
                        return Err(HistoryError::Corrupt(format!(
                            "duplicate history reference {}",
                            history_ref.id
                        )));
                    }
                }
            }
        }
        for history_ref in groups.values() {
            read_group_blob(&ledger.journal, history_ref)?;
        }
        let mut index = Self {
            ledger,
            groups,
            checkpoint: CheckpointStatus {
                revision: 0,
                journal_hash: String::new(),
                rebuilt: true,
                stale: true,
            },
        };
        let valid_checkpoint = read_checkpoint(&index.ledger.journal, &index.groups)?;
        let rebuilt = !valid_checkpoint;
        index.checkpoint = CheckpointStatus {
            revision: index.ledger.journal.revision(),
            journal_hash: index.ledger.journal.last_hash().to_owned(),
            rebuilt,
            stale: false,
        };
        if !valid_checkpoint {
            index.write_checkpoint()?;
            index.checkpoint.rebuilt = true;
        }
        Ok(index)
    }
    pub fn ledger(&self) -> &RequirementLedger {
        &self.ledger
    }
    pub fn ledger_mut(&mut self) -> &mut RequirementLedger {
        &mut self.ledger
    }

    /// Return the configuration directory used to open this index.  The
    /// journal stores its own `history/` child, so callers opening a fresh
    /// scope must pass this parent rather than appending `history` again.
    pub fn config_root(&self) -> &Path {
        let journal_root = self.ledger.journal().root();
        journal_root.parent().unwrap_or(journal_root)
    }

    pub fn runtime_message_ids(&self) -> Vec<String> {
        self.ledger.runtime_message_ids().cloned().collect()
    }

    /// Persist a runtime-generated user turn without projecting it into the
    /// user requirement ledger. The message remains available in the journal
    /// for provenance and receives duplicate-ID protection like an ordinary
    /// source message.
    pub fn record_runtime_message(
        &mut self,
        message_id: &str,
        text: &str,
        origin: RuntimeMessageOrigin,
    ) -> Result<u64, HistoryError> {
        if message_id.is_empty() {
            return Err(HistoryError::MissingMessageId);
        }
        if let Some(revision) = self.ledger.messages.get(message_id).copied() {
            return Ok(revision);
        }
        let event = EventPayload::RuntimeMessage {
            message_id: message_id.to_owned(),
            text: text.to_owned(),
            origin,
        };
        self.ledger.validate_events(std::slice::from_ref(&event))?;
        let revision = self
            .ledger
            .journal
            .append_transaction(vec![event.clone()])?;
        self.ledger.apply_event_with_revision(&event, revision)?;
        self.write_checkpoint()?;
        Ok(revision)
    }

    pub fn ingest(
        &mut self,
        message_id: Option<&str>,
        text: &str,
    ) -> Result<IngestReport, HistoryError> {
        self.ledger.ingest(message_id, text)
    }
    pub fn retrieve(&self, id: &str) -> Result<HistoricalRecord, HistoryError> {
        self.ledger.retrieve(id)
    }

    /// Persist an omitted dependency-closed component before dispatch.
    /// The blob is synced and verified before the journal publishes its ref.
    pub fn persist_omitted_group(
        &mut self,
        group_id: &str,
        sequence: u64,
        messages: Vec<Message>,
        reason: OmitReason,
    ) -> Result<HistoryRef, HistoryError> {
        self.persist_omitted_group_with_authority(
            group_id,
            sequence,
            messages,
            reason,
            HistoryAuthority::Historical,
        )
    }

    pub fn persist_omitted_group_with_authority(
        &mut self,
        group_id: &str,
        sequence: u64,
        messages: Vec<Message>,
        reason: OmitReason,
        authority: HistoryAuthority,
    ) -> Result<HistoryRef, HistoryError> {
        let source_message_ids = message_ids(&messages)?;
        if group_id.is_empty() || messages.is_empty() {
            return Err(HistoryError::InvalidHistoryGroup(
                "group ID and members must be non-empty".into(),
            ));
        }
        let source_range = SourceRange {
            message_id: source_message_ids[0].clone(),
            revision: sequence,
        };
        let reference = HistoryRef {
            id: history_ref_id(
                self.ledger.journal.scope(),
                group_id,
                sequence,
                &source_message_ids,
            )?,
            group_id: group_id.into(),
            sequence,
            source_range,
            source_message_ids,
            reason,
            authority,
        };
        if let Some(existing) = self.groups.get(&reference.id) {
            let stored = read_group_blob(&self.ledger.journal, existing)?;
            if serde_json::to_vec(&stored)? != serde_json::to_vec(&messages)? {
                return Err(HistoryError::Corrupt(format!(
                    "history reference {} has different content",
                    reference.id
                )));
            }
            if existing.reason != reference.reason || existing.authority != reference.authority {
                return Err(HistoryError::InvalidHistoryGroup(format!(
                    "history reference {} has contradictory provenance",
                    reference.id
                )));
            }
            return Ok(existing.clone());
        }
        let event = EventPayload::HistoryGroupStored {
            history_ref: reference.clone(),
        };
        let event_id = generate_uuid();
        let (revision, journal_hash, journal_bytes) = self
            .ledger
            .journal
            .prepare_append(std::slice::from_ref(&event), &event_id)?;
        let mut refs: Vec<HistoryRef> = self.groups.values().cloned().collect();
        refs.push(reference.clone());
        refs.sort_by(|left, right| left.id.cmp(&right.id));
        let snapshot_content = serde_json::to_vec(&(1u8, revision, &journal_hash, &refs))?;
        let snapshot = ProjectionSnapshot {
            version: 1,
            revision,
            journal_hash,
            refs,
            content_hash: sha256_hex(&snapshot_content),
        };
        let blob_bytes = encode_group_blob(&reference, messages.clone())?;
        if blob_bytes.len() > MAX_FRAME_BYTES {
            return Err(HistoryError::Quota(
                "history group blob is too large".into(),
            ));
        }
        let snapshot_bytes = serde_json::to_vec(&snapshot)?;
        let old_snapshot_bytes = file_size_if_present(&snapshot_path(&self.ledger.journal))?;
        let usage_without_old_snapshot =
            scope_usage(&self.ledger.journal.path)?.saturating_sub(old_snapshot_bytes);
        if usage_without_old_snapshot
            .saturating_add(blob_bytes.len())
            .saturating_add(journal_bytes)
            .saturating_add(snapshot_bytes.len())
            > self.ledger.journal.max_bytes
        {
            return Err(HistoryError::Quota(format!(
                "history scope limit {} bytes",
                self.ledger.journal.max_bytes
            )));
        }
        let group_path = group_path(&self.ledger.journal, &reference)?;
        write_group_blob(&self.ledger.journal, &reference, messages.clone())?;
        // Re-read after fsync and hash verification.  A crash can leave an
        // unreferenced blob, but it can never leave a published missing ref.
        read_group_blob(&self.ledger.journal, &reference)?;
        if let Err(error) = self
            .ledger
            .journal_mut()
            .append_transaction_with_id(vec![event], event_id)
        {
            if matches!(error, HistoryError::Quota(_)) {
                let _ = fs::remove_file(&group_path);
            }
            return Err(error);
        }
        self.groups.insert(reference.id.clone(), reference.clone());
        if let Err(error) = self.write_checkpoint() {
            return Err(HistoryError::Committed(error.to_string()));
        }
        Ok(reference)
    }

    /// Compatibility alias for the assembler's omission terminology.
    pub fn store_omitted_group(
        &mut self,
        group_id: &str,
        sequence: u64,
        messages: Vec<Message>,
        reason: OmitReason,
    ) -> Result<HistoryRef, HistoryError> {
        self.persist_omitted_group(group_id, sequence, messages, reason)
    }

    pub fn retrieve_group(&self, id: &str) -> Result<HistoricalGroup, HistoryError> {
        let history_ref = self
            .groups
            .get(id)
            .ok_or_else(|| HistoryError::HistoryUnavailable(id.into()))?
            .clone();
        let messages = read_group_blob(&self.ledger.journal, &history_ref)?;
        Ok(HistoricalGroup {
            history_ref,
            messages,
        })
    }

    pub fn retrieve_omitted_group(&self, id: &str) -> Result<HistoricalGroup, HistoryError> {
        self.retrieve_group(id)
    }

    pub fn omitted_refs(&self) -> impl Iterator<Item = &HistoryRef> {
        self.groups.values()
    }

    /// Return bounded historical references matching a caller-provided search
    /// term. This only searches the immutable omission manifest; it never
    /// projects a historical record back into the active ledger.
    pub fn search_omitted_refs(&self, query: &str, offset: usize, limit: usize) -> Vec<HistoryRef> {
        let query = query.trim().to_ascii_lowercase();
        let limit = limit.min(64);
        self.groups
            .values()
            .filter(|reference| {
                query.is_empty()
                    || reference.id.to_ascii_lowercase().contains(&query)
                    || reference.group_id.to_ascii_lowercase().contains(&query)
            })
            .skip(offset.min(1024))
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn checkpoint(&self) -> CheckpointStatus {
        let mut checkpoint = self.checkpoint.clone();
        checkpoint.stale = checkpoint.revision != self.ledger.journal.revision()
            || checkpoint.journal_hash != self.ledger.journal.last_hash();
        checkpoint
    }

    pub fn write_checkpoint(&mut self) -> Result<(), HistoryError> {
        let refs: Vec<HistoryRef> = self.groups.values().cloned().collect();
        let content = serde_json::to_vec(&(
            1u8,
            self.ledger.journal.revision(),
            self.ledger.journal.last_hash(),
            &refs,
        ))?;
        let snapshot = ProjectionSnapshot {
            version: 1,
            revision: self.ledger.journal.revision(),
            journal_hash: self.ledger.journal.last_hash().into(),
            refs,
            content_hash: sha256_hex(&content),
        };
        write_snapshot(&self.ledger.journal, &snapshot)?;
        self.checkpoint = CheckpointStatus {
            revision: snapshot.revision,
            journal_hash: snapshot.journal_hash,
            rebuilt: false,
            stale: false,
        };
        Ok(())
    }

    pub fn orphan_blob_bytes(&self) -> Result<usize, HistoryError> {
        let group_dir = self.ledger.journal.path().parent().unwrap().join(GROUP_DIR);
        let mut referenced = BTreeMap::new();
        for id in self.groups.keys() {
            referenced.insert(format!("{id}.json"), ());
        }
        let mut bytes = 0usize;
        let entries = match fs::read_dir(group_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(HistoryError::Io(error)),
        };
        for entry in entries {
            let entry = entry?;
            if !referenced.contains_key(&entry.file_name().to_string_lossy().to_string()) {
                bytes = bytes.saturating_add(entry.metadata()?.len() as usize);
            }
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug)]
enum Directive {
    Correct {
        id: String,
        anchor: Option<String>,
        replacement: String,
    },
    Rename {
        id: String,
        alias: String,
    },
    Status {
        id: String,
        state: RequirementStatus,
        reason: String,
    },
}

struct ParsedDirectives {
    valid: Vec<Directive>,
    prose: Vec<String>,
    raw_directives: Vec<String>,
    diagnostics: Vec<NoOpDiagnostic>,
}

fn parse_directives(text: &str) -> ParsedDirectives {
    let mut valid = Vec::new();
    let mut prose = Vec::new();
    let mut raw_directives = Vec::new();
    let mut diagnostics = Vec::new();
    let mut fenced = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            fenced = !fenced;
            prose.push(line.into());
            continue;
        }
        if !fenced && !trimmed.starts_with('>') && trimmed.starts_with("@forge/") {
            raw_directives.push(line.into());
            match parse_directive(trimmed) {
                Ok(directive) => valid.push(directive),
                Err(reason) => diagnostics.push(NoOpDiagnostic {
                    raw: line.into(),
                    reason,
                }),
            }
        } else {
            prose.push(line.into());
        }
    }
    ParsedDirectives {
        valid,
        prose,
        raw_directives,
        diagnostics,
    }
}

fn parse_directive(line: &str) -> Result<Directive, String> {
    if let Some(rest) = line.strip_prefix("@forge/correct ") {
        let (head, replacement) = rest
            .split_once(" :: ")
            .ok_or("missing correction separator")?;
        if replacement.trim().is_empty() {
            return Err("empty replacement".into());
        }
        let fields = parse_fields(head)?;
        let id = fields.get("id").ok_or("missing id")?.to_string();
        if fields.keys().any(|key| key != "id" && key != "anchor") {
            return Err("unknown or duplicate field".into());
        }
        if !valid_requirement_id(&id) {
            return Err("invalid requirement id".into());
        }
        return Ok(Directive::Correct {
            id,
            anchor: fields.get("anchor").cloned(),
            replacement: replacement.trim().into(),
        });
    }
    if let Some(rest) = line.strip_prefix("@forge/rename ") {
        let fields = parse_fields(rest)?;
        if fields.len() != 2 || !fields.contains_key("id") || !fields.contains_key("alias") {
            return Err("rename requires exactly id and alias".into());
        }
        let id = fields["id"].clone();
        let alias = fields["alias"].clone();
        if !valid_requirement_id(&id) || alias.is_empty() {
            return Err("invalid rename fields".into());
        }
        return Ok(Directive::Rename { id, alias });
    }
    if let Some(rest) = line.strip_prefix("@forge/status ") {
        let (head, reason) = rest.split_once(" :: ").ok_or("missing status reason")?;
        if reason.trim().is_empty() {
            return Err("empty status reason".into());
        }
        let fields = parse_fields(head)?;
        if fields.len() != 2 || !fields.contains_key("id") || !fields.contains_key("state") {
            return Err("status requires exactly id and state".into());
        }
        let id = fields["id"].clone();
        let state = match fields["state"].as_str() {
            "open" => RequirementStatus::Open,
            "blocked" => RequirementStatus::Blocked,
            "satisfied" => RequirementStatus::Satisfied,
            _ => return Err("invalid status".into()),
        };
        if !valid_requirement_id(&id) {
            return Err("invalid requirement id".into());
        }
        return Ok(Directive::Status {
            id,
            state,
            reason: reason.trim().into(),
        });
    }
    Err("unknown directive".into())
}

fn parse_fields(input: &str) -> Result<BTreeMap<String, String>, String> {
    let mut fields = BTreeMap::new();
    let mut rest = input.trim();
    while !rest.is_empty() {
        let eq = rest.find('=').ok_or("field is missing '='")?;
        let key = &rest[..eq];
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err("invalid field name".into());
        }
        rest = &rest[eq + 1..];
        let (value, tail) = if let Some(stripped) = rest.strip_prefix('"') {
            let end = stripped.find('"').ok_or("unterminated quoted field")?;
            (&stripped[..end], &stripped[end + 1..])
        } else {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            (&rest[..end], &rest[end..])
        };
        if fields.insert(key.into(), value.into()).is_some() {
            return Err("duplicate field".into());
        }
        rest = tail.trim_start();
    }
    Ok(fields)
}

fn valid_requirement_id(id: &str) -> bool {
    id.starts_with("req-")
        && id.len() > 4
        && id[4..]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
fn requirement_id(message_id: &str, index: usize) -> String {
    format!(
        "req-{}",
        &sha256_hex(format!("{message_id}\0{index}").as_bytes())[..24]
    )
}
fn scope_key(value: &str) -> String {
    sha256_hex(value.as_bytes())
}
fn conservative_tokens(text: &str) -> u64 {
    ((text.len() as u64).saturating_add(3)) / 4
}

fn message_ids(messages: &[Message]) -> Result<Vec<String>, HistoryError> {
    let mut ids = Vec::with_capacity(messages.len());
    for message in messages {
        let id = message
            .uuid
            .as_deref()
            .filter(|id| !id.is_empty())
            .ok_or(HistoryError::MissingMessageId)?;
        if ids.iter().any(|existing| existing == id) {
            return Err(HistoryError::InvalidHistoryGroup(
                "duplicate source message ID".into(),
            ));
        }
        ids.push(id.to_owned());
    }
    Ok(ids)
}

fn history_ref_id(
    scope: &JournalScope,
    group_id: &str,
    sequence: u64,
    source_message_ids: &[String],
) -> Result<String, HistoryError> {
    let seed = serde_json::to_vec(&(
        &scope.workspace,
        &scope.session,
        group_id,
        sequence,
        source_message_ids,
    ))?;
    Ok(format!("href-{}", &sha256_hex(&seed)[..32]))
}

fn validate_history_ref(scope: &JournalScope, reference: &HistoryRef) -> Result<(), HistoryError> {
    let valid_id = reference.id.len() == 37
        && reference.id.starts_with("href-")
        && reference.id[5..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit());
    let valid_group = !reference.group_id.is_empty()
        && reference
            .group_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte));
    let valid_ids = !reference.source_message_ids.is_empty()
        && reference
            .source_message_ids
            .iter()
            .all(|id| !id.is_empty() && !id.bytes().any(|byte| byte.is_ascii_control()))
        && reference
            .source_message_ids
            .iter()
            .enumerate()
            .all(|(index, id)| {
                reference.source_message_ids[..index]
                    .iter()
                    .all(|previous| previous != id)
            });
    let first_source_matches = reference
        .source_message_ids
        .first()
        .map(|id| id == &reference.source_range.message_id)
        .unwrap_or(false);
    let expected_id = history_ref_id(
        scope,
        &reference.group_id,
        reference.sequence,
        &reference.source_message_ids,
    )?;
    if !valid_id
        || !valid_group
        || !valid_ids
        || !first_source_matches
        || reference.source_range.revision != reference.sequence
        || expected_id != reference.id
    {
        return Err(HistoryError::Corrupt(format!(
            "history reference {} failed scope validation",
            reference.id
        )));
    }
    Ok(())
}

fn group_path(journal: &HistoryJournal, reference: &HistoryRef) -> Result<PathBuf, HistoryError> {
    validate_history_ref(&journal.scope, reference)?;
    Ok(journal
        .path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(GROUP_DIR)
        .join(format!("{}.json", reference.id)))
}

fn write_group_blob(
    journal: &HistoryJournal,
    reference: &HistoryRef,
    messages: Vec<Message>,
) -> Result<(), HistoryError> {
    let group_dir = journal
        .path
        .parent()
        .ok_or_else(|| HistoryError::Io(io::Error::other("journal has no parent")))?
        .join(GROUP_DIR);
    fs::create_dir_all(&group_dir)?;
    set_private(&group_dir)?;
    sync_directory(&group_dir)?;
    sync_parent_directory(&group_dir)?;
    let encoded = encode_group_blob(reference, messages)?;
    if encoded.len() > MAX_FRAME_BYTES {
        return Err(HistoryError::Quota(
            "history group blob is too large".into(),
        ));
    }
    let path = group_path(journal, reference)?;
    let temporary = group_dir.join(format!(".{}.{}.tmp", reference.id, generate_uuid()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    set_private_file(&temporary)?;
    fs::rename(&temporary, &path)?;
    sync_parent_directory(&group_dir)?;
    set_private_file(&path)?;
    Ok(())
}

fn encode_group_blob(
    reference: &HistoryRef,
    messages: Vec<Message>,
) -> Result<Vec<u8>, HistoryError> {
    let body = serde_json::to_vec(&(1u8, reference, &messages))?;
    let blob = GroupBlob {
        version: 1,
        history_ref: reference.clone(),
        messages,
        content_hash: sha256_hex(&body),
    };
    Ok(serde_json::to_vec(&blob)?)
}

fn read_group_blob(
    journal: &HistoryJournal,
    expected: &HistoryRef,
) -> Result<Vec<Message>, HistoryError> {
    let path = group_path(journal, expected)?;
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                HistoryError::HistoryUnavailable(expected.id.clone())
            } else {
                HistoryError::Io(error)
            }
        })?
        .read_to_end(&mut bytes)?;
    let blob: GroupBlob = serde_json::from_slice(&bytes)
        .map_err(|error| HistoryError::Corrupt(format!("history blob {}: {error}", expected.id)))?;
    let body = serde_json::to_vec(&(blob.version, &blob.history_ref, &blob.messages))?;
    if blob.version != 1 || blob.history_ref != *expected || blob.content_hash != sha256_hex(&body)
    {
        return Err(HistoryError::Corrupt(format!(
            "history blob {} failed verification",
            expected.id
        )));
    }
    if message_ids(&blob.messages)? != expected.source_message_ids {
        return Err(HistoryError::Corrupt(format!(
            "history blob {} source IDs mismatch",
            expected.id
        )));
    }
    Ok(blob.messages)
}

fn snapshot_path(journal: &HistoryJournal) -> PathBuf {
    journal
        .path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(CHECKPOINT_FILE)
}

fn write_snapshot(
    journal: &HistoryJournal,
    snapshot: &ProjectionSnapshot,
) -> Result<(), HistoryError> {
    let path = snapshot_path(journal);
    let temporary = path.with_extension(format!("snapshot.{}.tmp", generate_uuid()));
    let encoded = serde_json::to_vec(snapshot)?;
    let old_snapshot_bytes = file_size_if_present(&path)?;
    if scope_usage(&journal.path)?
        .saturating_sub(old_snapshot_bytes)
        .saturating_add(encoded.len())
        > journal.max_bytes
    {
        return Err(HistoryError::Quota(format!(
            "history scope limit {} bytes",
            journal.max_bytes
        )));
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    set_private_file(&temporary)?;
    fs::rename(&temporary, &path)?;
    sync_parent_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
    set_private_file(&path)?;
    Ok(())
}

fn read_checkpoint(
    journal: &HistoryJournal,
    groups: &BTreeMap<String, HistoryRef>,
) -> Result<bool, HistoryError> {
    let path = snapshot_path(journal);
    let mut bytes = Vec::new();
    match File::open(&path) {
        Ok(mut file) => {
            file.read_to_end(&mut bytes)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(HistoryError::Io(error)),
    }
    let snapshot: ProjectionSnapshot = match serde_json::from_slice(&bytes) {
        Ok(snapshot) => snapshot,
        Err(_) => return Ok(false),
    };
    let body = serde_json::to_vec(&(
        snapshot.version,
        snapshot.revision,
        &snapshot.journal_hash,
        &snapshot.refs,
    ))?;
    let expected_refs: Vec<HistoryRef> = groups.values().cloned().collect();
    Ok(snapshot.version == 1
        && snapshot.revision == journal.revision()
        && snapshot.journal_hash == journal.last_hash()
        && snapshot.refs == expected_refs
        && snapshot.content_hash == sha256_hex(&body))
}

fn scope_usage(journal_path: &Path) -> Result<usize, HistoryError> {
    let scope = journal_path.parent().unwrap_or_else(|| Path::new("."));
    fn sum(path: &Path) -> Result<usize, HistoryError> {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(HistoryError::Io(error)),
        };
        if metadata.is_file() {
            return Ok(metadata.len() as usize);
        }
        let mut total = 0usize;
        for entry in fs::read_dir(path)? {
            total = total.saturating_add(sum(&entry?.path())?);
        }
        Ok(total)
    }
    sum(scope)
}

fn file_size_if_present(path: &Path) -> Result<usize, HistoryError> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len() as usize),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(HistoryError::Io(error)),
    }
}

fn sync_parent_directory(path: &Path) -> Result<(), HistoryError> {
    sync_directory(path)
}

fn sync_directory(path: &Path) -> Result<(), HistoryError> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    Ok(())
}

fn replay(path: &Path) -> Result<(JournalState, Option<RecoveryStatus>), HistoryError> {
    let mut bytes = Vec::new();
    match File::open(path) {
        Ok(mut file) => {
            file.read_to_end(&mut bytes)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(HistoryError::Io(error)),
    }
    let mut state = JournalState {
        revision: 0,
        hash: String::new(),
        transactions: Vec::new(),
    };
    let mut offset = 0usize;
    let mut recovery = None;
    for (index, segment) in bytes.split_inclusive(|byte| *byte == b'\n').enumerate() {
        let has_newline = segment.ends_with(b"\n");
        let line = segment.strip_suffix(b"\n").unwrap_or(segment);
        if line.is_empty() {
            offset += segment.len();
            continue;
        }
        // A complete JSON object without its framing newline is still a
        // torn append: do not trust it or append after it until recovery has
        // truncated the unframed tail.
        if !has_newline && index + 1 == bytes.split(|byte| *byte == b'\n').count() {
            recovery = Some(RecoveryStatus {
                torn_tail: true,
                ignored_bytes: bytes.len() - offset,
                recovered_revision: state.revision,
            });
            break;
        }
        let frame: Frame = match serde_json::from_slice(line) {
            Ok(frame) => frame,
            Err(error) => {
                return Err(HistoryError::Corrupt(format!(
                    "frame {} is invalid: {error}",
                    index
                )))
            }
        };
        if frame.len > MAX_FRAME_BYTES {
            return Err(HistoryError::Corrupt(format!(
                "frame {} exceeds size bound",
                index
            )));
        }
        let body = serde_json::to_vec(&(
            frame.revision,
            &frame.event_id,
            &frame.previous_hash,
            &frame.transaction,
        ))?;
        if frame.len != body.len() {
            return Err(HistoryError::Corrupt(format!(
                "frame {} length mismatch",
                index
            )));
        }
        if frame.revision != state.revision + 1 || frame.previous_hash != state.hash {
            return Err(HistoryError::Corrupt(format!(
                "frame {} chain mismatch",
                index
            )));
        }
        let expected = sha256_hex(&[frame.previous_hash.as_bytes(), &body].concat());
        if frame.hash != expected {
            return Err(HistoryError::Corrupt(format!(
                "frame {} hash mismatch",
                index
            )));
        }
        state.revision = frame.revision;
        state.hash = frame.hash.clone();
        state
            .transactions
            .push((frame.revision, frame.event_id, frame.transaction));
        offset += segment.len();
    }
    if offset < bytes.len() && recovery.is_none() {
        recovery = Some(RecoveryStatus {
            torn_tail: true,
            ignored_bytes: bytes.len() - offset,
            recovered_revision: state.revision,
        });
    }
    Ok((state, recovery))
}

fn atomic_append(path: &Path, encoded: &[u8]) -> Result<(), HistoryError> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    set_private_file(path)?;
    file.write_all(encoded)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_private(path: &Path) -> Result<(), HistoryError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}
#[cfg(unix)]
fn set_private_file(path: &Path) -> Result<(), HistoryError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}
#[cfg(not(unix))]
fn set_private(_path: &Path) -> Result<(), HistoryError> {
    Ok(())
}
#[cfg(not(unix))]
fn set_private_file(_path: &Path) -> Result<(), HistoryError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn directives_are_strict_and_scoped() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        let first = ledger.ingest(Some("m1"), "Keep the old behavior").unwrap();
        let old = first.created[0].clone();
        let report = ledger
            .ingest(
                Some("m2"),
                &format!(
                    "@forge/correct id={old} anchor=\"old behavior\" :: Keep the new behavior"
                ),
            )
            .unwrap();
        assert_eq!(report.created.len(), 1);
        assert_eq!(
            ledger.record(&old).unwrap().status,
            RequirementStatus::Superseded
        );
        assert_eq!(
            ledger.retrieve(&old).unwrap().authority,
            HistoryAuthority::Superseded
        );
        let replacement = ledger.record(&report.created[0]).unwrap();
        assert_eq!(replacement.supersedes.as_deref(), Some(old.as_str()));
        let malformed = ledger
            .ingest(Some("m3"), "@forge/correct anchor=\"old\" :: bad")
            .unwrap();
        assert_eq!(malformed.diagnostics.len(), 1);
        assert_eq!(ledger.records().count(), 2);
    }

    #[test]
    fn malformed_directive_preserves_mixed_prose_once() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        let report = ledger
            .ingest(
                Some("mixed"),
                "Keep this sentence\n@forge/correct anchor=\"missing id\" :: ignored\nKeep this other sentence",
            )
            .unwrap();
        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.created.len(), 1);
        assert_eq!(
            ledger.record(&report.created[0]).unwrap().text,
            "Keep this sentence\nKeep this other sentence"
        );
        let message = ledger
            .journal()
            .transactions()
            .flat_map(|(_, _, transaction)| transaction.events.iter())
            .find_map(|event| match event {
                EventPayload::Message { message_id, text } if message_id == "mixed" => {
                    Some(text.clone())
                }
                _ => None,
            })
            .unwrap();
        assert!(message.contains("@forge/correct anchor=\"missing id\""));
    }

    #[test]
    fn quoted_and_fenced_markers_are_prose() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        let report = ledger.ingest(Some("m1"), "> @forge/status id=req-nope state=satisfied :: no\n```\n@forge/rename id=req-nope alias=\"x\"\n```").unwrap();
        assert_eq!(report.diagnostics.len(), 0);
        assert_eq!(report.created.len(), 1);
        assert!(ledger
            .record(&report.created[0])
            .unwrap()
            .text
            .contains("@forge/status"));
    }

    #[test]
    fn restart_idempotency_and_scope() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        let report = ledger.ingest(Some("m1"), "Persist this").unwrap();
        assert!(ledger.ingest(Some("m1"), "different").unwrap().duplicate);
        drop(ledger);
        let ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        assert!(ledger.record(&report.created[0]).is_some());
        assert_eq!(
            ledger.record(&report.created[0]).unwrap().revision,
            report.revision
        );
        assert!(RequirementLedger::open(dir.path(), "other", "s")
            .unwrap()
            .records()
            .next()
            .is_none());
    }

    #[test]
    fn status_records_source_message_provenance() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        let created = ledger.ingest(Some("request"), "Do this").unwrap();
        ledger
            .ingest(
                Some("directive"),
                &format!(
                    "@forge/status id={} state=blocked :: waiting on dependency",
                    created.created[0]
                ),
            )
            .unwrap();
        assert_eq!(
            ledger
                .record(&created.created[0])
                .unwrap()
                .status_source_message_id
                .as_deref(),
            Some("directive")
        );
        ledger
            .set_status_verified(
                &created.created[0],
                "verifier-message",
                RequirementStatus::Satisfied,
                "operation-1",
                &["evidence-1".into()],
                "verified",
            )
            .unwrap();
        assert_eq!(
            ledger
                .record(&created.created[0])
                .unwrap()
                .status_source_message_id
                .as_deref(),
            Some("verifier-message")
        );
    }

    #[test]
    fn torn_tail_is_visible_and_interior_corruption_fails_closed() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        ledger.ingest(Some("m1"), "one").unwrap();
        ledger.ingest(Some("m2"), "two").unwrap();
        let path = ledger.journal().path().to_owned();
        drop(ledger);
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(b"{\"torn");
        fs::write(&path, bytes).unwrap();
        let recovered = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        assert!(recovered.journal().recovery().is_some());
        drop(recovered);
        let mut bytes = fs::read(&path).unwrap();
        let newline = bytes.iter().position(|byte| *byte == b'\n').unwrap();
        bytes[newline / 2] = b'x';
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            RequirementLedger::open(dir.path(), "w", "s"),
            Err(HistoryError::Corrupt(_))
        ));
    }

    #[test]
    fn valid_terminal_frame_without_newline_is_recovered_before_append() {
        let dir = tempdir().unwrap();
        let mut ledger = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        ledger.ingest(Some("m1"), "one").unwrap();
        let path = ledger.journal().path().to_owned();
        drop(ledger);
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        fs::write(&path, bytes).unwrap();
        let mut recovered = RequirementLedger::open(dir.path(), "w", "s").unwrap();
        assert!(recovered.journal().recovery().unwrap().torn_tail);
        recovered.ingest(Some("m2"), "two").unwrap();
        assert_eq!(recovered.records().count(), 1);
        assert_eq!(recovered.records().next().unwrap().text, "two");
    }

    #[test]
    fn replay_rejects_unknown_ids_duplicate_records_and_status_contradictions() {
        let cases = [
            vec![
                EventPayload::Message {
                    message_id: "m".into(),
                    text: "raw".into(),
                },
                EventPayload::RequirementRenamed {
                    id: "req-missing".into(),
                    alias: "alias".into(),
                },
            ],
            vec![
                EventPayload::Message {
                    message_id: "m".into(),
                    text: "raw".into(),
                },
                EventPayload::RequirementCorrected {
                    old_id: "req-missing".into(),
                    replacement: test_record("req-new", "m"),
                },
            ],
            vec![
                EventPayload::Message {
                    message_id: "m".into(),
                    text: "raw".into(),
                },
                EventPayload::RequirementCreated(test_record("req-one", "m")),
                EventPayload::RequirementCreated(test_record("req-one", "m")),
            ],
            vec![
                EventPayload::Message {
                    message_id: "m".into(),
                    text: "raw".into(),
                },
                EventPayload::RequirementCreated(test_record("req-one", "m")),
                EventPayload::RequirementStatus {
                    id: "req-one".into(),
                    previous: RequirementStatus::Blocked,
                    new: RequirementStatus::Satisfied,
                    actor: EventActor::UserDirective,
                    source_message_id: "m".into(),
                    operation_id: None,
                    evidence_ids: Vec::new(),
                    reason: "wrong previous".into(),
                },
            ],
        ];
        for (index, events) in cases.into_iter().enumerate() {
            let dir = tempdir().unwrap();
            let mut journal = HistoryJournal::open(dir.path(), "w", &index.to_string()).unwrap();
            journal.append_transaction(events).unwrap();
            drop(journal);
            assert!(matches!(
                RequirementLedger::open(dir.path(), "w", &index.to_string()),
                Err(HistoryError::Corrupt(_))
            ));
        }
    }

    #[test]
    fn one_writer_per_scope_is_enforced_and_released() {
        let dir = tempdir().unwrap();
        let journal = HistoryJournal::open(dir.path(), "w", "s").unwrap();
        assert!(matches!(
            HistoryJournal::open(dir.path(), "w", "s"),
            Err(HistoryError::Locked)
        ));
        drop(journal);
        assert!(HistoryJournal::open(dir.path(), "w", "s").is_ok());
    }

    #[test]
    fn killed_child_releases_persistent_advisory_lock() {
        if let Ok(root) = std::env::var("FORGE_JOURNAL_LOCK_CHILD_ROOT") {
            let _journal = HistoryJournal::open(&root, "w", "s").unwrap();
            fs::write(std::path::Path::new(&root).join("child-ready"), b"ready").unwrap();
            thread::sleep(Duration::from_secs(60));
            return;
        }

        let dir = tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("context_history::tests::killed_child_releases_persistent_advisory_lock")
            .arg("--nocapture")
            .env("FORGE_JOURNAL_LOCK_CHILD_ROOT", &root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let ready = dir.path().join("child-ready");
        for _ in 0..200 {
            if ready.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(ready.exists(), "child never acquired journal lock");
        assert!(matches!(
            HistoryJournal::open(dir.path(), "w", "s"),
            Err(HistoryError::Locked)
        ));
        child.kill().unwrap();
        child.wait().unwrap();

        let reopened = HistoryJournal::open(dir.path(), "w", "s").unwrap();
        let lock_path = reopened.path().parent().unwrap().join(".journal.lock");
        assert!(lock_path.exists(), "persistent lock file was removed");
    }

    fn test_record(id: &str, source_turn: &str) -> RequirementRecord {
        RequirementRecord {
            id: id.into(),
            subject: RequirementSubject::Turn(source_turn.into()),
            source_turn: source_turn.into(),
            text: "text".into(),
            status: RequirementStatus::Open,
            status_source_message_id: None,
            supersedes: None,
            superseded_by: None,
            correction_scope: None,
            aliases: Vec::new(),
            revision: 0,
        }
    }

    #[test]
    fn reference_overflow_is_explicit() {
        let dir = tempdir().unwrap();
        let limits = ActiveLedgerLimits {
            max_full_records: 0,
            max_full_bytes: 0,
            max_full_tokens: 0,
            max_references: 1,
            max_reference_bytes: 1,
        };
        let mut ledger = RequirementLedger::open_with_limits(dir.path(), "w", "s", limits).unwrap();
        ledger.ingest(Some("m1"), "first").unwrap();
        ledger.ingest(Some("m2"), "second").unwrap();
        assert!(matches!(
            ledger.active_view(),
            Err(HistoryError::ActiveOverflow(_))
        ));
    }

    fn message(id: &str, text: &str) -> Message {
        let mut message = Message::user(text);
        message.uuid = Some(id.into());
        message
    }

    #[test]
    fn omitted_group_roundtrips_after_restart_without_reexecution() {
        let dir = tempdir().unwrap();
        let source = vec![
            message("m1", "exact user"),
            message("m2", "exact tool result"),
        ];
        let (reference, expected) = {
            let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
            let reference = index
                .persist_omitted_group("group-1", 7, source.clone(), OmitReason::ToolResultBudget)
                .unwrap();
            let retrieved = index.retrieve_group(&reference.id).unwrap();
            assert_eq!(retrieved.history_ref.source_message_ids, vec!["m1", "m2"]);
            (reference, serde_json::to_vec(&retrieved.messages).unwrap())
        };
        let index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        let retrieved = index.retrieve_omitted_group(&reference.id).unwrap();
        assert_eq!(serde_json::to_vec(&retrieved.messages).unwrap(), expected);
        assert_eq!(retrieved.history_ref.sequence, 7);
        assert_eq!(
            retrieved.history_ref.authority,
            HistoryAuthority::Historical
        );
    }

    #[test]
    fn omitted_group_scope_and_superseded_authority_are_explicit() {
        let dir = tempdir().unwrap();
        let reference = {
            let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
            index
                .persist_omitted_group_with_authority(
                    "group-1",
                    1,
                    vec![message("m1", "old instruction")],
                    OmitReason::Relevance,
                    HistoryAuthority::Superseded,
                )
                .unwrap()
        };
        let wrong_scope = HistoryIndex::open(dir.path(), "other", "session").unwrap();
        assert!(matches!(
            wrong_scope.retrieve_group(&reference.id),
            Err(HistoryError::HistoryUnavailable(_))
        ));
        let reopened = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        assert_eq!(
            reopened
                .retrieve_group(&reference.id)
                .unwrap()
                .history_ref
                .authority,
            HistoryAuthority::Superseded
        );
        assert!(reopened.ledger().records().next().is_none());
    }

    #[test]
    fn omitted_reference_search_is_bounded_and_read_only() {
        let dir = tempdir().unwrap();
        let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        index
            .persist_omitted_group(
                "old-requirement",
                1,
                vec![message("m1", "old")],
                OmitReason::Budget,
            )
            .unwrap();
        index
            .persist_omitted_group(
                "other-requirement",
                2,
                vec![message("m2", "other")],
                OmitReason::Budget,
            )
            .unwrap();
        let refs = index.search_omitted_refs("old", 0, 1000);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].group_id, "old-requirement");
        assert!(index.ledger().records().next().is_none());
    }

    #[test]
    fn missing_and_tampered_group_blobs_fail_closed() {
        let dir = tempdir().unwrap();
        let reference = {
            let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
            index
                .persist_omitted_group(
                    "group-1",
                    1,
                    vec![message("m1", "protected")],
                    OmitReason::Budget,
                )
                .unwrap()
        };
        let group_path = dir
            .path()
            .join("history")
            .join(scope_key("workspace"))
            .join(scope_key("session"))
            .join(GROUP_DIR)
            .join(format!("{}.json", reference.id));
        let mut bytes = fs::read(&group_path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        fs::write(&group_path, bytes).unwrap();
        assert!(matches!(
            HistoryIndex::open(dir.path(), "workspace", "session"),
            Err(HistoryError::Corrupt(_))
        ));
        fs::remove_file(&group_path).unwrap();
        assert!(matches!(
            HistoryIndex::open(dir.path(), "workspace", "session"),
            Err(HistoryError::HistoryUnavailable(_))
        ));
    }

    #[test]
    fn stale_or_corrupt_checkpoint_rebuilds_from_verified_journal() {
        let dir = tempdir().unwrap();
        let reference = {
            let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
            index
                .persist_omitted_group(
                    "group-1",
                    1,
                    vec![message("m1", "checkpointed")],
                    OmitReason::Budget,
                )
                .unwrap()
        };
        let snapshot_path = dir
            .path()
            .join("history")
            .join(scope_key("workspace"))
            .join(scope_key("session"))
            .join(CHECKPOINT_FILE);
        fs::write(&snapshot_path, b"not a snapshot").unwrap();
        let reopened = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        assert!(reopened.checkpoint().rebuilt);
        assert!(reopened.retrieve_group(&reference.id).is_ok());
    }

    #[test]
    fn forged_history_ref_is_rejected_before_group_path_access() {
        let dir = tempdir().unwrap();
        {
            let mut journal = HistoryJournal::open(dir.path(), "workspace", "session").unwrap();
            journal
                .append_transaction(vec![EventPayload::HistoryGroupStored {
                    history_ref: HistoryRef {
                        id: "../../outside".into(),
                        group_id: "../../escape".into(),
                        sequence: 1,
                        source_range: SourceRange {
                            message_id: "m1".into(),
                            revision: 1,
                        },
                        source_message_ids: vec!["m1".into()],
                        reason: OmitReason::Budget,
                        authority: HistoryAuthority::Historical,
                    },
                }])
                .unwrap();
        }
        assert!(matches!(
            HistoryIndex::open(dir.path(), "workspace", "session"),
            Err(HistoryError::Corrupt(_))
        ));
        assert!(!dir.path().join("outside.json").exists());
    }

    #[test]
    fn checkpoint_reports_stale_after_ledger_mutation() {
        let dir = tempdir().unwrap();
        let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        assert!(!index.checkpoint().stale);
        index
            .ledger_mut()
            .ingest(Some("m1"), "new requirement")
            .unwrap();
        assert!(index.checkpoint().stale);
        index.write_checkpoint().unwrap();
        assert!(!index.checkpoint().stale);
    }

    #[test]
    fn runtime_messages_are_durable_but_have_no_user_authority() {
        let dir = tempdir().unwrap();
        {
            let mut index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
            index
                .record_runtime_message(
                    "runtime-1",
                    "continue the previous turn",
                    RuntimeMessageOrigin::Continuation,
                )
                .unwrap();
            assert!(index.ledger().records().next().is_none());
        }
        let mut reopened = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        assert!(reopened.ledger().records().next().is_none());
        assert!(reopened
            .record_runtime_message(
                "runtime-1",
                "duplicate is harmless",
                RuntimeMessageOrigin::Continuation,
            )
            .is_ok());
    }

    #[test]
    fn orphan_blob_is_accounted_and_group_quota_is_visible() {
        let dir = tempdir().unwrap();
        let scope = dir
            .path()
            .join("history")
            .join(scope_key("workspace"))
            .join(scope_key("session"));
        {
            let _index = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
            let groups = scope.join(GROUP_DIR);
            fs::create_dir_all(&groups).unwrap();
            fs::write(groups.join("orphan.json"), vec![b'x'; 17]).unwrap();
        }
        let reopened = HistoryIndex::open(dir.path(), "workspace", "session").unwrap();
        assert_eq!(reopened.orphan_blob_bytes().unwrap(), 17);

        let tiny = tempdir().unwrap();
        let mut index = HistoryIndex::open_with_quota(tiny.path(), "w", "s", 512).unwrap();
        assert!(matches!(
            index.persist_omitted_group(
                "too-large",
                1,
                vec![message("m1", &"x".repeat(2048))],
                OmitReason::Budget,
            ),
            Err(HistoryError::Quota(_))
        ));
        assert!(!tiny
            .path()
            .join("history")
            .join(scope_key("w"))
            .join(scope_key("s"))
            .join(GROUP_DIR)
            .exists());
    }
}
