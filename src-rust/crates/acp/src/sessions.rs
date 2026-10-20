//! Per-session state for the ACP server.

use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol_schema as acp;
use anyhow::{anyhow, bail, Context};
use chrono::{DateTime, Utc};
use claurst_core::types::Message;
use claurst_tools::PendingPermissionStore;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

const SESSION_SCHEMA_VERSION: u32 = 1;
const MAX_SESSION_ID_BYTES: usize = 128;
const MAX_SESSION_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct WorkspaceBinding {
    canonical_path: PathBuf,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl WorkspaceBinding {
    fn capture(path: &std::path::Path) -> anyhow::Result<Self> {
        if !path.is_absolute() {
            bail!("ACP session workspace must be absolute");
        }
        let canonical_path = path
            .canonicalize()
            .context("ACP session workspace is unavailable")?;
        let metadata = std::fs::metadata(&canonical_path)
            .context("ACP session workspace metadata is unavailable")?;
        if !metadata.is_dir() {
            bail!("ACP session workspace is not a directory");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self {
                canonical_path,
                device: metadata.dev(),
                inode: metadata.ino(),
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self { canonical_path })
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedSession {
    schema_version: u32,
    session_id: String,
    workspace: WorkspaceBinding,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    messages: Vec<Message>,
}

/// One ACP session — a logical conversation with its own cwd, transcript,
/// MCP server roster, and cancellation token.
pub struct SessionState {
    pub runtime: parking_lot::Mutex<Option<Arc<crate::runtime::AgentRuntime>>>,
    pub session_id: acp::SessionId,
    pub cwd: PathBuf,
    pub messages: parking_lot::Mutex<Vec<Message>>,
    pub pending_permissions: Arc<parking_lot::Mutex<PendingPermissionStore>>,
    pub file_history: Arc<parking_lot::Mutex<claurst_core::file_history::FileHistory>>,
    pub current_turn: Arc<std::sync::atomic::AtomicUsize>,
    pub prompt_gate: tokio::sync::Mutex<()>,
    cancel_token: parking_lot::Mutex<CancellationToken>,
    workspace: WorkspaceBinding,
    storage_root: PathBuf,
    created_at: DateTime<Utc>,
}

impl SessionState {
    pub fn new(session_id: acp::SessionId, cwd: PathBuf) -> anyhow::Result<Arc<Self>> {
        Self::new_in(session_id, cwd, default_storage_root())
    }

    fn new_in(
        session_id: acp::SessionId,
        cwd: PathBuf,
        storage_root: PathBuf,
    ) -> anyhow::Result<Arc<Self>> {
        validate_session_id(&session_id.0)?;
        let workspace = WorkspaceBinding::capture(&cwd)?;
        let created_at = Utc::now();
        Ok(Arc::new(Self {
            runtime: parking_lot::Mutex::new(None),
            session_id,
            cwd: workspace.canonical_path.clone(),
            messages: parking_lot::Mutex::new(Vec::new()),
            pending_permissions: Arc::new(parking_lot::Mutex::new(
                PendingPermissionStore::default(),
            )),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            prompt_gate: tokio::sync::Mutex::new(()),
            cancel_token: parking_lot::Mutex::new(CancellationToken::new()),
            workspace,
            storage_root,
            created_at,
        }))
    }

    pub async fn load(session_id: acp::SessionId, cwd: PathBuf) -> anyhow::Result<Arc<Self>> {
        Self::load_in(session_id, cwd, default_storage_root()).await
    }

    async fn load_in(
        session_id: acp::SessionId,
        cwd: PathBuf,
        storage_root: PathBuf,
    ) -> anyhow::Result<Arc<Self>> {
        validate_session_id(&session_id.0)?;
        let requested_workspace = WorkspaceBinding::capture(&cwd)?;
        let path = session_file_path(&storage_root, &session_id.0);
        let file = tokio::fs::File::open(&path)
            .await
            .context("ACP session does not exist")?;
        let mut bytes = Vec::new();
        file.take(MAX_SESSION_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .context("ACP session could not be read")?;
        if bytes.len() as u64 > MAX_SESSION_BYTES {
            bail!("ACP session exceeds the storage limit");
        }
        let persisted: PersistedSession =
            serde_json::from_slice(&bytes).context("ACP session record is invalid")?;
        if persisted.schema_version != SESSION_SCHEMA_VERSION
            || persisted.session_id.as_str() != session_id.0.as_ref()
        {
            bail!("ACP session record identity is invalid");
        }
        if persisted.workspace != requested_workspace {
            bail!("ACP session workspace does not match");
        }
        Ok(Arc::new(Self {
            runtime: parking_lot::Mutex::new(None),
            session_id,
            cwd: requested_workspace.canonical_path.clone(),
            messages: parking_lot::Mutex::new(persisted.messages),
            pending_permissions: Arc::new(parking_lot::Mutex::new(
                PendingPermissionStore::default(),
            )),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            prompt_gate: tokio::sync::Mutex::new(()),
            cancel_token: parking_lot::Mutex::new(CancellationToken::new()),
            workspace: requested_workspace,
            storage_root,
            created_at: persisted.created_at,
        }))
    }

    pub async fn persist(&self) -> anyhow::Result<()> {
        let record = PersistedSession {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: self.session_id.0.to_string(),
            workspace: self.workspace.clone(),
            created_at: self.created_at,
            updated_at: Utc::now(),
            messages: self.messages.lock().clone(),
        };
        let bytes =
            serde_json::to_vec_pretty(&record).context("ACP session could not be encoded")?;
        if bytes.len() as u64 > MAX_SESSION_BYTES {
            bail!("ACP session exceeds the storage limit");
        }
        tokio::fs::create_dir_all(&self.storage_root)
            .await
            .context("ACP session directory could not be created")?;
        set_owner_only_dir_perms(&self.storage_root)
            .context("ACP session directory permissions could not be restricted")?;
        let final_path = session_file_path(&self.storage_root, &self.session_id.0);
        let temporary_path = self.storage_root.join(format!(
            ".{}.{}.tmp",
            self.session_id.0,
            uuid::Uuid::new_v4()
        ));
        let mut options = tokio::fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            options.mode(0o600);
        }
        let mut file = options
            .open(&temporary_path)
            .await
            .context("ACP session temporary file could not be created")?;
        if let Err(error) = file.write_all(&bytes).await {
            let _ = tokio::fs::remove_file(&temporary_path).await;
            return Err(error).context("ACP session could not be written");
        }
        if let Err(error) = file.sync_all().await {
            let _ = tokio::fs::remove_file(&temporary_path).await;
            return Err(error).context("ACP session could not be synchronized");
        }
        drop(file);
        if let Err(error) = tokio::fs::rename(&temporary_path, &final_path).await {
            let _ = tokio::fs::remove_file(&temporary_path).await;
            return Err(error).context("ACP session could not be committed");
        }
        set_owner_only_file_perms(&final_path)
            .context("ACP session file permissions could not be restricted")?;
        Ok(())
    }

    pub fn cancel_current_turn(&self) {
        self.cancel_token.lock().cancel();
    }

    pub fn cancel_token_for_turn(&self) -> CancellationToken {
        let mut token = self.cancel_token.lock();
        if token.is_cancelled() {
            *token = CancellationToken::new();
        }
        token.clone()
    }
}

fn default_storage_root() -> PathBuf {
    claurst_core::config::Settings::config_dir().join("acp-sessions")
}

fn validate_session_id(session_id: &str) -> anyhow::Result<()> {
    if session_id.is_empty()
        || session_id.len() > MAX_SESSION_ID_BYTES
        || !session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(anyhow!("invalid ACP session id"));
    }
    Ok(())
}

fn session_file_path(storage_root: &std::path::Path, session_id: &str) -> PathBuf {
    storage_root.join(format!("{session_id}.json"))
}

fn set_owner_only_dir_perms(path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn set_owner_only_file_perms(path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

/// Map of active sessions and in-flight load reservations keyed by session id.
#[derive(Default)]
pub struct SessionRegistry {
    inner: DashMap<acp::SessionId, Option<Arc<SessionState>>>,
}

/// A pending load occupies the same registry slot as its eventual session.
/// Dropping a failed or cancelled load makes that id available again.
pub(crate) struct SessionLoadReservation {
    registry: Arc<SessionRegistry>,
    session_id: acp::SessionId,
    activated: bool,
}

impl SessionLoadReservation {
    pub(crate) fn activate(mut self, state: Arc<SessionState>) -> anyhow::Result<()> {
        if state.session_id != self.session_id {
            bail!("ACP session reservation identity mismatch");
        }
        self.registry.insert(state);
        self.activated = true;
        Ok(())
    }
}

impl Drop for SessionLoadReservation {
    fn drop(&mut self) {
        if !self.activated {
            self.registry
                .inner
                .remove_if(&self.session_id, |_, state| state.is_none());
        }
    }
}

impl SessionRegistry {
    pub fn runtimes(&self) -> Vec<Arc<crate::runtime::AgentRuntime>> {
        self.inner
            .iter()
            .filter_map(|entry| entry.value().as_ref()?.runtime.lock().clone())
            .collect()
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, state: Arc<SessionState>) {
        self.inner.insert(state.session_id.clone(), Some(state));
    }

    pub(crate) fn reserve_load(
        self: &Arc<Self>,
        id: &acp::SessionId,
    ) -> Option<SessionLoadReservation> {
        match self.inner.entry(id.clone()) {
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(None);
                Some(SessionLoadReservation {
                    registry: Arc::clone(self),
                    session_id: id.clone(),
                    activated: false,
                })
            }
            dashmap::mapref::entry::Entry::Occupied(_) => None,
        }
    }

    pub fn get(&self, id: &acp::SessionId) -> Option<Arc<SessionState>> {
        self.inner.get(id).and_then(|r| r.value().clone())
    }

    pub fn remove(&self, id: &acp::SessionId) -> Option<Arc<SessionState>> {
        self.inner
            .remove_if(id, |_, state| state.is_some())
            .and_then(|(_, state)| state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("forge-acp-{label}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn new_session_starts_empty_with_a_fresh_cancel_token() {
        let root = temp_root("new");
        let id = acp::SessionId::new("session-1");
        let cwd = root.join("workspace");
        std::fs::create_dir(&cwd).unwrap();
        let state = SessionState::new(id.clone(), cwd.clone()).unwrap();

        assert_eq!(state.session_id, id);
        assert_eq!(state.cwd, cwd.canonicalize().unwrap());
        assert!(state.messages.lock().is_empty());
        assert_eq!(
            state.current_turn.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert!(!state.cancel_token_for_turn().is_cancelled());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_insert_get_remove_round_trip() {
        let registry = SessionRegistry::new();
        let id = acp::SessionId::new("session-2");
        let state = SessionState::new(id.clone(), PathBuf::from("/tmp")).unwrap();

        assert!(registry.get(&id).is_none());

        registry.insert(Arc::clone(&state));
        let fetched = registry
            .get(&id)
            .expect("session should be present after insert");
        assert!(Arc::ptr_eq(&fetched, &state));

        let removed = registry
            .remove(&id)
            .expect("session should be present to remove");
        assert!(Arc::ptr_eq(&removed, &state));
        assert!(registry.get(&id).is_none());
    }

    #[test]
    fn registry_remove_unknown_id_returns_none() {
        let registry = SessionRegistry::new();
        let id = acp::SessionId::new("missing");
        assert!(registry.remove(&id).is_none());
    }

    #[test]
    fn load_reservation_excludes_pending_and_active_owners() {
        let registry = Arc::new(SessionRegistry::new());
        let id = acp::SessionId::new("acp-reserved");
        let reservation = registry.reserve_load(&id).unwrap();
        assert!(registry.reserve_load(&id).is_none());
        assert!(registry.get(&id).is_none());
        assert!(registry.remove(&id).is_none());
        assert!(registry.reserve_load(&id).is_none());
        drop(reservation);

        let reservation = registry.reserve_load(&id).unwrap();
        let state = SessionState::new(id.clone(), PathBuf::from("/tmp")).unwrap();
        reservation.activate(Arc::clone(&state)).unwrap();
        assert!(Arc::ptr_eq(&registry.get(&id).unwrap(), &state));
        assert!(registry.reserve_load(&id).is_none());
        registry.remove(&id).unwrap();
        assert!(registry.reserve_load(&id).is_some());
    }

    #[tokio::test]
    async fn durable_session_round_trip_restores_transcript_with_fresh_runtime_state() {
        let root = temp_root("round-trip");
        let workspace = root.join("workspace");
        let store = root.join("store");
        std::fs::create_dir(&workspace).unwrap();
        let id = acp::SessionId::new("acp-round-trip");

        let state = SessionState::new_in(id.clone(), workspace.clone(), store.clone()).unwrap();
        {
            let mut messages = state.messages.lock();
            messages.push(Message::user("remember this"));
            messages.push(Message::assistant("remembered"));
        }
        state.cancel_current_turn();
        state.persist().await.unwrap();

        let loaded = SessionState::load_in(id, workspace.clone(), store.clone())
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&*loaded.messages.lock()).unwrap(),
            serde_json::json!([
                {"role":"user","content":"remember this"},
                {"role":"assistant","content":"remembered"}
            ])
        );
        assert_eq!(loaded.cwd, workspace.canonicalize().unwrap());
        assert!(!loaded.cancel_token_for_turn().is_cancelled());
        {
            let pending = loaded.pending_permissions.lock();
            assert!(pending.queue.is_empty());
            assert!(pending.waiting.is_empty());
        }
        assert_eq!(
            loaded
                .current_turn
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(session_file_path(&store, "acp-round-trip"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn load_rejects_invalid_id_and_foreign_workspace() {
        let root = temp_root("scope");
        let workspace = root.join("workspace");
        let foreign = root.join("foreign");
        let store = root.join("store");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&foreign).unwrap();

        assert!(SessionState::new_in(
            acp::SessionId::new("../escape"),
            workspace.clone(),
            store.clone()
        )
        .is_err());

        let id = acp::SessionId::new("acp-scope");
        let state = SessionState::new_in(id.clone(), workspace.clone(), store.clone()).unwrap();
        state.persist().await.unwrap();
        assert!(SessionState::load_in(id, foreign, store).await.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn load_rejects_same_path_workspace_replacement() {
        let root = temp_root("replacement");
        let workspace = root.join("workspace");
        let old_workspace = root.join("workspace-old");
        let store = root.join("store");
        std::fs::create_dir(&workspace).unwrap();
        let id = acp::SessionId::new("acp-replacement");
        let state = SessionState::new_in(id.clone(), workspace.clone(), store.clone()).unwrap();
        state.persist().await.unwrap();

        std::fs::rename(&workspace, &old_workspace).unwrap();
        std::fs::create_dir(&workspace).unwrap();
        assert!(SessionState::load_in(id, workspace, store).await.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn load_rejects_corrupt_record_without_activating_session() {
        let root = temp_root("corrupt");
        let workspace = root.join("workspace");
        let store = root.join("store");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&store).unwrap();
        std::fs::write(session_file_path(&store, "acp-corrupt"), b"not json").unwrap();

        assert!(
            SessionState::load_in(acp::SessionId::new("acp-corrupt"), workspace, store)
                .await
                .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn persist_atomically_replaces_the_previous_transcript() {
        let root = temp_root("replace-record");
        let workspace = root.join("workspace");
        let store = root.join("store");
        std::fs::create_dir(&workspace).unwrap();
        let id = acp::SessionId::new("acp-replace-record");
        let state = SessionState::new_in(id.clone(), workspace.clone(), store.clone()).unwrap();
        state.messages.lock().push(Message::user("first"));
        state.persist().await.unwrap();

        *state.messages.lock() = vec![Message::user("second")];
        state.persist().await.unwrap();
        let loaded = SessionState::load_in(id, workspace, store.clone())
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&*loaded.messages.lock()).unwrap(),
            serde_json::json!([{"role":"user","content":"second"}])
        );
        let leftovers = std::fs::read_dir(&store)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0);
        std::fs::remove_dir_all(root).unwrap();
    }
}
