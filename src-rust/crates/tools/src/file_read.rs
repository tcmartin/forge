// FileRead tool: bounded text windows with scoped in-flight freshness.

use crate::{PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult};
use async_trait::async_trait;
use base64::Engine;
use claurst_core::media::MediaLimits;
use claurst_core::types::{ContentBlock, ImageSource};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(not(unix))]
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncReadExt;
use tokio::sync::watch;
#[cfg(test)]
use tokio::sync::Notify;
use tracing::debug;

const MAX_SELECTED_BYTES: usize = 64 * 1024;
const MAX_LINES: usize = 2_000;
const MAX_SCAN_BYTES: u64 = 8 * 1024 * 1024;
const READ_DEADLINE: Duration = Duration::from_secs(5);
const MAX_FLIGHTS: usize = 128;
const READ_CHUNK_BYTES: usize = 8 * 1024;
const MAX_UTF8_BYTES: usize = 4;
const MAX_IMAGE_BYTES: usize = 3_750_000;
const IMAGE_PROBE_BYTES: usize = 16;
const MAX_FILE_PATH_BYTES: usize = 4096;
const MAX_FILE_PATH_CHARS: usize = 4096;

#[derive(Debug, Clone)]
enum ReadPayload {
    Text(String),
    Image(Arc<ToolResult>),
}

type FlightResult = Result<ReadPayload, String>;

type FlightSender = watch::Sender<Option<FlightResult>>;
type FlightReceiver = watch::Receiver<Option<FlightResult>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileIdentity {
    canonical: PathBuf,
    dev: u64,
    ino: u64,
    len: u64,
    mtime_ns: i128,
    ctime_ns: i128,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReadKey {
    scope_workspace: PathBuf,
    scope_session: String,
    file: FileIdentity,
    generation: String,
    epoch: u64,
    offset: usize,
    limit: usize,
    byte_offset: usize,
}

struct ReadFlight {
    sender: FlightSender,
    started: Instant,
    terminal: AtomicBool,
}

struct FlightGuard {
    key: ReadKey,
    flight: Arc<ReadFlight>,
    complete: bool,
}

enum FlightRegistration {
    Owner(FlightGuard),
    Follower(FlightReceiver),
    Bypass,
}

static ACTIVE_FLIGHTS: Lazy<Mutex<HashMap<ReadKey, Arc<ReadFlight>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
static TEST_READ_COUNTS: Lazy<Mutex<HashMap<PathBuf, usize>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
static TEST_SCAN_GATES: Lazy<Mutex<HashMap<PathBuf, Arc<Notify>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
static TEST_VALIDATION_GATES: Lazy<Mutex<HashMap<PathBuf, Arc<Notify>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
#[cfg(test)]
static TEST_VALIDATION_REACHED: Lazy<Mutex<HashMap<PathBuf, bool>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
struct ImageValidationGate {
    reached: AtomicBool,
    finished: AtomicBool,
    released: std::sync::Mutex<bool>,
    release_signal: std::sync::Condvar,
}

#[cfg(test)]
impl ImageValidationGate {
    fn new() -> Self {
        Self {
            reached: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            released: std::sync::Mutex::new(false),
            release_signal: std::sync::Condvar::new(),
        }
    }

    fn wait_for_release(&self) {
        self.reached.store(true, Ordering::SeqCst);
        let released = self.released.lock().unwrap();
        let (_released, _timeout) = self
            .release_signal
            .wait_timeout_while(released, Duration::from_secs(8), |released| !*released)
            .unwrap();
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.release_signal.notify_all();
    }
}

#[cfg(test)]
static TEST_IMAGE_VALIDATION_GATES: Lazy<Mutex<HashMap<PathBuf, Arc<ImageValidationGate>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug)]
enum ReadFailure {
    InvalidUtf8,
    ScanLimit,
    Cancelled,
    Deadline,
    Stale,
    Io(String),
}

impl fmt::Display for ReadFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => {
                formatter.write_str("File appears to be binary and cannot be displayed as text")
            }
            Self::ScanLimit => formatter.write_str(
                "Read exceeded the bounded scan window before reaching the requested line",
            ),
            Self::Cancelled => formatter.write_str("File read cancelled before completion"),
            Self::Deadline => formatter.write_str("File read exceeded its deadline"),
            Self::Stale => {
                formatter.write_str("File changed during read; no fresh observation was returned")
            }
            Self::Io(error) => formatter.write_str(error),
        }
    }
}

impl From<std::io::Error> for ReadFailure {
    fn from(error: std::io::Error) -> Self {
        if error.kind() == std::io::ErrorKind::InvalidData {
            Self::InvalidUtf8
        } else {
            Self::Io(error.to_string())
        }
    }
}

#[derive(Debug, Deserialize)]
struct FileReadInput {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    byte_offset: Option<usize>,
    #[serde(default)]
    refresh: bool,
}

#[derive(Debug)]
struct ScanResult {
    text: String,
    has_more: bool,
    next_offset: Option<usize>,
    next_byte_offset: Option<usize>,
}

impl FlightGuard {
    fn complete(&mut self, result: FlightResult) {
        if self.complete {
            return;
        }
        self.complete = true;
        if self
            .flight
            .terminal
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.flight.sender.send(Some(result));
        }
        remove_flight_if_current(&self.key, &self.flight);
    }
}

impl Drop for FlightGuard {
    fn drop(&mut self) {
        if !self.complete {
            self.complete(Err(ReadFailure::Cancelled.to_string()));
        }
    }
}

fn remove_flight_if_current(key: &ReadKey, flight: &Arc<ReadFlight>) {
    let mut flights = ACTIVE_FLIGHTS.lock();
    if flights
        .get(key)
        .is_some_and(|current| Arc::ptr_eq(current, flight))
    {
        flights.remove(key);
    }
}

fn register_flight(key: ReadKey, refresh: bool) -> FlightRegistration {
    register_flight_in(&mut ACTIVE_FLIGHTS.lock(), key, refresh)
}

fn register_flight_in(
    flights: &mut HashMap<ReadKey, Arc<ReadFlight>>,
    key: ReadKey,
    refresh: bool,
) -> FlightRegistration {
    let expired: Vec<ReadKey> = flights
        .iter()
        .filter(|(_, flight)| flight.started.elapsed() >= READ_DEADLINE)
        .map(|(key, _)| key.clone())
        .collect();
    for expired_key in expired {
        if let Some(flight) = flights.remove(&expired_key) {
            if flight
                .terminal
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let _ = flight
                    .sender
                    .send(Some(Err(ReadFailure::Deadline.to_string())));
            }
        }
    }

    if refresh {
        return FlightRegistration::Bypass;
    }

    if let Some(flight) = flights.get(&key) {
        return FlightRegistration::Follower(flight.sender.subscribe());
    }
    if flights.len() >= MAX_FLIGHTS {
        return FlightRegistration::Bypass;
    }

    let (sender, _receiver) = watch::channel(None);
    let flight = Arc::new(ReadFlight {
        sender,
        started: Instant::now(),
        terminal: AtomicBool::new(false),
    });
    flights.insert(key.clone(), Arc::clone(&flight));
    FlightRegistration::Owner(FlightGuard {
        key,
        flight,
        complete: false,
    })
}

async fn await_flight(
    mut receiver: FlightReceiver,
    ctx: &ToolContext,
    key: &ReadKey,
    requested_path: &Path,
) -> FlightResult {
    let deadline = tokio::time::sleep(READ_DEADLINE);
    tokio::pin!(deadline);
    loop {
        let completed = receiver.borrow().clone();
        if let Some(result) = completed {
            verify_current_key(ctx, key, requested_path).await?;
            return result;
        }
        tokio::select! {
            changed = receiver.changed() => {
                if changed.is_err() {
                    return Err(ReadFailure::Cancelled.to_string());
                }
            }
            _ = ctx.cancel_token.cancelled() => {
                return Err(ReadFailure::Cancelled.to_string());
            }
            _ = &mut deadline => {
                return Err(ReadFailure::Deadline.to_string());
            }
        }
    }
}

#[cfg(not(unix))]
fn unix_time_ns(time: Option<SystemTime>) -> i128 {
    time.and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as i128)
        .unwrap_or_default()
}

fn identity_from_metadata(canonical: PathBuf, metadata: &std::fs::Metadata) -> FileIdentity {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        FileIdentity {
            canonical,
            dev: metadata.dev(),
            ino: metadata.ino(),
            len: metadata.len(),
            mtime_ns: (metadata.mtime() as i128)
                .saturating_mul(1_000_000_000)
                .saturating_add(metadata.mtime_nsec() as i128),
            ctime_ns: (metadata.ctime() as i128)
                .saturating_mul(1_000_000_000)
                .saturating_add(metadata.ctime_nsec() as i128),
        }
    }
    #[cfg(not(unix))]
    {
        FileIdentity {
            canonical,
            dev: 0,
            ino: 0,
            len: metadata.len(),
            mtime_ns: unix_time_ns(metadata.modified().ok()),
            ctime_ns: unix_time_ns(metadata.created().ok()),
        }
    }
}

#[cfg(test)]
fn file_identity(path: &Path) -> Result<FileIdentity, ReadFailure> {
    let canonical = path
        .canonicalize()
        .map_err(|error| ReadFailure::Io(format!("Failed to resolve file: {error}")))?;
    let metadata = std::fs::metadata(&canonical)
        .map_err(|error| ReadFailure::Io(format!("Failed to inspect file: {error}")))?;
    if !metadata.is_file() {
        return Err(ReadFailure::Io(
            "Read target is not a regular file".to_string(),
        ));
    }
    Ok(identity_from_metadata(canonical, &metadata))
}

async fn file_identity_async(path: &Path) -> Result<FileIdentity, ReadFailure> {
    let canonical = tokio::fs::canonicalize(path)
        .await
        .map_err(|error| ReadFailure::Io(format!("Failed to resolve file: {error}")))?;
    let metadata = tokio::fs::metadata(&canonical)
        .await
        .map_err(|error| ReadFailure::Io(format!("Failed to inspect file: {error}")))?;
    if !metadata.is_file() {
        return Err(ReadFailure::Io(
            "Read target is not a regular file".to_string(),
        ));
    }
    Ok(identity_from_metadata(canonical, &metadata))
}

fn key_for_stamp(
    stamp: crate::mutation::MutationStamp,
    identity: FileIdentity,
    input: &FileReadInput,
) -> ReadKey {
    ReadKey {
        scope_workspace: stamp.scope.workspace,
        scope_session: stamp.scope.session,
        file: identity,
        generation: stamp.generation,
        epoch: stamp.epoch,
        offset: input.offset.unwrap_or(0).max(1),
        limit: input.limit.unwrap_or(2000).min(MAX_LINES),
        byte_offset: input.byte_offset.unwrap_or(0),
    }
}

#[cfg(test)]
fn key_for(ctx: &ToolContext, identity: FileIdentity, input: &FileReadInput) -> ReadKey {
    key_for_stamp(crate::mutation::current_stamp(ctx), identity, input)
}

async fn current_stamp_async(
    ctx: &ToolContext,
) -> Result<crate::mutation::MutationStamp, ReadFailure> {
    let context = ctx.clone();
    tokio::task::spawn_blocking(move || crate::mutation::current_stamp(&context))
        .await
        .map_err(|error| ReadFailure::Io(format!("Mutation stamp task failed: {error}")))
}

async fn key_for_async(
    ctx: &ToolContext,
    identity: FileIdentity,
    input: &FileReadInput,
) -> Result<ReadKey, ReadFailure> {
    let stamp = current_stamp_async(ctx).await?;
    Ok(key_for_stamp(stamp, identity, input))
}

fn stamp_matches_key(stamp: &crate::mutation::MutationStamp, key: &ReadKey) -> bool {
    stamp.generation == key.generation
        && stamp.epoch == key.epoch
        && stamp.scope.workspace == key.scope_workspace
        && stamp.scope.session == key.scope_session
}

async fn verify_current_key(
    ctx: &ToolContext,
    key: &ReadKey,
    requested_path: &Path,
) -> Result<(), String> {
    let before = current_stamp_async(ctx)
        .await
        .map_err(|error| error.to_string())?;
    if !stamp_matches_key(&before, key) {
        return Err(ReadFailure::Stale.to_string());
    }
    let identity = file_identity_async(requested_path)
        .await
        .map_err(|error| error.to_string())?;
    if identity != key.file {
        return Err(ReadFailure::Stale.to_string());
    }
    #[cfg(test)]
    {
        let gate = TEST_VALIDATION_GATES.lock().get(requested_path).cloned();
        if let Some(gate) = gate {
            TEST_VALIDATION_REACHED
                .lock()
                .insert(requested_path.to_path_buf(), true);
            gate.notified().await;
        }
    }
    let after = current_stamp_async(ctx)
        .await
        .map_err(|error| error.to_string())?;
    if !stamp_matches_key(&after, key) {
        return Err(ReadFailure::Stale.to_string());
    }
    let final_identity = file_identity_async(requested_path)
        .await
        .map_err(|error| error.to_string())?;
    if final_identity != key.file {
        return Err(ReadFailure::Stale.to_string());
    }
    let final_stamp = current_stamp_async(ctx)
        .await
        .map_err(|error| error.to_string())?;
    if !stamp_matches_key(&final_stamp, key) {
        return Err(ReadFailure::Stale.to_string());
    }
    Ok(())
}

struct Scanner {
    file: tokio::fs::File,
    buffer: [u8; READ_CHUNK_BYTES],
    position: usize,
    available: usize,
    pending: Option<u8>,
    eof: bool,
    scanned: u64,
    bytes_since_yield: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineByte {
    Content(u8),
    End,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LimitedEnd {
    MoreContent,
    End,
    Eof,
}

async fn open_verified_file(expected: &FileIdentity) -> Result<tokio::fs::File, ReadFailure> {
    let file = {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let path = expected.canonical.clone();
            let file = tokio::task::spawn_blocking(move || {
                let mut options = std::fs::OpenOptions::new();
                options
                    .read(true)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
                options.open(path)
            })
            .await
            .map_err(|error| ReadFailure::Io(format!("File open task failed: {error}")))??;
            tokio::fs::File::from_std(file)
        }
        #[cfg(not(unix))]
        {
            tokio::fs::File::open(&expected.canonical).await?
        }
    };
    let metadata = file.metadata().await?;
    if !metadata.is_file() {
        return Err(ReadFailure::Stale);
    }
    let opened = identity_from_metadata(expected.canonical.clone(), &metadata);
    if &opened != expected {
        return Err(ReadFailure::Stale);
    }
    Ok(file)
}

impl Scanner {
    async fn open(expected: &FileIdentity) -> Result<Self, ReadFailure> {
        let file = open_verified_file(expected).await?;
        Ok(Self {
            file,
            buffer: [0; READ_CHUNK_BYTES],
            position: 0,
            available: 0,
            pending: None,
            eof: false,
            scanned: 0,
            bytes_since_yield: 0,
        })
    }

    async fn verify_identity(&self, expected: &FileIdentity) -> Result<(), ReadFailure> {
        let metadata = self.file.metadata().await?;
        if !metadata.is_file() {
            return Err(ReadFailure::Stale);
        }
        let observed = identity_from_metadata(expected.canonical.clone(), &metadata);
        if &observed != expected {
            return Err(ReadFailure::Stale);
        }
        Ok(())
    }

    async fn next_raw(&mut self) -> Result<Option<u8>, ReadFailure> {
        if let Some(byte) = self.pending.take() {
            self.bytes_since_yield = self.bytes_since_yield.saturating_add(1);
            if self.bytes_since_yield >= READ_CHUNK_BYTES {
                self.bytes_since_yield = 0;
                tokio::task::yield_now().await;
            }
            return Ok(Some(byte));
        }
        loop {
            if self.position < self.available {
                let byte = self.buffer[self.position];
                self.position += 1;
                self.bytes_since_yield = self.bytes_since_yield.saturating_add(1);
                if self.bytes_since_yield >= READ_CHUNK_BYTES {
                    self.bytes_since_yield = 0;
                    tokio::task::yield_now().await;
                }
                return Ok(Some(byte));
            }
            if self.eof {
                return Ok(None);
            }
            let read = self.file.read(&mut self.buffer).await?;
            if read == 0 {
                self.eof = true;
                return Ok(None);
            }
            self.scanned = self
                .scanned
                .checked_add(read as u64)
                .ok_or(ReadFailure::ScanLimit)?;
            if self.scanned > MAX_SCAN_BYTES {
                return Err(ReadFailure::ScanLimit);
            }
            self.position = 0;
            self.available = read;
        }
    }

    async fn next_line_byte(&mut self) -> Result<LineByte, ReadFailure> {
        match self.next_raw().await? {
            None => Ok(LineByte::Eof),
            Some(b'\n') => Ok(LineByte::End),
            Some(b'\r') => match self.next_raw().await? {
                Some(b'\n') => Ok(LineByte::End),
                Some(byte) => {
                    self.pending = Some(byte);
                    Ok(LineByte::Content(b'\r'))
                }
                None => Ok(LineByte::Content(b'\r')),
            },
            Some(byte) => Ok(LineByte::Content(byte)),
        }
    }

    async fn has_next_line(&mut self) -> Result<bool, ReadFailure> {
        Ok(!matches!(self.next_line_byte().await?, LineByte::Eof))
    }
}

fn render_lines(lines: &[(usize, String)], width: usize) -> String {
    let mut output = String::new();
    for (line_number, line) in lines {
        output.push_str(&format!(
            "{:>width$}\t{}\n",
            line_number,
            line,
            width = width
        ));
    }
    output
}

fn utf8_prefix_len(candidate: &[u8]) -> Result<usize, ReadFailure> {
    match std::str::from_utf8(candidate) {
        Ok(_) => Ok(candidate.len()),
        Err(error) if error.error_len().is_some() => Err(ReadFailure::InvalidUtf8),
        Err(_) => {
            for length in (0..candidate.len()).rev() {
                if std::str::from_utf8(&candidate[..length]).is_ok() {
                    return Ok(length);
                }
            }
            Err(ReadFailure::InvalidUtf8)
        }
    }
}

async fn limited_fragment(
    scanner: &mut Scanner,
    fragment: Vec<u8>,
) -> Result<(Vec<u8>, LimitedEnd), ReadFailure> {
    let initial_len = fragment.len();
    let mut candidate = fragment;
    let mut ended = None;
    loop {
        match std::str::from_utf8(&candidate) {
            Ok(_) => break,
            Err(error)
                if error.error_len().is_none()
                    && candidate.len().saturating_sub(initial_len) < MAX_UTF8_BYTES =>
            {
                match scanner.next_line_byte().await? {
                    LineByte::Content(byte) => candidate.push(byte),
                    LineByte::End => {
                        ended = Some(LimitedEnd::End);
                        break;
                    }
                    LineByte::Eof => {
                        ended = Some(LimitedEnd::Eof);
                        break;
                    }
                }
            }
            Err(_) => return Err(ReadFailure::InvalidUtf8),
        }
    }
    if ended.is_some() && std::str::from_utf8(&candidate).is_err() {
        return Err(ReadFailure::InvalidUtf8);
    }
    let boundary = utf8_prefix_len(&candidate[..initial_len])?;
    if boundary < candidate.len() {
        return Ok((candidate[..boundary].to_vec(), LimitedEnd::MoreContent));
    }
    let end = match ended {
        Some(end) => end,
        None => match scanner.next_line_byte().await? {
            LineByte::Content(_) => LimitedEnd::MoreContent,
            LineByte::End => LimitedEnd::End,
            LineByte::Eof => LimitedEnd::Eof,
        },
    };
    Ok((candidate[..boundary].to_vec(), end))
}

async fn scan_text(
    expected: &FileIdentity,
    input: &FileReadInput,
) -> Result<ScanResult, ReadFailure> {
    #[cfg(test)]
    {
        let canonical = expected.canonical.clone();
        {
            let mut counts = TEST_READ_COUNTS.lock();
            *counts.entry(canonical.clone()).or_default() += 1;
        }
        let gate = TEST_SCAN_GATES.lock().get(&canonical).cloned();
        if let Some(gate) = gate {
            gate.notified().await;
        }
    }

    let mut scanner = Scanner::open(expected).await?;
    let result = scan_text_inner(&mut scanner, input).await;
    if result.is_ok() {
        scanner.verify_identity(expected).await?;
    }
    result
}

async fn scan_text_inner(
    scanner: &mut Scanner,
    input: &FileReadInput,
) -> Result<ScanResult, ReadFailure> {
    let requested_line = input.offset.unwrap_or(0).max(1);
    let requested_limit = input.limit.unwrap_or(2000).min(MAX_LINES);
    let requested_byte_offset = input.byte_offset.unwrap_or(0);
    if requested_limit == 0 {
        return Ok(ScanResult {
            text: String::new(),
            has_more: false,
            next_offset: None,
            next_byte_offset: None,
        });
    }

    let planned_width = requested_line
        .saturating_add(requested_limit.saturating_sub(1))
        .to_string()
        .len();
    let mut line_number = 1usize;
    let mut selected_lines: Vec<(usize, String)> = Vec::new();
    let mut selected_bytes = 0usize;
    let mut first_line = true;

    while line_number < requested_line {
        let mut skipped_line_has_content = false;
        loop {
            match scanner.next_line_byte().await? {
                LineByte::Content(_) => skipped_line_has_content = true,
                LineByte::End => {
                    line_number = line_number.saturating_add(1);
                    break;
                }
                LineByte::Eof => {
                    let total_lines = if skipped_line_has_content {
                        line_number
                    } else {
                        line_number.saturating_sub(1)
                    };
                    return Err(ReadFailure::Io(format!(
                        "Offset {} exceeds total line count {}",
                        input.offset.unwrap_or(0),
                        total_lines
                    )));
                }
            }
        }
    }

    loop {
        let byte_offset = if first_line { requested_byte_offset } else { 0 };
        let mut line_position = 0usize;
        let mut prefix_utf8 = Vec::new();
        let mut fragment = Vec::new();
        let fragment_limit = MAX_SELECTED_BYTES.saturating_sub(selected_bytes);
        let line_end;
        let mut saw_line = false;
        loop {
            match scanner.next_line_byte().await? {
                LineByte::Content(byte) => {
                    saw_line = true;
                    if line_position < byte_offset {
                        if byte == 0 {
                            return Err(ReadFailure::Io(
                                "NUL bytes are not valid UTF-8 text".to_string(),
                            ));
                        }
                        prefix_utf8.push(byte);
                        match std::str::from_utf8(&prefix_utf8) {
                            Ok(_) => prefix_utf8.clear(),
                            Err(error) if error.error_len().is_some() => {
                                return Err(ReadFailure::InvalidUtf8);
                            }
                            Err(_) if prefix_utf8.len() >= MAX_UTF8_BYTES => {
                                return Err(ReadFailure::InvalidUtf8);
                            }
                            Err(_) => {}
                        }
                        line_position = line_position.saturating_add(1);
                        continue;
                    }
                    line_position = line_position.saturating_add(1);
                    if fragment_limit == 0 {
                        return Ok(ScanResult {
                            text: render_lines(&selected_lines, planned_width),
                            has_more: true,
                            next_offset: Some(line_number),
                            next_byte_offset: Some(byte_offset),
                        });
                    }
                    if line_position - 1 == byte_offset
                        && ((byte & 0xc0) == 0x80 || !prefix_utf8.is_empty())
                    {
                        return Err(ReadFailure::Io(format!(
                            "byte_offset {} is not a UTF-8 boundary within line {}",
                            byte_offset, line_number
                        )));
                    }
                    if byte == 0 && line_position > byte_offset {
                        return Err(ReadFailure::Io(
                            "NUL bytes are not valid UTF-8 text".to_string(),
                        ));
                    }
                    fragment.push(byte);
                    if fragment.len() >= fragment_limit {
                        let (prefix, end) = limited_fragment(scanner, fragment).await?;
                        if !prefix.is_empty() {
                            let line = std::str::from_utf8(&prefix)
                                .map_err(|_| ReadFailure::InvalidUtf8)?
                                .to_string();
                            selected_lines.push((line_number, line));
                        }
                        let next_byte_offset = byte_offset.saturating_add(prefix.len());
                        let page = match end {
                            LimitedEnd::MoreContent => ScanResult {
                                text: render_lines(&selected_lines, planned_width),
                                has_more: true,
                                next_offset: Some(line_number),
                                next_byte_offset: Some(next_byte_offset),
                            },
                            LimitedEnd::End => {
                                if scanner.has_next_line().await? {
                                    ScanResult {
                                        text: render_lines(&selected_lines, planned_width),
                                        has_more: true,
                                        next_offset: Some(line_number.saturating_add(1)),
                                        next_byte_offset: Some(0),
                                    }
                                } else {
                                    ScanResult {
                                        text: render_lines(
                                            &selected_lines,
                                            line_number.to_string().len(),
                                        ),
                                        has_more: false,
                                        next_offset: None,
                                        next_byte_offset: None,
                                    }
                                }
                            }
                            LimitedEnd::Eof => ScanResult {
                                text: render_lines(&selected_lines, line_number.to_string().len()),
                                has_more: false,
                                next_offset: None,
                                next_byte_offset: None,
                            },
                        };
                        return Ok(page);
                    }
                }
                LineByte::End => {
                    line_end = Some(LimitedEnd::End);
                    break;
                }
                LineByte::Eof => {
                    if !saw_line {
                        if selected_lines.is_empty() {
                            if line_number > 1 {
                                return Err(ReadFailure::Io(format!(
                                    "Offset {} exceeds total line count {}",
                                    input.offset.unwrap_or(0),
                                    line_number.saturating_sub(1)
                                )));
                            }
                            return Ok(ScanResult {
                                text: String::new(),
                                has_more: false,
                                next_offset: None,
                                next_byte_offset: None,
                            });
                        }
                        return Ok(ScanResult {
                            text: render_lines(&selected_lines, planned_width),
                            has_more: false,
                            next_offset: None,
                            next_byte_offset: None,
                        });
                    }
                    line_end = Some(LimitedEnd::Eof);
                    break;
                }
            }
        }

        if !prefix_utf8.is_empty() {
            return Err(ReadFailure::InvalidUtf8);
        }
        if byte_offset > line_position {
            return Err(ReadFailure::Io(format!(
                "byte_offset {} exceeds line {} length {}",
                byte_offset, line_number, line_position
            )));
        }
        selected_bytes = selected_bytes.saturating_add(fragment.len());
        let line = std::str::from_utf8(&fragment)
            .map_err(|_| ReadFailure::InvalidUtf8)?
            .to_string();
        selected_lines.push((line_number, line));
        first_line = false;
        if selected_lines.len() >= requested_limit {
            let has_next =
                matches!(line_end, Some(LimitedEnd::End)) && scanner.has_next_line().await?;
            return Ok(ScanResult {
                text: render_lines(&selected_lines, planned_width),
                has_more: has_next,
                next_offset: has_next.then_some(line_number.saturating_add(1)),
                next_byte_offset: has_next.then_some(0),
            });
        }
        if matches!(line_end, Some(LimitedEnd::Eof)) {
            return Ok(ScanResult {
                text: render_lines(&selected_lines, line_number.to_string().len()),
                has_more: false,
                next_offset: None,
                next_byte_offset: None,
            });
        }
        line_number = line_number.saturating_add(1);
    }
}

async fn scan_with_deadline(
    expected: &FileIdentity,
    input: &FileReadInput,
    ctx: &ToolContext,
) -> Result<ScanResult, ReadFailure> {
    let scan = scan_text(expected, input);
    tokio::pin!(scan);
    let deadline = tokio::time::sleep(READ_DEADLINE);
    tokio::pin!(deadline);
    tokio::select! {
        result = &mut scan => result,
        _ = ctx.cancel_token.cancelled() => Err(ReadFailure::Cancelled),
        _ = &mut deadline => Err(ReadFailure::Deadline),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectedFileKind {
    Text,
    Image(&'static str),
    Unsupported(&'static str),
}

fn extension_is_media(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" | "svg" | "ico" | "pdf"
    )
}

fn detect_file_kind(prefix: &[u8], path: &Path) -> DetectedFileKind {
    if prefix.starts_with(b"\x89PNG\r\n\x1a\n") {
        return DetectedFileKind::Image("image/png");
    }
    if prefix.starts_with(b"\xff\xd8") {
        return DetectedFileKind::Image("image/jpeg");
    }
    if prefix.starts_with(b"GIF87a") || prefix.starts_with(b"GIF89a") {
        return DetectedFileKind::Unsupported("GIF image reading is unavailable");
    }
    if prefix.starts_with(b"RIFF") && prefix.len() >= 12 && &prefix[8..12] == b"WEBP" {
        return DetectedFileKind::Unsupported("WebP image reading is unavailable");
    }
    if prefix.starts_with(b"%PDF-") {
        return DetectedFileKind::Unsupported("PDF reading is unavailable");
    }
    if extension_is_media(path) {
        return DetectedFileKind::Unsupported(
            "This media format is unavailable; only PNG and JPEG are supported",
        );
    }
    DetectedFileKind::Text
}

async fn probe_file_kind(
    expected: &FileIdentity,
    path: &Path,
) -> Result<DetectedFileKind, ReadFailure> {
    let mut file = open_verified_file(expected).await?;
    let mut prefix = [0u8; IMAGE_PROBE_BYTES];
    let read = file.read(&mut prefix).await?;
    file.metadata()
        .await
        .map_err(ReadFailure::from)
        .and_then(|metadata| {
            if !metadata.is_file()
                || identity_from_metadata(expected.canonical.clone(), &metadata) != *expected
            {
                Err(ReadFailure::Stale)
            } else {
                Ok(())
            }
        })?;
    Ok(detect_file_kind(&prefix[..read], path))
}

async fn read_image_payload(
    expected: &FileIdentity,
    media_type: &'static str,
    ctx: &ToolContext,
) -> Result<Arc<ToolResult>, ReadFailure> {
    let mut file = open_verified_file(expected).await?;
    let capacity = usize::try_from(expected.len)
        .unwrap_or(MAX_IMAGE_BYTES)
        .min(MAX_IMAGE_BYTES);
    let mut raw = Vec::with_capacity(capacity);
    let mut buffer = [0u8; READ_CHUNK_BYTES];
    loop {
        if ctx.cancel_token.is_cancelled() {
            return Err(ReadFailure::Cancelled);
        }
        let remaining = MAX_IMAGE_BYTES.saturating_sub(raw.len());
        if remaining == 0 {
            let mut extra = [0u8; 1];
            if file.read(&mut extra).await? != 0 {
                return Err(ReadFailure::ScanLimit);
            }
            break;
        }
        let chunk_len = remaining.min(buffer.len());
        let read = file.read(&mut buffer[..chunk_len]).await?;
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..read]);
        tokio::task::yield_now().await;
    }
    let metadata = file.metadata().await?;
    if !metadata.is_file()
        || identity_from_metadata(expected.canonical.clone(), &metadata) != *expected
    {
        return Err(ReadFailure::Stale);
    }
    if raw.is_empty() {
        return Err(ReadFailure::Io("Image file is empty".to_string()));
    }

    let limits = MediaLimits::default();
    let media_type = media_type.to_string();
    #[cfg(test)]
    let validation_gate = TEST_IMAGE_VALIDATION_GATES
        .lock()
        .get(&expected.canonical)
        .cloned();
    let validation = tokio::task::spawn_blocking(move || {
        #[cfg(test)]
        if let Some(gate) = &validation_gate {
            gate.wait_for_release();
        }
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
        let blocks = vec![ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: Some(media_type),
                data: Some(encoded),
                url: None,
            },
        }];
        let mut result = ToolResult::success("validated image payload")
            .with_typed_content(blocks, &limits)
            .map_err(|error| error.to_string())?;
        let (media_type, dimensions, encoded_bytes, decoded_bytes) = {
            let manifest = result
                .typed_manifest()
                .ok_or_else(|| "image validation did not produce a manifest".to_string())?;
            let item = manifest.items().first();
            let dimensions = item
                .and_then(|item| item.width.zip(item.height))
                .map(|(width, height)| format!("{width}x{height}"))
                .unwrap_or_else(|| "unknown dimensions".to_string());
            (
                item.map(|item| item.media_type.clone())
                    .unwrap_or_else(|| "image".to_string()),
                dimensions,
                manifest.total_encoded_bytes(),
                manifest.total_decoded_bytes(),
            )
        };
        result.content = format!(
            "Read image {media_type} ({dimensions}, encoded_bytes={encoded_bytes}, decoded_bytes={decoded_bytes})"
        );
        #[cfg(test)]
        if let Some(gate) = &validation_gate {
            gate.finished.store(true, Ordering::SeqCst);
        }
        Ok::<ToolResult, String>(result)
    });
    tokio::pin!(validation);
    let deadline = tokio::time::sleep(READ_DEADLINE);
    tokio::pin!(deadline);
    let result = tokio::select! {
        result = &mut validation => result
            .map_err(|error| ReadFailure::Io(format!("Image validation task failed: {error}")))?
            .map_err(ReadFailure::Io)?,
        _ = ctx.cancel_token.cancelled() => return Err(ReadFailure::Cancelled),
        _ = &mut deadline => return Err(ReadFailure::Deadline),
    };
    Ok(Arc::new(result))
}

async fn read_image_with_deadline(
    expected: &FileIdentity,
    media_type: &'static str,
    ctx: &ToolContext,
) -> Result<Arc<ToolResult>, ReadFailure> {
    let read = read_image_payload(expected, media_type, ctx);
    tokio::pin!(read);
    let deadline = tokio::time::sleep(READ_DEADLINE);
    tokio::pin!(deadline);
    tokio::select! {
        result = &mut read => result,
        _ = ctx.cancel_token.cancelled() => Err(ReadFailure::Cancelled),
        _ = &mut deadline => Err(ReadFailure::Deadline),
    }
}

fn result_to_tool_result(result: FlightResult) -> ToolResult {
    match result {
        Err(error) => ToolResult::error(error),
        Ok(ReadPayload::Text(text)) => ToolResult::success(text),
        Ok(ReadPayload::Image(image)) => image.as_ref().clone(),
    }
}

async fn read_once(
    expected: &FileIdentity,
    input: &FileReadInput,
    ctx: &ToolContext,
    key: &ReadKey,
    requested_path: &Path,
) -> FlightResult {
    let kind = match probe_file_kind(expected, requested_path).await {
        Ok(kind) => kind,
        Err(error) => return Err(error.to_string()),
    };
    let payload = match kind {
        DetectedFileKind::Image(media_type) => {
            if input.offset.is_some() || input.limit.is_some() || input.byte_offset.is_some() {
                return Err(
                    "Image reads do not support text-only offset, limit, or byte_offset parameters"
                        .to_string(),
                );
            }
            match read_image_with_deadline(expected, media_type, ctx).await {
                Ok(image) => ReadPayload::Image(image),
                Err(error) => return Err(error.to_string()),
            }
        }
        DetectedFileKind::Unsupported(message) => return Err(message.to_string()),
        DetectedFileKind::Text => {
            let result = match scan_with_deadline(expected, input, ctx).await {
                Ok(scan) => scan,
                Err(error) => return Err(error.to_string()),
            };
            if expected.len == 0 {
                ReadPayload::Text(format!(
                    "[File {} exists but is empty]",
                    requested_path.display()
                ))
            } else {
                let mut text = result.text;
                if result.has_more {
                    text.push_str(&format!(
                        "\n... (has_more=true next_offset={} next_byte_offset={})\n",
                        result.next_offset.unwrap_or(0),
                        result.next_byte_offset.unwrap_or(0)
                    ));
                }
                ReadPayload::Text(text)
            }
        }
    };
    if ctx.cancel_token.is_cancelled() {
        return Err(ReadFailure::Cancelled.to_string());
    }
    verify_current_key(ctx, key, requested_path).await?;
    Ok(payload)
}

pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        claurst_core::constants::TOOL_NAME_FILE_READ
    }

    fn description(&self) -> &str {
        "Reads bounded UTF-8 text or validated PNG/JPEG media from a local file. Text uses offset/limit and byte_offset; image requests do not support text paging."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Observation
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read; maximum 4096 UTF-8 bytes and 4096 characters",
                    "maxLength": MAX_FILE_PATH_CHARS,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Text-only 1-based line number to start reading from; zero means the first line."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Text-only number of lines to read, capped at 2000."
                },
                "byte_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Text-only UTF-8 byte offset within the starting visible line."
                },
                "refresh": {
                    "type": "boolean",
                    "description": "Read independently of simultaneous requests."
                }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: FileReadInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(error) => return ToolResult::error(format!("Invalid input: {error}")),
        };
        if params.file_path.len() > MAX_FILE_PATH_BYTES
            || params.file_path.chars().count() > MAX_FILE_PATH_CHARS
        {
            return ToolResult::error("File path exceeds the maximum allowed length");
        }
        let path = ctx.resolve_path(&params.file_path);
        debug!("Reading file");

        if let Err(error) = ctx.check_permission_for_path(
            self.name(),
            &format!("Read {}", path.display()),
            path.clone(),
            true,
        ) {
            return ToolResult::error(error.to_string());
        }

        let read = execute_read_inner(params, path, ctx);
        tokio::pin!(read);
        let deadline = tokio::time::sleep(READ_DEADLINE);
        tokio::pin!(deadline);
        tokio::select! {
            result = &mut read => result,
            _ = ctx.cancel_token.cancelled() => {
                ToolResult::error(ReadFailure::Cancelled.to_string())
            }
            _ = &mut deadline => {
                ToolResult::error(ReadFailure::Deadline.to_string())
            }
        }
    }
}

async fn execute_read_inner(params: FileReadInput, path: PathBuf, ctx: &ToolContext) -> ToolResult {
    let metadata = match tokio::fs::metadata(&path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return ToolResult::error(format!("File not found: {}", path.display()));
        }
        Err(error) => {
            return ToolResult::error(format!("Failed to inspect file: {error}"));
        }
    };
    if metadata.is_dir() {
        return ToolResult::error(format!(
            "{} is a directory, not a file. Use Bash with `ls` to list directory contents.",
            path.display()
        ));
    }
    if !metadata.is_file() {
        return ToolResult::error("Read target is not a regular file".to_string());
    }

    let identity = match file_identity_async(&path).await {
        Ok(identity) => identity,
        Err(error) => return ToolResult::error(error.to_string()),
    };
    let key = match key_for_async(ctx, identity.clone(), &params).await {
        Ok(key) => key,
        Err(error) => return ToolResult::error(error.to_string()),
    };
    match register_flight(key.clone(), params.refresh) {
        FlightRegistration::Follower(receiver) => {
            result_to_tool_result(await_flight(receiver, ctx, &key, &path).await)
        }
        FlightRegistration::Bypass => {
            result_to_tool_result(read_once(&identity, &params, ctx, &key, &path).await)
        }
        FlightRegistration::Owner(mut guard) => {
            let result = read_once(&identity, &params, ctx, &key, &path).await;
            guard.complete(result.clone());
            result_to_tool_result(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    fn canonical(path: &std::path::Path) -> PathBuf {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    }

    fn reset_target(path: &std::path::Path) {
        let canonical = canonical(path);
        ACTIVE_FLIGHTS
            .lock()
            .retain(|key, _| key.file.canonical != canonical);
        TEST_READ_COUNTS.lock().remove(&canonical);
        TEST_SCAN_GATES.lock().remove(&canonical);
        TEST_IMAGE_VALIDATION_GATES.lock().remove(&canonical);
    }

    fn set_gate(path: &std::path::Path, gate: Arc<Notify>) {
        TEST_SCAN_GATES.lock().insert(canonical(path), gate);
    }

    fn read_count(path: &std::path::Path) -> usize {
        TEST_READ_COUNTS
            .lock()
            .get(&canonical(path))
            .copied()
            .unwrap_or(0)
    }

    async fn wait_for_flag(flag: &AtomicBool, message: &str) {
        tokio::time::timeout(Duration::from_secs(7), async {
            while !flag.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect(message);
    }

    fn payload(content: &str) -> String {
        content
            .lines()
            .filter_map(|line| line.split_once('\t').map(|(_, value)| value))
            .collect::<Vec<_>>()
            .join("\n")
    }

    struct DenyHandler;

    impl claurst_core::permissions::PermissionHandler for DenyHandler {
        fn check_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Ask {
                reason: "denied in test".to_string(),
            }
        }

        fn request_permission(
            &self,
            _request: &claurst_core::permissions::PermissionRequest,
        ) -> claurst_core::permissions::PermissionDecision {
            claurst_core::permissions::PermissionDecision::Ask {
                reason: "denied in test".to_string(),
            }
        }
    }

    #[tokio::test]
    async fn bounded_text_preserves_lines_and_crlf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        std::fs::write(&path, b"one\r\ntwo\r\nthree").unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = FileReadTool
            .execute(json!({"file_path": path, "offset": 2, "limit": 2}), &ctx)
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("2\ttwo"));
        assert!(result.content.contains("3\tthree"));
    }

    #[tokio::test]
    async fn zero_limit_and_extreme_offset_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        std::fs::write(&path, b"one\n").unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let empty = FileReadTool
            .execute(json!({"file_path": path, "limit": 0}), &ctx)
            .await;
        assert!(!empty.is_error);
        assert!(empty.content.is_empty());
        let error = FileReadTool
            .execute(
                json!({"file_path": dir.path().join("sample.txt"), "offset": usize::MAX}),
                &ctx,
            )
            .await;
        assert!(error.is_error);
    }

    #[tokio::test]
    async fn file_path_length_is_bounded_before_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let exact = FileReadTool
            .execute(json!({"file_path": "p".repeat(MAX_FILE_PATH_CHARS)}), &ctx)
            .await;
        assert!(!exact
            .content
            .contains("File path exceeds the maximum allowed length"));

        let oversized = FileReadTool
            .execute(
                json!({"file_path": "p".repeat(MAX_FILE_PATH_BYTES + 1)}),
                &ctx,
            )
            .await;
        assert_eq!(
            oversized.content,
            "File path exceeds the maximum allowed length"
        );

        let multibyte = FileReadTool
            .execute(
                json!({"file_path": "\u{1f980}".repeat(MAX_FILE_PATH_CHARS)}),
                &ctx,
            )
            .await;
        assert_eq!(
            multibyte.content,
            "File path exceeds the maximum allowed length"
        );

        let schema = FileReadTool.input_schema();
        assert_eq!(
            schema["properties"]["file_path"]["maxLength"],
            MAX_FILE_PATH_CHARS
        );
    }

    #[tokio::test]
    async fn invalid_utf8_and_media_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("binary.txt");
        std::fs::write(&binary, [0xff, 0xfe]).unwrap();
        reset_target(&binary);
        let image = dir.path().join("image.png");
        std::fs::write(&image, b"not pixels").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        assert!(
            FileReadTool
                .execute(json!({"file_path": binary}), &ctx)
                .await
                .is_error
        );
        assert!(
            FileReadTool
                .execute(json!({"file_path": image}), &ctx)
                .await
                .is_error
        );
    }

    const TINY_JPEG_BASE64: &str = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AYf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AYf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ap//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IYf/2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z";

    #[tokio::test]
    async fn bounded_png_and_jpeg_reads_preserve_typed_content() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("image.bin");
        let png_bytes = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H',
            b'D', b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, b'I', b'D', b'A', b'T', 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&png, &png_bytes).unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = FileReadTool.execute(json!({"file_path": png}), &ctx).await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("image/png") && result.content.contains("1x1"));
        let blocks = result.typed_blocks().expect("PNG payload must stay typed");
        assert!(
            matches!(blocks.first(), Some(ContentBlock::Image { source }) if source.media_type.as_deref() == Some("image/png"))
        );
        assert_eq!(
            result.typed_manifest().unwrap().items()[0].raw_bytes,
            png_bytes.len()
        );

        let jpeg = dir.path().join("image.data");
        let jpeg_bytes = base64::engine::general_purpose::STANDARD
            .decode(TINY_JPEG_BASE64)
            .unwrap();
        std::fs::write(&jpeg, &jpeg_bytes).unwrap();
        let result = FileReadTool.execute(json!({"file_path": jpeg}), &ctx).await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("image/jpeg") && result.content.contains("1x1"));
        assert!(result.typed_blocks().is_some());
    }

    #[tokio::test]
    async fn image_replacement_during_final_validation_never_delivers_typed_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stale.png");
        let bytes = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H',
            b'D', b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, b'I', b'D', b'A', b'T', 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&path, &bytes).unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        TEST_VALIDATION_GATES
            .lock()
            .insert(path.clone(), gate.clone());
        TEST_VALIDATION_REACHED.lock().remove(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let reader = start_read(path.clone(), ctx.clone());
        tokio::time::timeout(Duration::from_secs(5), async {
            while !TEST_VALIDATION_REACHED.lock().contains_key(&path) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("image validation did not reach the final freshness gate");
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let admission = crate::mutation::admit(&ctx, &crate::FileWriteTool).unwrap();
        drop(admission);
        gate.notify_waiters();
        let result = reader.await.unwrap();
        assert!(result.is_error && result.typed_blocks().is_none());
        TEST_VALIDATION_GATES.lock().remove(&path);
        TEST_VALIDATION_REACHED.lock().remove(&path);
    }

    #[tokio::test]
    async fn external_image_replacement_during_final_validation_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("external-replacement.jpg");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(TINY_JPEG_BASE64)
            .unwrap();
        std::fs::write(&path, &bytes).unwrap();
        reset_target(&path);
        let original = file_identity(&path).unwrap();
        let gate = Arc::new(Notify::new());
        TEST_VALIDATION_GATES
            .lock()
            .insert(path.clone(), gate.clone());
        TEST_VALIDATION_REACHED.lock().remove(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let stamp = crate::mutation::current_stamp(&ctx);
        let reader = start_read(path.clone(), ctx.clone());
        tokio::time::timeout(Duration::from_secs(2), async {
            while !TEST_VALIDATION_REACHED.lock().contains_key(&path) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("image did not reach final validation");
        let replacement = dir.path().join("replacement.jpg");
        std::fs::write(&replacement, &bytes).unwrap();
        #[cfg(not(unix))]
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        let replaced = file_identity(&path).unwrap();
        assert_eq!(original.len, replaced.len);
        assert_ne!(original, replaced);
        gate.notify_waiters();
        let result = reader.await.unwrap();
        TEST_VALIDATION_GATES.lock().remove(&path);
        TEST_VALIDATION_REACHED.lock().remove(&path);
        assert_eq!(stamp, crate::mutation::current_stamp(&ctx));
        assert!(
            result.is_error,
            "stale image was delivered: {}",
            result.content
        );
        assert!(result.content.contains("changed"), "{}", result.content);
        assert!(result.typed_blocks().is_none());
        assert!(!ACTIVE_FLIGHTS.lock().keys().any(|key| key.file == original));
        let fresh = FileReadTool.execute(json!({"file_path": path}), &ctx).await;
        assert!(!fresh.is_error, "{}", fresh.content);
        assert!(fresh.typed_blocks().is_some());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn external_image_parent_symlink_swap_during_final_validation_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("image-root");
        let parked = dir.path().join("original-root");
        let foreign = dir.path().join("foreign-root");
        std::fs::create_dir(&parent).unwrap();
        std::fs::create_dir(&foreign).unwrap();
        let path = parent.join("image.jpg");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(TINY_JPEG_BASE64)
            .unwrap();
        std::fs::write(&path, &bytes).unwrap();
        std::fs::write(foreign.join("image.jpg"), &bytes).unwrap();
        reset_target(&path);
        let original = file_identity(&path).unwrap();
        let gate = Arc::new(Notify::new());
        TEST_VALIDATION_GATES
            .lock()
            .insert(path.clone(), gate.clone());
        TEST_VALIDATION_REACHED.lock().remove(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let stamp = crate::mutation::current_stamp(&ctx);
        let reader = start_read(path.clone(), ctx.clone());
        tokio::time::timeout(Duration::from_secs(2), async {
            while !TEST_VALIDATION_REACHED.lock().contains_key(&path) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("image did not reach final validation");
        std::fs::rename(&parent, &parked).unwrap();
        std::os::unix::fs::symlink(&foreign, &parent).unwrap();
        assert_ne!(original, file_identity(&path).unwrap());
        gate.notify_waiters();
        let result = reader.await.unwrap();
        TEST_VALIDATION_GATES.lock().remove(&path);
        TEST_VALIDATION_REACHED.lock().remove(&path);
        assert_eq!(stamp, crate::mutation::current_stamp(&ctx));
        assert!(
            result.is_error,
            "retargeted image was delivered: {}",
            result.content
        );
        assert!(result.content.contains("changed"), "{}", result.content);
        assert!(result.typed_blocks().is_none());
        assert!(!ACTIVE_FLIGHTS.lock().keys().any(|key| key.file == original));
        let fresh = FileReadTool.execute(json!({"file_path": path}), &ctx).await;
        assert!(!fresh.is_error, "{}", fresh.content);
        assert!(fresh.typed_blocks().is_some());
    }

    #[tokio::test]
    async fn cancelling_image_validation_drops_late_typed_result_and_flight() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cancelled-validation.jpg");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(TINY_JPEG_BASE64)
            .unwrap();
        std::fs::write(&path, bytes).unwrap();
        reset_target(&path);
        let canonical_path = canonical(&path);
        let gate = Arc::new(ImageValidationGate::new());
        TEST_IMAGE_VALIDATION_GATES
            .lock()
            .insert(canonical_path.clone(), gate.clone());
        let ctx = allow_all_context(dir.path().to_path_buf());
        let cancel = ctx.cancel_token.clone();
        let reader = start_read(path.clone(), ctx);
        wait_for_flag(
            &gate.reached,
            "image validation worker did not reach its gate",
        )
        .await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), reader)
            .await
            .expect("cancelled image read did not return")
            .unwrap();
        assert!(result.is_error && result.content.contains("cancelled"));
        assert!(result.typed_blocks().is_none());
        assert!(!gate.finished.load(Ordering::SeqCst));
        assert!(!ACTIVE_FLIGHTS
            .lock()
            .keys()
            .any(|key| key.file.canonical == canonical_path));
        gate.release();
        wait_for_flag(
            &gate.finished,
            "late image validation worker did not finish",
        )
        .await;
        TEST_IMAGE_VALIDATION_GATES.lock().remove(&canonical_path);
        let fresh_ctx = allow_all_context(dir.path().to_path_buf());
        let fresh = FileReadTool
            .execute(json!({"file_path": path}), &fresh_ctx)
            .await;
        assert!(!fresh.is_error, "{}", fresh.content);
        assert!(fresh.typed_blocks().is_some());
    }

    #[tokio::test]
    async fn image_validation_deadline_drops_late_typed_result_and_flight() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deadline-validation.jpg");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(TINY_JPEG_BASE64)
            .unwrap();
        std::fs::write(&path, bytes).unwrap();
        reset_target(&path);
        let canonical_path = canonical(&path);
        let gate = Arc::new(ImageValidationGate::new());
        TEST_IMAGE_VALIDATION_GATES
            .lock()
            .insert(canonical_path.clone(), gate.clone());
        let ctx = allow_all_context(dir.path().to_path_buf());
        let reader = start_read(path.clone(), ctx);
        wait_for_flag(
            &gate.reached,
            "image validation worker did not reach its gate",
        )
        .await;
        let result = tokio::time::timeout(Duration::from_secs(7), reader)
            .await
            .expect("image deadline did not return")
            .unwrap();
        assert!(result.is_error && result.content.contains("deadline"));
        assert!(result.typed_blocks().is_none());
        assert!(!gate.finished.load(Ordering::SeqCst));
        assert!(!ACTIVE_FLIGHTS
            .lock()
            .keys()
            .any(|key| key.file.canonical == canonical_path));
        gate.release();
        wait_for_flag(
            &gate.finished,
            "late image validation worker did not finish",
        )
        .await;
        TEST_IMAGE_VALIDATION_GATES.lock().remove(&canonical_path);
        let fresh_ctx = allow_all_context(dir.path().to_path_buf());
        let fresh = FileReadTool
            .execute(json!({"file_path": path}), &fresh_ctx)
            .await;
        assert!(!fresh.is_error, "{}", fresh.content);
        assert!(fresh.typed_blocks().is_some());
    }

    #[tokio::test]
    async fn image_cap_accepts_exact_eof_and_rejects_one_extra_byte() {
        use std::io::{Seek, SeekFrom, Write};

        let dir = tempfile::tempdir().unwrap();
        let exact = dir.path().join("exact-cap.png");
        let mut file = std::fs::File::create(&exact).unwrap();
        file.set_len(MAX_IMAGE_BYTES as u64).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"\x89PNG\r\n\x1a\n").unwrap();
        file.flush().unwrap();
        let identity = file_identity(&exact).unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let exact_result = read_image_payload(&identity, "image/png", &ctx).await;
        assert!(!matches!(exact_result, Err(ReadFailure::ScanLimit)));

        let oversized = dir.path().join("over-cap.png");
        let mut file = std::fs::File::create(&oversized).unwrap();
        file.set_len((MAX_IMAGE_BYTES + 1) as u64).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"\x89PNG\r\n\x1a\n").unwrap();
        file.flush().unwrap();
        let identity = file_identity(&oversized).unwrap();
        let oversized_result = read_image_payload(&identity, "image/png", &ctx).await;
        assert!(matches!(oversized_result, Err(ReadFailure::ScanLimit)));
    }

    #[tokio::test]
    async fn image_magic_rejects_paging_and_unsupported_media_without_text_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("image.png");
        std::fs::write(&png, b"plain text").unwrap();
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = FileReadTool.execute(json!({"file_path": png}), &ctx).await;
        assert!(result.is_error);
        assert!(result.content.contains("media format") || result.content.contains("PNG"));

        let gif = dir.path().join("image.dat");
        std::fs::write(&gif, b"GIF89a").unwrap();
        let result = FileReadTool.execute(json!({"file_path": gif}), &ctx).await;
        assert!(result.is_error && result.content.contains("GIF"));

        let valid_png = dir.path().join("valid.png");
        let bytes = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H',
            b'D', b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, b'I', b'D', b'A', b'T', 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&valid_png, bytes).unwrap();
        let result = FileReadTool
            .execute(json!({"file_path": valid_png, "limit": 1}), &ctx)
            .await;
        assert!(result.is_error && result.content.contains("text-only"));
    }

    #[tokio::test]
    async fn huge_line_can_be_reconstructed_with_byte_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.txt");
        let line = format!("{}TAIL", "x".repeat(MAX_SELECTED_BYTES + 512));
        std::fs::write(&path, line.as_bytes()).unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let head = FileReadTool
            .execute(json!({"file_path": path, "limit": 1}), &ctx)
            .await;
        assert!(!head.is_error, "{}", head.content);
        assert!(head.content.contains("has_more=true"));
        let next = head
            .content
            .split("next_byte_offset=")
            .nth(1)
            .and_then(|value| value.split(')').next())
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap();
        let tail = FileReadTool
            .execute(
                json!({"file_path": dir.path().join("huge.txt"), "limit": 1, "byte_offset": next}),
                &ctx,
            )
            .await;
        assert!(!tail.is_error, "{}", tail.content);
        assert!(tail.content.contains("TAIL"));
        let reconstructed = payload(&head.content) + payload(&tail.content).as_str();
        assert_eq!(reconstructed, line);

        let unicode = dir.path().join("unicode.txt");
        std::fs::write(&unicode, "éclair\n").unwrap();
        let invalid = FileReadTool
            .execute(json!({"file_path": unicode, "byte_offset": 1}), &ctx)
            .await;
        assert!(invalid.is_error, "mid-codepoint offsets must fail closed");
    }

    #[tokio::test]
    async fn unicode_boundary_trailing_newline_and_nul_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let unicode = dir.path().join("unicode-boundary.txt");
        let line = format!("{}é", "a".repeat(MAX_SELECTED_BYTES - 1));
        std::fs::write(&unicode, line.as_bytes()).unwrap();
        reset_target(&unicode);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let page = FileReadTool
            .execute(json!({"file_path": unicode, "limit": 1}), &ctx)
            .await;
        assert!(!page.is_error, "{}", page.content);
        let head = payload(&page.content);
        assert!(head.len() <= MAX_SELECTED_BYTES);
        assert!(page.content.contains("has_more=true"));
        let tail = FileReadTool
            .execute(
                json!({
                    "file_path": dir.path().join("unicode-boundary.txt"), "limit": 1,
                    "byte_offset": head.len(),
                }),
                &ctx,
            )
            .await;
        assert!(!tail.is_error, "{}", tail.content);
        assert_eq!(head + &payload(&tail.content), line);
        assert!(!tail.content.contains("has_more=true"));

        let trailing = dir.path().join("trailing-newline.txt");
        std::fs::write(&trailing, b"one\n").unwrap();
        reset_target(&trailing);
        let page = FileReadTool
            .execute(json!({"file_path": trailing, "limit": 2}), &ctx)
            .await;
        assert!(!page.is_error, "{}", page.content);
        assert!(page.content.contains("1\tone"));
        assert!(!page.content.contains("2\t"));
        let missing = FileReadTool
            .execute(
                json!({"file_path": dir.path().join("trailing-newline.txt"), "offset": 2}),
                &ctx,
            )
            .await;
        assert!(missing.is_error);

        let two_lines = dir.path().join("two-lines.txt");
        std::fs::write(&two_lines, b"one\n\n").unwrap();
        reset_target(&two_lines);
        let page = FileReadTool
            .execute(json!({"file_path": two_lines, "limit": 2}), &ctx)
            .await;
        assert!(!page.is_error, "{}", page.content);
        assert!(page.content.contains("1\tone"));
        assert!(page.content.contains("2\t\n"));

        let nul = dir.path().join("nul.txt");
        std::fs::write(&nul, b"first\nskip\0\n").unwrap();
        reset_target(&nul);
        let first = FileReadTool
            .execute(json!({"file_path": nul, "limit": 1}), &ctx)
            .await;
        assert!(
            !first.is_error,
            "unselected NUL must not poison a bounded read"
        );
        let selected = FileReadTool
            .execute(
                json!({"file_path": dir.path().join("nul.txt"), "offset": 2}),
                &ctx,
            )
            .await;
        assert!(selected.is_error, "selected NUL must fail closed");

        for (index, (bytes, offset)) in [
            (vec![0xff, b'o', b'k'], 1),
            (vec![0xe9, 0x9b], 2),
            (vec![0xe9, b'a'], 1),
        ]
        .into_iter()
        .enumerate()
        {
            let malformed = dir.path().join(format!("malformed-prefix-{index}.txt"));
            std::fs::write(&malformed, bytes).unwrap();
            reset_target(&malformed);
            let result = FileReadTool
                .execute(json!({"file_path": malformed, "byte_offset": offset}), &ctx)
                .await;
            assert!(
                result.is_error,
                "invalid skipped UTF-8 prefix must fail closed"
            );
        }
    }

    #[tokio::test]
    async fn selected_byte_cap_reconstructs_multiple_lines_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multiline-cap.txt");
        let expected = format!("{}\n{}", "a".repeat(40_000), "b".repeat(40_000));
        std::fs::write(&path, expected.as_bytes()).unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let head = FileReadTool
            .execute(json!({"file_path": path, "limit": 2000}), &ctx)
            .await;
        assert!(!head.is_error, "{}", head.content);
        assert!(head.content.contains("has_more=true"));
        let next_offset = head
            .content
            .split("next_offset=")
            .nth(1)
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap();
        let next_byte_offset = head
            .content
            .split("next_byte_offset=")
            .nth(1)
            .and_then(|value| value.split(')').next())
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap();
        let tail = FileReadTool
            .execute(
                json!({
                    "file_path": dir.path().join("multiline-cap.txt"),
                    "offset": next_offset,
                    "byte_offset": next_byte_offset,
                    "limit": 2000,
                }),
                &ctx,
            )
            .await;
        assert!(!tail.is_error, "{}", tail.content);
        assert_eq!(
            payload(&head.content) + payload(&tail.content).as_str(),
            expected
        );
    }

    #[tokio::test]
    async fn first_page_does_not_scan_a_64mib_line_or_huge_tail() {
        use std::io::{Seek, SeekFrom, Write};

        let dir = tempfile::tempdir().unwrap();
        let single = dir.path().join("single-line-64m.txt");
        let mut file = std::fs::File::create(&single).unwrap();
        file.set_len(64 * 1024 * 1024).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&vec![b'x'; MAX_SELECTED_BYTES]).unwrap();
        file.flush().unwrap();
        reset_target(&single);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = FileReadTool
            .execute(json!({"file_path": single, "limit": 1}), &ctx)
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("has_more=true"));

        let tail = dir.path().join("short-line-64m-tail.txt");
        let mut file = std::fs::File::create(&tail).unwrap();
        file.set_len(64 * 1024 * 1024).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"head\n").unwrap();
        file.flush().unwrap();
        reset_target(&tail);
        let result = FileReadTool
            .execute(json!({"file_path": tail, "limit": 1}), &ctx)
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("has_more=true"));
    }

    #[tokio::test]
    async fn replacement_between_capture_and_open_is_stale_even_same_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replace.txt");
        std::fs::write(&path, b"before").unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        set_gate(&path, Arc::clone(&gate));
        let ctx = allow_all_context(dir.path().to_path_buf());
        let path_for_reader = path.clone();
        let reader = tokio::spawn(async move {
            FileReadTool
                .execute(json!({"file_path": path_for_reader}), &ctx)
                .await
        });
        wait_for_scan_count(&path, 1).await;
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"after!").unwrap();
        gate.notify_waiters();
        let result = reader.await.unwrap();
        assert!(
            result.is_error,
            "replacement must not return stale text: {}",
            result.content
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn alias_retarget_is_stale_for_owner_delivery() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let second = dir.path().join("second.txt");
        let alias = dir.path().join("alias.txt");
        std::fs::write(&first, b"first").unwrap();
        std::fs::write(&second, b"second").unwrap();
        symlink(&first, &alias).unwrap();
        reset_target(&first);
        let gate = Arc::new(Notify::new());
        set_gate(&first, Arc::clone(&gate));
        let ctx = allow_all_context(dir.path().to_path_buf());
        let alias_for_reader = alias.clone();
        let reader = tokio::spawn(async move {
            FileReadTool
                .execute(json!({"file_path": alias_for_reader}), &ctx)
                .await
        });
        wait_for_scan_count(&first, 1).await;
        std::fs::remove_file(&alias).unwrap();
        symlink(&second, &alias).unwrap();
        gate.notify_waiters();
        assert!(reader.await.unwrap().is_error);
    }

    #[tokio::test]
    async fn cancellation_while_scanning_releases_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cancel.txt");
        std::fs::write(&path, b"cancel").unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        set_gate(&path, Arc::clone(&gate));
        let ctx = allow_all_context(dir.path().to_path_buf());
        let cancel = ctx.cancel_token.clone();
        let path_for_reader = path.clone();
        let reader = tokio::spawn(async move {
            FileReadTool
                .execute(json!({"file_path": path_for_reader}), &ctx)
                .await
        });
        wait_for_scan_count(&path, 1).await;
        cancel.cancel();
        gate.notify_waiters();
        assert!(reader.await.unwrap().is_error);
        assert!(matches!(
            register_flight(
                key_for(
                    &allow_all_context(dir.path().to_path_buf()),
                    file_identity(&path).unwrap(),
                    &FileReadInput {
                        file_path: path.display().to_string(),
                        offset: None,
                        limit: None,
                        byte_offset: None,
                        refresh: false,
                    },
                ),
                false,
            ),
            FlightRegistration::Owner(_)
        ));
    }

    #[tokio::test]
    async fn empty_file_keeps_legacy_marker_and_permission_denial_skips_scan() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.txt");
        std::fs::write(&empty, b"").unwrap();
        reset_target(&empty);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let result = FileReadTool
            .execute(json!({"file_path": empty}), &ctx)
            .await;
        assert!(!result.is_error);
        assert!(result.content.contains("exists but is empty"));

        let denied_path = dir.path().join("denied.txt");
        std::fs::write(&denied_path, b"secret").unwrap();
        reset_target(&denied_path);
        let mut denied = allow_all_context(dir.path().to_path_buf());
        denied.permission_handler = std::sync::Arc::new(DenyHandler);
        let result = FileReadTool
            .execute(json!({"file_path": denied_path}), &denied)
            .await;
        assert!(result.is_error);
        assert_eq!(read_count(&denied_path), 0);
    }

    async fn wait_for_scan_count(path: &std::path::Path, expected: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while read_count(path) < expected {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("scanner did not reach the expected count");
    }

    #[tokio::test]
    async fn matching_reads_coalesce_but_refresh_and_scope_bypass() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coalesce.txt");
        std::fs::write(&path, b"one\ntwo\n").unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        set_gate(&path, Arc::clone(&gate));
        let first_ctx = allow_all_context(dir.path().to_path_buf());
        let second_ctx = allow_all_context(dir.path().to_path_buf());
        let path_for_first = path.clone();
        let first = tokio::spawn(async move {
            FileReadTool
                .execute(json!({"file_path": path_for_first}), &first_ctx)
                .await
        });
        wait_for_scan_count(&path, 1).await;

        let path_for_second = path.clone();
        let second = tokio::spawn(async move {
            FileReadTool
                .execute(json!({"file_path": path_for_second}), &second_ctx)
                .await
        });
        tokio::task::yield_now().await;
        assert_eq!(read_count(&path), 1);

        let refresh_ctx = allow_all_context(dir.path().to_path_buf());
        let path_for_refresh = path.clone();
        let refresh = tokio::spawn(async move {
            FileReadTool
                .execute(
                    json!({"file_path": path_for_refresh, "refresh": true}),
                    &refresh_ctx,
                )
                .await
        });
        wait_for_scan_count(&path, 2).await;

        let mut isolated_ctx = allow_all_context(dir.path().to_path_buf());
        isolated_ctx.session_id = "different-session".to_string();
        let path_for_isolated = path.clone();
        let isolated = tokio::spawn(async move {
            FileReadTool
                .execute(json!({"file_path": path_for_isolated}), &isolated_ctx)
                .await
        });
        wait_for_scan_count(&path, 3).await;
        gate.notify_waiters();

        for result in [
            first.await.unwrap(),
            second.await.unwrap(),
            refresh.await.unwrap(),
            isolated.await.unwrap(),
        ] {
            assert!(!result.is_error, "{}", result.content);
        }
        assert_eq!(read_count(&path), 3);
    }

    #[tokio::test]
    async fn dropped_owner_wakes_follower_and_allows_fresh_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drop.txt");
        std::fs::write(&path, b"drop").unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let params = FileReadInput {
            file_path: path.display().to_string(),
            offset: None,
            limit: None,
            byte_offset: None,
            refresh: false,
        };
        let key = key_for(&ctx, file_identity(&path).unwrap(), &params);
        let follower = match register_flight(key.clone(), false) {
            FlightRegistration::Owner(owner) => {
                let follower = match register_flight(key.clone(), false) {
                    FlightRegistration::Follower(receiver) => receiver,
                    _ => panic!("matching read did not join"),
                };
                drop(owner);
                follower
            }
            _ => panic!("first read did not become owner"),
        };
        assert_eq!(
            follower.borrow().as_ref().unwrap().as_ref().unwrap_err(),
            "File read cancelled before completion"
        );
        assert!(matches!(
            register_flight(key, false),
            FlightRegistration::Owner(_)
        ));
    }

    #[tokio::test]
    async fn expired_flight_publishes_deadline_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("expired.txt");
        std::fs::write(&path, b"expired").unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let params = FileReadInput {
            file_path: path.display().to_string(),
            offset: None,
            limit: None,
            byte_offset: None,
            refresh: false,
        };
        let key = key_for(&ctx, file_identity(&path).unwrap(), &params);
        let (sender, receiver) = watch::channel(None);
        let stale_flight = Arc::new(ReadFlight {
            sender,
            started: Instant::now() - READ_DEADLINE,
            terminal: AtomicBool::new(false),
        });
        let stale_guard = FlightGuard {
            key: key.clone(),
            flight: Arc::clone(&stale_flight),
            complete: false,
        };
        ACTIVE_FLIGHTS.lock().insert(key.clone(), stale_flight);
        assert!(matches!(
            register_flight(key.clone(), true),
            FlightRegistration::Bypass
        ));
        assert_eq!(
            receiver.borrow().as_ref().unwrap().as_ref().unwrap_err(),
            "File read exceeded its deadline"
        );
        drop(stale_guard);
        let mut replacement = match register_flight(key.clone(), false) {
            FlightRegistration::Owner(guard) => guard,
            _ => panic!("expired flight must be replaced"),
        };
        replacement.complete(Ok(ReadPayload::Text("fresh".to_string())));
        assert_eq!(
            receiver.borrow().as_ref().unwrap().as_ref().unwrap_err(),
            "File read exceeded its deadline"
        );
    }

    #[tokio::test]
    async fn active_flight_joins_and_completion_evicts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flight.txt");
        std::fs::write(&path, b"flight").unwrap();
        reset_target(&path);
        let ctx = allow_all_context(dir.path().to_path_buf());
        let params = FileReadInput {
            file_path: path.display().to_string(),
            offset: None,
            limit: None,
            byte_offset: None,
            refresh: false,
        };
        let identity = file_identity(&path).unwrap();
        let key = key_for(&ctx, identity, &params);
        let owner = register_flight(key.clone(), false);
        let receiver = match owner {
            FlightRegistration::Owner(mut guard) => {
                let follower = register_flight(key.clone(), false);
                guard.complete(Ok(ReadPayload::Text("joined".into())));
                match follower {
                    FlightRegistration::Follower(receiver) => receiver,
                    _ => panic!("matching read did not join"),
                }
            }
            _ => panic!("first read did not become owner"),
        };
        assert!(
            matches!(receiver.borrow().as_ref().unwrap().as_ref().unwrap(), ReadPayload::Text(text) if text == "joined")
        );
        assert!(matches!(
            register_flight(key, false),
            FlightRegistration::Owner(_)
        ));

        let mut flights = HashMap::new();
        let mut guards = Vec::new();
        for offset in 0..MAX_FLIGHTS {
            let mut capacity_key = key_for(&ctx, file_identity(&path).unwrap(), &params);
            capacity_key.offset = offset;
            match register_flight_in(&mut flights, capacity_key, false) {
                FlightRegistration::Owner(guard) => guards.push(guard),
                _ => panic!("flight capacity should admit owner {offset}"),
            }
        }
        let mut overflow_key = key_for(&ctx, file_identity(&path).unwrap(), &params);
        overflow_key.offset = MAX_FLIGHTS;
        assert!(matches!(
            register_flight_in(&mut flights, overflow_key, false),
            FlightRegistration::Bypass
        ));
        drop(guards);
    }

    async fn wait_for_joined_reader(path: &Path) {
        let target = canonical(path);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let joined = ACTIVE_FLIGHTS.lock().iter().any(|(key, flight)| {
                    key.file.canonical == target && flight.sender.receiver_count() > 0
                });
                if joined {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("reader did not join its active owner");
    }

    fn start_read(path: PathBuf, ctx: ToolContext) -> tokio::task::JoinHandle<ToolResult> {
        tokio::spawn(async move { FileReadTool.execute(json!({"file_path": path}), &ctx).await })
    }

    #[tokio::test]
    async fn aborted_owner_wakes_actual_follower_and_later_read_is_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("owner-abort.txt");
        std::fs::write(&path, b"before\n").unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        set_gate(&path, gate);
        let owner = start_read(path.clone(), allow_all_context(dir.path().into()));
        wait_for_scan_count(&path, 1).await;
        let follower = start_read(path.clone(), allow_all_context(dir.path().into()));
        wait_for_joined_reader(&path).await;
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        let result = tokio::time::timeout(Duration::from_secs(2), follower)
            .await
            .unwrap()
            .unwrap();
        assert!(result.is_error && result.content.contains("cancelled"));
        TEST_SCAN_GATES.lock().remove(&canonical(&path));
        std::fs::write(&path, b"after!\n").unwrap();
        let result = FileReadTool
            .execute(
                json!({"file_path": path}),
                &allow_all_context(dir.path().into()),
            )
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(payload(&result.content), "after!");
        assert_eq!(read_count(&path), 2);
    }

    #[tokio::test]
    async fn cancelling_follower_keeps_owner_alive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("follower-cancel.txt");
        std::fs::write(&path, b"shared\n").unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        set_gate(&path, gate.clone());
        let owner = start_read(path.clone(), allow_all_context(dir.path().into()));
        wait_for_scan_count(&path, 1).await;
        let follower_ctx = allow_all_context(dir.path().into());
        let cancel = follower_ctx.cancel_token.clone();
        let follower = start_read(path.clone(), follower_ctx);
        wait_for_joined_reader(&path).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), follower)
            .await
            .unwrap()
            .unwrap();
        assert!(result.is_error && result.content.contains("cancelled"));
        gate.notify_waiters();
        let result = owner.await.unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(read_count(&path), 1);
    }

    #[tokio::test]
    async fn owner_deadline_releases_actual_follower() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("owner-deadline.txt");
        std::fs::write(&path, b"deadline\n").unwrap();
        reset_target(&path);
        set_gate(&path, Arc::new(Notify::new()));
        let owner = start_read(path.clone(), allow_all_context(dir.path().into()));
        wait_for_scan_count(&path, 1).await;
        let follower = start_read(path.clone(), allow_all_context(dir.path().into()));
        wait_for_joined_reader(&path).await;
        let results = tokio::time::timeout(Duration::from_secs(7), async {
            (owner.await.unwrap(), follower.await.unwrap())
        })
        .await
        .unwrap();
        for result in [results.0, results.1] {
            assert!(
                result.is_error
                    && (result.content.contains("deadline")
                        || result.content.contains("cancelled"))
            );
        }
        assert!(!ACTIVE_FLIGHTS
            .lock()
            .keys()
            .any(|key| key.file.canonical == canonical(&path)));
    }

    #[tokio::test]
    async fn mutation_during_final_identity_validation_invalidates_capture() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch-during-validation.txt");
        std::fs::write(&path, b"unchanged\n").unwrap();
        reset_target(&path);
        let gate = Arc::new(Notify::new());
        TEST_VALIDATION_GATES
            .lock()
            .insert(path.clone(), gate.clone());
        let ctx = allow_all_context(dir.path().into());
        let reader = start_read(path.clone(), ctx.clone());
        tokio::time::timeout(Duration::from_secs(5), async {
            while !TEST_VALIDATION_REACHED.lock().contains_key(&path) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        let admission = crate::mutation::admit(&ctx, &crate::FileWriteTool).unwrap();
        drop(admission);
        gate.notify_waiters();
        let result = reader.await.unwrap();
        assert!(
            result.is_error && result.content.contains("changed"),
            "{}",
            result.content
        );
        TEST_VALIDATION_GATES.lock().remove(&path);
    }

    #[tokio::test]
    async fn bounded_scanner_never_reads_the_unrequested_huge_tail() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("instrumented-tail.txt");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"wanted\n").unwrap();
        file.set_len(64 * 1024 * 1024).unwrap();
        let identity = file_identity(&path).unwrap();
        let mut scanner = Scanner::open(&identity).await.unwrap();
        let input = FileReadInput {
            file_path: path.display().to_string(),
            offset: Some(1),
            limit: Some(1),
            byte_offset: None,
            refresh: false,
        };
        let result = scan_text_inner(&mut scanner, &input).await.unwrap();
        assert_eq!(payload(&result.text), "wanted");
        assert!(result.has_more);
        assert!(scanner.scanned <= READ_CHUNK_BYTES as u64);
    }
}
