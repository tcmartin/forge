// upgrade.rs — `forge upgrade` subcommand.
//
// Downloads the latest release from GitHub, extracts it, and atomically
// replaces the running binary.  Mirrors the logic in install.sh / install.ps1
// so that an `upgrade` from inside Forge feels identical to a fresh install.
//
// Extraction shells out to the platform `tar`/bsdtar implementation. Keeping
// one listing and streamed-member path avoids separate shell extraction
// semantics and saves us pulling archive crates into the cli crate.

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

const APP: &str = "forge";
const MAX_LATEST_BYTES: usize = 1024 * 1024;
const MAX_CHECKSUM_BYTES: usize = 1024 * 1024;
const MAX_ARCHIVE_BYTES: usize = 128 * 1024 * 1024;
const MAX_COMMAND_OUTPUT_BYTES: usize = 1024 * 1024;
const LATEST_TIMEOUT: Duration = Duration::from_secs(10);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);
const EXTRACTION_TIMEOUT: Duration = Duration::from_secs(30);
const PROCESS_REAP_TIMEOUT: Duration = Duration::from_millis(250);
const PIPE_DRAIN_TIMEOUT: Duration = Duration::from_millis(100);

pub async fn run_upgrade(args: &[String]) -> Result<()> {
    // -------- arg parsing --------
    let mut requested_version: Option<String> = None;
    let mut requested_repository: Option<String> = None;
    let mut force = false;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "-v" | "--version" => {
                requested_version = iter.next().cloned();
                if requested_version.is_none() {
                    bail!("--version requires an argument");
                }
            }
            "--repository" => {
                requested_repository = iter.next().cloned();
                if requested_repository.is_none() {
                    bail!("--repository requires owner/repository");
                }
            }
            "--force" | "-f" => force = true,
            unknown => bail!("Unknown option: {}", unknown),
        }
    }

    // Validate all caller-controlled release coordinates before any HTTP.
    let repository = resolve_repository(requested_repository.as_deref())?;
    let requested_version = requested_version
        .as_deref()
        .map(normalize_version)
        .transpose()?;

    let current_version = env!("CARGO_PKG_VERSION").to_string();
    println!("Current version: {}", current_version);

    // -------- detect target --------
    let target = detect_target()?;
    println!("Detected target: {}", target);

    // -------- resolve target version --------
    let target_version = if let Some(v) = requested_version {
        v
    } else {
        fetch_latest_version(&repository).await?
    };
    println!("Target version:  {}", target_version);

    if target_version == current_version && !force {
        println!("\nAlready running v{} — nothing to do.", current_version);
        println!("Use --force to reinstall the same version, or --version <v> to install a specific one.");
        return Ok(());
    }

    // -------- locate current exe --------
    let exe_path =
        std::env::current_exe().context("could not determine current executable path")?;
    let exe_path = std::fs::canonicalize(&exe_path).unwrap_or(exe_path);
    println!("Installed at:    {}", exe_path.display());

    // -------- download --------
    let (archive_name, is_zip) = archive_name_for_target(&target);
    let release_base = format!(
        "https://github.com/{}/releases/download/v{}",
        repository, target_version
    );
    let archive_url = format!("{release_base}/{archive_name}");
    let checksums_url = format!("{release_base}/SHA256SUMS");

    let tmp_dir = UpgradeTempDir::create()?;
    let archive_path = tmp_dir.path.join(&archive_name);
    let checksums_path = tmp_dir.path.join("SHA256SUMS");

    println!("\nDownloading {}", archive_url);
    download_to_file(&checksums_url, &checksums_path, MAX_CHECKSUM_BYTES).await?;
    download_to_file(&archive_url, &archive_path, MAX_ARCHIVE_BYTES).await?;
    verify_archive_checksum(&archive_path, &checksums_path, &archive_name)?;
    println!("Checksum verified.");

    // -------- extract --------
    let bin_name = binary_name_for_target(&target);
    let extract_dir = tmp_dir.path.join("extracted");
    std::fs::create_dir_all(&extract_dir)?;
    println!("Extracting…");
    extract_archive(&archive_path, &extract_dir, is_zip, bin_name).await?;

    // The archive validator only permits the exact root member. Re-check the
    // filesystem without following links before any bytes reach swap_binary.
    let new_binary = extract_dir.join(bin_name);
    validate_extracted_binary(&extract_dir, &new_binary, bin_name)?;

    // -------- swap --------
    println!("Replacing running binary…");
    swap_binary(&exe_path, &new_binary)?;

    // -------- macOS quarantine strip --------
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("xattr")
            .args(["-dr", "com.apple.quarantine"])
            .arg(&exe_path)
            .status();
    }

    println!("\nUpgraded to v{}.", target_version);
    println!("Run `forge --version` in a new shell to verify.");
    Ok(())
}

fn print_help() {
    println!(
        "Usage: forge upgrade [options]\n\n\
         Options:\n\
           -v, --version <v>   Install a specific version (default: latest)\n\
           --repository <r>    Release repository (owner/repository)\n\
           -f, --force         Reinstall even if already up to date\n\
           -h, --help          Show this help\n\n\
         Downloads a Forge release from the explicitly selected GitHub repository\n\
         and replaces this binary in place. Set FORGE_RELEASE_REPO when the\n\
         --repository option is not supplied. Your settings and sessions are preserved."
    );
}

fn validate_repository(repository: &str) -> Result<()> {
    let Some((owner, name)) = repository.split_once('/') else {
        bail!("release repository must be owner/repository");
    };
    if owner.is_empty()
        || name.is_empty()
        || owner == "."
        || owner == ".."
        || name == "."
        || name == ".."
        || name.contains('/')
        || !owner
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
    {
        bail!("release repository must be owner/repository");
    }
    Ok(())
}

fn resolve_repository(cli_value: Option<&str>) -> Result<String> {
    let repository = cli_value
        .map(str::to_owned)
        .or_else(|| std::env::var("FORGE_RELEASE_REPO").ok())
        .unwrap_or_default();
    validate_repository(&repository)?;
    Ok(repository)
}

fn normalize_version(value: &str) -> Result<String> {
    let trimmed = value.trim();
    let version = trimmed.strip_prefix('v').unwrap_or(trimmed);
    if version.is_empty()
        || version == "."
        || version == ".."
        || version.len() > 128
        || version.chars().any(|character| {
            character.is_whitespace() || !character.is_ascii() || "/?#%".contains(character)
        })
    {
        bail!("release version is invalid");
    }
    Ok(version.to_string())
}

// ---------------------------------------------------------------------------
// Target detection
// ---------------------------------------------------------------------------

fn detect_target() -> Result<String> {
    let os = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        bail!("Unsupported OS for upgrade");
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        bail!("Unsupported architecture for upgrade");
    };

    target_for_platform(os, arch)
}

fn target_for_platform(os: &str, arch: &str) -> Result<String> {
    if os == "windows" && arch == "aarch64" {
        bail!("Windows aarch64 is not a supported Forge release target");
    }
    if !matches!(os, "linux" | "macos" | "windows") || !matches!(arch, "x86_64" | "aarch64") {
        bail!("Unsupported target for upgrade");
    }
    Ok(format!("{os}-{arch}"))
}

fn archive_name_for_target(target: &str) -> (String, bool) {
    if target.starts_with("windows") {
        (format!("{}-{}.zip", APP, target), true)
    } else {
        (format!("{}-{}.tar.gz", APP, target), false)
    }
}

fn binary_name_for_target(target: &str) -> &'static str {
    if target.starts_with("windows") {
        "forge.exe"
    } else {
        "forge"
    }
}

// ---------------------------------------------------------------------------
// GitHub API: latest version
// ---------------------------------------------------------------------------

async fn fetch_latest_version(repository: &str) -> Result<String> {
    let url = format!("https://api.github.com/repos/{repository}/releases/latest");
    let client = reqwest::Client::builder()
        .timeout(LATEST_TIMEOUT)
        .user_agent(format!("forge-upgrade/{}", env!("CARGO_PKG_VERSION")))
        .build()?;
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        bail!("GitHub API returned {}", resp.status());
    }
    let body = bounded_response_bytes(resp, MAX_LATEST_BYTES).await?;
    let json: serde_json::Value =
        serde_json::from_slice(&body).context("latest release response was not valid JSON")?;
    let tag = json
        .get("tag_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("no tag_name in GitHub API response"))?;
    normalize_version(tag)
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

async fn bounded_response_bytes(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        bail!("download exceeds the {} byte limit", max_bytes);
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len().saturating_add(chunk.len()) > max_bytes {
            bail!("download exceeds the {} byte limit", max_bytes);
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn download_to_file(url: &str, dest: &Path, max_bytes: usize) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .user_agent(format!("forge-upgrade/{}", env!("CARGO_PKG_VERSION")))
        .build()?;
    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        bail!(
            "Download failed: HTTP {}. Check that this version exists in the releases page.",
            response.status()
        );
    }
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        bail!("download exceeds the {} byte limit", max_bytes);
    }
    let mut response = response;
    let mut output = File::create(dest)?;
    let mut written = 0usize;
    while let Some(chunk) = response.chunk().await? {
        written = written
            .checked_add(chunk.len())
            .ok_or_else(|| anyhow!("download size overflow"))?;
        if written > max_bytes {
            bail!("download exceeds the {} byte limit", max_bytes);
        }
        output.write_all(&chunk)?;
    }
    output.flush()?;
    Ok(())
}

fn expected_sha256(manifest: &[u8], archive_name: &str) -> Result<String> {
    let text = std::str::from_utf8(manifest).context("SHA256SUMS is not UTF-8")?;
    let mut expected = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2
            || fields[0].len() != 64
            || !fields[0].bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("SHA256SUMS contains a malformed entry");
        }
        let name = fields[1].strip_prefix('*').unwrap_or(fields[1]);
        if name == archive_name && expected.replace(fields[0].to_ascii_lowercase()).is_some() {
            bail!("SHA256SUMS contains duplicate entries for {}", archive_name);
        }
    }
    expected.ok_or_else(|| anyhow!("SHA256SUMS does not list {}", archive_name))
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn verify_archive_checksum(archive: &Path, manifest: &Path, archive_name: &str) -> Result<()> {
    let expected = expected_sha256(&std::fs::read(manifest)?, archive_name)?;
    let mut file = File::open(archive)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = hex_digest(&hasher.finalize());
    if actual != expected {
        bail!("checksum verification failed for {}", archive_name);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extraction and bounded helper-process supervision.
// ---------------------------------------------------------------------------

struct BoundedCommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

async fn read_bounded(mut reader: impl AsyncRead + Unpin, max_bytes: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            return Ok(output);
        }
        if output.len().saturating_add(count) > max_bytes {
            bail!("helper process output exceeds {} bytes", max_bytes);
        }
        output.extend_from_slice(&buffer[..count]);
    }
}

async fn reap_after_kill(child: &mut Child) -> Result<()> {
    child
        .start_kill()
        .context("failed to signal helper process for termination")?;
    match tokio::time::timeout(PROCESS_REAP_TIMEOUT, child.wait()).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(error).context("failed to reap helper process"),
        Err(_) => bail!("helper process could not be reaped within the reap limit"),
    }
}

struct OwnedReaderTask(Option<tokio::task::JoinHandle<Result<Vec<u8>>>>);

impl OwnedReaderTask {
    fn is_some(&self) -> bool {
        self.0.is_some()
    }

    fn future(&mut self) -> &mut tokio::task::JoinHandle<Result<Vec<u8>>> {
        self.0.as_mut().expect("reader task exists")
    }

    fn take(&mut self) -> Option<tokio::task::JoinHandle<Result<Vec<u8>>>> {
        self.0.take()
    }

    async fn abort_and_wait(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for OwnedReaderTask {
    fn drop(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
        }
    }
}

async fn run_bounded_command_with_limits(
    program: &str,
    args: &[String],
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<BoundedCommandOutput> {
    let mut child = Command::new(program)
        .args(args)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn helper process {program}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("helper process stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("helper process stderr was not piped"))?;
    let mut stdout_task =
        OwnedReaderTask(Some(tokio::spawn(read_bounded(stdout, max_output_bytes))));
    let mut stderr_task =
        OwnedReaderTask(Some(tokio::spawn(read_bounded(stderr, max_output_bytes))));
    let mut stdout_result: Option<Result<Vec<u8>>> = None;
    let mut stderr_result: Option<Result<Vec<u8>>> = None;
    let mut child_wait = Box::pin(child.wait());
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    enum WaitOutcome {
        Exited(Result<ExitStatus>),
        ReaderError(anyhow::Error),
        TimedOut,
    }
    let outcome = loop {
        tokio::select! {
            status = &mut child_wait => {
                break WaitOutcome::Exited(status.context("failed to wait for helper process"));
            }
            output = async { stdout_task.future().await }, if stdout_task.is_some() => {
                let error = match output {
                    Ok(result) => {
                        let error = result
                            .as_ref()
                            .err()
                            .map(|error| anyhow!("helper stdout reader failed: {error}"));
                        stdout_result = Some(result);
                        error
                    }
                    Err(error) => {
                        let message = format!("helper stdout reader task failed: {error}");
                        stdout_result = Some(Err(anyhow!("{message}")));
                        Some(anyhow!("{message}"))
                    }
                };
                let _ = stdout_task.take();
                if let Some(error) = error {
                    break WaitOutcome::ReaderError(error);
                }
            }
            output = async { stderr_task.future().await }, if stderr_task.is_some() => {
                let error = match output {
                    Ok(result) => {
                        let error = result
                            .as_ref()
                            .err()
                            .map(|error| anyhow!("helper stderr reader failed: {error}"));
                        stderr_result = Some(result);
                        error
                    }
                    Err(error) => {
                        let message = format!("helper stderr reader task failed: {error}");
                        stderr_result = Some(Err(anyhow!("{message}")));
                        Some(anyhow!("{message}"))
                    }
                };
                let _ = stderr_task.take();
                if let Some(error) = error {
                    break WaitOutcome::ReaderError(error);
                }
            }
            _ = &mut deadline => {
                break WaitOutcome::TimedOut;
            }
        }
    };
    drop(child_wait);
    let status = match outcome {
        WaitOutcome::Exited(Ok(status)) => status,
        WaitOutcome::Exited(Err(error)) => {
            if let Some(task) = stdout_task.take() {
                task.abort();
                let _ = task.await;
            }
            if let Some(task) = stderr_task.take() {
                task.abort();
                let _ = task.await;
            }
            reap_after_kill(&mut child).await?;
            return Err(error);
        }
        WaitOutcome::ReaderError(error) => {
            if let Some(task) = stdout_task.take() {
                task.abort();
                let _ = task.await;
            }
            if let Some(task) = stderr_task.take() {
                task.abort();
                let _ = task.await;
            }
            reap_after_kill(&mut child).await?;
            return Err(error);
        }
        WaitOutcome::TimedOut => {
            if let Some(task) = stdout_task.take() {
                task.abort();
                let _ = task.await;
            }
            if let Some(task) = stderr_task.take() {
                task.abort();
                let _ = task.await;
            }
            reap_after_kill(&mut child).await?;
            bail!("helper process timed out after {:?}", timeout);
        }
    };
    // A direct child can exit while a descendant retains an inherited pipe.
    // Give owned readers at most the fixed drain window, then abort their
    // Tokio tasks. This bounds cleanup without claiming descendant control.
    let drain = async {
        if stdout_result.is_none() && stdout_task.is_some() {
            let result = stdout_task.future().await;
            let _ = stdout_task.take();
            stdout_result =
                Some(result.map_err(|error| anyhow!("helper stdout reader task failed: {error}"))?);
        }
        if stderr_result.is_none() && stderr_task.is_some() {
            let result = stderr_task.future().await;
            let _ = stderr_task.take();
            stderr_result =
                Some(result.map_err(|error| anyhow!("helper stderr reader task failed: {error}"))?);
        }
        let stdout = stdout_result
            .take()
            .ok_or_else(|| anyhow!("helper stdout reader produced no result"))??;
        let stderr = stderr_result
            .take()
            .ok_or_else(|| anyhow!("helper stderr reader produced no result"))??;
        Ok::<_, anyhow::Error>((stdout, stderr))
    };
    let (stdout, stderr) = match tokio::time::timeout(PIPE_DRAIN_TIMEOUT, drain).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            stdout_task.abort_and_wait().await;
            stderr_task.abort_and_wait().await;
            return Err(error);
        }
        Err(_) => {
            stdout_task.abort_and_wait().await;
            stderr_task.abort_and_wait().await;
            bail!("helper output remained open after the pipe drain limit");
        }
    };
    Ok(BoundedCommandOutput {
        status,
        stdout,
        stderr,
    })
}

async fn run_bounded_command(program: &str, args: &[String]) -> Result<BoundedCommandOutput> {
    run_bounded_command_with_limits(program, args, EXTRACTION_TIMEOUT, MAX_COMMAND_OUTPUT_BYTES)
        .await
}

fn validate_archive_listing(output: &[u8], expected_name: &str) -> Result<()> {
    let listing = std::str::from_utf8(output).context("archive member listing was not UTF-8")?;
    let mut members = 0usize;
    let mut exact_regular_member = false;
    for line in listing
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        members = members.saturating_add(1);
        let mode = line
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("archive member listing contained a malformed line"))?;
        if !mode.starts_with('-') {
            bail!("archive member is not a regular file: {line}");
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let date_start = fields
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, field)| {
                field.contains('-')
                    || matches!(
                        **field,
                        "Jan"
                            | "Feb"
                            | "Mar"
                            | "Apr"
                            | "May"
                            | "Jun"
                            | "Jul"
                            | "Aug"
                            | "Sep"
                            | "Oct"
                            | "Nov"
                            | "Dec"
                    )
            })
            .map(|(index, _)| index)
            .ok_or_else(|| anyhow!("archive member listing omitted a date field"))?;
        let size = fields[..date_start]
            .iter()
            .rev()
            .find_map(|field| field.parse::<u64>().ok())
            .ok_or_else(|| anyhow!("archive member listing omitted a byte size"))?;
        if size > MAX_ARCHIVE_BYTES as u64 {
            bail!(
                "archive member {} exceeds {} bytes",
                fields.last().unwrap_or(&"<unknown>"),
                MAX_ARCHIVE_BYTES
            );
        }
        if line.split_whitespace().last() == Some(expected_name) {
            exact_regular_member = true;
        }
        if members > 1 {
            bail!("archive must contain exactly one root member");
        }
    }
    if members != 1 || !exact_regular_member {
        bail!("archive must contain exactly one regular root member {expected_name}");
    }
    Ok(())
}

fn validate_extracted_binary(dest: &Path, binary: &Path, expected_name: &str) -> Result<()> {
    let mut entries = std::fs::read_dir(dest)?;
    let first = entries
        .next()
        .ok_or_else(|| anyhow!("archive did not extract {expected_name}"))??;
    if entries.next().is_some() {
        bail!("archive extracted more than one root entry");
    }
    if first.file_name() != expected_name {
        bail!("archive extracted an unexpected root entry");
    }
    let metadata = std::fs::symlink_metadata(binary)?;
    if !metadata.file_type().is_file() {
        bail!("extracted {expected_name} is not a regular file");
    }
    if metadata.len() > MAX_ARCHIVE_BYTES as u64 {
        bail!(
            "extracted {expected_name} exceeds {} bytes",
            MAX_ARCHIVE_BYTES
        );
    }
    Ok(())
}

async fn archive_listing(archive: &Path, is_zip: bool, expected_name: &str) -> Result<()> {
    let archive_arg = archive.to_string_lossy().into_owned();
    let args = if is_zip {
        vec!["-tvf".to_string(), archive_arg]
    } else {
        vec!["-tvzf".to_string(), archive_arg]
    };
    let output = run_bounded_command("tar", &args).await?;
    if !output.status.success() {
        bail!(
            "tar archive listing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    validate_archive_listing(&output.stdout, expected_name)
}

async fn extract_archive(
    archive: &Path,
    dest: &Path,
    is_zip: bool,
    expected_name: &str,
) -> Result<()> {
    archive_listing(archive, is_zip, expected_name).await?;
    let archive_arg = archive.to_string_lossy().into_owned();
    let extract_flag = if is_zip { "-xOf" } else { "-xzOf" };
    let args = vec![
        extract_flag.to_string(),
        archive_arg,
        "--".to_string(),
        expected_name.to_string(),
    ];
    // Capture the exact validated member through a bounded pipe first. The
    // destination is created only after decompressed bytes pass the cap.
    let output =
        run_bounded_command_with_limits("tar", &args, EXTRACTION_TIMEOUT, MAX_ARCHIVE_BYTES)
            .await?;
    if !output.status.success() {
        bail!(
            "archive extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if output.stdout.is_empty() {
        bail!("extracted {expected_name} is empty");
    }
    let binary = dest.join(expected_name);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&binary)
        .with_context(|| format!("failed to create extracted member {}", binary.display()))?;
    file.write_all(&output.stdout)?;
    file.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Atomic binary swap
// ---------------------------------------------------------------------------

#[cfg(unix)]
struct StagedBinary {
    path: PathBuf,
}

#[cfg(unix)]
impl StagedBinary {
    fn create_next_to(current: &Path) -> Result<(Self, std::fs::File)> {
        use std::ffi::OsString;
        use std::fs::OpenOptions;
        use std::io::ErrorKind;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_STAGE_ID: AtomicU64 = AtomicU64::new(0);
        const MAX_ATTEMPTS: usize = 128;

        let parent = current
            .parent()
            .ok_or_else(|| anyhow!("installed binary path has no parent: {}", current.display()))?;
        let file_name = current.file_name().ok_or_else(|| {
            anyhow!(
                "installed binary path has no file name: {}",
                current.display()
            )
        })?;

        for _ in 0..MAX_ATTEMPTS {
            let id = NEXT_STAGE_ID.fetch_add(1, Ordering::Relaxed);
            let mut staged_name = OsString::from(".");
            staged_name.push(file_name);
            staged_name.push(format!(".upgrade-{}-{}", std::process::id(), id));
            let path = parent.join(staged_name);

            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((Self { path }, file)),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to create upgrade staging file next to {}",
                            current.display()
                        )
                    });
                }
            }
        }

        bail!(
            "failed to create a unique upgrade staging file next to {} after {} attempts",
            current.display(),
            MAX_ATTEMPTS
        )
    }
}

#[cfg(unix)]
impl Drop for StagedBinary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn swap_binary(current: &Path, new: &Path) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        // Windows holds an exclusive lock on the running .exe — we can rename
        // it but not unlink it, so move-aside, then move-in-place.
        let mut sidelined = current.to_path_buf();
        sidelined.set_extension("exe.old");
        let _ = std::fs::remove_file(&sidelined);
        std::fs::rename(current, &sidelined).with_context(|| {
            format!("failed to sideline current exe to {}", sidelined.display())
        })?;
        if let Err(e) = std::fs::copy(new, current) {
            // Try to roll back the rename so the user isn't left without claurst.
            let _ = std::fs::rename(&sidelined, current);
            bail!("failed to install new binary: {}", e);
        }
        // Best-effort cleanup of the old exe (Windows may keep it locked
        // until the running process exits — that's fine).
        let _ = std::fs::remove_file(&sidelined);
        Ok(())
    }

    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        // std::fs::copy() writes into the existing inode (open + O_TRUNC), and
        // that inode is the running process's own text segment — the kernel
        // rejects the write with ETXTBSY ("Text file busy"). rename() instead
        // swaps the directory entry to point at a new inode, which the kernel
        // allows even while the old inode is still mapped and executing.
        // rename() requires src/dest on the same filesystem, so stage the
        // temp file next to `current` rather than relying on the OS tmp dir.
        let (staged, mut staged_file) = StagedBinary::create_next_to(current)?;
        let mut source = std::fs::File::open(new)
            .with_context(|| format!("failed to open new binary at {}", new.display()))?;
        std::io::copy(&mut source, &mut staged_file)
            .with_context(|| format!("failed to stage new binary at {}", staged.path.display()))?;
        staged_file
            .set_permissions(std::fs::Permissions::from_mode(0o755))
            .with_context(|| {
                format!(
                    "failed to make staged binary executable at {}",
                    staged.path.display()
                )
            })?;
        staged_file.flush().with_context(|| {
            format!("failed to flush staged binary at {}", staged.path.display())
        })?;
        staged_file.sync_all().with_context(|| {
            format!("failed to sync staged binary at {}", staged.path.display())
        })?;
        drop(staged_file);

        std::fs::rename(&staged.path, current)
            .with_context(|| format!("failed to swap new binary into {}", current.display()))?;
        Ok(())
    }

    #[cfg(not(any(target_os = "windows", unix)))]
    {
        let _ = (current, new);
        bail!("Unsupported OS for upgrade")
    }
}

// ---------------------------------------------------------------------------
// Tempdir
// ---------------------------------------------------------------------------

struct UpgradeTempDir {
    path: PathBuf,
}

impl UpgradeTempDir {
    fn create() -> Result<Self> {
        use std::io::ErrorKind;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir();
        let pid = std::process::id();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        for _ in 0..128 {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("forge-upgrade-{}-{}-{}", pid, now, id));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        bail!("failed to create a unique Forge upgrade directory")
    }
}

impl Drop for UpgradeTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::{swap_binary, StagedBinary};
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "claurst-upgrade-test-{}-{}-{}",
                std::process::id(),
                name,
                id
            ));
            std::fs::create_dir(&path).expect("failed to create test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    #[ignore = "spawned by swap_binary_replaces_a_running_executable"]
    fn running_executable_fixture() {
        std::thread::park();
    }

    #[test]
    fn staging_files_are_unique_and_removed_on_drop() {
        let dir = TestDir::new("unique-stage");
        let current = dir.path().join("claurst");

        let (first, first_file) =
            StagedBinary::create_next_to(&current).expect("first staging file");
        let first_path = first.path.clone();
        let (second, second_file) =
            StagedBinary::create_next_to(&current).expect("second staging file");
        let second_path = second.path.clone();

        assert_ne!(first_path, second_path);
        assert!(first_path.exists());
        assert!(second_path.exists());

        drop(first_file);
        drop(second_file);
        drop(first);
        drop(second);

        assert!(!first_path.exists());
        assert!(!second_path.exists());
    }

    // Regression test for the ETXTBSY bug: swap_binary must be able to
    // replace a binary while it is actively running (mirrors `claurst
    // upgrade` replacing its own executable). The current test harness is
    // itself a portable native executable, so copying and spawning it avoids
    // assumptions about whether system tools live in /bin or /usr/bin.
    #[test]
    fn swap_binary_replaces_a_running_executable() {
        let dir = TestDir::new("running-executable");
        let current = dir.path().join("running");
        let replacement = dir.path().join("replacement");
        let legacy_staging_path = current.with_extension("new");
        let replacement_bytes = b"#!/bin/sh\nprintf 'replacement binary\n'\n";

        std::fs::copy(
            std::env::current_exe().expect("current test executable"),
            &current,
        )
        .expect("failed to copy test executable");
        std::fs::write(&replacement, replacement_bytes).expect("failed to write replacement");
        std::fs::write(&legacy_staging_path, b"must not be overwritten")
            .expect("failed to write legacy staging sentinel");

        let mut child = Command::new(&current)
            .args([
                "--ignored",
                "--exact",
                "upgrade::tests::running_executable_fixture",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn fixture binary");
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            child.try_wait().expect("failed to inspect fixture process"),
            None,
            "fixture process exited before the swap"
        );

        // Sanity check: overwriting the file in place while it's running
        // is exactly what triggers ETXTBSY. This confirms the fixture
        // actually reproduces the bug this test guards against.
        #[cfg(target_os = "linux")]
        {
            let direct_overwrite = std::fs::copy(&replacement, &current);
            assert!(
                direct_overwrite.is_err(),
                "expected direct overwrite of a running binary to fail with ETXTBSY"
            );
        }

        let result = swap_binary(&current, &replacement);

        child.kill().ok();
        child.wait().ok();
        result.expect("swap_binary should replace a running binary via rename, not in-place write");

        assert_eq!(
            std::fs::read(&current).expect("failed to read installed replacement"),
            replacement_bytes
        );
        assert_eq!(
            std::fs::metadata(&current)
                .expect("failed to stat installed replacement")
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
        assert_eq!(
            std::fs::read(&legacy_staging_path).expect("failed to read staging sentinel"),
            b"must not be overwritten"
        );
        assert!(
            std::fs::read_dir(dir.path())
                .expect("failed to inspect test directory")
                .flatten()
                .all(|entry| !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".running.upgrade-")),
            "staging file was not cleaned up"
        );
    }
}

#[cfg(test)]
mod contract_tests {
    use super::{
        archive_listing, archive_name_for_target, binary_name_for_target, download_to_file,
        expected_sha256, extract_archive, normalize_version, validate_archive_listing,
        validate_extracted_binary, validate_repository, verify_archive_checksum,
    };
    #[cfg(unix)]
    use super::{run_bounded_command_with_limits, MAX_ARCHIVE_BYTES};
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "forge-upgrade-contract-test-{}-{}",
                std::process::id(),
                id
            ));
            std::fs::create_dir(&path).expect("test directory");
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn digest(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn target_matrix_uses_forge_release_assets_and_members() {
        for target in [
            "linux-x86_64",
            "linux-aarch64",
            "macos-x86_64",
            "macos-aarch64",
            "windows-x86_64",
        ] {
            let (archive, is_zip) = archive_name_for_target(target);
            let expected_extension = if target.starts_with("windows") {
                ".zip"
            } else {
                ".tar.gz"
            };
            assert_eq!(archive, format!("forge-{target}{expected_extension}"));
            assert_eq!(is_zip, target.starts_with("windows"));
            assert_eq!(
                binary_name_for_target(target),
                if target.starts_with("windows") {
                    "forge.exe"
                } else {
                    "forge"
                }
            );
        }
    }

    #[test]
    fn windows_aarch64_is_rejected_without_inventing_an_asset() {
        assert!(
            super::target_for_platform("windows", "aarch64").is_err(),
            "the release workflow has no Windows aarch64 asset"
        );
    }

    #[test]
    fn repository_and_version_validation_fail_closed() {
        for repository in [
            "owner/repository",
            "owner-name/repository_name",
            "org.with.dots/forge",
        ] {
            validate_repository(repository).expect("valid release repository");
        }
        for repository in [
            "",
            "owner",
            "/repository",
            "owner/",
            "owner/repo/extra",
            "owner/repo?redirect=1",
            "owner repo/repository",
        ] {
            assert!(
                validate_repository(repository).is_err(),
                "accepted {repository}"
            );
        }

        assert_eq!(normalize_version("v1.2.3").unwrap(), "1.2.3");
        for version in ["", "v", "1/2", "1 2", "../forge", "v#latest"] {
            assert!(normalize_version(version).is_err(), "accepted {version}");
        }
    }

    #[tokio::test]
    async fn invalid_coordinates_fail_before_any_http_attempt() {
        assert!(super::run_upgrade(&[
            "--repository".into(),
            "not-a-repository".into(),
            "--version".into(),
            "1.2.3".into(),
        ])
        .await
        .is_err());
        assert!(super::run_upgrade(&[
            "--repository".into(),
            "owner/repository".into(),
            "--version".into(),
            "../escape".into(),
        ])
        .await
        .is_err());
    }

    #[test]
    fn checksum_manifest_rejects_missing_malformed_duplicate_and_mismatch() {
        let archive = "forge-linux-x86_64.tar.gz";
        let body = b"forge payload";
        let valid = format!("{}  {archive}\n", digest(body));
        assert_eq!(
            expected_sha256(valid.as_bytes(), archive).unwrap(),
            digest(body)
        );
        assert!(expected_sha256(b"", archive).is_err());
        assert!(expected_sha256(b"not-a-digest  forge-linux-x86_64.tar.gz\n", archive).is_err());
        assert!(expected_sha256(
            format!("{}  {archive}\n{}  {archive}\n", digest(body), digest(body)).as_bytes(),
            archive
        )
        .is_err());
    }

    #[test]
    fn archive_listing_rejects_malformed_missing_nested_and_nonregular_members() {
        assert!(validate_archive_listing(b"", "forge").is_err());
        assert!(validate_archive_listing(b"not-a-tar-line\n", "forge").is_err());
        assert!(validate_archive_listing(
            b"drwxr-xr-x user/group 0 2026-01-01 00:00 forge\n",
            "forge"
        )
        .is_err());
        assert!(validate_archive_listing(
            b"-rw-r--r-- user/group 1 2026-01-01 00:00 nested/forge\n",
            "forge"
        )
        .is_err());
        assert!(validate_archive_listing(
            b"lrwxr-xr-x user/group 0 2026-01-01 00:00 forge -> payload\n",
            "forge"
        )
        .is_err());
        assert!(validate_archive_listing(
            b"-rw-r--r-- user/group 1 2026-01-01 00:00 forge\n-rw-r--r-- user/group 1 2026-01-01 00:00 extra\n",
            "forge"
        )
        .is_err());
        assert!(validate_archive_listing(&[0xff, 0xfe], "forge").is_err());
    }

    fn make_tar(source: &Path, archive: &Path, members: &[&str]) {
        let mut command = Command::new("tar");
        command.arg("-czf").arg(archive).arg("-C").arg(source);
        for member in members {
            command.arg(member);
        }
        let status = command.status().expect("spawn tar fixture");
        assert!(status.success(), "tar fixture failed: {status}");
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn make_zip(archive: &Path, member: &str, payload: &[u8]) {
        let name = member.as_bytes();
        let crc = crc32(payload);
        let size = payload.len() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        bytes.extend_from_slice(&20u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(payload);
        let central_offset = bytes.len() as u32;
        bytes.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        bytes.extend_from_slice(&20u16.to_le_bytes());
        bytes.extend_from_slice(&20u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(name);
        let central_size = bytes.len() as u32 - central_offset;
        bytes.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&central_size.to_le_bytes());
        bytes.extend_from_slice(&central_offset.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        std::fs::write(archive, bytes).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_tar_archive_accepts_only_regular_root_forge() {
        let dir = TestDir::new();
        let source = dir.path().join("source");
        let destination = dir.path().join("destination");
        let archive = dir.path().join("forge.tar.gz");
        std::fs::create_dir(&source).unwrap();
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(source.join("forge"), b"forge binary").unwrap();
        make_tar(&source, &archive, &["forge"]);

        archive_listing(&archive, false, "forge").await.unwrap();
        extract_archive(&archive, &destination, false, "forge")
            .await
            .unwrap();
        validate_extracted_binary(&destination, &destination.join("forge"), "forge").unwrap();
        assert_eq!(
            std::fs::read(destination.join("forge")).unwrap(),
            b"forge binary"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_tar_archive_rejects_extra_nested_and_symlink_members() {
        let dir = TestDir::new();
        let source = dir.path().join("source");
        let destination = dir.path().join("destination");
        let archive = dir.path().join("invalid.tar.gz");
        std::fs::create_dir(&source).unwrap();
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(source.join("forge"), b"forge binary").unwrap();
        std::fs::write(source.join("extra"), b"extra").unwrap();
        make_tar(&source, &archive, &["forge", "extra"]);
        assert!(extract_archive(&archive, &destination, false, "forge")
            .await
            .is_err());

        let nested_source = dir.path().join("nested-source");
        let nested_destination = dir.path().join("nested-destination");
        let nested_archive = dir.path().join("nested.tar.gz");
        std::fs::create_dir_all(nested_source.join("nested")).unwrap();
        std::fs::create_dir(&nested_destination).unwrap();
        std::fs::write(nested_source.join("nested/forge"), b"nested").unwrap();
        make_tar(&nested_source, &nested_archive, &["nested/forge"]);
        assert!(
            extract_archive(&nested_archive, &nested_destination, false, "forge")
                .await
                .is_err()
        );

        let link_source = dir.path().join("link-source");
        let link_destination = dir.path().join("link-destination");
        let link_archive = dir.path().join("link.tar.gz");
        std::fs::create_dir(&link_source).unwrap();
        std::fs::create_dir(&link_destination).unwrap();
        std::fs::write(link_source.join("payload"), b"payload").unwrap();
        std::os::unix::fs::symlink("payload", link_source.join("forge")).unwrap();
        make_tar(&link_source, &link_archive, &["forge"]);
        assert!(
            extract_archive(&link_archive, &link_destination, false, "forge")
                .await
                .is_err()
        );

        let directory_source = dir.path().join("directory-source");
        let directory_destination = dir.path().join("directory-destination");
        let directory_archive = dir.path().join("directory.tar.gz");
        std::fs::create_dir(&directory_source).unwrap();
        std::fs::create_dir(directory_source.join("forge")).unwrap();
        std::fs::create_dir(&directory_destination).unwrap();
        make_tar(&directory_source, &directory_archive, &["forge"]);
        assert!(
            extract_archive(&directory_archive, &directory_destination, false, "forge")
                .await
                .is_err()
        );

        let hardlink_source = dir.path().join("hardlink-source");
        let hardlink_destination = dir.path().join("hardlink-destination");
        let hardlink_archive = dir.path().join("hardlink.tar.gz");
        std::fs::create_dir(&hardlink_source).unwrap();
        std::fs::write(hardlink_source.join("payload"), b"payload").unwrap();
        std::fs::hard_link(
            hardlink_source.join("payload"),
            hardlink_source.join("forge"),
        )
        .unwrap();
        std::fs::create_dir(&hardlink_destination).unwrap();
        // A single filesystem hard link is serialized as a regular file.
        // Archive both inode names first, then retain the actual hard-link
        // header as a one-member archive; no extra member can mask this case.
        let pair_archive = dir.path().join("hardlink-pair.tar");
        let status = Command::new("tar")
            .arg("-cf")
            .arg(&pair_archive)
            .arg("-C")
            .arg(&hardlink_source)
            .args(["payload", "forge"])
            .status()
            .unwrap();
        assert!(status.success());
        let pair = std::fs::read(pair_archive).unwrap();
        let header = pair
            .as_chunks::<512>()
            .0
            .iter()
            .find(|header| header[..6] == *b"forge\0" && header[156] == b'1')
            .expect("fixture must contain a real hard-link header");
        let mut member = header.to_vec();
        member.resize(512 + 1024, 0);
        let raw_archive = dir.path().join("hardlink-only.tar");
        std::fs::write(&raw_archive, member).unwrap();
        let compressed = Command::new("gzip")
            .arg("-c")
            .arg(&raw_archive)
            .output()
            .unwrap();
        assert!(compressed.status.success());
        std::fs::write(&hardlink_archive, compressed.stdout).unwrap();
        assert!(
            extract_archive(&hardlink_archive, &hardlink_destination, false, "forge")
                .await
                .is_err()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn real_tar_archive_rejects_oversized_extracted_binary() {
        let dir = TestDir::new();
        let source = dir.path().join("source");
        let destination = dir.path().join("destination");
        let archive = dir.path().join("oversized.tar.gz");
        std::fs::create_dir(&source).unwrap();
        std::fs::create_dir(&destination).unwrap();
        let file = std::fs::File::create(source.join("forge")).unwrap();
        file.set_len(MAX_ARCHIVE_BYTES as u64 + 1).unwrap();
        make_tar(&source, &archive, &["forge"]);
        assert!(extract_archive(&archive, &destination, false, "forge")
            .await
            .is_err());
        assert!(std::fs::read_dir(&destination).unwrap().next().is_none());
    }

    #[tokio::test]
    async fn streamed_oversize_helper_rejects_before_destination_create() {
        let dir = TestDir::new();
        let destination = dir.path().join("destination").join("forge");
        let result = run_bounded_command_with_limits(
            "sh",
            &["-c".to_string(), "printf 1234567890".to_string()],
            std::time::Duration::from_secs(2),
            4,
        )
        .await;
        assert!(result.is_err());
        assert!(
            !destination.exists(),
            "oversize bytes must precede destination creation"
        );
    }

    #[tokio::test]
    async fn real_tar_archive_rejects_malformed_input() {
        let dir = TestDir::new();
        let malformed = dir.path().join("malformed.tar.gz");
        let malformed_destination = dir.path().join("malformed-destination");
        std::fs::write(&malformed, b"not a tar archive").unwrap();
        std::fs::create_dir(&malformed_destination).unwrap();
        assert!(
            extract_archive(&malformed, &malformed_destination, false, "forge")
                .await
                .is_err()
        );
        assert!(std::fs::read_dir(&malformed_destination)
            .unwrap()
            .next()
            .is_none());
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[tokio::test]
    async fn platform_bsdtar_accepts_valid_zip_root_and_rejects_wrong_root() {
        let dir = TestDir::new();
        let zip = dir.path().join("forge.zip");
        let destination = dir.path().join("zip-destination");
        make_zip(&zip, "forge.exe", b"forge windows binary");
        std::fs::create_dir(&destination).unwrap();
        archive_listing(&zip, true, "forge.exe").await.unwrap();
        extract_archive(&zip, &destination, true, "forge.exe")
            .await
            .unwrap();
        validate_extracted_binary(&destination, &destination.join("forge.exe"), "forge.exe")
            .unwrap();
        assert_eq!(
            std::fs::read(destination.join("forge.exe")).unwrap(),
            b"forge windows binary"
        );

        let wrong_root = dir.path().join("wrong-root.zip");
        let wrong_destination = dir.path().join("wrong-destination");
        make_zip(&wrong_root, "other.exe", b"wrong root");
        std::fs::create_dir(&wrong_destination).unwrap();
        assert!(
            extract_archive(&wrong_root, &wrong_destination, true, "forge.exe")
                .await
                .is_err()
        );
        assert!(std::fs::read_dir(&wrong_destination)
            .unwrap()
            .next()
            .is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bounded_helper_timeout_and_output_limit_reap_owned_child() {
        let started = std::time::Instant::now();
        let timeout = run_bounded_command_with_limits(
            "sleep",
            &["5".to_string()],
            std::time::Duration::from_millis(50),
            1024,
        )
        .await;
        assert!(timeout.is_err());
        assert!(started.elapsed() < std::time::Duration::from_secs(1));

        let output = run_bounded_command_with_limits(
            "sh",
            &["-c".to_string(), "printf 12345".to_string()],
            std::time::Duration::from_secs(2),
            4,
        )
        .await;
        assert!(output.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn direct_exit_with_descendant_holding_stdout_returns_after_bounded_drain() {
        let started = std::time::Instant::now();
        let result = run_bounded_command_with_limits(
            "sh",
            &[
                "-c".to_string(),
                "sleep 5 >/dev/stdout 2>&1 & exit 0".to_string(),
            ],
            std::time::Duration::from_secs(2),
            1024,
        )
        .await;
        let error = result.err().expect("incomplete output must fail");
        assert!(error.to_string().contains("pipe drain limit"));
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn completed_stdout_and_inherited_stderr_fail_without_repolling_reader() {
        let result = run_bounded_command_with_limits(
            "sh",
            &[
                "-c".to_string(),
                "exec 1>/dev/null; sleep 5 >&2 & exit 0".to_string(),
            ],
            std::time::Duration::from_secs(2),
            1024,
        )
        .await;
        assert!(result
            .err()
            .expect("incomplete stderr must fail")
            .to_string()
            .contains("pipe drain limit"));
    }

    #[tokio::test]
    async fn empty_archive_member_fails_before_destination_create() {
        let dir = TestDir::new();
        let source = dir.path().join("source");
        let destination = dir.path().join("destination");
        std::fs::create_dir(&source).unwrap();
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(source.join("forge"), []).unwrap();
        let archive = dir.path().join("empty.tar.gz");
        make_tar(&source, &archive, &["forge"]);
        assert!(extract_archive(&archive, &destination, false, "forge")
            .await
            .is_err());
        assert!(std::fs::read_dir(destination).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_helper_future_aborts_owned_readers() {
        let args = vec![
            "-c".to_string(),
            "sleep 5 >/dev/stdout 2>&1 & exit 0".to_string(),
        ];
        let task = tokio::spawn(async move {
            run_bounded_command_with_limits("sh", &args, std::time::Duration::from_secs(2), 1024)
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        task.abort();
        let _ = task.await;
    }

    async fn loopback_body(body: Vec<u8>) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes()).await;
            let _ = socket.write_all(&body).await;
        });
        (format!("http://{address}/release"), task)
    }

    #[tokio::test]
    async fn bounded_loopback_download_verifies_valid_archive() {
        let payload = b"forge release payload".to_vec();
        let (url, server) = loopback_body(payload.clone()).await;
        let dir = TestDir::new();
        let archive = dir.path().join("forge-linux-x86_64.tar.gz");
        let manifest = dir.path().join("SHA256SUMS");
        download_to_file(&url, &archive, 1024).await.unwrap();
        std::fs::write(
            &manifest,
            format!("{}  forge-linux-x86_64.tar.gz\n", digest(&payload)),
        )
        .unwrap();
        verify_archive_checksum(&archive, &manifest, "forge-linux-x86_64.tar.gz").unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn bounded_loopback_download_rejects_oversize_body() {
        let (url, server) = loopback_body(b"12345".to_vec()).await;
        let dir = TestDir::new();
        let archive = dir.path().join("forge-linux-x86_64.tar.gz");
        assert!(download_to_file(&url, &archive, 4).await.is_err());
        server.await.unwrap();
        assert!(!archive.exists() || std::fs::metadata(&archive).unwrap().len() <= 4);
    }

    #[tokio::test]
    async fn bounded_loopback_download_rejects_checksum_mismatch() {
        let payload = b"tampered forge release".to_vec();
        let (url, server) = loopback_body(payload).await;
        let dir = TestDir::new();
        let archive = dir.path().join("forge-linux-x86_64.tar.gz");
        let manifest = dir.path().join("SHA256SUMS");
        download_to_file(&url, &archive, 1024).await.unwrap();
        std::fs::write(
            &manifest,
            format!(
                "{}  forge-linux-x86_64.tar.gz\n",
                digest(b"different payload")
            ),
        )
        .unwrap();
        assert!(verify_archive_checksum(&archive, &manifest, "forge-linux-x86_64.tar.gz").is_err());
        server.await.unwrap();
    }
}
