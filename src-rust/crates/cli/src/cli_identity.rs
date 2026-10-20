//! Forge product identity and isolated data-root initialization.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const PRODUCT_NAME: &str = "forge";
pub const FORGE_HOME_ENV: &str = "FORGE_HOME";
pub const CLAURST_HOME_ENV: &str = "CLAURST_HOME";

/// The query planner has the same upper bound, keeping CLI validation
/// fail-closed before any provider or inference work starts.
pub const MAX_CONTEXT_LIMIT_TOKENS: u64 = 100_000_000;
/// A context smaller than this cannot hold a useful system/request frame.
pub const MIN_CONTEXT_LIMIT_TOKENS: u64 = 256;
/// A wire budget below one request frame cannot carry even a minimal prompt.
pub const MIN_WIRE_LIMIT_BYTES: u64 = 4 * 1024;
/// Keep an accidental unit conversion from requesting an unbounded body.
pub const MAX_WIRE_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;

pub fn parse_context_limit(raw: &str) -> Result<u64, String> {
    parse_finite_limit(
        raw,
        "context token limit",
        MIN_CONTEXT_LIMIT_TOKENS,
        MAX_CONTEXT_LIMIT_TOKENS,
    )
}

pub fn parse_wire_limit_bytes(raw: &str) -> Result<u64, String> {
    parse_finite_limit(
        raw,
        "wire byte limit",
        MIN_WIRE_LIMIT_BYTES,
        MAX_WIRE_LIMIT_BYTES,
    )
}

fn parse_finite_limit(raw: &str, label: &str, minimum: u64, maximum: u64) -> Result<u64, String> {
    let value = raw
        .parse::<u64>()
        .map_err(|_| format!("{label} must be an integer between {minimum} and {maximum}"))?;
    if !(minimum..=maximum).contains(&value) {
        return Err(format!("{label} must be between {minimum} and {maximum}"));
    }
    Ok(value)
}

/// Return a stable, filesystem-safe scope identifier for a workspace.
///
/// The absolute path is intentionally hashed: it keeps journals scoped to the
/// real workspace while avoiding path separators and leaking the path in the
/// history directory layout.
pub fn workspace_scope_id(path: &Path) -> String {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let digest = Sha256::digest(canonical.to_string_lossy().as_bytes());
    let mut id = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        use std::fmt::Write;
        let _ = write!(id, "{byte:02x}");
    }
    format!("workspace-{id}")
}

/// Select Forge's data root before any Claurst-owned path helper is called.
///
/// `Settings::config_dir` currently reads `CLAURST_HOME`; setting that bridge
/// early keeps the internal crates unchanged while preventing a fresh Forge
/// process from falling back to Claurst's legacy `~/.claurst` directory.
pub fn configure_data_root() {
    if let Some(root) = resolve_data_root(
        non_empty_env(FORGE_HOME_ENV).map(PathBuf::from),
        non_empty_env(CLAURST_HOME_ENV).map(PathBuf::from),
        dirs::home_dir(),
    ) {
        std::env::set_var(CLAURST_HOME_ENV, root);
    }
}

pub fn resolve_data_root(
    forge_home: Option<PathBuf>,
    claurst_home: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    forge_home
        .into_iter()
        .chain(claurst_home)
        .find(|path| !path.as_os_str().is_empty())
        .or_else(|| default_data_root(home))
}

pub fn default_data_root(home: Option<PathBuf>) -> Option<PathBuf> {
    home.map(|path| path.join(".config").join(PRODUCT_NAME))
}

fn non_empty_env(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_root_is_product_specific() {
        assert_eq!(
            default_data_root(Some(PathBuf::from("/tmp/home"))),
            Some(PathBuf::from("/tmp/home/.config/forge"))
        );
    }

    #[test]
    fn missing_home_has_no_default_root() {
        assert_eq!(default_data_root(None), None);
    }

    #[test]
    fn explicit_forge_root_wins_over_legacy_bridge() {
        assert_eq!(
            resolve_data_root(
                Some(PathBuf::from("/tmp/forge")),
                Some(PathBuf::from("/tmp/claurst")),
                Some(PathBuf::from("/tmp/home")),
            ),
            Some(PathBuf::from("/tmp/forge"))
        );
    }

    #[test]
    fn empty_legacy_bridge_uses_forge_default() {
        assert_eq!(
            resolve_data_root(None, Some(PathBuf::new()), Some(PathBuf::from("/tmp/home")),),
            Some(PathBuf::from("/tmp/home/.config/forge"))
        );
    }

    #[test]
    fn finite_budget_parsers_reject_zero_and_implausible_values() {
        assert_eq!(parse_context_limit("32000"), Ok(32000));
        assert!(parse_context_limit("0").is_err());
        assert!(parse_context_limit("255").is_err());
        assert!(parse_context_limit("100000001").is_err());
        assert_eq!(parse_wire_limit_bytes("4096"), Ok(4096));
        assert!(parse_wire_limit_bytes("0").is_err());
        assert!(parse_wire_limit_bytes("1023").is_err());
        assert!(parse_wire_limit_bytes("1073741825").is_err());
    }

    #[test]
    fn workspace_scope_is_stable_and_safe() {
        let first = workspace_scope_id(Path::new("/tmp/forge-workspace"));
        let second = workspace_scope_id(Path::new("/tmp/forge-workspace"));
        assert_eq!(first, second);
        assert!(first.starts_with("workspace-"));
        assert!(first.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
    }
}
