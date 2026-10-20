//! Build script for Claurst CLI
//!
//! Embeds build-time metadata (version, timestamp, git info) into the binary
//! for display and debugging purposes.

use std::process::Command;

fn main() {
    // The Windows MSVC default main-thread stack is 1 MiB. The Forge CLI's
    // tokio entry future is large enough to overflow that fixed stack before
    // ACP can emit its first response. Increase only this executable's stack;
    // other targets and Tokio worker stacks retain their defaults.
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(std::ffi::OsStr::new("windows"))
        && std::env::var_os("CARGO_CFG_TARGET_ENV").as_deref() == Some(std::ffi::OsStr::new("msvc"))
    {
        println!("cargo:rustc-link-arg-bin=forge=/STACK:8388608");
    }

    // Embed build timestamp (RFC 3339 format)
    let now = chrono::Utc::now().to_rfc3339();
    println!("cargo:rustc-env=BUILD_TIME={}", now);

    // Embed short git commit hash
    let commit = get_git_commit().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_COMMIT={}", commit);

    // Package/distribution metadata
    println!("cargo:rustc-env=PACKAGE_URL=claurst-source-snapshot");
    println!("cargo:rustc-env=FEEDBACK_CHANNEL=github");
    println!("cargo:rustc-env=ISSUES_EXPLAINER=This build does not include Anthropic internal issue routing.");

    // Trigger rebuild if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
}

/// Get the short git commit hash, or None if git is not available.
fn get_git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;

    if output.status.success() {
        let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !commit.is_empty() {
            return Some(commit);
        }
    }

    None
}
