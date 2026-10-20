//! End-to-end smoke test: spawn the `forge` binary in ACP mode, send a
//! short JSON-RPC conversation over its stdin, and verify the responses on
//! stdout match what the Agent Client Protocol spec mandates.
//!
//! This guards the wire-format and capability surface that registry-listed
//! ACP clients (Zed, Neovim, JetBrains, …) rely on. Runs against the
//! debug binary produced by `cargo build` — Cargo provides the path via
//! `CARGO_BIN_EXE_forge`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

fn binary_path() -> String {
    env!("CARGO_BIN_EXE_forge").to_string()
}

fn run_with_input(stdin: &str, timeout: Duration) -> (String, String) {
    let root = test_root("isolated");
    let output = run_with_input_in_root(stdin, timeout, &root);
    std::fs::remove_dir_all(root).expect("remove isolated ACP root");
    output
}

fn test_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("forge-acp-{label}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(root.join("home")).expect("create isolated home");
    std::fs::create_dir_all(root.join("data")).expect("create isolated data root");
    root
}

fn run_with_input_in_root(stdin: &str, timeout: Duration, root: &Path) -> (String, String) {
    let mut child = Command::new(binary_path())
        .arg("acp")
        .env("HOME", root.join("home"))
        .env("FORGE_HOME", root.join("data"))
        .env("CLAURST_HOME", root.join("data"))
        .env("CLAURST_DISABLE_MODELS_FETCH", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn forge acp");

    {
        let mut stdin_handle = child.stdin.take().expect("stdin");
        stdin_handle
            .write_all(stdin.as_bytes())
            .expect("write stdin");
        // Dropping stdin signals EOF — the agent will finish in-flight work
        // and then exit cleanly.
    }

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait().expect("try_wait") {
            Some(_status) => break,
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("forge acp did not exit within {timeout:?}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }

    let output = child.wait_with_output().expect("wait_with_output");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn find_response(stdout: &str, id: i64) -> serde_json::Value {
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if v.get("id") == Some(&serde_json::json!(id)) {
                return v;
            }
        }
    }
    panic!("no response for id={id} in stdout:\n{stdout}");
}

#[test]
fn initialize_returns_spec_compliant_response() {
    let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":true,"writeTextFile":true},"terminal":false},"clientInfo":{"name":"smoke","version":"0.0.0"}}}
"#;
    let (stdout, _stderr) = run_with_input(request, Duration::from_secs(20));
    let resp = find_response(&stdout, 1);
    assert_eq!(resp["jsonrpc"], "2.0");
    let result = &resp["result"];
    assert_eq!(result["protocolVersion"], 1);
    // Agent identifies itself.
    assert_eq!(result["agentInfo"]["name"], "forge");
    assert_eq!(result["agentInfo"]["title"], "Forge");
    assert!(result["agentInfo"]["version"].is_string());
    // authMethods MUST be an array (even if empty).
    assert!(result["authMethods"].is_array());
    // Standard capability blocks present.
    let caps = &result["agentCapabilities"];
    assert!(caps["promptCapabilities"].is_object());
    assert!(caps["mcpCapabilities"].is_object());
    assert_eq!(caps["loadSession"], true);
}

#[test]
fn session_new_returns_session_id() {
    // Use a path that's absolute on both Unix and Windows.
    let cwd = std::env::current_dir().expect("cwd");
    let cwd_str = cwd.to_string_lossy().replace('\\', "/");
    let request = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1,"clientCapabilities":{{}},"clientInfo":{{"name":"smoke","version":"0.0.0"}}}}}}
{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{cwd_str}","mcpServers":[]}}}}
"#
    );
    let (stdout, _stderr) = run_with_input(&request, Duration::from_secs(20));
    let resp = find_response(&stdout, 2);
    let session_id = resp["result"]["sessionId"]
        .as_str()
        .expect("sessionId should be a string");
    assert!(
        session_id.starts_with("acp-"),
        "sessionId not prefixed: {session_id}"
    );
}

#[test]
fn session_load_survives_restart_and_enforces_workspace_binding() {
    let root = test_root("restart");
    let workspace = root.join("workspace");
    let foreign = root.join("foreign");
    std::fs::create_dir(&workspace).expect("create workspace");
    std::fs::create_dir(&foreign).expect("create foreign workspace");
    let cwd = workspace.to_string_lossy().replace('\\', "/");
    let foreign_cwd = foreign.to_string_lossy().replace('\\', "/");
    let new_request = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1,"clientCapabilities":{{}},"clientInfo":{{"name":"smoke","version":"0.0.0"}}}}}}
{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{"cwd":"{cwd}","mcpServers":[]}}}}
"#
    );
    let (stdout, stderr) = run_with_input_in_root(&new_request, Duration::from_secs(20), &root);
    assert!(stderr.is_empty(), "unexpected ACP stderr: {stderr}");
    let session_id = find_response(&stdout, 2)["result"]["sessionId"]
        .as_str()
        .expect("session id")
        .to_string();
    let record = root
        .join("data/acp-sessions")
        .join(format!("{session_id}.json"));
    assert!(record.is_file(), "durable ACP record missing: {record:?}");

    let load_request = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1,"clientCapabilities":{{}},"clientInfo":{{"name":"smoke","version":"0.0.0"}}}}}}
{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{cwd}","mcpServers":[]}}}}
"#
    );
    let (stdout, stderr) = run_with_input_in_root(&load_request, Duration::from_secs(20), &root);
    assert!(stderr.is_empty(), "unexpected ACP stderr: {stderr}");
    let loaded = find_response(&stdout, 2);
    assert!(
        loaded["result"].is_object(),
        "load did not return an empty result"
    );
    assert!(loaded.get("error").is_none());

    let foreign_request = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1,"clientCapabilities":{{}},"clientInfo":{{"name":"smoke","version":"0.0.0"}}}}}}
{{"jsonrpc":"2.0","id":2,"method":"session/load","params":{{"sessionId":"{session_id}","cwd":"{foreign_cwd}","mcpServers":[]}}}}
"#
    );
    let (stdout, _stderr) =
        run_with_input_in_root(&foreign_request, Duration::from_secs(20), &root);
    assert_eq!(find_response(&stdout, 2)["error"]["code"], -32602);
    std::fs::remove_dir_all(root).expect("remove restart ACP root");
}

#[test]
fn cancel_notification_is_silent() {
    // session/cancel is a notification — the agent MUST NOT respond, even
    // for an unknown session.
    let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{},"clientInfo":{"name":"smoke","version":"0.0.0"}}}
{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"nonexistent"}}
"#;
    let (stdout, _stderr) = run_with_input(request, Duration::from_secs(20));
    // Only the initialize response (id=1) should appear; no extra responses.
    let response_count = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter(|l| serde_json::from_str::<serde_json::Value>(l).is_ok())
        .count();
    assert_eq!(
        response_count, 1,
        "unexpected extra responses in:\n{stdout}"
    );
}
