/// Plugin hook execution — ported from `loadPluginHooks.ts`.
///
/// Hooks let plugins run shell commands in response to lifecycle events.
/// This module defines the data model and bounded asynchronous dispatch.
use crate::manifest::{HookEventKind, PluginHookMatcher, PluginHooksConfig};
use claurst_core::hooks::{HookCancellation, HookContext, HookEnvironment};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Registered hook set
// ---------------------------------------------------------------------------

/// A hook entry enriched with the plugin context it came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredHook {
    /// Shell command to execute.
    pub command: String,
    /// Optional tool-name matcher (glob accepted by the shell runner).
    pub matcher: Option<String>,
    /// Whether a non-zero exit code should block the operation.
    pub blocking: bool,
    /// Absolute path to the plugin root directory.
    pub plugin_root: String,
    /// Plugin name for display / logging.
    pub plugin_name: String,
    /// Plugin source identifier (e.g. `"myplugin@builtin"`).
    pub plugin_source: String,
}

/// All registered hooks for a running session, keyed by event name.
pub type HookRegistry = HashMap<String, Vec<RegisteredHook>>;

// ---------------------------------------------------------------------------
// Building the registry from loaded plugins
// ---------------------------------------------------------------------------

/// Convert raw `PluginHooksConfig` (from `hooks/hooks.json` or inline in the
/// manifest) into `RegisteredHook` entries and merge them into `registry`.
pub fn register_plugin_hooks(
    config: &PluginHooksConfig,
    plugin_root: &str,
    plugin_name: &str,
    plugin_source: &str,
    registry: &mut HookRegistry,
) {
    for (event_name, matchers) in &config.events {
        let registered_hooks = registry.entry(event_name.clone()).or_default();

        for matcher in matchers {
            for hook in &matcher.hooks {
                registered_hooks.push(RegisteredHook {
                    command: hook.command.clone(),
                    matcher: matcher.matcher.clone().or_else(|| hook.matcher.clone()),
                    blocking: hook.blocking,
                    plugin_root: plugin_root.to_string(),
                    plugin_name: plugin_name.to_string(),
                    plugin_source: plugin_source.to_string(),
                });
            }
        }
    }
}

/// Remove all hooks whose `plugin_root` is not in `active_roots`.
/// Used when plugins are disabled / uninstalled so their hooks stop firing.
pub fn prune_hooks(registry: &mut HookRegistry, active_roots: &std::collections::HashSet<String>) {
    for hooks in registry.values_mut() {
        hooks.retain(|h| active_roots.contains(&h.plugin_root));
    }
}

/// Parse a raw `serde_json::Value` that may be either a flat hooks object
/// (`{ "PreToolUse": [...] }`) or a wrapped object with a `hooks` key
/// (`{ "description": "...", "hooks": { "PreToolUse": [...] } }`).
///
/// Returns `None` when the value cannot be parsed (errors are logged).
pub fn parse_hooks_value(value: &serde_json::Value) -> Option<PluginHooksConfig> {
    // Try the wrapped form first.
    if let Some(inner) = value.get("hooks") {
        let description = value
            .get("description")
            .and_then(|d| d.as_str())
            .map(String::from);
        let events = parse_hooks_events_map(inner)?;
        return Some(PluginHooksConfig {
            description,
            events,
        });
    }

    // Fall back: the whole value is the events map.
    let events = parse_hooks_events_map(value)?;
    Some(PluginHooksConfig {
        description: None,
        events,
    })
}

/// Parse the events map `{ "PreToolUse": [ { matcher: ..., hooks: [...] } ] }`.
fn parse_hooks_events_map(
    value: &serde_json::Value,
) -> Option<HashMap<String, Vec<PluginHookMatcher>>> {
    let obj = value.as_object()?;
    let mut events: HashMap<String, Vec<PluginHookMatcher>> = HashMap::new();

    for (event_name, matchers_val) in obj {
        let matchers_arr = match matchers_val.as_array() {
            Some(a) => a,
            None => {
                tracing::warn!(
                    "Plugin hooks: expected array for event '{}', got something else",
                    event_name
                );
                continue;
            }
        };

        let mut parsed_matchers: Vec<PluginHookMatcher> = Vec::new();
        for mv in matchers_arr {
            match serde_json::from_value::<PluginHookMatcher>(mv.clone()) {
                Ok(m) => parsed_matchers.push(m),
                Err(e) => {
                    tracing::warn!(
                        "Plugin hooks: failed to parse matcher for '{}': {}",
                        event_name,
                        e
                    );
                }
            }
        }

        if !parsed_matchers.is_empty() {
            events.insert(event_name.clone(), parsed_matchers);
        }
    }

    Some(events)
}

// ---------------------------------------------------------------------------
// Hook execution result
// ---------------------------------------------------------------------------

/// What happened when a hook ran.
#[derive(Debug, Clone)]
pub enum HookOutcome {
    /// Hook exited 0 — allow the operation.
    Allow,
    /// Hook exited non-zero and was blocking — deny the operation.
    Deny(String),
    /// Hook failed to start / timed out — treated as allow (non-blocking failure).
    Error(String),
}

// ---------------------------------------------------------------------------
// Hook runner
// ---------------------------------------------------------------------------

fn plugin_environment<'a>(
    hook: &'a RegisteredHook,
    additional: &'a [(&'a str, &'a str)],
) -> Result<Vec<HookEnvironment<'a>>, String> {
    let mut environment = Vec::with_capacity(additional.len() + 2);
    for (name, value) in [
        ("CLAUDE_PLUGIN_ROOT", hook.plugin_root.as_str()),
        ("CLAUDE_PLUGIN_NAME", hook.plugin_name.as_str()),
    ]
    .into_iter()
    .chain(additional.iter().copied())
    {
        environment.push(HookEnvironment::new(name, value).map_err(|error| {
            format!(
                "Plugin '{}' hook environment rejected: {error}",
                hook.plugin_name
            )
        })?);
    }
    Ok(environment)
}

async fn run_registered_hook_core<'a>(
    hook: &RegisteredHook,
    context: &HookContext,
    working_dir: &Path,
    cancel: &HookCancellation,
    additional_environment: &'a [(&'a str, &'a str)],
) -> Result<claurst_core::hooks::HookOutcome, String> {
    let environment = plugin_environment(hook, additional_environment)?;
    let entry = claurst_core::config::HookEntry {
        command: hook.command.clone(),
        tool_filter: hook.matcher.clone(),
        blocking: hook.blocking,
    };
    Ok(claurst_core::hooks::run_hook_with_environment(
        &entry,
        context,
        working_dir,
        cancel,
        &environment,
    )
    .await)
}

/// Execute one plugin hook through core's bounded async supervisor.
pub async fn run_registered_hook(
    hook: &RegisteredHook,
    context: &HookContext,
    working_dir: &Path,
    cancel: &HookCancellation,
    additional_environment: &[(&str, &str)],
) -> HookOutcome {
    match run_registered_hook_core(hook, context, working_dir, cancel, additional_environment).await
    {
        Err(reason) => HookOutcome::Error(reason),
        Ok(claurst_core::hooks::HookOutcome::Allowed)
        | Ok(claurst_core::hooks::HookOutcome::Modified(_)) => HookOutcome::Allow,
        Ok(claurst_core::hooks::HookOutcome::Blocked(reason))
        | Ok(claurst_core::hooks::HookOutcome::Vetoed(reason)) => HookOutcome::Deny(reason),
        Ok(claurst_core::hooks::HookOutcome::Failed(reason)) => {
            tracing::debug!(plugin = %hook.plugin_name, %reason, "non-blocking plugin hook failed");
            HookOutcome::Allow
        }
        Ok(claurst_core::hooks::HookOutcome::SupervisionFailed(reason)) => {
            HookOutcome::Error(reason)
        }
    }
}

/// Return the canonical string key for a `HookEventKind`.
pub fn event_key(kind: &HookEventKind) -> String {
    kind.to_string()
}

// ---------------------------------------------------------------------------
// PostToolUse hook dispatch (T2-14)
// ---------------------------------------------------------------------------

/// Dispatch PostToolUse hooks for a completed tool call.
///
/// Mirrors the TS plugin hook dispatch path in src/plugins/hooks.ts.
/// All matching hooks run through the shared bounded supervisor; results are
/// collected in registry order.
/// Non-blocking: non-zero exit logs a warning but doesn't fail the tool call.
pub async fn dispatch_post_tool_hooks(
    registry: &HookRegistry,
    tool_name: &str,
    tool_input_json: &str,
    tool_result_json: &str,
) -> Vec<String> {
    dispatch_post_tool_hooks_with_cancel(
        registry,
        tool_name,
        tool_input_json,
        tool_result_json,
        false,
        PathBuf::from("."),
        HookCancellation::new(),
    )
    .await
}

/// Dispatch PostToolUse hooks through the bounded supervisor with the query's
/// working directory and cancellation signal.
pub async fn dispatch_post_tool_hooks_with_cancel(
    registry: &HookRegistry,
    tool_name: &str,
    tool_input_json: &str,
    tool_result_json: &str,
    is_error: bool,
    working_dir: PathBuf,
    cancel: HookCancellation,
) -> Vec<String> {
    let event_name = event_key(&HookEventKind::PostToolUse);

    let matching_hooks: Vec<RegisteredHook> = match registry.get(&event_name) {
        None => return Vec::new(),
        Some(hooks) => hooks
            .iter()
            .filter(|h| {
                let pattern = h.matcher.as_deref().unwrap_or("*");
                glob_match(pattern, tool_name)
            })
            .cloned()
            .collect(),
    };

    if matching_hooks.is_empty() {
        return Vec::new();
    }

    let context = HookContext {
        event: "PostToolUse".to_string(),
        tool_name: Some(tool_name.to_string()),
        tool_input: serde_json::from_str(tool_input_json).ok(),
        tool_output: Some(tool_result_json.to_string()),
        is_error: Some(is_error),
        session_id: None,
    };
    let additional_environment = [
        ("CLAUDE_TOOL_NAME", tool_name),
        ("CLAUDE_TOOL_INPUT", tool_input_json),
        ("CLAUDE_TOOL_RESULT", tool_result_json),
    ];
    let mut outputs = Vec::new();
    for hook in matching_hooks {
        match run_registered_hook_core(
            &hook,
            &context,
            &working_dir,
            &cancel,
            &additional_environment,
        )
        .await
        {
            Ok(claurst_core::hooks::HookOutcome::Modified(output)) if !output.trim().is_empty() => {
                outputs.push(output.trim().to_string())
            }
            Ok(claurst_core::hooks::HookOutcome::SupervisionFailed(reason)) => {
                tracing::warn!(plugin = %hook.plugin_name, %reason, "PostToolUse hook supervision failed");
            }
            Err(reason) => {
                tracing::warn!(plugin = %hook.plugin_name, %reason, "PostToolUse hook environment rejected");
            }
            Ok(claurst_core::hooks::HookOutcome::Blocked(reason))
            | Ok(claurst_core::hooks::HookOutcome::Vetoed(reason))
            | Ok(claurst_core::hooks::HookOutcome::Failed(reason)) => {
                tracing::debug!(plugin = %hook.plugin_name, %reason, "PostToolUse hook returned non-success");
            }
            Ok(claurst_core::hooks::HookOutcome::Allowed)
            | Ok(claurst_core::hooks::HookOutcome::Modified(_)) => {}
        }
    }
    outputs
}

/// Simple glob pattern matching: `*` matches any sequence of chars.
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if pattern == text {
        return true;
    }
    // Handle patterns like "File*" or "*Tool"
    if let Some(prefix) = pattern.strip_suffix('*') {
        return text.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return text.ends_with(suffix);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_wrapped_hooks_value() {
        let raw = serde_json::json!({
            "description": "test hooks",
            "hooks": {
                "PreToolUse": [
                    { "matcher": "Bash", "hooks": [{ "command": "echo pre", "blocking": false }] }
                ]
            }
        });
        let config = parse_hooks_value(&raw).unwrap();
        assert_eq!(config.description.as_deref(), Some("test hooks"));
        assert!(config.events.contains_key("PreToolUse"));
    }

    #[test]
    fn parse_flat_hooks_value() {
        let raw = serde_json::json!({
            "Stop": [
                { "hooks": [{ "command": "echo stop", "blocking": false }] }
            ]
        });
        let config = parse_hooks_value(&raw).unwrap();
        assert!(config.events.contains_key("Stop"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bounded_post_runner_preserves_plugin_and_tool_environment() {
        let mut registry = HookRegistry::new();
        registry.insert(
            HookEventKind::PostToolUse.to_string(),
            vec![RegisteredHook {
                command: "printf '%s|%s|%s|%s|%s' \"$CLAUDE_PLUGIN_ROOT\" \"$CLAUDE_PLUGIN_NAME\" \"$CLAUDE_TOOL_NAME\" \"$CLAUDE_TOOL_INPUT\" \"$CLAUDE_TOOL_RESULT\"".to_string(),
                matcher: None,
                blocking: false,
                plugin_root: "/plugin/root".to_string(),
                plugin_name: "fixture".to_string(),
                plugin_source: "fixture@user".to_string(),
            }],
        );
        let output = dispatch_post_tool_hooks_with_cancel(
            &registry,
            "Bash",
            "{\"arg\":1}",
            "tool-result",
            false,
            PathBuf::from("."),
            HookCancellation::new(),
        )
        .await;
        assert_eq!(
            output,
            vec!["/plugin/root|fixture|Bash|{\"arg\":1}|tool-result".to_string()]
        );
    }
}
