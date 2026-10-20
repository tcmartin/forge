/// Plugin discovery and loading — ported from `pluginLoader.ts` / `pluginDirectories.ts`.
///
/// Scan order (matches TS precedence):
/// 1. `~/.claurst/plugins/<name>/`  — user-global plugins
/// 2. `<project>/.claurst/plugins/<name>/`  — project-local plugins
/// 3. Extra paths from `settings.plugin_paths` (if the field exists)
///
/// Each plugin directory must contain a `plugin.json` or `plugin.toml`
/// manifest file.  A bare manifest file (no containing directory) is also
/// accepted.
use crate::manifest::{PluginHooksConfig, PluginManifest};
use crate::plugin::{LoadedPlugin, PluginError, PluginSource};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Return the default user-level plugins directory: `<claurst home>/plugins`.
pub fn default_user_plugins_dir() -> Option<PathBuf> {
    Some(claurst_core::config::Settings::config_dir().join("plugins"))
}

/// Return the project-level plugins directory: `<project>/.claurst/plugins`.
pub fn project_plugins_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(".claurst").join("plugins")
}

// ---------------------------------------------------------------------------
// Core loader
// ---------------------------------------------------------------------------

/// Discover and load all plugins from the given root directories.
///
/// Each directory in `search_dirs` is scanned at depth 1: every immediate
/// subdirectory (or manifest file) is treated as a candidate plugin.
pub async fn discover_plugins(
    search_dirs: &[PathBuf],
    source: PluginSource,
) -> (Vec<LoadedPlugin>, Vec<PluginError>) {
    let mut plugins: Vec<LoadedPlugin> = Vec::new();
    let mut errors: Vec<PluginError> = Vec::new();

    for dir in search_dirs {
        if !dir.exists() {
            continue;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                errors.push(PluginError::Io {
                    path: dir.to_string_lossy().into_owned(),
                    message: e.to_string(),
                });
                continue;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            match try_load_from_path(&path, source.clone()) {
                Ok(Some(plugin)) => plugins.push(plugin),
                Ok(None) => {}
                Err(e) => errors.push(e),
            }
        }
    }

    (plugins, errors)
}

/// Try to load a plugin from a filesystem path.
///
/// `path` can be:
/// - A directory containing `plugin.json` or `plugin.toml`
/// - A direct `plugin.json` or `plugin.toml` file
///
/// Returns `Ok(None)` if the path does not look like a plugin (no manifest
/// found) without adding an error.
pub fn try_load_from_path(
    path: &Path,
    source: PluginSource,
) -> Result<Option<LoadedPlugin>, PluginError> {
    let (plugin_dir, manifest_path) = if path.is_dir() {
        // Look for manifest inside the directory.
        let json_path = path.join("plugin.json");
        let toml_path = path.join("plugin.toml");

        if json_path.exists() {
            (path.to_path_buf(), json_path)
        } else if toml_path.exists() {
            (path.to_path_buf(), toml_path)
        } else {
            // Directory with no manifest — not a plugin, skip silently.
            return Ok(None);
        }
    } else if path.is_file() {
        // Accept a bare manifest file.
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == "plugin.json" || name == "plugin.toml" {
            let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            (parent, path.to_path_buf())
        } else {
            return Ok(None);
        }
    } else {
        return Ok(None);
    };

    let manifest = load_manifest(&manifest_path)?;

    // Resolve sub-paths.
    let commands_path = {
        let p = plugin_dir.join("commands");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    let agents_path = {
        let p = plugin_dir.join("agents");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    let skills_path = {
        let p = plugin_dir.join("skills");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };
    let output_styles_path = {
        let p = plugin_dir.join("output-styles");
        if p.is_dir() {
            Some(p)
        } else {
            None
        }
    };

    // Load hooks config (hooks/hooks.json takes priority over inline manifest field).
    let hooks_config = load_hooks_config(&plugin_dir, &manifest);

    let plugin_name = manifest.name.clone();
    let plugin_source_id = format!("{}@{}", plugin_name, source.label());

    Ok(Some(LoadedPlugin {
        name: plugin_name,
        path: plugin_dir,
        source: source.clone(),
        source_id: plugin_source_id,
        manifest,
        enabled: true,
        commands_path,
        agents_path,
        skills_path,
        output_styles_path,
        hooks_config,
    }))
}

// ---------------------------------------------------------------------------
// Manifest loading
// ---------------------------------------------------------------------------

fn load_manifest(path: &Path) -> Result<PluginManifest, PluginError> {
    let bytes = std::fs::read(path).map_err(|e| PluginError::Io {
        path: path.to_string_lossy().into_owned(),
        message: e.to_string(),
    })?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("json");

    let manifest = match ext {
        "toml" => PluginManifest::from_toml(&bytes).map_err(|e| PluginError::InvalidManifest {
            path: path.to_string_lossy().into_owned(),
            message: e.to_string(),
        })?,
        _ => PluginManifest::from_json(&bytes).map_err(|e| PluginError::InvalidManifest {
            path: path.to_string_lossy().into_owned(),
            message: e.to_string(),
        })?,
    };

    Ok(manifest)
}

// ---------------------------------------------------------------------------
// Hooks loading
// ---------------------------------------------------------------------------

/// Load hooks for a plugin.
///
/// Priority:
/// 1. `hooks/hooks.json` inside the plugin directory
/// 2. Inline `hooks` field in the manifest
pub fn load_hooks_config(
    plugin_dir: &Path,
    manifest: &PluginManifest,
) -> Option<PluginHooksConfig> {
    // 1. File-based hooks.
    let hooks_file = plugin_dir.join("hooks").join("hooks.json");
    if hooks_file.exists() {
        if let Ok(bytes) = std::fs::read(&hooks_file) {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(config) = crate::hooks::parse_hooks_value(&value) {
                    return Some(config);
                }
            }
        }
    }

    // 2. Inline hooks in manifest.
    if let Some(ref inline) = manifest.hooks {
        if let Some(config) = crate::hooks::parse_hooks_value(inline) {
            return Some(config);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Command definitions collected from a plugin
// ---------------------------------------------------------------------------

/// Collect plugin commands and skill entrypoints under the plugin namespace.
pub fn collect_command_defs(plugin: &LoadedPlugin) -> Vec<crate::plugin::PluginCommandDef> {
    let mut defs: Vec<crate::plugin::PluginCommandDef> = Vec::new();
    let capabilities = plugin.manifest.capabilities.clone();

    // Commands from the `commands/` directory.
    if let Some(ref cmd_dir) = plugin.commands_path {
        collect_markdown_commands(
            cmd_dir,
            &plugin.path,
            &plugin.name,
            capabilities.clone(),
            &mut defs,
        );
    }

    if let Some(ref skill_dir) = plugin.skills_path {
        for entrypoint in claurst_core::skill_discovery::discover_skill_files(skill_dir) {
            collect_markdown_commands(
                &entrypoint,
                &plugin.path,
                &plugin.name,
                capabilities.clone(),
                &mut defs,
            );
        }
    }

    // Extra commands declared in the manifest.
    for rel_path in &plugin.manifest.commands {
        let abs = plugin.path.join(rel_path.trim_start_matches("./"));
        if abs.is_file() && abs.extension().map(|e| e == "md").unwrap_or(false) {
            let cmd_name = command_name_from_file(&abs, &plugin.name);
            defs.push(crate::plugin::PluginCommandDef {
                name: cmd_name,
                description: if crate::plugin_capability_granted(
                    capabilities.as_deref(),
                    "read_files",
                ) {
                    extract_description_from_markdown_file(&abs)
                        .unwrap_or_else(|| "Plugin command".to_string())
                } else {
                    "Plugin command".to_string()
                },
                plugin_name: plugin.name.clone(),
                plugin_source_id: plugin.source_id.clone(),
                run_action: crate::plugin::CommandRunAction::MarkdownPrompt {
                    file_path: abs.to_string_lossy().into_owned(),
                    plugin_root: plugin.path.to_string_lossy().into_owned(),
                },
                plugin_capabilities: capabilities.clone(),
            });
        } else if abs.is_dir() {
            collect_markdown_commands(
                &abs,
                &plugin.path,
                &plugin.name,
                capabilities.clone(),
                &mut defs,
            );
        }
    }

    defs
}

/// Recursively collect .md files from `dir` into `PluginCommandDef` items.
fn collect_markdown_commands(
    dir: &Path,
    plugin_root: &Path,
    plugin_name: &str,
    capabilities: Option<Vec<String>>,
    defs: &mut Vec<crate::plugin::PluginCommandDef>,
) {
    use walkdir::WalkDir;

    for entry in WalkDir::new(dir)
        .min_depth(usize::from(dir.is_dir()))
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // SKILL.md — use parent directory name as command name.
        if file_name.eq_ignore_ascii_case("skill.md") {
            let skill_dir = path.parent().unwrap_or(dir);
            let base_name = skill_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("skill");
            let cmd_name = format!("{}:{}", plugin_name, base_name);
            defs.push(crate::plugin::PluginCommandDef {
                name: cmd_name,
                description: if crate::plugin_capability_granted(
                    capabilities.as_deref(),
                    "read_files",
                ) {
                    extract_description_from_markdown_file(path)
                        .unwrap_or_else(|| "Plugin skill".to_string())
                } else {
                    "Plugin skill".to_string()
                },
                plugin_name: plugin_name.to_string(),
                plugin_source_id: String::new(),
                run_action: crate::plugin::CommandRunAction::MarkdownPrompt {
                    file_path: path.to_string_lossy().into_owned(),
                    plugin_root: plugin_root.to_string_lossy().into_owned(),
                },
                plugin_capabilities: capabilities.clone(),
            });
            continue;
        }

        if path.extension().map(|e| e == "md").unwrap_or(false) {
            let cmd_name = command_name_from_file(path, plugin_name);
            defs.push(crate::plugin::PluginCommandDef {
                name: cmd_name,
                description: if crate::plugin_capability_granted(
                    capabilities.as_deref(),
                    "read_files",
                ) {
                    extract_description_from_markdown_file(path)
                        .unwrap_or_else(|| "Plugin command".to_string())
                } else {
                    "Plugin command".to_string()
                },
                plugin_name: plugin_name.to_string(),
                plugin_source_id: String::new(),
                run_action: crate::plugin::CommandRunAction::MarkdownPrompt {
                    file_path: path.to_string_lossy().into_owned(),
                    plugin_root: plugin_root.to_string_lossy().into_owned(),
                },
                plugin_capabilities: capabilities.clone(),
            });
        }
    }
}

/// Derive a slash-command name from a markdown file path.
///
/// e.g. `<plugin_dir>/commands/build/deploy.md` → `myplugin:build:deploy`
fn command_name_from_file(path: &Path, plugin_name: &str) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("cmd");
    format!("{}:{}", plugin_name, stem)
}

/// Read a short description from the first 4 KiB without loading the prompt body.
fn extract_description_from_markdown_file(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    for line in BufReader::new(file.take(4096)).lines() {
        let line = line.ok()?;
        let trimmed = line.trim_start_matches('#').trim();
        if !trimmed.is_empty() {
            return Some(trimmed.chars().take(200).collect());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_entrypoints_share_plugin_namespace_and_capability_gate() {
        let root = tempfile::tempdir().unwrap();
        let skills = root.path().join("skills");
        std::fs::create_dir_all(skills.join("review/references")).unwrap();
        std::fs::write(
            root.path().join("plugin.json"),
            r#"{"name":"fixture","capabilities":["read_files"]}"#,
        )
        .unwrap();
        std::fs::write(skills.join("review/SKILL.md"), "Review the change.").unwrap();
        std::fs::write(
            skills.join("review/references/example.md"),
            "Reference material.",
        )
        .unwrap();
        std::fs::write(skills.join("flat.md"), "Flat skill.").unwrap();
        let mut plugin = try_load_from_path(root.path(), PluginSource::Project)
            .unwrap()
            .unwrap();
        let defs = collect_command_defs(&plugin);
        assert_eq!(defs.len(), 2);
        let entry = defs
            .iter()
            .find(|def| def.name == "fixture:review")
            .unwrap();
        assert!(crate::check_plugin_capability(entry).is_ok());
        assert!(
            matches!(&entry.run_action, crate::plugin::CommandRunAction::MarkdownPrompt {
            file_path, plugin_root,
        } if Path::new(file_path) == skills.join("review/SKILL.md") && Path::new(plugin_root) == root.path())
        );
        plugin.manifest.capabilities = None;
        let denied = collect_command_defs(&plugin);
        assert!(denied
            .iter()
            .all(|def| crate::check_plugin_capability(def).is_err()));
        assert!(denied
            .iter()
            .all(|def| matches!(def.description.as_str(), "Plugin skill" | "Plugin command")));
    }

    #[test]
    fn metadata_descriptions_are_bounded_and_unicode_safe() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("large.md");
        std::fs::write(&file, format!("# {}\nBody.", "界".repeat(500))).unwrap();
        let description = extract_description_from_markdown_file(&file).unwrap();
        assert_eq!(description.chars().count(), 200);
        assert!(description.chars().all(|c| c == '界'));
        std::fs::write(
            &file,
            format!("{}Outside metadata window.", "\n".repeat(4096)),
        )
        .unwrap();
        assert!(extract_description_from_markdown_file(&file).is_none());
    }
}
