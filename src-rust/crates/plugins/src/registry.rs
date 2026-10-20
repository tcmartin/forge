/// Plugin registry — holds all loaded plugins and provides queries.
///
/// Ported from the TS "enabled plugins" concept in `pluginLoader.ts` and the
/// app-state plugin arrays.
use crate::hooks::{register_plugin_hooks, HookRegistry};
use crate::plugin::{LoadedPlugin, PluginCommandDef, PluginError, ReloadDiff};
use claurst_core::config::{McpServerConfig, McpServerOrigin};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PluginRegistry
// ---------------------------------------------------------------------------

/// Central store for all discovered plugins in a session.
///
/// Methods follow the TS pattern: `enabled()` returns only enabled plugins,
/// `all()` returns every plugin including disabled ones.
#[derive(Debug, Default)]
pub struct PluginRegistry {
    /// All plugins keyed by name.
    plugins: HashMap<String, LoadedPlugin>,
    /// Names of plugins that are currently enabled.
    enabled_names: std::collections::HashSet<String>,
    /// Accumulated load errors.
    pub errors: Vec<PluginError>,
}

/// An MCP definition together with the immutable plugin identity that owns it.
/// Ownership is runtime metadata and is never serialized into Config.
#[derive(Debug, Clone)]
pub struct PluginMcpServer {
    pub owner_source_id: String,
    pub config: McpServerConfig,
}

impl PluginRegistry {
    // ---- Construction & population ----------------------------------------

    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a loaded plugin.  Emits a duplicate error if
    /// a different path already holds a plugin with the same name.
    pub fn insert(&mut self, plugin: LoadedPlugin) {
        let name = plugin.name.clone();
        let enabled = plugin.enabled;

        if let Some(existing) = self.plugins.get(&name) {
            if existing.path != plugin.path {
                self.errors.push(PluginError::DuplicateName {
                    name: name.clone(),
                    first: existing.path.to_string_lossy().into_owned(),
                    second: plugin.path.to_string_lossy().into_owned(),
                });
                // Keep the first one (same behaviour as TS: first-wins).
                return;
            }
        }

        self.plugins.insert(name.clone(), plugin);
        if enabled {
            self.enabled_names.insert(name);
        }
    }

    /// Append multiple plugins at once, updating errors inline.
    pub fn extend(&mut self, plugins: Vec<LoadedPlugin>, errors: Vec<PluginError>) {
        self.errors.extend(errors);
        for p in plugins {
            self.insert(p);
        }
    }

    // ---- Queries ----------------------------------------------------------

    /// All loaded plugins (enabled + disabled).
    pub fn all(&self) -> Vec<&LoadedPlugin> {
        self.plugins.values().collect()
    }

    /// Only the enabled plugins.
    pub fn enabled(&self) -> Vec<&LoadedPlugin> {
        self.plugins
            .values()
            .filter(|p| self.enabled_names.contains(&p.name))
            .collect()
    }

    /// Look up a plugin by name.
    pub fn get(&self, name: &str) -> Option<&LoadedPlugin> {
        self.plugins.get(name)
    }

    /// Whether a plugin is enabled.
    pub fn is_enabled(&self, name: &str) -> bool {
        self.enabled_names.contains(name)
    }

    /// Return whether any loaded plugin declares at least one hook entry.
    ///
    /// This intentionally inspects loaded definitions, including disabled
    /// plugins.  The eligibility gate is conservative and must account for
    /// hook configuration that is present in the immutable startup registry;
    /// hook execution still uses [`Self::enabled`] when building its runtime
    /// registry.
    pub fn has_configured_hooks(&self) -> bool {
        self.plugins.values().any(|plugin| {
            plugin.hooks_config.as_ref().is_some_and(|config| {
                config
                    .events
                    .values()
                    .any(|matchers| matchers.iter().any(|matcher| !matcher.hooks.is_empty()))
            })
        })
    }

    // ---- Enable / disable -------------------------------------------------

    /// Enable a plugin by name.  Returns `false` if the plugin is not loaded.
    pub fn enable(&mut self, name: &str) -> bool {
        if self.plugins.contains_key(name) {
            self.enabled_names.insert(name.to_string());
            if let Some(p) = self.plugins.get_mut(name) {
                p.enabled = true;
            }
            true
        } else {
            false
        }
    }

    /// Disable a plugin by name.  Returns `false` if the plugin is not loaded.
    pub fn disable(&mut self, name: &str) -> bool {
        if self.plugins.contains_key(name) {
            self.enabled_names.remove(name);
            if let Some(p) = self.plugins.get_mut(name) {
                p.enabled = false;
            }
            true
        } else {
            false
        }
    }

    // ---- Derived collections from enabled plugins -------------------------

    /// Collect all `PluginCommandDef` items from enabled plugins.
    pub fn all_command_defs(&self) -> Vec<PluginCommandDef> {
        let mut defs: Vec<PluginCommandDef> = Vec::new();
        for plugin in self.enabled() {
            let mut plugin_defs = crate::loader::collect_command_defs(plugin);
            // Patch source_id now that we have it.
            for d in &mut plugin_defs {
                d.plugin_source_id = plugin.source_id.clone();
            }
            defs.extend(plugin_defs);
        }
        defs
    }

    /// Publish enabled prompt sources without exposing plugin internals to tools.
    /// Config clones keep the same shared metadata across explicit reloads.
    pub fn publish_skill_sources(
        &self,
        config: &claurst_core::config::Config,
        project_dir: &std::path::Path,
    ) {
        use claurst_core::skill_discovery::{SkillOrigin, SkillSource};
        let mut sources = Vec::new();
        for plugin in self.enabled() {
            for definition in crate::loader::collect_command_defs(plugin) {
                if let crate::plugin::CommandRunAction::MarkdownPrompt { ref file_path, .. } =
                    definition.run_action
                {
                    sources.push(SkillSource {
                        name: definition.name.clone(),
                        path: file_path.into(),
                        origin: SkillOrigin::Plugin {
                            denial: crate::check_plugin_capability(&definition).err(),
                            project_root: (plugin.source == crate::plugin::PluginSource::Project)
                                .then(|| project_dir.to_path_buf()),
                        },
                    });
                }
            }
        }
        // Keep directory-entrypoint discovery available even when a loader
        // path is not represented as a command definition.  The source
        // carries the same capability and project-origin checks consumed by
        // SkillTool; avoid duplicate names when both surfaces expose the
        // same SKILL.md entrypoint.
        for source in self.all_skill_sources(project_dir) {
            if !sources
                .iter()
                .any(|existing| existing.name == source.name && existing.path == source.path)
            {
                sources.push(source);
            }
        }
        sources.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.path.cmp(&b.path)));
        *config.runtime_skills.write() = sources;
    }

    /// Collect paths to all `skills/` directories contributed by enabled plugins.
    pub fn all_skill_paths(&self) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(
                plugin.manifest.capabilities.as_deref(),
                "read_files",
            ) {
                continue;
            }
            if let Some(ref p) = plugin.skills_path {
                paths.push(p.clone());
            }
        }
        paths
    }

    /// Discover enabled plugin directory skills with their immutable origin
    /// and capability metadata attached. SkillTool consumes this snapshot so
    /// directory skills follow the same trust and reload lifecycle as plugin
    /// prompt commands.
    pub fn all_skill_sources(
        &self,
        project_dir: &std::path::Path,
    ) -> Vec<claurst_core::skill_discovery::SkillSource> {
        use claurst_core::skill_discovery::{skill_name_from_path, SkillOrigin, SkillSource};

        let mut sources = Vec::new();
        for plugin in self.enabled() {
            let read_files_granted = crate::plugin_capability_granted(
                plugin.manifest.capabilities.as_deref(),
                "read_files",
            );
            let denial =
                (!read_files_granted).then(|| "Plugin requires read_files capability".to_string());
            let project_root = (plugin.source == crate::plugin::PluginSource::Project)
                .then(|| project_dir.to_path_buf());
            let Some(skills_path) = &plugin.skills_path else {
                continue;
            };
            for path in claurst_core::skill_discovery::discover_skill_files(skills_path) {
                sources.push(SkillSource {
                    name: format!("{}:{}", plugin.name, skill_name_from_path(&path)),
                    path,
                    origin: SkillOrigin::Plugin {
                        denial: denial.clone(),
                        project_root: project_root.clone(),
                    },
                });
            }
        }
        sources.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.path.cmp(&b.path)));
        sources
    }

    /// Collect paths to all `agents/` directories contributed by enabled plugins.
    pub fn all_agent_paths(&self) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(
                plugin.manifest.capabilities.as_deref(),
                "read_files",
            ) {
                continue;
            }
            if let Some(ref p) = plugin.agents_path {
                paths.push(p.clone());
            }
        }
        paths
    }

    /// Collect paths to all `output-styles/` directories contributed by enabled plugins.
    pub fn all_output_style_paths(&self) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(
                plugin.manifest.capabilities.as_deref(),
                "read_files",
            ) {
                continue;
            }
            if let Some(ref p) = plugin.output_styles_path {
                paths.push(p.clone());
            }
        }
        paths
    }

    /// Build the `HookRegistry` from all enabled plugins.
    pub fn build_hook_registry(&self) -> HookRegistry {
        let mut registry: HookRegistry = HashMap::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(plugin.manifest.capabilities.as_deref(), "shell") {
                continue;
            }
            if let Some(ref hooks_config) = plugin.hooks_config {
                register_plugin_hooks(
                    hooks_config,
                    &plugin.path.to_string_lossy(),
                    &plugin.name,
                    &plugin.source_id,
                    &mut registry,
                );
            }
        }
        registry
    }

    /// Collect all MCP server configs contributed by enabled plugins.
    pub fn all_mcp_servers(&self) -> Vec<claurst_core::config::McpServerConfig> {
        let mut servers: Vec<claurst_core::config::McpServerConfig> = Vec::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(plugin.manifest.capabilities.as_deref(), "mcp") {
                continue;
            }
            for mcp in &plugin.manifest.mcp_servers {
                servers.push(claurst_core::config::McpServerConfig {
                    name: mcp.name.clone(),
                    command: mcp.command.clone(),
                    args: mcp.args.clone(),
                    env: mcp.env.clone(),
                    url: mcp.url.clone(),
                    server_type: mcp.server_type.clone(),
                    origin: if matches!(plugin.source, crate::plugin::PluginSource::Project) {
                        claurst_core::config::McpServerOrigin::Project
                    } else {
                        claurst_core::config::McpServerOrigin::User
                    },
                });
            }
        }
        servers
    }

    /// Collect enabled plugin MCP definitions while retaining ownership for
    /// same-session reload reconciliation. The compatibility helper above
    /// intentionally returns only plain configs.
    pub fn all_mcp_servers_with_owners(&self) -> Vec<PluginMcpServer> {
        let mut servers = Vec::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(plugin.manifest.capabilities.as_deref(), "mcp") {
                continue;
            }
            for mcp in &plugin.manifest.mcp_servers {
                servers.push(PluginMcpServer {
                    owner_source_id: plugin.source_id.clone(),
                    config: McpServerConfig {
                        name: mcp.name.clone(),
                        command: mcp.command.clone(),
                        args: mcp.args.clone(),
                        env: mcp.env.clone(),
                        url: mcp.url.clone(),
                        server_type: mcp.server_type.clone(),
                        origin: if matches!(plugin.source, crate::plugin::PluginSource::Project) {
                            McpServerOrigin::Project
                        } else {
                            McpServerOrigin::User
                        },
                    },
                });
            }
        }
        servers
    }

    /// Collect all LSP server configs contributed by enabled plugins.
    pub fn all_lsp_servers(&self) -> Vec<crate::manifest::PluginLspServer> {
        let mut servers: Vec<crate::manifest::PluginLspServer> = Vec::new();
        for plugin in self.enabled() {
            if !crate::plugin_capability_granted(plugin.manifest.capabilities.as_deref(), "shell") {
                continue;
            }
            for lsp in &plugin.manifest.lsp_servers {
                servers.push(lsp.clone());
            }
        }
        servers
    }

    // ---- Statistics -------------------------------------------------------

    /// Total number of plugins (enabled + disabled).
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Number of enabled plugins.
    pub fn enabled_count(&self) -> usize {
        self.enabled_names.len()
    }

    /// Number of load errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    // ---- Reload diff ------------------------------------------------------

    /// Compare this registry against `old` and produce a diff report.
    pub fn diff_against(&self, old: &PluginRegistry) -> ReloadDiff {
        let old_names: std::collections::HashSet<&str> =
            old.plugins.keys().map(|s| s.as_str()).collect();
        let new_names: std::collections::HashSet<&str> =
            self.plugins.keys().map(|s| s.as_str()).collect();

        let added: Vec<String> = new_names
            .difference(&old_names)
            .map(|&s| s.to_string())
            .collect();
        let removed: Vec<String> = old_names
            .difference(&new_names)
            .map(|&s| s.to_string())
            .collect();
        let updated: Vec<String> = new_names
            .intersection(&old_names)
            .filter(|&&name| {
                let new_ver = self
                    .plugins
                    .get(name)
                    .and_then(|p| p.manifest.version.as_deref());
                let old_ver = old
                    .plugins
                    .get(name)
                    .and_then(|p| p.manifest.version.as_deref());
                new_ver != old_ver
            })
            .map(|&s| s.to_string())
            .collect();

        ReloadDiff {
            added,
            removed,
            updated,
            error_count: self.errors.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        PluginHookEntry, PluginHookMatcher, PluginHooksConfig, PluginLspServer, PluginManifest,
        PluginMcpServer,
    };
    use crate::plugin::{CommandRunAction, PluginCommandDef, PluginSource};
    use std::path::PathBuf;

    fn make_plugin(name: &str) -> LoadedPlugin {
        LoadedPlugin {
            name: name.to_string(),
            path: PathBuf::from(format!("/tmp/{}", name)),
            source: PluginSource::User,
            source_id: format!("{}@user", name),
            manifest: PluginManifest {
                name: name.to_string(),
                ..Default::default()
            },
            enabled: true,
            commands_path: None,
            agents_path: None,
            skills_path: None,
            output_styles_path: None,
            hooks_config: None,
        }
    }

    #[test]
    fn published_prompt_sources_preserve_capabilities_and_reload_removals() {
        let root = tempfile::tempdir().unwrap();
        let skills = root.path().join("skills/review");
        std::fs::create_dir_all(&skills).unwrap();
        std::fs::write(skills.join("SKILL.md"), "Review $ARGUMENTS").unwrap();
        let mut plugin = make_plugin("fixture");
        plugin.path = root.path().to_path_buf();
        plugin.source = PluginSource::Project;
        plugin.skills_path = Some(root.path().join("skills"));
        let mut registry = PluginRegistry::new();
        registry.insert(plugin);
        let config = claurst_core::config::Config::default();
        let cloned = config.clone();
        registry.publish_skill_sources(&config, root.path());
        {
            let sources = cloned.runtime_skills.read();
            assert_eq!(sources.len(), 1);
            assert_eq!(sources[0].name, "fixture:review");
            assert!(matches!(&sources[0].origin,
                claurst_core::skill_discovery::SkillOrigin::Plugin { denial: Some(reason), project_root: Some(project) }
                if reason.contains("read_files") && project == root.path()));
        }
        registry.disable("fixture");
        registry.publish_skill_sources(&config, root.path());
        assert!(cloned.runtime_skills.read().is_empty());
    }

    #[test]
    fn directory_skill_sources_preserve_capability_and_project_scope() {
        let root = tempfile::tempdir().unwrap();
        let mut allowed = make_plugin("allowed-directory");
        allowed.source = PluginSource::Project;
        allowed.path = root.path().join("allowed-directory");
        allowed.skills_path = Some(allowed.path.join("skills"));
        allowed.manifest.capabilities = Some(vec!["read_files".to_string()]);
        std::fs::create_dir_all(allowed.skills_path.as_ref().unwrap().join("review")).unwrap();
        std::fs::write(
            allowed
                .skills_path
                .as_ref()
                .unwrap()
                .join("review/SKILL.md"),
            "Review",
        )
        .unwrap();

        let mut denied = make_plugin("denied-directory");
        denied.source = PluginSource::Project;
        denied.path = root.path().join("denied-directory");
        denied.skills_path = Some(denied.path.join("skills"));
        std::fs::create_dir_all(denied.skills_path.as_ref().unwrap().join("review")).unwrap();
        std::fs::write(
            denied.skills_path.as_ref().unwrap().join("review/SKILL.md"),
            "Review",
        )
        .unwrap();

        let mut registry = PluginRegistry::new();
        registry.insert(allowed);
        registry.insert(denied);
        let sources = registry.all_skill_sources(root.path());
        assert_eq!(sources.len(), 2);
        let allowed_source = sources
            .iter()
            .find(|source| source.name == "allowed-directory:review")
            .unwrap();
        assert!(matches!(
            &allowed_source.origin,
            claurst_core::skill_discovery::SkillOrigin::Plugin {
                denial: None,
                project_root: Some(project),
            } if project == root.path()
        ));
        let denied_source = sources
            .iter()
            .find(|source| source.name == "denied-directory:review")
            .unwrap();
        assert!(matches!(
            &denied_source.origin,
            claurst_core::skill_discovery::SkillOrigin::Plugin {
                denial: Some(reason),
                project_root: Some(project),
            } if reason.contains("read_files") && project == root.path()
        ));
    }

    fn executable_plugin(name: &str, capabilities: Option<Vec<&str>>) -> LoadedPlugin {
        let mut plugin = make_plugin(name);
        plugin.manifest.capabilities = capabilities.map(|entries| {
            entries
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        });
        plugin.skills_path = Some(plugin.path.join("skills"));
        plugin.agents_path = Some(plugin.path.join("agents"));
        plugin.output_styles_path = Some(plugin.path.join("output-styles"));
        plugin.manifest.mcp_servers = vec![PluginMcpServer {
            name: "fixture-mcp".to_string(),
            command: Some("fixture-server".to_string()),
            ..Default::default()
        }];
        plugin.manifest.lsp_servers = vec![PluginLspServer {
            name: "fixture-lsp".to_string(),
            command: "fixture-lsp".to_string(),
            ..Default::default()
        }];
        plugin.hooks_config = Some(PluginHooksConfig {
            description: None,
            events: [(
                "PreToolUse".to_string(),
                vec![PluginHookMatcher {
                    matcher: None,
                    hooks: vec![PluginHookEntry {
                        command: "true".to_string(),
                        matcher: None,
                        blocking: false,
                    }],
                }],
            )]
            .into_iter()
            .collect(),
        });
        plugin
    }

    #[test]
    fn command_capabilities_are_fail_closed() {
        let shell = |capabilities| PluginCommandDef {
            name: "fixture:shell".to_string(),
            description: "fixture".to_string(),
            plugin_name: "fixture".to_string(),
            plugin_source_id: "fixture@user".to_string(),
            run_action: CommandRunAction::ShellCommand {
                command: "true".to_string(),
                plugin_root: "/tmp/fixture".to_string(),
            },
            plugin_capabilities: capabilities,
        };

        assert!(crate::check_plugin_capability(&shell(None)).is_err());
        assert!(crate::check_plugin_capability(&shell(Some(Vec::new()))).is_err());
        assert!(crate::check_plugin_capability(&shell(Some(vec!["shell".to_string()]))).is_ok());

        let static_command = PluginCommandDef {
            name: "fixture:static".to_string(),
            description: "fixture".to_string(),
            plugin_name: "fixture".to_string(),
            plugin_source_id: "fixture@user".to_string(),
            run_action: CommandRunAction::StaticResponse("ok".to_string()),
            plugin_capabilities: None,
        };
        assert!(crate::check_plugin_capability(&static_command).is_ok());
    }

    #[test]
    fn plugin_mcp_preserves_project_origin() {
        use claurst_core::config::McpServerOrigin;
        for (source, expected) in [
            (PluginSource::User, McpServerOrigin::User),
            (PluginSource::Project, McpServerOrigin::Project),
            (
                PluginSource::Extra("explicit".to_string()),
                McpServerOrigin::User,
            ),
        ] {
            let mut plugin = executable_plugin("origin", Some(vec!["mcp"]));
            plugin.source = source;
            let mut registry = PluginRegistry::new();
            registry.insert(plugin);
            assert_eq!(registry.all_mcp_servers()[0].origin, expected);
            let owned = registry.all_mcp_servers_with_owners();
            assert_eq!(owned.len(), 1);
            assert_eq!(owned[0].owner_source_id, "origin@user");
            assert_eq!(owned[0].config.origin, expected);
        }
    }

    #[test]
    fn derived_plugin_surfaces_require_explicit_capabilities() {
        let mut denied = PluginRegistry::new();
        denied.insert(executable_plugin("denied", None));
        assert!(denied.all_skill_paths().is_empty());
        assert!(denied.all_agent_paths().is_empty());
        assert!(denied.all_output_style_paths().is_empty());
        assert!(denied.build_hook_registry().is_empty());
        assert!(denied.all_mcp_servers().is_empty());
        assert!(denied.all_lsp_servers().is_empty());

        let mut allowed = PluginRegistry::new();
        allowed.insert(executable_plugin(
            "allowed",
            Some(vec!["read_files", "shell", "mcp"]),
        ));
        assert_eq!(allowed.all_skill_paths().len(), 1);
        assert_eq!(allowed.all_agent_paths().len(), 1);
        assert_eq!(allowed.all_output_style_paths().len(), 1);
        assert_eq!(
            allowed
                .build_hook_registry()
                .values()
                .map(Vec::len)
                .sum::<usize>(),
            1
        );
        assert_eq!(allowed.all_mcp_servers().len(), 1);
        assert_eq!(allowed.all_lsp_servers().len(), 1);
    }

    #[test]
    fn enable_disable() {
        let mut reg = PluginRegistry::new();
        reg.insert(make_plugin("alpha"));
        assert!(reg.is_enabled("alpha"));

        reg.disable("alpha");
        assert!(!reg.is_enabled("alpha"));
        assert_eq!(reg.enabled().len(), 0);

        reg.enable("alpha");
        assert!(reg.is_enabled("alpha"));
        assert_eq!(reg.enabled().len(), 1);
    }

    #[test]
    fn duplicate_name_kept_first() {
        let mut reg = PluginRegistry::new();
        reg.insert(make_plugin("beta"));
        let mut dup = make_plugin("beta");
        dup.path = PathBuf::from("/tmp/beta2");
        reg.insert(dup);
        assert_eq!(reg.plugin_count(), 1);
        assert_eq!(reg.error_count(), 1);
    }

    #[test]
    fn configured_hooks_require_a_nonempty_hook_entry() {
        let mut reg = PluginRegistry::new();
        assert!(!reg.has_configured_hooks());

        let mut empty = make_plugin("empty-hooks");
        empty.hooks_config = Some(PluginHooksConfig::default());
        reg.insert(empty);
        assert!(!reg.has_configured_hooks());

        let mut configured = make_plugin("configured-hooks");
        configured.hooks_config = Some(PluginHooksConfig {
            description: None,
            events: [(
                "PreToolUse".to_string(),
                vec![PluginHookMatcher {
                    matcher: None,
                    hooks: vec![PluginHookEntry {
                        command: "true".to_string(),
                        matcher: None,
                        blocking: false,
                    }],
                }],
            )]
            .into_iter()
            .collect(),
        });
        reg.insert(configured);
        assert!(reg.has_configured_hooks());
    }

    #[test]
    fn configured_hooks_are_conservative_for_disabled_plugins() {
        let mut reg = PluginRegistry::new();
        let mut plugin = make_plugin("disabled-hooks");
        plugin.enabled = false;
        plugin.hooks_config = Some(PluginHooksConfig {
            description: None,
            events: [(
                "Stop".to_string(),
                vec![PluginHookMatcher {
                    matcher: None,
                    hooks: vec![PluginHookEntry {
                        command: "true".to_string(),
                        matcher: None,
                        blocking: false,
                    }],
                }],
            )]
            .into_iter()
            .collect(),
        });
        reg.insert(plugin);
        assert!(reg.has_configured_hooks());
        assert!(reg.enabled().is_empty());
    }

    #[test]
    fn diff_detects_added_removed() {
        let mut old_reg = PluginRegistry::new();
        old_reg.insert(make_plugin("kept"));
        old_reg.insert(make_plugin("gone"));

        let mut new_reg = PluginRegistry::new();
        new_reg.insert(make_plugin("kept"));
        new_reg.insert(make_plugin("new-plugin"));

        let diff = new_reg.diff_against(&old_reg);
        assert_eq!(diff.added, vec!["new-plugin"]);
        assert_eq!(diff.removed, vec!["gone"]);
        assert!(diff.updated.is_empty());
    }
}
