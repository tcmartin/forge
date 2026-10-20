// SkillTool: execute user-defined skill (prompt template) files programmatically.
//
// Skills include legacy command files, core-discovered directory/configured
// sources, existing Git caches, and host-published enabled plugin prompts.
//
// Bundled skills (defined in bundled_skills.rs) are checked first before the
// disk directories, so they take precedence over same-named .md files.
//
// The model invokes this tool to expand a skill's prompt inline.
// Supports $ARGUMENTS placeholder substitution.
// Use skill="list" to discover available skills.

use crate::bundled_skills::{expand_prompt, find_bundled_skill, user_invocable_skills};
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use claurst_core::skill_discovery::{SkillOrigin, SkillSource};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::io::AsyncReadExt;
use tracing::debug;

pub struct SkillTool;

#[derive(Debug, Deserialize)]
struct SkillInput {
    skill: String,
    #[serde(default)]
    args: Option<String>,
}

#[async_trait]
impl Tool for SkillTool {
    // Gates itself: calls `ctx.check_permission_for_path` in `execute()` (#210).
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "Skill"
    }

    fn description(&self) -> &str {
        "Execute a bundled or configured skill (custom prompt template) by name. \
         Supports command files, SKILL.md directories and enabled plugin prompts. \
         Use skill=\"list\" to discover available skills. \
         The expanded skill prompt is returned for you to act on."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Skill name (without .md extension), or \"list\" to enumerate skills"
                },
                "args": {
                    "type": "string",
                    "description": "Arguments passed to the skill — replaces $ARGUMENTS in the template"
                }
            },
            "required": ["skill"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: SkillInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let dirs = skill_search_dirs(ctx);
        if params.skill == "list" {
            return list_skills(&dirs, ctx).await;
        }
        let skill_name = params.skill.trim_end_matches(".md");
        debug!(skill = skill_name, "Loading skill");

        // Check bundled skills first — they take precedence over disk files.
        if let Some(bundled) = find_bundled_skill(skill_name) {
            let args = params.args.as_deref().unwrap_or("");
            let prompt = expand_prompt(bundled, args);
            let prompt = prompt.trim().to_string();
            if prompt.is_empty() {
                return ToolResult::error(format!(
                    "Bundled skill '{}' expanded to empty content.",
                    skill_name
                ));
            }
            return ToolResult::success(prompt);
        }

        let sources = skill_sources(&dirs, ctx);
        let mut approved = false;
        let mut selected = sources
            .iter()
            .find(|source| source.name == skill_name)
            .cloned();
        // Frontmatter aliases require bounded metadata reads and the same path
        // approval as descriptions. Never load other prompt bodies to search.
        if selected.is_none() {
            for source in &sources {
                if !matches!(source.origin, SkillOrigin::Local) {
                    continue;
                }
                if ctx
                    .check_permission_for_path(
                        self.name(),
                        "Read skill metadata",
                        source.path.clone(),
                        true,
                    )
                    .is_err()
                {
                    continue;
                }
                if read_skill_metadata(&source.path)
                    .await
                    .is_some_and(|skill| skill.name == skill_name)
                {
                    selected = Some(source.clone());
                    approved = true;
                    break;
                }
            }
        }
        let source = match selected {
            Some(source) => source,
            None => {
                return ToolResult::error(format!(
                    "Skill '{}' not found. Use skill=\"list\" to see available skills.",
                    skill_name
                ));
            }
        };

        if let SkillOrigin::Plugin {
            denial: Some(reason),
            ..
        } = &source.origin
        {
            return ToolResult::error(reason.clone());
        }
        let skill_path = source.path;
        if !approved {
            if let Err(e) = ctx.check_permission_for_path(
                self.name(),
                &format!("Load skill {}", params.skill),
                skill_path.clone(),
                true,
            ) {
                return ToolResult::error(e.to_string());
            }
        }

        let raw = match tokio::fs::read_to_string(&skill_path).await {
            Ok(content) => content,
            Err(_) => {
                return ToolResult::error(format!("Skill '{}' could not be read.", skill_name))
            }
        };

        // Strip YAML frontmatter if present (--- ... ---)
        let content = strip_frontmatter(&raw);

        // Substitute $ARGUMENTS
        let prompt = if let Some(args) = &params.args {
            content.replace("$ARGUMENTS", args)
        } else {
            content.replace("$ARGUMENTS", "")
        };

        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return ToolResult::error(format!("Skill '{}' expanded to empty content.", skill_name));
        }

        ToolResult::success(prompt)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn skill_search_dirs(ctx: &ToolContext) -> Vec<PathBuf> {
    if ctx.config.disable_extensions {
        return Vec::new();
    }
    let mut dirs = vec![ctx.working_dir.join(".claurst").join("commands")];
    dirs.push(claurst_core::config::Settings::config_dir().join("commands"));
    dirs.extend(claurst_core::skill_discovery::skill_search_dirs(
        &ctx.working_dir,
        &ctx.config.skills,
    ));
    dirs.extend(claurst_core::skill_discovery::cached_git_skill_dirs(
        &ctx.config.skills,
    ));
    dirs
}

fn skill_sources(dirs: &[PathBuf], ctx: &ToolContext) -> Vec<SkillSource> {
    if ctx.config.disable_extensions {
        return Vec::new();
    }
    let mut sources: Vec<SkillSource> = Vec::new();
    for dir in dirs {
        for path in claurst_core::skill_discovery::discover_skill_files(dir) {
            let name = claurst_core::skill_discovery::skill_name_from_path(&path);
            if !sources.iter().any(|source| source.name == name) {
                sources.push(SkillSource {
                    name,
                    path,
                    origin: SkillOrigin::Local,
                });
            }
        }
    }
    for source in ctx.config.runtime_skills.read().iter() {
        if let SkillOrigin::Plugin {
            project_root: Some(root),
            ..
        } = &source.origin
        {
            let within_project = match (ctx.working_dir.canonicalize(), root.canonicalize()) {
                (Ok(cwd), Ok(root)) => cwd.starts_with(root),
                _ => false,
            };
            if !within_project {
                continue;
            }
        }
        if !sources.iter().any(|existing| existing.name == source.name) {
            sources.push(source.clone());
        }
    }
    sources
}

async fn list_skills(dirs: &[PathBuf], ctx: &ToolContext) -> ToolResult {
    // Start with the bundled skills.
    let mut lines: Vec<String> = Vec::new();
    let bundled = user_invocable_skills();
    for (name, desc) in &bundled {
        lines.push(format!("  {} — {} [bundled]", name, desc));
    }
    let mut names: std::collections::HashSet<String> = bundled
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();
    let mut sources = skill_sources(dirs, ctx);
    sources.sort_by(|a, b| a.name.cmp(&b.name));
    for source in sources {
        if names.contains(&source.name) {
            continue;
        }
        let mut name = source.name;
        let desc = if let SkillOrigin::Plugin {
            denial: Some(reason),
            ..
        } = &source.origin
        {
            reason.clone()
        } else if ctx
            .check_permission_for_path(
                "Skill",
                &format!("Read skill description {}", name),
                source.path.clone(),
                true,
            )
            .is_ok()
        {
            if matches!(source.origin, SkillOrigin::Local) {
                if let Some(metadata) = read_skill_metadata(&source.path).await {
                    name = metadata.name;
                }
            }
            read_skill_description(&source.path).await
        } else {
            "(description unavailable)".to_string()
        };
        if names.insert(name.clone()) {
            lines.push(format!("  {} — {}", name, desc));
        }
    }

    let total = lines.len();
    if total == 0 {
        return ToolResult::success(
            "No skills found. Create .md files in .claurst/commands/ to define skills.\n\
             Example: .claurst/commands/review.md"
                .to_string(),
        );
    }

    ToolResult::success(format!(
        "Available skills ({}):\n{}",
        total,
        lines.join("\n")
    ))
}

async fn read_skill_metadata(path: &std::path::Path) -> Option<claurst_core::DiscoveredSkill> {
    let file = tokio::fs::File::open(path).await.ok()?;
    let mut content = String::new();
    file.take(4096).read_to_string(&mut content).await.ok()?;
    claurst_core::parse_skill_file(&content, path)
}

async fn read_skill_description(path: &std::path::Path) -> String {
    let Ok(file) = tokio::fs::File::open(path).await else {
        return "(no description)".to_string();
    };
    let mut content = String::new();
    if file.take(4096).read_to_string(&mut content).await.is_err() {
        return "(no description)".to_string();
    }
    let body = strip_frontmatter(&content);
    // First non-empty, non-heading line
    for line in body.lines() {
        let t = line.trim().trim_start_matches('#').trim();
        if !t.is_empty() {
            return t.chars().take(80).collect();
        }
    }
    "(no description)".to_string()
}

/// Remove YAML frontmatter delimited by `---` at the start of the file.
fn strip_frontmatter(content: &str) -> String {
    if let Some(after_open) = content.strip_prefix("---") {
        // Find closing ---
        if let Some(close_pos) = after_open.find("\n---") {
            // Skip past the closing delimiter and any leading newline
            let rest = &after_open[close_pos + 4..];
            return rest.trim_start_matches('\n').to_string();
        }
    }
    content.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct AllowReads;

    impl PermissionHandler for AllowReads {
        fn check_permission(&self, _request: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }

        fn request_permission(&self, _request: &PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    #[tokio::test]
    async fn directory_configured_and_aliased_sources_load_current_bodies() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join(".agents/skills/directory/SKILL.md");
        let configured = root.path().join("extra/configured.md");
        let alias = root.path().join("extra/filename.md");
        for path in [&directory, &configured, &alias] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        }
        std::fs::write(&directory, "Directory $ARGUMENTS").unwrap();
        std::fs::write(&configured, "Configured $ARGUMENTS").unwrap();
        std::fs::write(&alias, "---\nname: alias\n---\nAliased $ARGUMENTS").unwrap();
        let mut ctx = context(root.path().to_path_buf(), Arc::new(AllowReads));
        ctx.config.skills.paths.push("extra".to_string());
        for (name, expected) in [
            ("directory", "Directory"),
            ("configured", "Configured"),
            ("alias", "Aliased"),
        ] {
            let result = SkillTool
                .execute(json!({"skill": name, "args": "alpha"}), &ctx)
                .await;
            assert!(!result.is_error, "{}", result.content);
            assert_eq!(result.content, format!("{expected} alpha"));
        }
        std::fs::write(&directory, "Edited $ARGUMENTS").unwrap();
        let updated = SkillTool
            .execute(json!({"skill": "directory", "args": "beta"}), &ctx)
            .await;
        assert_eq!(updated.content, "Edited beta");
        let listed = SkillTool.execute(json!({"skill": "list"}), &ctx).await;
        assert!(listed.content.contains("directory —"));
        assert!(listed.content.contains("configured —"));
        assert!(listed.content.contains("alias —"));
        assert!(!listed.content.contains("filename —"));
    }

    #[tokio::test]
    async fn plugin_grant_is_checked_before_path_approval_or_prompt_read() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("skills/denied/SKILL.md");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, [0xff]).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = context(
            root.path().to_path_buf(),
            Arc::new(RewriteOnApproval {
                path: path.clone(),
                calls: Arc::clone(&calls),
                allow: true,
            }),
        );
        ctx.config.runtime_skills.write().push(SkillSource {
            name: "fixture:denied".to_string(),
            path,
            origin: SkillOrigin::Plugin {
                denial: Some("Plugin requires read_files capability".to_string()),
                project_root: None,
            },
        });
        let result = SkillTool
            .execute(json!({"skill": "fixture:denied"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("read_files"));
        let listed = SkillTool.execute(json!({"skill": "list"}), &ctx).await;
        assert!(listed.content.contains("fixture:denied"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn runtime_sources_follow_clone_reload_scope_and_bare_mode() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let path = root.path().join("skills/review/SKILL.md");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "Plugin body").unwrap();
        let original = context(root.path().to_path_buf(), Arc::new(AllowReads));
        let mut cloned = context(root.path().to_path_buf(), Arc::new(AllowReads));
        cloned.config = original.config.clone();
        original.config.runtime_skills.write().push(SkillSource {
            name: "fixture:review".to_string(),
            path,
            origin: SkillOrigin::Plugin {
                denial: None,
                project_root: Some(root.path().to_path_buf()),
            },
        });
        let loaded = SkillTool
            .execute(json!({"skill": "fixture:review"}), &cloned)
            .await;
        assert_eq!(loaded.content, "Plugin body");
        cloned.working_dir = other.path().to_path_buf();
        assert!(
            SkillTool
                .execute(json!({"skill": "fixture:review"}), &cloned)
                .await
                .is_error
        );
        cloned.working_dir = root.path().to_path_buf();
        cloned.config.disable_extensions = true;
        assert!(
            SkillTool
                .execute(json!({"skill": "fixture:review"}), &cloned)
                .await
                .is_error
        );
        cloned.config.disable_extensions = false;
        original.config.runtime_skills.write().clear();
        assert!(
            SkillTool
                .execute(json!({"skill": "fixture:review"}), &cloned)
                .await
                .is_error
        );
    }

    #[test]
    fn settings_never_serialize_or_restore_runtime_grants() {
        let config = claurst_core::config::Config::default();
        config.runtime_skills.write().push(SkillSource {
            name: "not-persisted".to_string(),
            path: PathBuf::from("/runtime-only-prompt"),
            origin: SkillOrigin::Plugin {
                denial: None,
                project_root: None,
            },
        });
        let mut json = serde_json::to_value(&config).unwrap();
        assert!(json.get("runtime_skills").is_none());
        json["runtime_skills"] = json!([{"name": "injected", "denial": null}]);
        let restored: claurst_core::config::Config = serde_json::from_value(json).unwrap();
        assert!(restored.runtime_skills.read().is_empty());
    }

    struct RewriteOnApproval {
        path: PathBuf,
        calls: Arc<AtomicUsize>,
        allow: bool,
    }

    impl PermissionHandler for RewriteOnApproval {
        fn check_permission(&self, _request: &PermissionRequest) -> PermissionDecision {
            if self.allow {
                PermissionDecision::Allow
            } else {
                PermissionDecision::Deny
            }
        }

        fn request_permission(&self, request: &PermissionRequest) -> PermissionDecision {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(request.path.as_deref(), self.path.to_str());
            std::fs::write(&self.path, "Approved current body: $ARGUMENTS").unwrap();
            self.check_permission(request)
        }
    }

    fn context(root: PathBuf, handler: Arc<dyn PermissionHandler>) -> ToolContext {
        ToolContext {
            working_dir: root,
            permission_mode: claurst_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "skill-read-order".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: claurst_core::config::Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            trusted_checker_catalog: Arc::new(
                crate::trusted_checker::TrustedCheckerCatalog::empty(),
            ),
            catalog_view: None,
        }
    }

    #[tokio::test]
    async fn prompt_is_read_after_approval_and_denied_before_read() {
        for allow in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(".claurst/commands/read-order-fixture.md");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            // Invalid UTF-8 makes any premature body read observable: the old
            // lookup returns "not found" without ever requesting permission.
            std::fs::write(&path, [0xff]).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let ctx = context(
                root.path().to_path_buf(),
                Arc::new(RewriteOnApproval {
                    path,
                    calls: Arc::clone(&calls),
                    allow,
                }),
            );
            let result = SkillTool
                .execute(
                    json!({"skill": "read-order-fixture", "args": "alpha"}),
                    &ctx,
                )
                .await;
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            if allow {
                assert!(!result.is_error, "{}", result.content);
                assert_eq!(result.content, "Approved current body: alpha");
            } else {
                assert!(result.is_error);
                assert!(result.content.contains("Permission denied"));
                assert!(!result.content.contains("Approved current body"));
            }
        }
    }

    #[tokio::test]
    async fn descriptions_are_unicode_safe_and_read_only_a_bounded_prefix() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("unicode.md");
        std::fs::write(&path, "界".repeat(100)).unwrap();
        assert_eq!(read_skill_description(&path).await, "界".repeat(80));
        std::fs::write(
            &path,
            format!("{}Outside metadata window", "\n".repeat(4096)),
        )
        .unwrap();
        assert_eq!(read_skill_description(&path).await, "(no description)");
    }

    #[tokio::test]
    async fn list_checks_permission_before_loading_descriptions() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("denied-description.md");
        std::fs::write(&path, [0xff]).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = context(
            root.path().to_path_buf(),
            Arc::new(RewriteOnApproval {
                path,
                calls: Arc::clone(&calls),
                allow: false,
            }),
        );
        let result = list_skills(&[root.path().to_path_buf()], &ctx).await;
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(!result.is_error);
        assert!(result.content.contains("denied-description"));
        assert!(result.content.contains("description unavailable"));
        assert!(!result.content.contains("Approved current body"));
    }
}
