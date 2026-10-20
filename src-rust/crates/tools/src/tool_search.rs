//! Bounded, host-scoped tool discovery.
//!
//! Search reads only the immutable catalog bound to the current ToolContext.
//! It never rebuilds a registry or supplements names that were not authorized
//! by the host.

use crate::tool_catalog::{
    DiscoveryReceipt, ToolCatalogEntry, ToolCatalogError, ToolCatalogView,
    MAX_DISCOVERY_QUERY_BYTES, MAX_DISCOVERY_TERMS,
};
use crate::{PermissionLevel, Tool, ToolContext, ToolEffect, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

pub struct ToolSearchTool;

#[derive(Debug, Deserialize)]
struct ToolSearchInput {
    query: String,
    #[serde(default = "default_max")]
    max_results: usize,
}

fn default_max() -> usize {
    5
}

/// Extra search synonyms for high-value tools, keyed by canonical name.
fn keywords_for(name: &str) -> &'static [&'static str] {
    match name {
        "Bash" => &["shell", "run", "command", "exec", "terminal"],
        "Read" => &["file", "cat", "content", "open"],
        "Write" => &["file", "create", "save", "new"],
        "Edit" => &["file", "modify", "replace", "patch", "change"],
        "Glob" => &["find", "pattern", "files", "filename"],
        "Grep" => &["search", "regex", "content", "ripgrep"],
        "WebFetch" => &["web", "url", "http", "download", "browse", "internet"],
        "WebSearch" => &["web", "internet", "google", "browse", "news"],
        "NotebookEdit" => &["notebook", "jupyter", "ipynb", "cell"],
        "TodoWrite" => &["todo", "task", "plan", "checklist"],
        "AskUserQuestion" => &["ask", "question", "clarify", "choose"],
        "Agent" => &["agent", "subagent", "delegate", "parallel", "spawn"],
        "Skill" => &["skill", "slash", "command", "template", "prompt"],
        "Config" => &["config", "settings", "model", "permission"],
        "SendMessage" => &["message", "broadcast", "inbox", "communicate"],
        _ => &[],
    }
}

fn score_entry(entry: &ToolCatalogEntry, terms: &[&str]) -> usize {
    let name_lower = entry.name().to_lowercase();
    let desc_lower = entry.description().to_lowercase();
    let mut score = 0usize;

    for term in terms {
        if name_lower == *term {
            score += 25;
        } else if name_lower.contains(term) {
            score += 10;
        }
        for keyword in keywords_for(entry.name()) {
            if keyword == term {
                score += 8;
            } else if keyword.contains(term) {
                score += 3;
            }
        }
        if desc_lower
            .split_whitespace()
            .any(|word| word.trim_matches(|c: char| !c.is_alphanumeric()) == *term)
        {
            score += 5;
        } else if desc_lower.contains(term) {
            score += 2;
        }
    }
    score
}

fn query_error(error: ToolCatalogError) -> ToolResult {
    ToolResult::error(format!("Tool discovery unavailable: {error}"))
}

fn bound_catalog(ctx: &ToolContext) -> Result<&ToolCatalogView, ToolCatalogError> {
    let view = ctx
        .catalog_view
        .as_deref()
        .ok_or(ToolCatalogError::MissingCatalog)?;
    if !view.matches_context(ctx) {
        return Err(ToolCatalogError::ScopeMismatch);
    }
    Ok(view)
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "ToolSearch"
    }

    fn description(&self) -> &str {
        "Find the right tool for a task. Search authorized tools by name or keyword and get back the best-matching tool names with a one-line description each. Use a natural phrase or 'select:ToolName' for direct lookup."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Observation
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A task description or keywords, or 'select:ToolName' for direct lookup"
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum results (default: 5, max: 20)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ToolSearchInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(error) => return ToolResult::error(format!("Invalid input: {error}")),
        };
        if params.query.len() > MAX_DISCOVERY_QUERY_BYTES {
            return query_error(ToolCatalogError::QueryTooLarge {
                bytes: params.query.len(),
                limit: MAX_DISCOVERY_QUERY_BYTES,
            });
        }
        let view = match bound_catalog(ctx) {
            Ok(view) => view,
            Err(error) => return query_error(error),
        };
        let query = params.query.trim();
        let max = params.max_results.clamp(1, 20);

        if let Some(names_str) = query.strip_prefix("select:").map(str::trim) {
            let requested: Vec<&str> = names_str
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .collect();
            if requested.is_empty() {
                return query_error(ToolCatalogError::EmptySelection);
            }
            if requested.len() > MAX_DISCOVERY_TERMS {
                return query_error(ToolCatalogError::TooManyTerms {
                    actual: requested.len(),
                    limit: MAX_DISCOVERY_TERMS,
                });
            }
            let mut unique_requested = Vec::with_capacity(requested.len());
            for requested_name in requested {
                if !unique_requested
                    .iter()
                    .any(|name: &&str| name.eq_ignore_ascii_case(requested_name))
                {
                    unique_requested.push(requested_name);
                }
            }
            if unique_requested.len() > max {
                return query_error(ToolCatalogError::SelectionTooLarge {
                    actual: unique_requested.len(),
                    limit: max,
                });
            }
            let mut found = Vec::new();
            let mut selected = Vec::new();
            let mut missing = Vec::new();
            for requested_name in unique_requested {
                if let Some(entry) = view
                    .entries()
                    .iter()
                    .find(|entry| entry.name().eq_ignore_ascii_case(requested_name))
                {
                    if !selected.iter().any(|name| name == entry.name()) {
                        found.push(format!("{}: {}", entry.name(), entry.description()));
                        selected.push(entry.name().to_string());
                    }
                } else {
                    missing.push(requested_name.to_string());
                }
            }
            if !missing.is_empty() {
                return query_error(ToolCatalogError::SelectionUnavailable(missing));
            }
            let output = found.join("\n");
            return ToolResult::success(output)
                .with_discovery_receipt(DiscoveryReceipt::new(view, selected));
        }

        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower
            .split(|character: char| !character.is_alphanumeric())
            .filter(|term| term.len() > 1)
            .collect();
        if terms.len() > MAX_DISCOVERY_TERMS {
            return query_error(ToolCatalogError::TooManyTerms {
                actual: terms.len(),
                limit: MAX_DISCOVERY_TERMS,
            });
        }
        if terms.is_empty() {
            return ToolResult::success(format!(
                "Empty query. Provide keywords or use 'select:ToolName'. {} tools available.",
                view.entries().len()
            ));
        }

        let mut scored: Vec<(usize, &ToolCatalogEntry)> = view
            .entries()
            .iter()
            .filter_map(|entry| {
                let score = score_entry(entry, &terms);
                (score > 0).then_some((score, entry))
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| left.1.name().cmp(right.1.name()))
        });
        scored.truncate(max);
        if scored.is_empty() {
            return ToolResult::success(format!(
                "No tools matched '{}'. Try broader keywords or use 'select:ToolName'. {} tools available.",
                query,
                view.entries().len()
            ));
        }
        let selected: Vec<String> = scored
            .iter()
            .map(|(_, entry)| entry.name().to_string())
            .collect();
        let lines: Vec<String> = scored
            .iter()
            .map(|(_, entry)| format!("{}: {}", entry.name(), entry.description()))
            .collect();
        ToolResult::success(format!(
            "Tools matching '{}' (use one of these for the task):\n\n{}\n\n{} of {} tools shown.",
            query,
            lines.join("\n"),
            scored.len(),
            view.entries().len()
        ))
        .with_discovery_receipt(DiscoveryReceipt::new(view, selected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::all_tools;

    struct ActualAgent;

    #[async_trait::async_trait]
    impl Tool for ActualAgent {
        fn name(&self) -> &str {
            "Agent"
        }
        fn description(&self) -> &str {
            "Launch a real test agent"
        }
        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _: Value, _: &ToolContext) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    fn ctx() -> ToolContext {
        let mut ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools = all_tools();
        ctx.catalog_view = Some(ToolCatalogView::from_tools(&ctx, &tools).expect("catalog"));
        ctx
    }

    async fn run(query: &str) -> ToolResult {
        ToolSearchTool
            .execute(json!({ "query": query }), &ctx())
            .await
    }

    #[tokio::test]
    async fn web_query_surfaces_web_tools() {
        let out = run("search the web").await;
        assert!(out.content.contains("WebSearch"), "{}", out.content);
        let ws = out.content.find("WebSearch");
        let wf = out.content.find("WebFetch");
        if let (Some(ws), Some(wf)) = (ws, wf) {
            assert!(ws < wf, "WebSearch should rank above WebFetch");
        }
        assert!(out.discovery_receipt.is_some());
    }

    #[tokio::test]
    async fn exact_name_ranks_first() {
        let out = run("grep").await;
        let first_line = out
            .content
            .lines()
            .find(|line| line.contains(": "))
            .unwrap_or_default();
        assert!(
            first_line.starts_with("Grep:"),
            "{first_line:?}\n{}",
            out.content
        );
    }

    #[tokio::test]
    async fn select_prefix_requires_every_name_to_be_available() {
        let out = run("select:WebFetch,DoesNotExist").await;
        assert!(out.is_error, "{}", out.content);
        assert!(out.content.contains("DoesNotExist"), "{}", out.content);
        assert!(out.discovery_receipt.is_none());
    }

    #[tokio::test]
    async fn select_deduplicates_within_max_and_rejects_overflow() {
        let duplicate = ToolSearchTool
            .execute(
                json!({ "query": "select:WebFetch,WebFetch", "max_results": 2 }),
                &ctx(),
            )
            .await;
        assert_eq!(
            duplicate.discovery_receipt.unwrap().selected_names(),
            &[String::from("WebFetch")]
        );
        let overflow = ToolSearchTool
            .execute(
                json!({ "query": "select:WebFetch,WebSearch", "max_results": 1 }),
                &ctx(),
            )
            .await;
        assert!(overflow.is_error);
        assert!(overflow.content.contains("selection has 2 names"));
    }

    #[tokio::test]
    async fn select_denied_name_has_no_partial_authority() {
        let mut context = ctx();
        context.config.disallowed_tools = vec!["WebFetch".to_string()];
        let tools = crate::all_tools();
        context.catalog_view = Some(ToolCatalogView::from_tools(&context, &tools).unwrap());
        let out = ToolSearchTool
            .execute(json!({ "query": "select:WebSearch,WebFetch" }), &context)
            .await;
        assert!(out.is_error, "{}", out.content);
        assert!(out.content.contains("WebFetch"), "{}", out.content);
        assert!(out.discovery_receipt.is_none());
    }

    #[tokio::test]
    async fn missing_catalog_is_explicit_and_has_no_authority() {
        let ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let out = ToolSearchTool
            .execute(json!({ "query": "web" }), &ctx)
            .await;
        assert!(out.is_error);
        assert!(out.content.contains("not bound"));
        assert!(out.discovery_receipt.is_none());
    }

    #[tokio::test]
    async fn custom_inventory_requires_an_actual_agent_entry() {
        let mut ctx = crate::test_support::allow_all_context(std::env::temp_dir());
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(ActualAgent)];
        ctx.catalog_view = Some(ToolCatalogView::from_tools(&ctx, &tools).unwrap());
        let out = ToolSearchTool
            .execute(json!({ "query": "delegate agent" }), &ctx)
            .await;
        assert!(out.content.contains("Agent:"));
        assert_eq!(
            out.discovery_receipt.unwrap().selected_names(),
            &[String::from("Agent")]
        );
    }

    #[tokio::test]
    async fn oversized_query_is_rejected() {
        let out = run(&"x".repeat(MAX_DISCOVERY_QUERY_BYTES + 1)).await;
        assert!(out.is_error);
        assert!(out.content.contains("4096"));
    }
}
