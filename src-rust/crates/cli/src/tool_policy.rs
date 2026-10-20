//! CLI tool selector parsing and exposed-tool filtering.

use claurst_core::config::Config;
use claurst_tools::{
    tool_policy::{selector_matches, validate_selectors},
    Tool,
};
use std::sync::Arc;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ToolPolicy {
    allowed: Vec<String>,
    denied: Vec<String>,
}

impl ToolPolicy {
    pub(crate) fn deny_all() -> Self {
        Self {
            allowed: vec!["__forge_no_tool_matches__".to_string()],
            denied: vec!["*".to_string()],
        }
    }

    pub(crate) fn from_cli(allowed: Option<&str>, denied: Option<&str>) -> Result<Self, String> {
        Ok(Self {
            allowed: parse_selectors(allowed, "--allowed-tools")?,
            denied: parse_selectors(denied, "--disallowed-tools")?,
        })
    }

    pub(crate) fn from_config(config: &Config) -> Result<Self, String> {
        let allowed = validate_selectors(&config.allowed_tools, "allowed_tools")?;
        let denied = validate_selectors(&config.disallowed_tools, "disallowed_tools")?;
        Ok(Self { allowed, denied })
    }

    pub(crate) fn apply_cli_overrides(
        &self,
        config: &mut Config,
        has_allowed: bool,
        has_denied: bool,
    ) {
        if has_allowed {
            config.allowed_tools = self.allowed.clone();
        }
        if has_denied {
            config.disallowed_tools = self.denied.clone();
        }
    }

    pub(crate) fn allows(&self, tool_name: &str) -> bool {
        if self
            .denied
            .iter()
            .any(|selector| selector_matches(selector, tool_name))
        {
            return false;
        }
        self.allowed.is_empty()
            || self
                .allowed
                .iter()
                .any(|selector| selector_matches(selector, tool_name))
    }
}

/// Filter a newly-created owned tool vector without cloning trait objects.
pub(crate) fn filter_tool_names(
    tools: Vec<Box<dyn Tool>>,
    policy: &ToolPolicy,
) -> Arc<Vec<Box<dyn Tool>>> {
    Arc::new(
        tools
            .into_iter()
            .filter(|tool| policy.allows(tool.name()))
            .collect(),
    )
}

fn parse_selectors(value: Option<&str>, flag: &str) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let selectors: Vec<String> = value
        .split(',')
        .map(str::trim)
        .map(str::to_string)
        .collect();
    if selectors.iter().any(String::is_empty) {
        return Err(format!("{} contains an empty tool selector", flag));
    }
    validate_selectors(&selectors, flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comma_separated_cli_selectors() {
        let policy = ToolPolicy::from_cli(Some("Read, mcp__server__*"), Some("Write"))
            .expect("selectors are valid");
        assert_eq!(policy.allowed, vec!["Read", "mcp__server__*"]);
        assert_eq!(policy.denied, vec!["Write"]);
    }

    #[test]
    fn deny_wins_over_allow_and_globs_match_full_name() {
        let policy = ToolPolicy::from_cli(Some("mcp__server__*"), Some("mcp__server__secret"))
            .expect("selectors are valid");
        assert!(policy.allows("mcp__server__read"));
        assert!(!policy.allows("mcp__server__secret"));
        assert!(!policy.allows("mcp__other__read"));
    }

    #[test]
    fn unsupported_argument_selector_is_rejected() {
        let error = ToolPolicy::from_cli(Some("Bash(git:*)"), None).expect_err("must reject");
        assert!(error.contains("unsupported selector"));
    }

    #[test]
    fn absent_allow_list_exposes_all_except_denied() {
        let policy = ToolPolicy::from_cli(None, Some("Write")).expect("selector is valid");
        assert!(policy.allows("Read"));
        assert!(!policy.allows("Write"));
    }

    #[test]
    fn filtering_inventory_keeps_only_selected_tools() {
        let policy = ToolPolicy::from_cli(Some("Read"), None).expect("selector is valid");
        let filtered = filter_tool_names(claurst_tools::all_tools(), &policy);
        let names: Vec<&str> = filtered.iter().map(|tool| tool.name()).collect();

        assert_eq!(names, vec!["Read"]);
    }
}
