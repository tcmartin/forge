//! Shared configuration policy for tool names.
//!
//! Selectors intentionally support only complete tool names and `*` globs.
//! Argument-aware selectors are rejected at validation time rather than being
//! silently treated as names that could grant or deny the wrong invocation.

use claurst_core::config::Config;

/// Validate tool selectors from configuration or CLI input.
pub fn validate_selectors(selectors: &[String], source: &str) -> Result<Vec<String>, String> {
    for selector in selectors {
        if selector.is_empty() {
            return Err(format!("{source} contains an empty tool selector"));
        }
        if selector.chars().any(|ch| {
            matches!(ch, '(' | ')' | '[' | ']' | '{' | '}' | ':' | '?' | '\\') || ch.is_whitespace()
        }) {
            return Err(format!(
                "unsupported selector '{selector}' in {source}; use a tool name or '*' glob"
            ));
        }
    }
    Ok(selectors.to_vec())
}

/// Match a selector against the complete tool name.
pub fn selector_matches(selector: &str, tool_name: &str) -> bool {
    let pattern: Vec<char> = selector.chars().collect();
    let name: Vec<char> = tool_name.chars().collect();
    let mut row = vec![false; name.len() + 1];
    row[0] = true;
    for pattern_char in pattern {
        if pattern_char == '*' {
            for index in 1..=name.len() {
                row[index] |= row[index - 1];
            }
        } else {
            let mut next = vec![false; name.len() + 1];
            for index in 1..=name.len() {
                next[index] = row[index - 1] && name[index - 1] == pattern_char;
            }
            row = next;
        }
    }
    row[name.len()]
}

/// Apply configured tool policy with deny taking precedence over allow.
pub fn tool_allowed(config: &Config, tool_name: &str) -> Result<bool, String> {
    let allowed = validate_selectors(&config.allowed_tools, "allowed_tools")?;
    let denied = validate_selectors(&config.disallowed_tools, "disallowed_tools")?;
    if denied
        .iter()
        .any(|selector| selector_matches(selector, tool_name))
    {
        return Ok(false);
    }
    Ok(allowed.is_empty()
        || allowed
            .iter()
            .any(|selector| selector_matches(selector, tool_name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_wins_and_globs_match_full_name() {
        let config = Config {
            allowed_tools: vec!["mcp__server__*".into()],
            disallowed_tools: vec!["mcp__server__secret".into()],
            ..Config::default()
        };
        assert_eq!(tool_allowed(&config, "mcp__server__read"), Ok(true));
        assert_eq!(tool_allowed(&config, "mcp__server__secret"), Ok(false));
        assert_eq!(tool_allowed(&config, "mcp__other__read"), Ok(false));
    }

    #[test]
    fn unsupported_argument_selectors_are_rejected() {
        let config = Config {
            allowed_tools: vec!["Bash(git:*)".into()],
            ..Config::default()
        };
        let error = tool_allowed(&config, "Bash").expect_err("argument selector must reject");
        assert!(error.contains("unsupported selector"));
    }
}
