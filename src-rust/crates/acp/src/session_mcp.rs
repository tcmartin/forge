//! Bounded validation before any client-supplied MCP process can be launched.

use agent_client_protocol_schema as acp;
use claurst_core::config::{McpServerConfig, McpServerOrigin};
use std::collections::{HashMap, HashSet};

pub(crate) const MAX_SESSION_MCP_CONNECTIONS: u32 = 8;
const MAX_ROSTER_BYTES: usize = 256 * 1024;
const MAX_ITEMS: usize = 128;
const MAX_VALUE_BYTES: usize = 16 * 1024;

pub(crate) fn unavailable(reason: &'static str) -> acp::Error {
    acp::Error::internal_error().data(Some(
        serde_json::json!({"kind":"session_mcp","reason":reason}),
    ))
}

fn invalid(reason: &'static str) -> acp::Error {
    acp::Error::invalid_params().data(Some(
        serde_json::json!({"kind":"session_mcp","reason":reason}),
    ))
}

pub(crate) fn validate(
    roster: &[acp::McpServer],
    configured: &[McpServerConfig],
) -> Result<Vec<McpServerConfig>, acp::Error> {
    if roster.len() > MAX_SESSION_MCP_CONNECTIONS as usize {
        return Err(invalid("server_count_limit"));
    }
    let mut names = configured
        .iter()
        .map(|server| server.name.to_lowercase())
        .collect::<HashSet<_>>();
    let mut bytes = 0usize;
    let mut configs = Vec::with_capacity(roster.len());
    for server in roster {
        let acp::McpServer::Stdio(server) = server else {
            return Err(invalid("unsupported_transport"));
        };
        if server.name.is_empty()
            || server.name.len() > 256
            || server.name.chars().any(char::is_control)
            || !names.insert(server.name.to_lowercase())
        {
            return Err(invalid("invalid_or_duplicate_name"));
        }
        let Some(command) = server.command.to_str() else {
            return Err(invalid("invalid_command"));
        };
        if !server.command.is_absolute()
            || !server.command.is_file()
            || command.len() > 4096
            || command.contains('\0')
        {
            return Err(invalid("invalid_command"));
        }
        if server.args.len() > MAX_ITEMS || server.env.len() > MAX_ITEMS {
            return Err(invalid("argument_or_environment_count_limit"));
        }
        bytes = bytes
            .saturating_add(server.name.len())
            .saturating_add(command.len());
        for arg in &server.args {
            if arg.len() > MAX_VALUE_BYTES || arg.contains('\0') {
                return Err(invalid("invalid_argument"));
            }
            bytes = bytes.saturating_add(arg.len());
            if bytes > MAX_ROSTER_BYTES {
                return Err(invalid("roster_byte_limit"));
            }
        }
        let mut env = HashMap::new();
        for entry in &server.env {
            if entry.name.is_empty()
                || entry.name.len() > 256
                || entry.name.contains(['=', '\0'])
                || entry.value.len() > MAX_VALUE_BYTES
                || entry.value.contains('\0')
                || env.contains_key(&entry.name)
            {
                return Err(invalid("invalid_or_duplicate_environment"));
            }
            bytes = bytes
                .saturating_add(entry.name.len())
                .saturating_add(entry.value.len());
            if bytes > MAX_ROSTER_BYTES {
                return Err(invalid("roster_byte_limit"));
            }
            env.insert(entry.name.clone(), entry.value.clone());
        }
        if bytes > MAX_ROSTER_BYTES {
            return Err(invalid("roster_byte_limit"));
        }
        configs.push(McpServerConfig {
            name: server.name.clone(),
            command: Some(command.to_string()),
            args: server.args.clone(),
            env,
            url: None,
            server_type: "stdio".to_string(),
            origin: McpServerOrigin::User,
        });
    }
    Ok(configs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(name: &str) -> acp::McpServer {
        serde_json::from_value(serde_json::json!({
            "name":name,"command":std::env::current_exe().unwrap(),
            "args":["$LITERAL_ARG"],"env":[{"name":"FIXTURE","value":"$LITERAL_ENV"}]
        }))
        .unwrap()
    }

    #[test]
    fn human_names_and_literal_values_are_preserved() {
        let configs = validate(&[server("Équipe Files")], &[]).unwrap();
        assert_eq!(configs[0].name, "Équipe Files");
        assert_eq!(configs[0].args, ["$LITERAL_ARG"]);
        assert_eq!(configs[0].env["FIXTURE"], "$LITERAL_ENV");
        assert!(validate(&[server("équipe files")], &configs).is_err());
    }

    #[test]
    fn complete_roster_rejects_late_invalid_entries_and_unicode_collisions() {
        assert!(validate(&[server("valid"), server("bad\nname")], &[]).is_err());
        assert!(validate(&[server("Équipe"), server("équipe")], &[]).is_err());
        let mut duplicate = server("env");
        if let acp::McpServer::Stdio(server) = &mut duplicate {
            server.env.push(server.env[0].clone());
        }
        let error = validate(&[duplicate], &[]).unwrap_err();
        let encoded = serde_json::to_string(&error).unwrap();
        assert!(!encoded.contains("$LITERAL_ENV"));
        assert!(encoded.contains("invalid_or_duplicate_environment"));
    }

    #[test]
    fn count_and_aggregate_bytes_fail_before_configuration_is_returned() {
        assert!(validate(&vec![server("entry"); 9], &[]).is_err());
        let mut large = server("large");
        if let acp::McpServer::Stdio(server) = &mut large {
            server.args = vec!["x".repeat(MAX_VALUE_BYTES); 17];
        }
        let error = validate(&[large], &[]).unwrap_err();
        assert!(serde_json::to_string(&error)
            .unwrap()
            .contains("roster_byte_limit"));
    }
}
