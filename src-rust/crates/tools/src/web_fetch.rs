// WebFetch tool: HTTP GET with HTML-to-text conversion and LLM-powered semantic extraction
// for edge cases (JS-heavy pages, minimal content).

use crate::evidence::EvidenceStore;
use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, warn};

pub struct WebFetchTool;

#[derive(Debug, Deserialize)]
struct WebFetchInput {
    url: String,
    #[serde(default)]
    #[allow(dead_code)]
    prompt: Option<String>,
}

fn capture_evidence(ctx: &ToolContext, text: &str) -> Option<Value> {
    EvidenceStore::for_context(ctx)
        .and_then(|store| store.capture(text, false))
        .map(|reference| json!(reference))
        .map_err(|_| debug!("Failed to retain web observation"))
        .ok()
}

fn safe_prefix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn semantic_extraction_allowed(ctx: &ToolContext, prompt: Option<&str>) -> bool {
    std::env::var("FORGE_ALLOW_WEB_EXTRACTION").as_deref() == Ok("1")
        && prompt.is_some_and(|value| !value.trim().is_empty())
        && ctx.config.selected_provider_id() == "anthropic"
}

/// Detect if HTML is likely a JS-heavy page with minimal semantic content.
fn is_edge_case_html(html: &str, extracted_text: &str) -> bool {
    // Check word count (rough estimate)
    let word_count = extracted_text.split_whitespace().count();
    if word_count < 100 {
        debug!(word_count, "Edge case: low word count");
        return true;
    }

    // Check for semantic HTML tags
    let lower = html.to_lowercase();
    let has_semantic =
        lower.contains("<article") || lower.contains("<main") || lower.contains("<body");

    if !has_semantic {
        debug!("Edge case: no semantic HTML tags");
        return true;
    }

    false
}

const EXTRACTION_MODEL: &str = "claude-haiku-4-5";
const EXTRACTION_OUTPUT_TOKENS: u32 = 2_000;

fn extraction_wire_budget(
    config: &claurst_core::config::Config,
) -> Result<claurst_api::WireBudget, String> {
    let explicit = config
        .model_overrides
        .get(&format!("anthropic/{EXTRACTION_MODEL}"))
        .and_then(|entry| entry.context_window);
    let registry = claurst_api::ModelRegistry::new();
    let (context_limit, source) = match explicit {
        Some(limit) => (u64::from(limit), "explicit"),
        None => {
            let limit = registry
                .get("anthropic", EXTRACTION_MODEL)
                .map(|entry| u64::from(entry.info.context_window))
                .filter(|limit| *limit > 0 && *limit <= 100_000_000)
                .ok_or_else(|| {
                    "Web extraction model has no known finite context limit".to_string()
                })?;
            (limit, "registry")
        }
    };
    if context_limit == 0 || context_limit > 100_000_000 {
        return Err("Web extraction context override is invalid".into());
    }
    let output_reserve = u64::from(EXTRACTION_OUTPUT_TOKENS);
    let input_limit = context_limit
        .checked_sub(output_reserve)
        .ok_or_else(|| "Web extraction context does not leave room for input".to_string())?;
    claurst_api::WireBudget::finite(
        context_limit,
        input_limit,
        output_reserve,
        8 * 1024 * 1024,
        claurst_api::MediaAccounting::None,
    )
    .and_then(|budget| budget.with_source(source))
    .map_err(|error| error.to_string())
}

/// Call Claude Haiku to extract main content from HTML.
async fn semantic_extraction(html: &str, prompt: &str, ctx: &ToolContext) -> Option<String> {
    let wire_budget = match extraction_wire_budget(&ctx.config) {
        Ok(budget) => budget,
        Err(_) => {
            warn!("Skipping semantic extraction without a valid finite budget");
            return None;
        }
    };
    // Try to create an Anthropic client from the config
    let client = match claurst_api::AnthropicClient::from_config(&ctx.config) {
        Ok(c) => c,
        Err(_) => {
            warn!("Failed to create Anthropic client for semantic extraction");
            return None;
        }
    };

    // Truncate HTML to avoid exceeding token limits
    let html_excerpt = if html.len() > 20000 {
        format!("{}...", safe_prefix(html, 20000))
    } else {
        html.to_string()
    };

    let system = "You are a content extraction expert. Given HTML, extract and return only the main text content. Return just plain text, no markdown or formatting.";
    let user_message = format!(
        "{}\n\nExtract the main content from this HTML and return only the text:\n\n{}",
        prompt.trim(),
        html_excerpt
    );

    // Use the builder API to construct the request
    let api_messages = vec![claurst_api::ApiMessage {
        role: "user".to_string(),
        content: serde_json::Value::String(user_message),
    }];

    let request =
        claurst_api::CreateMessageRequest::builder(EXTRACTION_MODEL, EXTRACTION_OUTPUT_TOKENS)
            .wire_budget(wire_budget)
            .messages(api_messages)
            .system(claurst_api::SystemPrompt::Text(system.to_string()))
            .build();

    match client.create_message(request).await {
        Ok(response) => {
            // Extract text from the response content (Vec<Value>)
            // Response content is JSON objects like {"type": "text", "text": "..."}
            let text = response.content.iter().find_map(|block| {
                if block.get("type")?.as_str()? == "text" {
                    block.get("text")?.as_str().map(str::to_owned)
                } else {
                    None
                }
            });

            if let Some(extracted) = text {
                debug!(
                    extracted_len = extracted.len(),
                    "Semantic extraction successful"
                );
                return Some(extracted);
            }

            warn!("No text block in semantic extraction response");
            None
        }
        Err(_) => {
            warn!("Semantic extraction API call failed");
            None
        }
    }
}

/// Naively strip HTML tags and decode common entities.
fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;

    let lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if !in_tag && chars[i] == '<' {
            in_tag = true;
            // Check for script/style open/close
            let rest: String = lower_chars[i..].iter().take(20).collect();
            if rest.starts_with("<script") {
                in_script = true;
            } else if rest.starts_with("</script") {
                in_script = false;
            } else if rest.starts_with("<style") {
                in_style = true;
            } else if rest.starts_with("</style") {
                in_style = false;
            }
            // Block tags => newline
            let block_tags = [
                "<br", "<p ", "<p>", "</p>", "<div", "</div>", "<h1", "<h2", "<h3", "<h4", "<h5",
                "<h6", "</h1", "</h2", "</h3", "</h4", "</h5", "</h6", "<li", "</li", "<tr",
                "</tr", "<hr",
            ];
            for tag in &block_tags {
                if rest.starts_with(tag) {
                    result.push('\n');
                    break;
                }
            }
            i += 1;
            continue;
        }

        if in_tag {
            if chars[i] == '>' {
                in_tag = false;
            }
            i += 1;
            continue;
        }

        if in_script || in_style {
            i += 1;
            continue;
        }

        // Decode basic entities
        if chars[i] == '&' {
            let rest: String = chars[i..].iter().take(10).collect();
            if rest.starts_with("&amp;") {
                result.push('&');
                i += 5;
            } else if rest.starts_with("&lt;") {
                result.push('<');
                i += 4;
            } else if rest.starts_with("&gt;") {
                result.push('>');
                i += 4;
            } else if rest.starts_with("&quot;") {
                result.push('"');
                i += 6;
            } else if rest.starts_with("&#39;") || rest.starts_with("&apos;") {
                result.push('\'');
                i += if rest.starts_with("&#39;") { 5 } else { 6 };
            } else if rest.starts_with("&nbsp;") {
                result.push(' ');
                i += 6;
            } else {
                result.push('&');
                i += 1;
            }
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }

    // Collapse multiple blank lines
    let mut collapsed = String::new();
    let mut blank_count = 0;
    for line in result.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                collapsed.push('\n');
            }
        } else {
            blank_count = 0;
            collapsed.push_str(trimmed);
            collapsed.push('\n');
        }
    }

    collapsed.trim().to_string()
}

#[async_trait]
impl Tool for WebFetchTool {
    // Gates itself: calls `ctx.check_permission` in `execute()` (#210).
    fn self_gates(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        claurst_core::constants::TOOL_NAME_WEB_FETCH
    }

    fn description(&self) -> &str {
        "Fetches a web page URL and returns its content as text. HTML is \
         automatically converted to plain text. Use this for reading documentation, \
         APIs, and other web resources."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt for how to process the content"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: WebFetchInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Permission check
        if let Err(e) = ctx.check_permission(
            self.name(),
            "Fetch requested web URL",
            true, // read-only
        ) {
            return ToolResult::error(e.to_string());
        }

        debug!("Fetching web page");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build();

        let client = match client {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to create HTTP client: {}", e)),
        };

        let resp = match client
            .get(&params.url)
            .header(
                "User-Agent",
                concat!("Forge/", env!("CARGO_PKG_VERSION"), " (web-fetch)"),
            )
            .send()
            .await
        {
            Ok(r) => r,
            Err(_e) => return ToolResult::error("Failed to fetch requested URL"),
        };

        let status = resp.status();
        if !status.is_success() {
            return ToolResult::error(format!("HTTP {} when fetching requested URL", status));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = match resp.text().await {
            Ok(b) => b,
            Err(_) => return ToolResult::error("Failed to read response body"),
        };

        let source_evidence = capture_evidence(ctx, &body);

        // Convert HTML to text if applicable. Semantic extraction is opt-in:
        // it requires the explicit environment switch, a non-empty prompt in
        // this tool request, and the Anthropic provider that supports Haiku.
        let mut text = if content_type.contains("html") {
            strip_html(&body)
        } else {
            body.clone()
        };
        let mut extraction = if content_type.contains("html") {
            "deterministic_html"
        } else {
            "identity"
        };

        // Detect and handle edge cases with semantic extraction
        if content_type.contains("html")
            && is_edge_case_html(&body, &text)
            && semantic_extraction_allowed(ctx, params.prompt.as_deref())
        {
            debug!("Attempting opted-in semantic extraction for edge case");
            if let Some(extracted) =
                semantic_extraction(&body, params.prompt.as_deref().unwrap_or_default(), ctx).await
            {
                text = extracted;
                extraction = "semantic_model";
            } else {
                debug!("Semantic extraction failed, using basic HTML stripping");
            }
        }

        let text_evidence = capture_evidence(ctx, &text);

        // Truncate very long content
        const MAX_LEN: usize = 100_000;
        let text = if text.len() > MAX_LEN {
            format!(
                "{}\n\n... (truncated, {} total UTF-8 bytes)",
                safe_prefix(&text, MAX_LEN),
                text.len()
            )
        } else {
            text
        };

        let mut provenance = json!({"extraction": extraction});
        if let Some(reference) = source_evidence {
            provenance["source_evidence"] = reference;
        }
        if let Some(reference) = text_evidence {
            provenance["text_evidence"] = reference;
        }
        ToolResult::success(text).with_metadata(json!({"web_fetch": provenance}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::allow_all_context;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn serve_responses(
        responses: Vec<(String, String)>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let captured = requests.clone();
        let server = tokio::spawn(async move {
            for (body, content_type) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0; 8192];
                let length = stream.read(&mut buffer).await.unwrap();
                captured
                    .lock()
                    .push(String::from_utf8_lossy(&buffer[..length]).into_owned());
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
            captured.lock().clone()
        });
        let url = format!("http://{address}/fixture");
        (url, server)
    }

    fn context(tempdir: &TempDir) -> ToolContext {
        let mut ctx = allow_all_context(tempdir.path().to_path_buf());
        // The default test configuration is Anthropic-compatible, while this
        // fixture proves deterministic HTML conversion does not make an
        // auxiliary model call under another configured provider.
        ctx.config.provider = Some("openai".to_string());
        ctx
    }

    #[test]
    fn extraction_budget_honors_finite_override_and_rejects_invalid_pin() {
        let mut config = claurst_core::config::Config::default();
        config.model_overrides.insert(
            format!("anthropic/{EXTRACTION_MODEL}"),
            claurst_core::config::ModelOverride {
                context_window: Some(12_000),
                ..Default::default()
            },
        );
        let budget = extraction_wire_budget(&config).unwrap();
        assert_eq!(budget.context_limit_tokens, 12_000);
        assert_eq!(budget.input_limit_tokens, 10_000);
        assert_eq!(budget.budget_source, "explicit");
        assert_eq!(budget.media, claurst_api::MediaAccounting::None);
        config
            .model_overrides
            .get_mut(&format!("anthropic/{EXTRACTION_MODEL}"))
            .unwrap()
            .context_window = Some(0);
        assert!(extraction_wire_budget(&config).is_err());
    }

    #[tokio::test]
    async fn large_utf8_response_keeps_source_and_normalized_evidence() {
        let middle = "中".repeat(60_000);
        let body = format!("<html><body>head {middle} MIDDLE tail</body></html>");
        let (url, server) = serve_responses(vec![(body.clone(), "text/html".to_string())]).await;
        let tempdir = tempfile::tempdir().unwrap();
        let ctx = context(&tempdir);

        let result = WebFetchTool.execute(json!({"url": url}), &ctx).await;
        let _requests = server.await.unwrap();

        assert!(!result.is_error);
        assert!(result.content.len() <= 100_000 + 80);
        let provenance = &result.metadata.unwrap()["web_fetch"];
        assert_eq!(provenance["extraction"], "deterministic_html");
        let source_id = provenance["source_evidence"]["id"].as_str().unwrap();
        let text_id = provenance["text_evidence"]["id"].as_str().unwrap();
        let store = EvidenceStore::for_context(&ctx).unwrap();
        let (_, source) = store.read(source_id).unwrap();
        let (_, normalized) = store.read(text_id).unwrap();
        assert_eq!(source, body);
        assert!(normalized.contains("MIDDLE"));
        assert!(normalized.ends_with("tail"));
    }

    #[tokio::test]
    async fn fresh_response_is_not_replaced_by_stale_cached_text() {
        let (url, server) = serve_responses(vec![
            ("fresh-one".to_string(), "text/plain".to_string()),
            ("fresh-two".to_string(), "text/plain".to_string()),
        ])
        .await;
        let tempdir = tempfile::tempdir().unwrap();
        let ctx = context(&tempdir);

        let first = WebFetchTool.execute(json!({"url": url}), &ctx).await;
        let second = WebFetchTool.execute(json!({"url": url}), &ctx).await;
        let requests = server.await.unwrap();

        assert!(first.content.starts_with("fresh-one"));
        assert!(second.content.starts_with("fresh-two"));
        assert_eq!(requests.len(), 2);
    }

    #[tokio::test]
    async fn default_small_html_is_one_get_without_auxiliary_inference() {
        let (url, server) = serve_responses(vec![(
            "<html><body>Hello fixture</body></html>".to_string(),
            "text/html".to_string(),
        )])
        .await;
        let tempdir = tempfile::tempdir().unwrap();
        let ctx = context(&tempdir);

        let result = WebFetchTool.execute(json!({"url": url}), &ctx).await;
        let requests = server.await.unwrap();

        assert!(!result.is_error);
        assert!(result.content.starts_with("Hello fixture"));
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("GET /fixture HTTP/1.1"));
        let headers = requests[0].to_ascii_lowercase();
        assert!(headers.contains("user-agent: forge/"));
        assert!(!requests[0].contains("Claude-Code"));
    }
}
