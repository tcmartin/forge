//! Bounded conversion of native MCP tool content into provider-facing blocks.
//!
//! MCP responses are remote input.  This module keeps their order, validates
//! inline image bytes through the shared core media policy, and keeps encoded
//! payloads out of the model-visible display summary.

use crate::{ExecutionStatus, ToolContext, ToolResult, TypedProvenance};
use claurst_core::media::MediaLimits;
use claurst_core::types::{ContentBlock, ImageSource};
use claurst_mcp::{CallToolResult, McpContent, ResourceContents};

const MAX_MCP_CONTENT_ITEMS: usize = 100;
const MAX_MCP_INPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_MCP_URI_BYTES: usize = 16 * 1024;
const MAX_MCP_MIME_BYTES: usize = 256;
const MAX_DISPLAY_BYTES: usize = 32 * 1024;
const MAX_DISPLAY_PART_BYTES: usize = 16 * 1024;
const MCP_VALIDATION_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Convert an MCP tool result to a bounded tool result while retaining native
/// text/image/resource order in validated typed blocks.
pub async fn mcp_result_to_tool_result(result: CallToolResult, ctx: &ToolContext) -> ToolResult {
    mcp_result_to_tool_result_with_deadline(result, ctx, MCP_VALIDATION_DEADLINE).await
}

async fn mcp_result_to_tool_result_with_deadline(
    result: CallToolResult,
    ctx: &ToolContext,
    deadline: std::time::Duration,
) -> ToolResult {
    if ctx.cancel_token.is_cancelled() {
        return ToolResult::error("MCP tool result discarded after cancellation")
            .with_execution_status(ExecutionStatus::Cancelled);
    }

    if let Err(error) = preflight(&result) {
        return ToolResult::error(format!("MCP content rejected locally: {error}"));
    }

    let limits = MediaLimits::default();
    let cancellation = ctx.cancel_token.clone();
    let worker_cancellation = cancellation.child_token();
    let worker_for_task = worker_cancellation.clone();
    let validation =
        tokio::task::spawn_blocking(move || convert_owned(result, &limits, &worker_for_task));
    finish_validation(validation, cancellation, worker_cancellation, deadline).await
}

async fn finish_validation(
    validation: tokio::task::JoinHandle<Result<ToolResult, String>>,
    cancellation: tokio_util::sync::CancellationToken,
    worker_cancellation: tokio_util::sync::CancellationToken,
    deadline: std::time::Duration,
) -> ToolResult {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            worker_cancellation.cancel();
            ToolResult::error("MCP tool result discarded after cancellation")
                .with_execution_status(ExecutionStatus::Cancelled)
        },
        _ = tokio::time::sleep(deadline) => {
            if cancellation.is_cancelled() {
                worker_cancellation.cancel();
                ToolResult::error("MCP tool result discarded after cancellation")
                    .with_execution_status(ExecutionStatus::Cancelled)
            } else {
                worker_cancellation.cancel();
                ToolResult::error("MCP content validation deadline exceeded")
            }
        },
        joined = validation => {
            if cancellation.is_cancelled() {
                ToolResult::error("MCP tool result discarded after cancellation")
                    .with_execution_status(ExecutionStatus::Cancelled)
            } else {
                match joined {
                    Ok(Ok(result)) => result,
                    Ok(Err(error)) => ToolResult::error(format!("MCP content rejected locally: {error}")),
                    Err(error) => ToolResult::error(format!("MCP content validation failed: {error}")),
                }
            }
        },
    }
}

fn preflight(result: &CallToolResult) -> Result<(), String> {
    if result.content.len() > MAX_MCP_CONTENT_ITEMS {
        return Err(format!(
            "content item count exceeds {}",
            MAX_MCP_CONTENT_ITEMS
        ));
    }
    let mut total = 0usize;
    for content in &result.content {
        match content {
            McpContent::Text { text } => {
                checked_add(&mut total, text.len(), "input size")?;
            }
            McpContent::Image { data, mime_type } | McpContent::Audio { data, mime_type } => {
                checked_add(&mut total, data.len(), "input size")?;
                validate_mime_size(mime_type)?;
                checked_add(&mut total, mime_type.len(), "input size")?;
            }
            McpContent::Resource { resource } => preflight_resource(resource, &mut total)?,
            McpContent::ResourceLink {
                uri,
                name,
                description,
                mime_type,
            } => {
                validate_uri_size(uri)?;
                checked_add(&mut total, uri.len(), "input size")?;
                checked_add(&mut total, name.len(), "input size")?;
                if let Some(description) = description {
                    checked_add(&mut total, description.len(), "input size")?;
                }
                if let Some(mime_type) = mime_type {
                    validate_mime_size(mime_type)?;
                    checked_add(&mut total, mime_type.len(), "input size")?;
                }
            }
        }
        if total > MAX_MCP_INPUT_BYTES {
            return Err(format!("input exceeds {} bytes", MAX_MCP_INPUT_BYTES));
        }
    }
    Ok(())
}

fn preflight_resource(resource: &ResourceContents, total: &mut usize) -> Result<(), String> {
    validate_uri_size(&resource.uri)?;
    checked_add(total, resource.uri.len(), "input size")?;
    if let Some(mime_type) = &resource.mime_type {
        validate_mime_size(mime_type)?;
        checked_add(total, mime_type.len(), "input size")?;
    }
    if resource.text.is_some() && resource.blob.is_some() {
        return Err("resource contains both text and blob payloads".to_string());
    }
    if let Some(text) = &resource.text {
        checked_add(total, text.len(), "input size")?;
    }
    if let Some(blob) = &resource.blob {
        checked_add(total, blob.len(), "input size")?;
    }
    Ok(())
}

fn checked_add(total: &mut usize, amount: usize, label: &str) -> Result<(), String> {
    *total = total
        .checked_add(amount)
        .ok_or_else(|| format!("{label} arithmetic overflow"))?;
    Ok(())
}

fn validate_uri_size(uri: &str) -> Result<(), String> {
    if uri.len() > MAX_MCP_URI_BYTES {
        Err(format!("resource URI exceeds {} bytes", MAX_MCP_URI_BYTES))
    } else {
        Ok(())
    }
}

fn validate_mime_size(mime_type: &str) -> Result<(), String> {
    if mime_type.is_empty() {
        return Err("MCP MIME type is empty".to_string());
    }
    if mime_type.len() > MAX_MCP_MIME_BYTES {
        return Err(format!(
            "MCP MIME type exceeds {} bytes",
            MAX_MCP_MIME_BYTES
        ));
    }
    Ok(())
}

fn convert_owned(
    result: CallToolResult,
    limits: &MediaLimits,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<ToolResult, String> {
    let is_error = result.is_error;
    let mut blocks = Vec::with_capacity(result.content.len());
    let mut typed_provenance = TypedProvenance::default();
    let mut display_parts = Vec::with_capacity(result.content.len());
    for content in result.content {
        if cancellation.is_cancelled() {
            return Err("MCP content validation cancelled".to_string());
        }
        match content {
            McpContent::Text { text } => {
                display_parts.push(truncate_utf8(&text, MAX_DISPLAY_PART_BYTES));
                blocks.push(ContentBlock::Text { text });
            }
            McpContent::Image { data, mime_type } => {
                display_parts.push(format!("[Image: {mime_type}]"));
                blocks.push(ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: Some(mime_type),
                        data: Some(data),
                        url: None,
                    },
                });
            }
            McpContent::Audio { mime_type, .. } => {
                return Err(format!("audio content is unsupported ({mime_type})"));
            }
            McpContent::Resource { resource } => {
                let ResourceContents {
                    uri,
                    mime_type,
                    text,
                    blob,
                } = resource;
                let display_provenance = match mime_type.as_deref() {
                    Some(mime_type) => format!("[Resource: {uri} | {mime_type}]"),
                    None => format!("[Resource: {uri}]"),
                };
                match (text, blob) {
                    (Some(text), None) => {
                        display_parts.push(format!(
                            "{display_provenance}\n{}",
                            truncate_utf8(&text, MAX_DISPLAY_PART_BYTES)
                        ));
                        let provenance_index = blocks.len();
                        blocks.push(ContentBlock::Text {
                            text: display_provenance,
                        });
                        typed_provenance
                            .resource_text_indices
                            .push(provenance_index);
                        blocks.push(ContentBlock::Text { text });
                    }
                    (None, Some(data)) => {
                        let mime_type = mime_type
                            .ok_or_else(|| "resource blob is missing its MIME type".to_string())?;
                        if !is_supported_image_mime(&mime_type) {
                            return Err(format!(
                                "resource blob MIME type is unsupported: {mime_type}"
                            ));
                        }
                        display_parts.push(display_provenance);
                        let provenance_index = blocks.len();
                        blocks.push(ContentBlock::Text {
                            text: format!("[Resource: {uri} | {mime_type}]"),
                        });
                        typed_provenance
                            .resource_text_indices
                            .push(provenance_index);
                        blocks.push(ContentBlock::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type: Some(mime_type),
                                data: Some(data),
                                url: None,
                            },
                        });
                    }
                    (None, None) => return Err("resource has no text or blob payload".to_string()),
                    (Some(_), Some(_)) => {
                        return Err("resource contains both text and blob payloads".to_string())
                    }
                }
            }
            McpContent::ResourceLink {
                uri,
                name,
                description,
                mime_type,
            } => {
                // URI and MIME identify the remote resource and remain
                // private provenance. Name/description are payload text so
                // large descriptions stay pageable and count in the
                // canonical projection.
                let mut provenance = format!("[Resource link: {uri}]");
                if let Some(mime_type) = mime_type {
                    provenance.push_str(&format!(" [{mime_type}]"));
                }
                let mut payload = name;
                if let Some(description) = description {
                    payload.push(' ');
                    payload.push_str(&description);
                }
                display_parts.push(truncate_utf8(
                    &format!("{provenance}\n{payload}"),
                    MAX_DISPLAY_PART_BYTES,
                ));
                let provenance_index = blocks.len();
                blocks.push(ContentBlock::Text { text: provenance });
                typed_provenance
                    .resource_text_indices
                    .push(provenance_index);
                blocks.push(ContentBlock::Text { text: payload });
            }
        }
    }

    if cancellation.is_cancelled() {
        return Err("MCP content validation cancelled".to_string());
    }
    let display = truncate_utf8(&display_parts.join("\n"), MAX_DISPLAY_BYTES);
    let base = if is_error {
        ToolResult::error(display)
    } else {
        ToolResult::success(display)
    };
    if cancellation.is_cancelled() {
        return Err("MCP content validation cancelled".to_string());
    }
    let result = base.with_mcp_typed_content(blocks, typed_provenance, limits)?;
    if cancellation.is_cancelled() {
        return Err("MCP content validation cancelled".to_string());
    }
    Ok(result)
}

fn is_supported_image_mime(mime_type: &str) -> bool {
    mime_type.eq_ignore_ascii_case("image/png") || mime_type.eq_ignore_ascii_case("image/jpeg")
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes.saturating_sub("…".len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Genuine RGB 1px PNGs from the shared provider-wire media fixtures.
    const RED: &str =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
    const BLUE: &str =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC";
    const RED_SHA256: &str = "b1ff9c8ea3a780bad09b346c423d2d0e46815926879b18e841d928376a946640";
    const BLUE_SHA256: &str = "fce481932ea5d07a91c7991c09fdadb4bf78f9b4cc8f927188384231f9d12679";

    fn context() -> (tempfile::TempDir, ToolContext) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = PathBuf::from(dir.path());
        let context = crate::test_support::allow_all_context(path);
        (dir, context)
    }

    #[tokio::test]
    async fn mcp_typed_ingress_preserves_mixed_order_and_hashes() {
        let (_dir, context) = context();
        let result = CallToolResult {
            content: vec![
                McpContent::Text {
                    text: "before".to_string(),
                },
                McpContent::Image {
                    data: RED.to_string(),
                    mime_type: "image/png".to_string(),
                },
                McpContent::Resource {
                    resource: ResourceContents {
                        uri: "file:///image.png".to_string(),
                        mime_type: Some("image/png".to_string()),
                        text: None,
                        blob: Some(BLUE.to_string()),
                    },
                },
            ],
            is_error: true,
        };
        let converted = mcp_result_to_tool_result(result, &context).await;
        assert!(converted.is_error);
        let blocks = converted.typed_blocks().expect("typed blocks");
        assert!(matches!(blocks[0], ContentBlock::Text { .. }));
        match &blocks[1] {
            ContentBlock::Image { source } => {
                assert_eq!(source.media_type.as_deref(), Some("image/png"));
                assert_eq!(source.data.as_deref(), Some(RED));
            }
            other => panic!("expected inline image, got {other:?}"),
        }
        match &blocks[2] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "[Resource: file:///image.png | image/png]");
            }
            other => panic!("expected resource provenance, got {other:?}"),
        }
        match &blocks[3] {
            ContentBlock::Image { source } => {
                assert_eq!(source.media_type.as_deref(), Some("image/png"));
                assert_eq!(source.data.as_deref(), Some(BLUE));
            }
            other => panic!("expected resource image, got {other:?}"),
        }
        let manifest = converted.typed_manifest().expect("manifest");
        assert_eq!(manifest.items().len(), 2);
        assert_eq!(manifest.items()[0].sha256, RED_SHA256);
        assert_eq!(manifest.items()[1].sha256, BLUE_SHA256);
        assert_ne!(manifest.items()[0].sha256, manifest.items()[1].sha256);
    }

    #[tokio::test]
    async fn mcp_typed_ingress_rejects_mismatch_and_malformed_without_payload_and_retains_large_text(
    ) {
        let (_dir, context) = context();
        for data in ["aGVsbG8=", "not-base64??"] {
            let converted = mcp_result_to_tool_result(
                CallToolResult {
                    content: vec![McpContent::Image {
                        data: data.to_string(),
                        mime_type: "image/png".to_string(),
                    }],
                    is_error: false,
                },
                &context,
            )
            .await;
            assert!(converted.is_error);
            assert!(converted.typed_blocks().is_none());
        }
        let converted = mcp_result_to_tool_result(
            CallToolResult {
                content: vec![McpContent::Text {
                    text: "x".repeat(512 * 1024),
                }],
                is_error: false,
            },
            &context,
        )
        .await;
        assert!(!converted.is_error);
        let blocks = converted.typed_blocks().expect("large text is retained");
        match &blocks[0] {
            ContentBlock::Text { text } => assert_eq!(text, &"x".repeat(512 * 1024)),
            other => panic!("expected large text, got {other:?}"),
        }
        assert!(converted.content.len() <= MAX_DISPLAY_BYTES);
    }

    #[tokio::test]
    async fn mcp_typed_ingress_rejects_aggregate_count_and_uri_bounds_without_payload() {
        let (_dir, context) = context();
        let cases = [
            CallToolResult {
                content: vec![McpContent::Text {
                    text: "x".repeat(MAX_MCP_INPUT_BYTES + 1),
                }],
                is_error: false,
            },
            CallToolResult {
                content: (0..=MAX_MCP_CONTENT_ITEMS)
                    .map(|_| McpContent::Text {
                        text: "x".to_string(),
                    })
                    .collect(),
                is_error: false,
            },
            CallToolResult {
                content: vec![McpContent::ResourceLink {
                    uri: "u".repeat(MAX_MCP_URI_BYTES + 1),
                    name: "link".to_string(),
                    description: None,
                    mime_type: None,
                }],
                is_error: false,
            },
        ];
        for result in cases {
            let converted = mcp_result_to_tool_result(result, &context).await;
            assert!(converted.is_error);
            assert!(converted.typed_blocks().is_none());
        }
    }

    #[tokio::test]
    async fn mcp_typed_ingress_accepts_json_and_xml_resource_text_with_provenance() {
        let (_dir, context) = context();
        let converted = mcp_result_to_tool_result(
            CallToolResult {
                content: vec![
                    McpContent::Resource {
                        resource: ResourceContents {
                            uri: "file:///data.json".to_string(),
                            mime_type: Some("application/json".to_string()),
                            text: Some("{\"ok\":true}".to_string()),
                            blob: None,
                        },
                    },
                    McpContent::Resource {
                        resource: ResourceContents {
                            uri: "file:///data.xml".to_string(),
                            mime_type: Some("application/xml".to_string()),
                            text: Some("<ok>true</ok>".to_string()),
                            blob: None,
                        },
                    },
                ],
                is_error: false,
            },
            &context,
        )
        .await;
        let blocks = converted.typed_blocks().expect("typed resource text");
        assert_eq!(blocks.len(), 4);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Text { text } if text == "[Resource: file:///data.json | application/json]"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text } if text == "{\"ok\":true}"
        ));
        assert!(matches!(
            &blocks[2],
            ContentBlock::Text { text } if text == "[Resource: file:///data.xml | application/xml]"
        ));
        assert!(matches!(
            &blocks[3],
            ContentBlock::Text { text } if text == "<ok>true</ok>"
        ));
    }

    #[tokio::test]
    async fn mcp_typed_ingress_keeps_same_content_resources_associated_by_uri() {
        let (_dir, context) = context();
        let converted = mcp_result_to_tool_result(
            CallToolResult {
                content: vec![
                    McpContent::Resource {
                        resource: ResourceContents {
                            uri: "file:///red-a.png".to_string(),
                            mime_type: Some("image/png".to_string()),
                            text: None,
                            blob: Some(RED.to_string()),
                        },
                    },
                    McpContent::Resource {
                        resource: ResourceContents {
                            uri: "file:///red-b.png".to_string(),
                            mime_type: Some("image/png".to_string()),
                            text: None,
                            blob: Some(RED.to_string()),
                        },
                    },
                ],
                is_error: false,
            },
            &context,
        )
        .await;
        let blocks = converted.typed_blocks().expect("typed resource images");
        assert_eq!(blocks.len(), 4);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Text { text } if text == "[Resource: file:///red-a.png | image/png]"
        ));
        assert!(matches!(
            &blocks[2],
            ContentBlock::Text { text } if text == "[Resource: file:///red-b.png | image/png]"
        ));
        for index in [1, 3] {
            match &blocks[index] {
                ContentBlock::Image { source } => {
                    assert_eq!(source.media_type.as_deref(), Some("image/png"));
                    assert_eq!(source.data.as_deref(), Some(RED));
                }
                other => panic!("expected resource image at {index}, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn mcp_typed_ingress_retains_full_resource_link_description() {
        let (_dir, context) = context();
        let description = "d".repeat(512 * 1024);
        let converted = mcp_result_to_tool_result(
            CallToolResult {
                content: vec![McpContent::ResourceLink {
                    uri: "file:///large.txt".to_string(),
                    name: "large".to_string(),
                    description: Some(description.clone()),
                    mime_type: Some("text/plain".to_string()),
                }],
                is_error: false,
            },
            &context,
        )
        .await;
        let blocks = converted.typed_blocks().expect("typed resource link");
        match &blocks[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "[Resource link: file:///large.txt] [text/plain]");
            }
            other => panic!("expected typed resource link provenance, got {other:?}"),
        }
        match &blocks[1] {
            ContentBlock::Text { text } => {
                assert_eq!(text, &format!("large {description}"));
            }
            other => panic!("expected typed resource link payload, got {other:?}"),
        }
        assert!(converted.content.len() <= MAX_DISPLAY_BYTES);
    }

    #[tokio::test]
    async fn mcp_typed_ingress_deadline_discards_payload_without_not_run_status() {
        let (_dir, context) = context();
        let worker_cancellation = tokio_util::sync::CancellationToken::new();
        let worker_token = worker_cancellation.clone();
        let worker_observed_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_stop = worker_observed_stop.clone();
        let validation = tokio::task::spawn_blocking(move || {
            while !worker_token.is_cancelled() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            observed_stop.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok::<ToolResult, String>(ToolResult::success("late"))
        });
        let converted = finish_validation(
            validation,
            context.cancel_token.clone(),
            worker_cancellation,
            std::time::Duration::from_millis(1),
        )
        .await;
        assert!(converted.is_error);
        assert!(converted.typed_blocks().is_none());
        assert_ne!(converted.execution_status(), Some(ExecutionStatus::NotRun));
        assert!(!context.cancel_token.is_cancelled());
        tokio::time::timeout(std::time::Duration::from_millis(100), async {
            while !worker_observed_stop.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("worker should observe deadline cancellation");
    }

    #[tokio::test]
    async fn mcp_typed_ingress_active_cancellation_revokes_waiting_validation() {
        let (_dir, context) = context();
        let cancellation = context.cancel_token.clone();
        let worker_cancellation = tokio_util::sync::CancellationToken::new();
        let validation = tokio::task::spawn_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok::<ToolResult, String>(ToolResult::success("late"))
        });
        let waiting = tokio::spawn(finish_validation(
            validation,
            cancellation.clone(),
            worker_cancellation,
            std::time::Duration::from_secs(5),
        ));
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        cancellation.cancel();
        let converted = waiting.await.expect("validation waiter");
        assert_eq!(
            converted.execution_status(),
            Some(ExecutionStatus::Cancelled)
        );
        assert!(converted.typed_blocks().is_none());
    }

    #[tokio::test]
    async fn mcp_typed_ingress_rejects_unsupported_and_ambiguous_content() {
        let (_dir, context) = context();
        let cases = [
            vec![McpContent::Audio {
                data: RED.to_string(),
                mime_type: "audio/wav".to_string(),
            }],
            vec![McpContent::Resource {
                resource: ResourceContents {
                    uri: "file:///x.pdf".to_string(),
                    mime_type: Some("application/pdf".to_string()),
                    text: None,
                    blob: Some("JVBERi0xLjQK".to_string()),
                },
            }],
            vec![McpContent::Resource {
                resource: ResourceContents {
                    uri: "file:///ambiguous".to_string(),
                    mime_type: Some("text/plain".to_string()),
                    text: Some("text".to_string()),
                    blob: Some(RED.to_string()),
                },
            }],
        ];
        for content in cases {
            let converted = mcp_result_to_tool_result(
                CallToolResult {
                    content,
                    is_error: false,
                },
                &context,
            )
            .await;
            assert!(converted.is_error);
            assert!(converted.typed_blocks().is_none());
        }
    }

    #[tokio::test]
    async fn mcp_typed_ingress_cancellation_revokes_payload_before_validation() {
        let (_dir, context) = context();
        context.cancel_token.cancel();
        let converted = mcp_result_to_tool_result(
            CallToolResult {
                content: vec![McpContent::Image {
                    data: RED.to_string(),
                    mime_type: "image/png".to_string(),
                }],
                is_error: false,
            },
            &context,
        )
        .await;
        assert_eq!(
            converted.execution_status(),
            Some(ExecutionStatus::Cancelled)
        );
        assert!(converted.typed_blocks().is_none());
    }

    #[tokio::test]
    async fn mcp_typed_ingress_display_never_contains_encoded_payload() {
        let (_dir, context) = context();
        let converted = mcp_result_to_tool_result(
            CallToolResult {
                content: vec![McpContent::Image {
                    data: RED.to_string(),
                    mime_type: "image/png".to_string(),
                }],
                is_error: false,
            },
            &context,
        )
        .await;
        assert!(!converted.content.contains(RED));
        assert!(!converted.content.contains("base64"));
    }
}
