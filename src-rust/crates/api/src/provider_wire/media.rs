//! Revisioned image estimates and ordered canonical-to-wire verification.
//!
//! Policies use documented vision rules, with conservative integer rounding
//! and no enlargement. Encoded payload bytes remain subject to the wire cap.
//! References (checked 2026-10-14):
//! https://developers.openai.com/api/docs/guides/images-vision
//! https://platform.claude.com/docs/en/build-with-claude/vision
//! https://ai.google.dev/gemini-api/docs/image-understanding
//! https://ai.google.dev/gemini-api/docs/generate-content/media-resolution
//! https://ai.google.dev/gemini-api/docs/generate-content/function-calling
//! https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html
//!
//! These are local admission estimates, not provider billing quotations.

use super::{MediaAccounting, WireBudgetError};
use claurst_core::media::{MediaLimits, MediaManifest, MediaManifestBuilder};
use claurst_core::types::{ContentBlock, Message, MessageContent, ToolResultContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const POLICY_REVISION: &str = "inline-images-v1-20261014";
const MAX_IMAGES: usize = 20;
const MAX_IMAGE_ENCODED_BYTES: usize = 5 * 1024 * 1024;
const MAX_BEDROCK_IMAGE_DECODED_BYTES: usize = 3_750_000;

/// A payload-free record of a freshly validated image selection. Deserialized
/// records are untrusted: preparation recomputes the policy and wire manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageAccounting {
    provider: String,
    model: String,
    policy_revision: String,
    tokens: u64,
    manifest: MediaManifest,
}

impl ImageAccounting {
    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn policy_revision(&self) -> &str {
        &self.policy_revision
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    pub fn manifest(&self) -> &MediaManifest {
        &self.manifest
    }
}

#[derive(Clone, Copy)]
enum Estimate {
    Claude { visual_cap: u64 },
    OpenAiTiles { base: u64, tile: u64 },
    OpenAiPatches { numerator: u64, denominator: u64 },
    Gemini2,
    Gemini3,
}

fn policy(provider: &str, model: &str) -> Result<Estimate, WireBudgetError> {
    if provider == "amazon-bedrock" {
        // Recognize only explicit foundation model IDs and documented regional
        // namespaces. Arbitrary profile names and ARNs carry no image policy.
        let model = ["us.", "eu.", "au.", "jp.", "apac.", "global."]
            .iter()
            .find_map(|prefix| model.strip_prefix(prefix))
            .unwrap_or(model);
        return match model {
            "anthropic.claude-sonnet-4-6"
            | "anthropic.claude-sonnet-4-5-20250929-v1:0"
            | "anthropic.claude-opus-4-6-v1"
            | "anthropic.claude-opus-4-5-20251101-v1:0"
            | "anthropic.claude-3-7-sonnet-20250219-v1:0"
            | "anthropic.claude-3-5-sonnet-20241022-v2:0"
            | "anthropic.claude-haiku-4-5-20251001-v1:0" => {
                Ok(Estimate::Claude { visual_cap: 1568 })
            }
            "anthropic.claude-opus-4-7"
            | "anthropic.claude-opus-4-8"
            | "anthropic.claude-opus-5"
            | "anthropic.claude-opus-5-5"
            | "anthropic.claude-sonnet-5"
            | "anthropic.claude-sonnet-5-5"
            | "anthropic.claude-haiku-5-5" => Ok(Estimate::Claude { visual_cap: 4784 }),
            _ => Err(WireBudgetError::UnsupportedMedia),
        };
    }
    let estimate = match (provider, model) {
        (
            "anthropic",
            "claude-sonnet-4"
            | "claude-sonnet-4-20250514"
            | "claude-sonnet-4-5"
            | "claude-sonnet-4-5-20250929"
            | "claude-sonnet-4-6"
            | "claude-opus-4"
            | "claude-opus-4-20250514"
            | "claude-opus-4-1"
            | "claude-opus-4-5"
            | "claude-opus-4-6"
            | "claude-haiku-4-5"
            | "claude-haiku-4-5-20251001",
        ) => Estimate::Claude { visual_cap: 1568 },
        ("anthropic", "claude-opus-4-7" | "claude-opus-4-8" | "claude-opus-5-5") => {
            Estimate::Claude { visual_cap: 4784 }
        }
        ("openai", "gpt-4o" | "gpt-4.1") => Estimate::OpenAiTiles {
            base: 85,
            tile: 170,
        },
        ("openai", "gpt-4o-mini") => Estimate::OpenAiTiles {
            base: 2833,
            tile: 5667,
        },
        ("openai", "gpt-4.1-mini") => Estimate::OpenAiPatches {
            numerator: 162,
            denominator: 100,
        },
        (
            "google",
            "gemini-2.0-flash" | "gemini-2.5-pro" | "gemini-2.5-flash" | "gemini-2.5-flash-lite",
        ) => Estimate::Gemini2,
        (
            "google",
            "gemini-3-flash-preview"
            | "gemini-3-pro-preview"
            | "gemini-3.1-pro-preview"
            | "gemini-3.5-flash"
            | "gemini-3.8-flash",
        ) => Estimate::Gemini3,
        _ => return Err(WireBudgetError::UnsupportedMedia),
    };
    Ok(estimate)
}

pub(super) fn account_messages(
    provider: &str,
    model: &str,
    messages: &[Message],
) -> Result<MediaAccounting, WireBudgetError> {
    let provider = canonical_media_provider(provider);
    let manifest = MediaManifest::from_messages(messages, &MediaLimits::default())?;
    // These input formats accept images only in user content. Reject an
    // assistant-role image before transformation can silently flatten it.
    if !manifest.items().is_empty()
        && matches!(
            provider,
            "anthropic" | "openai" | "google" | "amazon-bedrock"
        ) && messages.iter().any(|message| {
        message.role == claurst_core::types::Role::Assistant
            && matches!(&message.content, MessageContent::Blocks(blocks) if contains_image(blocks))
    }) {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    if !manifest.items().is_empty()
        && provider == "google"
        && matches!(policy(provider, model)?, Estimate::Gemini2)
    {
        for message in messages {
            if let MessageContent::Blocks(blocks) = &message.content {
                for block in blocks {
                    if let ContentBlock::ToolResult {
                        content: ToolResultContent::Blocks(inner),
                        ..
                    } = block
                    {
                        if !MediaManifest::from_blocks(inner, &MediaLimits::default())?
                            .items()
                            .is_empty()
                        {
                            return Err(WireBudgetError::UnsupportedMedia);
                        }
                    }
                }
            }
        }
    }
    account_manifest(provider, model, manifest)
}

fn account_manifest(
    provider: &str,
    model: &str,
    manifest: MediaManifest,
) -> Result<MediaAccounting, WireBudgetError> {
    if manifest.items().is_empty() {
        return Ok(MediaAccounting::None);
    }
    let estimate = policy(provider, model)?;
    if manifest.items().len() > MAX_IMAGES {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    let mut tokens = 0u64;
    for item in manifest.items() {
        let (Some(width), Some(height)) = (item.width, item.height) else {
            return Err(WireBudgetError::MediaMismatch);
        };
        if width > 8000
            || height > 8000
            || item.encoded_bytes > MAX_IMAGE_ENCODED_BYTES
            || (provider == "amazon-bedrock" && item.raw_bytes > MAX_BEDROCK_IMAGE_DECODED_BYTES)
        {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        let cost = estimate_image(estimate, u64::from(width), u64::from(height))?;
        tokens = tokens
            .checked_add(cost)
            .ok_or(WireBudgetError::ArithmeticOverflow)?;
    }
    Ok(MediaAccounting::Images(ImageAccounting {
        provider: provider.to_owned(),
        model: model.to_owned(),
        policy_revision: POLICY_REVISION.to_owned(),
        tokens,
        manifest,
    }))
}

fn ceil_div(value: u64, divisor: u64) -> Result<u64, WireBudgetError> {
    if divisor == 0 {
        return Err(WireBudgetError::ArithmeticOverflow);
    }
    value
        .checked_add(divisor - 1)
        .map(|v| v / divisor)
        .ok_or(WireBudgetError::ArithmeticOverflow)
}

fn product(a: u64, b: u64) -> Result<u64, WireBudgetError> {
    a.checked_mul(b).ok_or(WireBudgetError::ArithmeticOverflow)
}

fn estimate_image(estimate: Estimate, width: u64, height: u64) -> Result<u64, WireBudgetError> {
    match estimate {
        Estimate::Claude { visual_cap } => {
            // The documented resize rules cap visual tokens. Before resizing,
            // ceil coverage also bounds padding to whole 28px patches.
            let patches = product(ceil_div(width, 28)?, ceil_div(height, 28)?)?;
            Ok(patches.min(visual_cap))
        }
        Estimate::OpenAiPatches {
            numerator,
            denominator,
        } => {
            // Unresized patch coverage conservatively includes every supported
            // detail level, including original. Provider downsizing only lowers it.
            let patches = product(ceil_div(width, 32)?, ceil_div(height, 32)?)?;
            // Provider documentation notes possible one-token rounding variation.
            ceil_div(product(patches, numerator)?, denominator)?
                .checked_add(1)
                .ok_or(WireBudgetError::ArithmeticOverflow)
        }
        Estimate::OpenAiTiles { base, tile } => {
            let long = width.max(height);
            let (mut upper_w, mut upper_h, lower_short) = if long > 2048 {
                (
                    ceil_div(product(width, 2048)?, long)?,
                    ceil_div(product(height, 2048)?, long)?,
                    product(width.min(height), 2048)? / long,
                )
            } else {
                (width, height, width.min(height))
            };
            if lower_short > 768 {
                // A lower bound on the first rounded short side and upper
                // dimensions prevent integer rounding from undercounting tiles.
                upper_w = ceil_div(product(upper_w, 768)?, lower_short)?;
                upper_h = ceil_div(product(upper_h, 768)?, lower_short)?;
            }
            let tiles = product(ceil_div(upper_w, 512)?, ceil_div(upper_h, 512)?)?;
            product(tiles, tile)?
                .checked_add(base)
                .ok_or(WireBudgetError::ArithmeticOverflow)
        }
        Estimate::Gemini2 => {
            if width <= 384 && height <= 384 {
                return Ok(258);
            }
            // Use no larger than a 384px crop unit, and no larger than the
            // documented floor(short_side / 1.5) estimate. This deliberately
            // overestimates large inputs instead of assuming provider resizing.
            let crop = (product(width.min(height), 2)? / 3).clamp(1, 384);
            product(
                product(ceil_div(width, crop)?, ceil_div(height, crop)?)?,
                258,
            )
        }
        // The mapped GenerateContent profiles document at most 1120 image
        // tokens at default/high resolution. Per-part overrides are rejected.
        Estimate::Gemini3 => Ok(1120),
    }
}

/// Verify actual wire occurrences against the charged selection. The expected
/// value is never trusted by itself, including after deserialization.
pub(super) fn verify_wire(
    provider: &str,
    model: Option<&str>,
    body: &Value,
    expected: &MediaAccounting,
) -> Result<MediaAccounting, WireBudgetError> {
    let provider = canonical_media_provider(provider);
    let manifest = wire_manifest(provider, model, body)?;
    if !manifest.items().is_empty() {
        for field in ["system", "systemInstruction", "prompt", "input"] {
            if body.get(field).is_some_and(super::contains_media) {
                return Err(WireBudgetError::UnsupportedMedia);
            }
        }
        let alternate = if provider == "google" {
            "messages"
        } else {
            "contents"
        };
        if body.get(alternate).is_some_and(super::contains_media) {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        if let Some(resolution) = body
            .get("generationConfig")
            .and_then(|c| c.get("mediaResolution"))
        {
            if provider != "google"
                || !matches!(
                    model.map(|m| policy(provider, m)),
                    Some(Ok(Estimate::Gemini3))
                )
                || !matches!(
                    resolution.as_str(),
                    Some(
                        "MEDIA_RESOLUTION_UNSPECIFIED"
                            | "MEDIA_RESOLUTION_LOW"
                            | "MEDIA_RESOLUTION_MEDIUM"
                            | "MEDIA_RESOLUTION_HIGH"
                    )
                )
            {
                return Err(WireBudgetError::UnsupportedMedia);
            }
        }
    }
    match expected {
        MediaAccounting::None if manifest.items().is_empty() => Ok(MediaAccounting::None),
        MediaAccounting::None => Err(WireBudgetError::UnsupportedMedia),
        MediaAccounting::Images(expected) => {
            let model = model.ok_or(WireBudgetError::MediaMismatch)?;
            if body
                .get("model")
                .is_some_and(|value| value.as_str() != Some(model))
            {
                return Err(WireBudgetError::MediaMismatch);
            }
            let fresh = account_manifest(provider, model, manifest)?;
            if fresh != MediaAccounting::Images(expected.clone()) {
                return Err(WireBudgetError::MediaMismatch);
            }
            Ok(fresh)
        }
    }
}

fn wire_manifest(
    provider: &str,
    model: Option<&str>,
    body: &Value,
) -> Result<MediaManifest, WireBudgetError> {
    let mut builder = MediaManifestBuilder::new(MediaLimits::default());
    let (messages_field, content_field) = if provider == "google" {
        ("contents", "parts")
    } else {
        ("messages", "content")
    };
    if let Some(messages) = body.get(messages_field) {
        let messages = messages.as_array().ok_or(WireBudgetError::MediaMismatch)?;
        for message in messages {
            if let Some(content) = message.get(content_field) {
                visit_content(provider, model, content, &mut builder, 0)?;
            }
        }
    }
    let manifest = builder.finish();
    // Preserve the fail-closed handling of unsupported top-level shapes, but
    // never walk tool schemas or arbitrary tool-result JSON as image payloads.
    if manifest.items().is_empty() && super::contains_media(body) {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    Ok(manifest)
}

fn visit_content(
    provider: &str,
    model: Option<&str>,
    value: &Value,
    builder: &mut MediaManifestBuilder,
    depth: usize,
) -> Result<(), WireBudgetError> {
    if depth > MediaLimits::default().max_nesting {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    if let Some(values) = value.as_array() {
        for value in values {
            visit_content(provider, model, value, builder, depth)?;
        }
        return Ok(());
    }
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    let kind = object.get("type").and_then(Value::as_str);
    if matches!(kind, Some("document" | "input_file")) || object.contains_key("document") {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    if kind == Some("image") {
        if provider != "anthropic" {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        let source = object.get("source").ok_or(WireBudgetError::MediaMismatch)?;
        only_keys(value, &["type", "source", "cache_control"])?;
        only_keys(source, &["type", "media_type", "data"])?;
        if source.get("type").and_then(Value::as_str) != Some("base64")
            || source.get("url").is_some()
            || source.get("file_id").is_some()
        {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        builder.push_inline_image(
            required_str(source, "media_type")?,
            required_str(source, "data")?,
        )?;
    } else if kind == Some("image_url") {
        if provider != "openai" {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        let source = object
            .get("image_url")
            .ok_or(WireBudgetError::MediaMismatch)?;
        only_keys(value, &["type", "image_url"])?;
        only_keys(source, &["url", "detail"])?;
        if source
            .get("detail")
            .is_some_and(|detail| !matches!(detail.as_str(), Some("auto" | "high" | "low")))
        {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        let uri = required_str(source, "url")?;
        let (mime, data) = uri
            .strip_prefix("data:")
            .and_then(|s| s.split_once(";base64,"))
            .ok_or(WireBudgetError::UnsupportedMedia)?;
        builder.push_inline_image(mime, data)?;
    } else if let Some(source) = object.get("inlineData") {
        if provider != "google" {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        only_keys(value, &["inlineData"])?;
        only_keys(source, &["mimeType", "data"])?;
        builder.push_inline_image(
            required_str(source, "mimeType")?,
            required_str(source, "data")?,
        )?;
    } else if let Some(result) = object
        .get("functionResponse")
        .filter(|_| provider == "google")
    {
        if let Some(parts) = result.get("parts") {
            if !matches!(
                model.map(|m| policy(provider, m)),
                Some(Ok(Estimate::Gemini3))
            ) {
                return Err(WireBudgetError::UnsupportedMedia);
            }
            visit_content(provider, model, parts, builder, depth + 1)?;
        }
    } else if let Some(image) = object.get("image") {
        if provider != "amazon-bedrock" {
            return Err(WireBudgetError::UnsupportedMedia);
        }
        let mime = match required_str(image, "format")? {
            "png" => "image/png",
            "jpeg" => "image/jpeg",
            _ => return Err(WireBudgetError::UnsupportedMedia),
        };
        only_keys(value, &["image"])?;
        only_keys(image, &["format", "source"])?;
        let source = image.get("source").ok_or(WireBudgetError::MediaMismatch)?;
        only_keys(source, &["bytes"])?;
        builder.push_inline_image(mime, required_str(source, "bytes")?)?;
    } else if kind == Some("tool_result") && provider == "anthropic" {
        if let Some(content) = object.get("content") {
            visit_content(provider, model, content, builder, depth + 1)?;
        }
    } else if let Some(result) = object
        .get("toolResult")
        .filter(|_| provider == "amazon-bedrock")
    {
        if let Some(content) = result.get("content") {
            visit_content(provider, model, content, builder, depth + 1)?;
        }
    } else if kind == Some("input_image")
        || [
            "image_url",
            "inline_data",
            "fileData",
            "file_data",
            "file_url",
        ]
        .iter()
        .any(|field| object.contains_key(*field))
    {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    Ok(())
}

fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, WireBudgetError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or(WireBudgetError::MediaMismatch)
}

fn canonical_media_provider(provider: &str) -> &str {
    match provider {
        "bedrock" => "amazon-bedrock",
        _ => provider,
    }
}

fn contains_image(blocks: &[ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        ContentBlock::Image { .. } => true,
        ContentBlock::ToolResult {
            content: ToolResultContent::Blocks(blocks),
            ..
        } => contains_image(blocks),
        _ => false,
    })
}

fn only_keys(value: &Value, allowed: &[&str]) -> Result<(), WireBudgetError> {
    let object = value.as_object().ok_or(WireBudgetError::MediaMismatch)?;
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(WireBudgetError::UnsupportedMedia);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
