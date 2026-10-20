//! Bounded validation and metadata for inline media content.
//!
//! This module records only validated metadata.  It never retains decoded
//! payloads, and a deserialized manifest is not evidence that a later body
//! still contains the same media.  Callers must derive a fresh manifest from
//! the content selected for each request.
//! Image checks are bounded container/header validation; they do not decode
//! pixels or guarantee that every provider can render the payload.

use crate::types::{ContentBlock, ImageSource, Message, MessageContent, ToolResultContent};
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

// These are hard safety ceilings.  Callers may select stricter route-specific
// limits, but a caller-supplied MediaLimits value can never widen them.
const HARD_MAX_ITEMS: usize = 100;
const HARD_MAX_TOTAL_ENCODED_BYTES: usize = 8 * 1024 * 1024;
const HARD_MAX_TOTAL_DECODED_BYTES: usize = 6 * 1024 * 1024;
const HARD_MAX_WIDTH: u32 = 8192;
const HARD_MAX_HEIGHT: u32 = 8192;
const HARD_MAX_PIXELS: u64 = 16 * 1024 * 1024;
const HARD_MAX_NESTING: usize = 8;

/// Finite limits for one validated media manifest.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct MediaLimits {
    pub max_items: usize,
    pub max_total_encoded_bytes: usize,
    pub max_total_decoded_bytes: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub max_nesting: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_items: HARD_MAX_ITEMS,
            max_total_encoded_bytes: HARD_MAX_TOTAL_ENCODED_BYTES,
            max_total_decoded_bytes: HARD_MAX_TOTAL_DECODED_BYTES,
            max_width: HARD_MAX_WIDTH,
            max_height: HARD_MAX_HEIGHT,
            max_pixels: HARD_MAX_PIXELS,
            max_nesting: HARD_MAX_NESTING,
        }
    }
}

impl MediaLimits {
    /// Clamp caller-provided limits to the core safety ceilings.  Route
    /// policies may only tighten these bounds, never widen them.
    fn bounded(self) -> Self {
        Self {
            max_items: self.max_items.min(HARD_MAX_ITEMS),
            max_total_encoded_bytes: self
                .max_total_encoded_bytes
                .min(HARD_MAX_TOTAL_ENCODED_BYTES),
            max_total_decoded_bytes: self
                .max_total_decoded_bytes
                .min(HARD_MAX_TOTAL_DECODED_BYTES),
            max_width: self.max_width.min(HARD_MAX_WIDTH),
            max_height: self.max_height.min(HARD_MAX_HEIGHT),
            max_pixels: self.max_pixels.min(HARD_MAX_PIXELS),
            max_nesting: self.max_nesting.min(HARD_MAX_NESTING),
        }
    }
}

/// The media kinds currently admitted by the core validator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
}

/// One ordered occurrence of validated media.  Payload bytes are deliberately
/// absent; this descriptor is safe to persist only as untrusted metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MediaItem {
    pub kind: MediaKind,
    pub media_type: String,
    pub raw_bytes: usize,
    pub encoded_bytes: usize,
    pub sha256: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub page_count: Option<u32>,
}

/// Ordered, bounded metadata for all accepted media occurrences.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MediaManifest {
    items: Vec<MediaItem>,
    total_encoded_bytes: usize,
    total_decoded_bytes: usize,
}

impl MediaManifest {
    pub fn items(&self) -> &[MediaItem] {
        &self.items
    }

    pub fn total_encoded_bytes(&self) -> usize {
        self.total_encoded_bytes
    }

    pub fn total_decoded_bytes(&self) -> usize {
        self.total_decoded_bytes
    }

    /// Derive a fresh manifest from message content.  Persisted manifests
    /// must never be used in place of this revalidation step.
    pub fn from_messages(messages: &[Message], limits: &MediaLimits) -> Result<Self, MediaError> {
        let mut builder = MediaManifestBuilder::new(*limits);
        for message in messages {
            if let MessageContent::Blocks(blocks) = &message.content {
                builder.push_blocks(blocks, 0)?;
            }
        }
        Ok(builder.finish())
    }

    /// Derive a fresh manifest from an ordered block sequence, including
    /// recursively nested tool-result blocks.
    pub fn from_blocks(blocks: &[ContentBlock], limits: &MediaLimits) -> Result<Self, MediaError> {
        let mut builder = MediaManifestBuilder::new(*limits);
        builder.push_blocks(blocks, 0)?;
        Ok(builder.finish())
    }
}

/// Errors returned by bounded media validation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum MediaError {
    #[error("document media is unsupported")]
    UnsupportedDocument,
    #[error("image source type is unsupported")]
    UnsupportedImageSourceType,
    #[error("image URLs are unsupported")]
    UrlUnsupported,
    #[error("image MIME type is missing")]
    MissingMimeType,
    #[error("image data is missing")]
    MissingImageData,
    #[error("unsupported image MIME type")]
    UnsupportedMimeType,
    #[error("invalid base64 image data: {0}")]
    InvalidBase64(String),
    #[error("media item limit exceeded")]
    ItemLimitExceeded,
    #[error("encoded media byte limit exceeded")]
    EncodedLimitExceeded,
    #[error("decoded media byte limit exceeded")]
    DecodedLimitExceeded,
    #[error("media nesting limit exceeded")]
    NestingLimitExceeded,
    #[error("media arithmetic overflow")]
    ArithmeticOverflow,
    #[error("image MIME does not match its encoded bytes")]
    MimeMismatch,
    #[error("malformed image: {0}")]
    MalformedImage(&'static str),
    #[error("truncated image: {0}")]
    TruncatedImage(&'static str),
    #[error("image dimensions exceed the configured limit")]
    DimensionLimitExceeded,
    #[error("image pixel count exceeds the configured limit")]
    PixelLimitExceeded,
}

/// Incrementally builds a bounded manifest without retaining media payloads.
pub struct MediaManifestBuilder {
    limits: MediaLimits,
    items: Vec<MediaItem>,
    total_encoded_bytes: usize,
    total_decoded_bytes: usize,
}

impl MediaManifestBuilder {
    pub fn new(limits: MediaLimits) -> Self {
        Self {
            limits: limits.bounded(),
            items: Vec::new(),
            total_encoded_bytes: 0,
            total_decoded_bytes: 0,
        }
    }

    /// Validate and append one raw base64 image occurrence.
    pub fn push_inline_image(&mut self, media_type: &str, encoded: &str) -> Result<(), MediaError> {
        let media_type = canonical_image_mime(media_type)?;
        if encoded.starts_with("data:") {
            return Err(MediaError::UrlUnsupported);
        }
        if encoded.is_empty() {
            return Err(MediaError::MissingImageData);
        }
        if self.items.len() >= self.limits.max_items {
            return Err(MediaError::ItemLimitExceeded);
        }

        let encoded_bytes = encoded.len();
        let next_encoded = self
            .total_encoded_bytes
            .checked_add(encoded_bytes)
            .ok_or(MediaError::ArithmeticOverflow)?;
        if next_encoded > self.limits.max_total_encoded_bytes {
            return Err(MediaError::EncodedLimitExceeded);
        }

        let decoded_upper =
            estimated_decoded_len(encoded_bytes).ok_or(MediaError::ArithmeticOverflow)?;
        let remaining_decoded = self
            .limits
            .max_total_decoded_bytes
            .checked_sub(self.total_decoded_bytes)
            .ok_or(MediaError::ArithmeticOverflow)?;
        if decoded_upper > remaining_decoded {
            return Err(MediaError::DecodedLimitExceeded);
        }

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded.as_bytes())
            .map_err(|error| MediaError::InvalidBase64(error.to_string()))?;
        let decoded_bytes = decoded.len();
        let next_decoded = self
            .total_decoded_bytes
            .checked_add(decoded_bytes)
            .ok_or(MediaError::ArithmeticOverflow)?;
        if next_decoded > self.limits.max_total_decoded_bytes {
            return Err(MediaError::DecodedLimitExceeded);
        }

        let (width, height) = match media_type {
            "image/png" => parse_png(&decoded)?,
            "image/jpeg" => parse_jpeg(&decoded)?,
            _ => return Err(MediaError::UnsupportedMimeType),
        };
        validate_dimensions(width, height, &self.limits)?;

        let sha256 = hex::encode(Sha256::digest(&decoded));
        self.items.push(MediaItem {
            kind: MediaKind::Image,
            media_type: media_type.to_owned(),
            raw_bytes: decoded_bytes,
            encoded_bytes,
            sha256,
            width: Some(width),
            height: Some(height),
            page_count: None,
        });
        self.total_encoded_bytes = next_encoded;
        self.total_decoded_bytes = next_decoded;
        Ok(())
    }

    pub fn finish(self) -> MediaManifest {
        MediaManifest {
            items: self.items,
            total_encoded_bytes: self.total_encoded_bytes,
            total_decoded_bytes: self.total_decoded_bytes,
        }
    }

    fn push_blocks(&mut self, blocks: &[ContentBlock], depth: usize) -> Result<(), MediaError> {
        for block in blocks {
            match block {
                ContentBlock::Image { source } => self.push_image_source(source)?,
                ContentBlock::Document { .. } => return Err(MediaError::UnsupportedDocument),
                ContentBlock::ToolResult {
                    content: ToolResultContent::Blocks(inner),
                    ..
                } => {
                    let next_depth = depth.checked_add(1).ok_or(MediaError::ArithmeticOverflow)?;
                    if next_depth > self.limits.max_nesting {
                        return Err(MediaError::NestingLimitExceeded);
                    }
                    self.push_blocks(inner, next_depth)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn push_image_source(&mut self, source: &ImageSource) -> Result<(), MediaError> {
        if source.url.is_some() {
            return Err(MediaError::UrlUnsupported);
        }
        if source.source_type != "base64" {
            return Err(MediaError::UnsupportedImageSourceType);
        }
        let media_type = source
            .media_type
            .as_deref()
            .ok_or(MediaError::MissingMimeType)?;
        let data = source.data.as_deref().ok_or(MediaError::MissingImageData)?;
        self.push_inline_image(media_type, data)
    }
}

fn canonical_image_mime(media_type: &str) -> Result<&'static str, MediaError> {
    if media_type.eq_ignore_ascii_case("image/png") {
        Ok("image/png")
    } else if media_type.eq_ignore_ascii_case("image/jpeg") {
        Ok("image/jpeg")
    } else if media_type.trim().is_empty() {
        Err(MediaError::MissingMimeType)
    } else {
        Err(MediaError::UnsupportedMimeType)
    }
}

fn estimated_decoded_len(encoded_len: usize) -> Option<usize> {
    encoded_len
        .checked_add(3)
        .and_then(|rounded| rounded.checked_div(4))
        .and_then(|groups| groups.checked_mul(3))
}

fn validate_dimensions(width: u32, height: u32, limits: &MediaLimits) -> Result<(), MediaError> {
    if width == 0 || height == 0 || width > limits.max_width || height > limits.max_height {
        return Err(MediaError::DimensionLimitExceeded);
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(MediaError::ArithmeticOverflow)?;
    if pixels > limits.max_pixels {
        return Err(MediaError::PixelLimitExceeded);
    }
    Ok(())
}

fn parse_png(data: &[u8]) -> Result<(u32, u32), MediaError> {
    if data.len() < PNG_SIGNATURE.len() {
        return Err(MediaError::TruncatedImage("PNG signature"));
    }
    if &data[..PNG_SIGNATURE.len()] != PNG_SIGNATURE {
        return Err(MediaError::MimeMismatch);
    }

    let mut offset = PNG_SIGNATURE.len();
    let mut saw_ihdr = false;
    let mut saw_idat = false;
    let mut dimensions = None;
    let mut saw_iend = false;
    while offset < data.len() {
        let header_end = offset
            .checked_add(8)
            .ok_or(MediaError::ArithmeticOverflow)?;
        if header_end > data.len() {
            return Err(MediaError::TruncatedImage("PNG chunk header"));
        }
        let chunk_len = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let chunk_len = usize::try_from(chunk_len).map_err(|_| MediaError::ArithmeticOverflow)?;
        let type_start = offset
            .checked_add(4)
            .ok_or(MediaError::ArithmeticOverflow)?;
        let payload_start = offset
            .checked_add(8)
            .ok_or(MediaError::ArithmeticOverflow)?;
        let crc_offset = payload_start
            .checked_add(chunk_len)
            .ok_or(MediaError::ArithmeticOverflow)?;
        let chunk_end = crc_offset
            .checked_add(4)
            .ok_or(MediaError::ArithmeticOverflow)?;
        if chunk_end > data.len() {
            return Err(MediaError::TruncatedImage("PNG chunk payload"));
        }
        let kind = &data[type_start..payload_start];
        let chunk_data = &data[payload_start..crc_offset];
        let expected_crc = u32::from_be_bytes([
            data[crc_offset],
            data[crc_offset + 1],
            data[crc_offset + 2],
            data[crc_offset + 3],
        ]);
        if png_crc(kind, chunk_data) != expected_crc {
            return Err(MediaError::MalformedImage("PNG CRC"));
        }

        if !saw_ihdr {
            if kind != b"IHDR" || chunk_len != 13 {
                return Err(MediaError::MalformedImage("PNG IHDR"));
            }
            let bit_depth = chunk_data[8];
            let color_type = chunk_data[9];
            let compression_method = chunk_data[10];
            let filter_method = chunk_data[11];
            let interlace_method = chunk_data[12];
            let legal_bit_depth = match color_type {
                0 => matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
                2 => matches!(bit_depth, 8 | 16),
                3 => matches!(bit_depth, 1 | 2 | 4 | 8),
                4 | 6 => matches!(bit_depth, 8 | 16),
                _ => false,
            };
            if !legal_bit_depth
                || compression_method != 0
                || filter_method != 0
                || !matches!(interlace_method, 0 | 1)
            {
                return Err(MediaError::MalformedImage("PNG IHDR fields"));
            }
            let width =
                u32::from_be_bytes([chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3]]);
            let height =
                u32::from_be_bytes([chunk_data[4], chunk_data[5], chunk_data[6], chunk_data[7]]);
            saw_ihdr = true;
            dimensions = Some((width, height));
        } else if kind == b"IHDR" {
            return Err(MediaError::MalformedImage("duplicate PNG IHDR"));
        }

        if kind == b"IDAT" {
            saw_idat = true;
        }
        if kind == b"IEND" {
            if chunk_len != 0 {
                return Err(MediaError::MalformedImage("PNG IEND"));
            }
            saw_iend = true;
            offset = chunk_end;
            break;
        }
        offset = chunk_end;
    }

    if !saw_iend || offset != data.len() {
        return Err(MediaError::TruncatedImage("PNG IEND"));
    }
    if !saw_ihdr || !saw_idat {
        return Err(MediaError::MalformedImage("PNG image data"));
    }
    dimensions.ok_or(MediaError::MalformedImage("PNG dimensions"))
}

fn png_crc(kind: &[u8], data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in kind.iter().chain(data) {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn parse_jpeg(data: &[u8]) -> Result<(u32, u32), MediaError> {
    if data.len() < 2 {
        return Err(MediaError::TruncatedImage("JPEG signature"));
    }
    if data[..2] != [0xff, 0xd8] {
        return Err(MediaError::MimeMismatch);
    }

    let mut pos = 2;
    let mut saw_sos = false;
    let mut saw_entropy = false;
    let mut dimensions = None;
    let mut frame_component_ids = [0u8; 4];
    let mut frame_component_count = 0usize;
    while pos < data.len() {
        if data[pos] != 0xff {
            return Err(MediaError::MalformedImage("JPEG marker"));
        }
        while pos < data.len() && data[pos] == 0xff {
            pos += 1;
        }
        if pos >= data.len() {
            return Err(MediaError::TruncatedImage("JPEG marker"));
        }
        let marker = data[pos];
        pos += 1;
        if marker == 0xd9 {
            if !saw_sos || !saw_entropy || dimensions.is_none() {
                return Err(MediaError::MalformedImage("JPEG scan"));
            }
            if pos != data.len() {
                return Err(MediaError::MalformedImage("JPEG trailing bytes"));
            }
            return dimensions.ok_or(MediaError::MalformedImage("JPEG dimensions"));
        }
        if marker == 0x00 || marker == 0xd8 {
            return Err(MediaError::MalformedImage("JPEG marker"));
        }
        if (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
            continue;
        }

        let segment_length_end = pos.checked_add(2).ok_or(MediaError::ArithmeticOverflow)?;
        if segment_length_end > data.len() {
            return Err(MediaError::TruncatedImage("JPEG segment length"));
        }
        let segment_len = usize::from(u16::from_be_bytes([data[pos], data[pos + 1]]));
        if segment_len < 2 {
            return Err(MediaError::MalformedImage("JPEG segment length"));
        }
        let payload_start = segment_length_end;
        let payload_end = payload_start
            .checked_add(segment_len - 2)
            .ok_or(MediaError::ArithmeticOverflow)?;
        if payload_end > data.len() {
            return Err(MediaError::TruncatedImage("JPEG segment payload"));
        }
        let payload = &data[payload_start..payload_end];

        if is_jpeg_sof(marker) {
            if segment_len < 8 || payload.len() < 6 {
                return Err(MediaError::MalformedImage("JPEG frame header"));
            }
            let components = usize::from(payload[5]);
            if components == 0 || components > frame_component_ids.len() || payload[0] == 0 {
                return Err(MediaError::MalformedImage("JPEG frame components"));
            }
            let expected_len = 8usize
                .checked_add(
                    components
                        .checked_mul(3)
                        .ok_or(MediaError::ArithmeticOverflow)?,
                )
                .ok_or(MediaError::ArithmeticOverflow)?;
            if segment_len != expected_len {
                return Err(MediaError::MalformedImage("JPEG frame components"));
            }
            let height = u32::from(u16::from_be_bytes([payload[1], payload[2]]));
            let width = u32::from(u16::from_be_bytes([payload[3], payload[4]]));
            frame_component_count = components;
            for index in 0..components {
                let component_offset = 6 + index * 3;
                let component_id = payload[component_offset];
                let sampling = payload[component_offset + 1];
                let quantization_table = payload[component_offset + 2];
                let horizontal_sampling = sampling >> 4;
                let vertical_sampling = sampling & 0x0f;
                if frame_component_ids[..index].contains(&component_id)
                    || horizontal_sampling == 0
                    || horizontal_sampling > 4
                    || vertical_sampling == 0
                    || vertical_sampling > 4
                    || quantization_table > 3
                {
                    return Err(MediaError::MalformedImage("JPEG frame components"));
                }
                frame_component_ids[index] = component_id;
            }
            if dimensions.replace((width, height)).is_some() {
                return Err(MediaError::MalformedImage("duplicate JPEG frame"));
            }
        }

        if marker == 0xda {
            if payload.len() < 4 {
                return Err(MediaError::MalformedImage("JPEG scan header"));
            }
            let components = usize::from(payload[0]);
            if components == 0 || components > frame_component_count {
                return Err(MediaError::MalformedImage("JPEG scan components"));
            }
            let expected_len = 6usize
                .checked_add(
                    components
                        .checked_mul(2)
                        .ok_or(MediaError::ArithmeticOverflow)?,
                )
                .ok_or(MediaError::ArithmeticOverflow)?;
            if components == 0 || segment_len != expected_len {
                return Err(MediaError::MalformedImage("JPEG scan components"));
            }
            for index in 0..components {
                let component_id = payload[1 + index * 2];
                let tables = payload[2 + index * 2];
                if !frame_component_ids[..frame_component_count].contains(&component_id)
                    || tables >> 4 > 3
                    || tables & 0x0f > 3
                {
                    return Err(MediaError::MalformedImage("JPEG scan components"));
                }
            }
            saw_sos = true;
            pos = payload_end;
            loop {
                if pos >= data.len() {
                    return Err(MediaError::TruncatedImage("JPEG EOI"));
                }
                if data[pos] != 0xff {
                    saw_entropy = true;
                    pos += 1;
                    continue;
                }
                let entropy_marker_start = pos;
                while pos < data.len() && data[pos] == 0xff {
                    pos += 1;
                }
                if pos >= data.len() {
                    return Err(MediaError::TruncatedImage("JPEG entropy marker"));
                }
                let entropy_marker = data[pos];
                pos += 1;
                if entropy_marker == 0x00 || (0xd0..=0xd7).contains(&entropy_marker) {
                    saw_entropy = true;
                    continue;
                }
                if entropy_marker == 0xd9 {
                    if !saw_entropy || dimensions.is_none() || pos != data.len() {
                        return Err(MediaError::MalformedImage("JPEG scan"));
                    }
                    return dimensions.ok_or(MediaError::MalformedImage("JPEG dimensions"));
                }
                pos = entropy_marker_start;
                break;
            }
        } else {
            pos = payload_end;
        }
    }
    Err(MediaError::TruncatedImage("JPEG EOI"))
}

fn is_jpeg_sof(marker: u8) -> bool {
    matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, ImageSource, Message, ToolResultContent};

    const TINY_JPEG_BASE64: &str = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/AYf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/AYf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ap//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IYf/2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z";

    fn tiny_png() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H',
            b'D', b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, b'I', b'D', b'A', b'T', 0x78,
            0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82,
        ]
    }

    fn encoded(data: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(data)
    }

    fn png_with_ihdr_fields(
        bit_depth: u8,
        color_type: u8,
        compression_method: u8,
        filter_method: u8,
        interlace_method: u8,
    ) -> Vec<u8> {
        let mut png = tiny_png();
        png[24] = bit_depth;
        png[25] = color_type;
        png[26] = compression_method;
        png[27] = filter_method;
        png[28] = interlace_method;
        let crc = png_crc(b"IHDR", &png[16..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
        png
    }

    fn png_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let mut png = tiny_png();
        png[16..20].copy_from_slice(&width.to_be_bytes());
        png[20..24].copy_from_slice(&height.to_be_bytes());
        let crc = png_crc(b"IHDR", &png[16..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
        png
    }

    fn image(media_type: &str, data: &str) -> ContentBlock {
        ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".into(),
                media_type: Some(media_type.into()),
                data: Some(data.into()),
                url: None,
            },
        }
    }

    #[test]
    fn validates_tiny_png_and_jpeg_with_digest_and_dimensions() {
        let png = encoded(&tiny_png());
        let png_manifest =
            MediaManifest::from_blocks(&[image("image/png", &png)], &MediaLimits::default())
                .expect("tiny PNG should validate");
        assert_eq!(png_manifest.items().len(), 1);
        assert_eq!(png_manifest.items()[0].width, Some(1));
        assert_eq!(png_manifest.items()[0].height, Some(1));
        assert_eq!(png_manifest.items()[0].raw_bytes, tiny_png().len());
        assert_eq!(png_manifest.items()[0].sha256.len(), 64);

        let jpeg_manifest = MediaManifest::from_blocks(
            &[image("image/jpeg", TINY_JPEG_BASE64)],
            &MediaLimits::default(),
        )
        .expect("tiny JPEG should validate");
        assert_eq!(jpeg_manifest.items()[0].width, Some(1));
        assert_eq!(jpeg_manifest.items()[0].height, Some(1));
    }

    #[test]
    fn rejects_bad_base64_truncation_and_mime_mismatch() {
        let png = encoded(&tiny_png());
        assert!(matches!(
            MediaManifest::from_blocks(
                &[image("image/png", "not-base64")],
                &MediaLimits::default()
            ),
            Err(MediaError::InvalidBase64(_))
        ));
        assert!(MediaManifest::from_blocks(
            &[image(
                "image/png",
                &encoded(&tiny_png()[..tiny_png().len() - 4])
            )],
            &MediaLimits::default(),
        )
        .is_err());
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/jpeg", &png)], &MediaLimits::default()),
            Err(MediaError::MimeMismatch)
        ));
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/gif", &png)], &MediaLimits::default()),
            Err(MediaError::UnsupportedMimeType)
        ));
    }

    #[test]
    fn rejects_illegal_png_ihdr_fields() {
        for fields in [
            (3, 0, 0, 0, 0),  // grayscale does not allow three-bit samples
            (16, 3, 0, 0, 0), // indexed color does not allow sixteen-bit samples
            (8, 1, 0, 0, 0),  // reserved color type
            (8, 6, 1, 0, 0),  // unsupported compression method
            (8, 6, 0, 1, 0),  // unsupported filter method
            (8, 6, 0, 0, 2),  // unsupported interlace method
        ] {
            let png = encoded(&png_with_ihdr_fields(
                fields.0, fields.1, fields.2, fields.3, fields.4,
            ));
            assert!(matches!(
                MediaManifest::from_blocks(&[image("image/png", &png)], &MediaLimits::default(),),
                Err(MediaError::MalformedImage("PNG IHDR fields"))
            ));
        }
    }

    #[test]
    fn rejects_zero_or_oversized_dimensions_before_admission() {
        let mut png = tiny_png();
        png[19..23].copy_from_slice(&0u32.to_be_bytes());
        assert!(matches!(
            MediaManifest::from_blocks(
                &[image("image/png", &encoded(&png))],
                &MediaLimits::default(),
            ),
            Err(MediaError::MalformedImage("PNG CRC"))
        ));

        let mut limits = MediaLimits {
            max_width: 0,
            ..MediaLimits::default()
        };
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/png", &encoded(&tiny_png()))], &limits,),
            Err(MediaError::DimensionLimitExceeded)
        ));
        limits.max_width = 1;
        limits.max_pixels = 0;
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/png", &encoded(&tiny_png()))], &limits,),
            Err(MediaError::PixelLimitExceeded)
        ));
    }

    #[test]
    fn bounds_count_bytes_depth_and_checked_arithmetic() {
        let png = encoded(&tiny_png());
        let mut limits = MediaLimits {
            max_items: 1,
            ..MediaLimits::default()
        };
        assert!(matches!(
            MediaManifest::from_blocks(
                &[image("image/png", &png), image("image/png", &png)],
                &limits,
            ),
            Err(MediaError::ItemLimitExceeded)
        ));
        limits.max_items = 100;
        limits.max_total_encoded_bytes = png.len() - 1;
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/png", &png)], &limits),
            Err(MediaError::EncodedLimitExceeded)
        ));
        limits.max_total_encoded_bytes = usize::MAX;
        limits.max_total_decoded_bytes = tiny_png().len() - 1;
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/png", &png)], &limits),
            Err(MediaError::DecodedLimitExceeded)
        ));
        assert!(estimated_decoded_len(usize::MAX).is_none());

        limits.max_total_decoded_bytes = usize::MAX;
        limits.max_nesting = 0;
        let nested = ContentBlock::ToolResult {
            tool_use_id: "tool".into(),
            content: ToolResultContent::Blocks(vec![image("image/png", &png)]),
            is_error: None,
        };
        assert!(matches!(
            MediaManifest::from_blocks(&[nested], &limits),
            Err(MediaError::NestingLimitExceeded)
        ));
    }

    #[test]
    fn all_max_caller_limits_are_clamped_to_core_ceilings() {
        let limits = MediaLimits {
            max_items: usize::MAX,
            max_total_encoded_bytes: usize::MAX,
            max_total_decoded_bytes: usize::MAX,
            max_width: u32::MAX,
            max_height: u32::MAX,
            max_pixels: u64::MAX,
            max_nesting: usize::MAX,
        };
        let builder = MediaManifestBuilder::new(limits);
        assert_eq!(builder.limits, MediaLimits::default());

        let png = encoded(&tiny_png());
        let blocks = (0..=HARD_MAX_ITEMS)
            .map(|_| image("image/png", &png))
            .collect::<Vec<_>>();
        assert!(matches!(
            MediaManifest::from_blocks(&blocks, &limits),
            Err(MediaError::ItemLimitExceeded)
        ));

        let oversized = encoded(&png_with_dimensions(HARD_MAX_WIDTH + 1, 1));
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/png", &oversized)], &limits),
            Err(MediaError::DimensionLimitExceeded)
        ));

        let too_many_pixels = encoded(&png_with_dimensions(4096, 4097));
        assert!(matches!(
            MediaManifest::from_blocks(&[image("image/png", &too_many_pixels)], &limits),
            Err(MediaError::PixelLimitExceeded)
        ));

        let mut nested = image("image/png", &png);
        for index in 0..=HARD_MAX_NESTING {
            nested = ContentBlock::ToolResult {
                tool_use_id: format!("tool-{index}"),
                content: ToolResultContent::Blocks(vec![nested]),
                is_error: None,
            };
        }
        assert!(matches!(
            MediaManifest::from_blocks(&[nested], &limits),
            Err(MediaError::NestingLimitExceeded)
        ));
    }

    #[test]
    fn rejects_impossible_jpeg_component_headers() {
        let mut jpeg = base64::engine::general_purpose::STANDARD
            .decode(TINY_JPEG_BASE64)
            .expect("fixture JPEG should decode");
        let sof = jpeg
            .windows(2)
            .position(|window| window == [0xff, 0xc0])
            .expect("fixture should contain baseline SOF");
        jpeg[sof + 11] = 0; // first component has zero horizontal sampling
        assert!(matches!(
            MediaManifest::from_blocks(
                &[image("image/jpeg", &encoded(&jpeg))],
                &MediaLimits::default(),
            ),
            Err(MediaError::MalformedImage("JPEG frame components"))
        ));
    }

    #[test]
    fn preserves_nested_order_and_duplicate_occurrences() {
        let png = encoded(&tiny_png());
        let blocks = vec![
            image("image/png", &png),
            ContentBlock::ToolResult {
                tool_use_id: "tool".into(),
                content: ToolResultContent::Blocks(vec![image("image/jpeg", TINY_JPEG_BASE64)]),
                is_error: None,
            },
            image("image/png", &png),
        ];
        let manifest =
            MediaManifest::from_messages(&[Message::user_blocks(blocks)], &MediaLimits::default())
                .expect("nested images should validate");
        assert_eq!(manifest.items().len(), 3);
        assert_eq!(manifest.items()[0].media_type, "image/png");
        assert_eq!(manifest.items()[1].media_type, "image/jpeg");
        assert_eq!(manifest.items()[2].media_type, "image/png");
        assert_eq!(manifest.total_decoded_bytes(), tiny_png().len() * 2 + 516);
    }

    #[test]
    fn rejects_urls_and_documents() {
        let source = ImageSource {
            source_type: "url".into(),
            media_type: Some("image/png".into()),
            data: None,
            url: Some("https://example.invalid/image.png".into()),
        };
        assert!(matches!(
            MediaManifest::from_blocks(&[ContentBlock::Image { source }], &MediaLimits::default()),
            Err(MediaError::UrlUnsupported)
        ));
        let document = ContentBlock::Document {
            source: crate::types::DocumentSource {
                source_type: "base64".into(),
                media_type: Some("application/pdf".into()),
                data: Some("AA==".into()),
                url: None,
            },
            title: None,
            context: None,
            citations: None,
        };
        assert!(matches!(
            MediaManifest::from_blocks(&[document], &MediaLimits::default()),
            Err(MediaError::UnsupportedDocument)
        ));
    }
}
