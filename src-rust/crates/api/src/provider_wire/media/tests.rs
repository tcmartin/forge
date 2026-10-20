use super::*;
use crate::provider_types::ProviderRequest;
use crate::provider_wire::{
    prepare_bytes, prepare_json_checked, prepare_provider_json, WireBudget,
};
use claurst_core::types::{ContentBlock, ImageSource, ToolResultContent};
use serde_json::json;
use sha2::{Digest, Sha256};

// Genuine RGB 1px PNGs from Anthropic's public vision examples.
const RED: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const BLUE: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC";

fn image(data: &str) -> ContentBlock {
    ContentBlock::Image {
        source: ImageSource {
            source_type: "base64".into(),
            media_type: Some("image/png".into()),
            data: Some(data.into()),
            url: None,
        },
    }
}

fn request(model: &str, blocks: Vec<ContentBlock>) -> ProviderRequest {
    ProviderRequest {
        model: model.into(),
        messages: vec![Message::user_blocks(blocks)],
        system_prompt: None,
        tools: vec![],
        max_tokens: 100,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: None,
        provider_options: json!({}),
        cache_scope: None,
        response_continuation: None,
        wire_budget: Some(
            WireBudget::finite(30_000, 20_000, 1000, 8 * 1024 * 1024, MediaAccounting::None)
                .unwrap(),
        ),
        dispatch: None,
    }
}

fn body(provider: &str, model: &str, data: &[&str]) -> Value {
    match provider {
        "anthropic" => json!({"model":model,"messages":[{"role":"user","content":
            data.iter().map(|s| json!({"type":"image","source":{
                "type":"base64","media_type":"image/png","data":s
            }})).collect::<Vec<_>>()
        }]}),
        "openai" => json!({"model":model,"messages":[{"role":"user","content":
            data.iter().map(|s| json!({"type":"image_url","image_url":{
                "url":format!("data:image/png;base64,{s}")
            }})).collect::<Vec<_>>()
        }]}),
        "google" => json!({"contents":[{"role":"user","parts":
            data.iter().map(|s| json!({"inlineData":{"mimeType":"image/png","data":s}}))
                .collect::<Vec<_>>()
        }]}),
        "bedrock" | "amazon-bedrock" => json!({"messages":[{"role":"user","content":
            data.iter().map(|s| json!({"image":{"format":"png","source":{"bytes":s}}}))
                .collect::<Vec<_>>()
        }]}),
        _ => unreachable!(),
    }
}

#[test]
fn selected_images_bind_to_each_implemented_wire_layout_and_exact_bytes() {
    for (provider, model) in [
        ("anthropic", "claude-sonnet-4-6"),
        ("openai", "gpt-4o"),
        ("google", "gemini-2.5-flash"),
        ("google", "gemini-3.5-flash"),
        ("amazon-bedrock", "anthropic.claude-sonnet-4-6"),
        ("bedrock", "global.anthropic.claude-sonnet-4-6"),
    ] {
        let request = request(model, vec![image(RED), image(BLUE), image(RED)]);
        let body = body(provider, model, &[RED, BLUE, RED]);
        let prepared =
            prepare_provider_json(provider, "/inference", true, &body, &request).unwrap();
        let MediaAccounting::Images(images) = &prepared.media else {
            panic!("missing manifest");
        };
        assert_eq!(images.manifest().items().len(), 3);
        assert_eq!(images.manifest().items()[0], images.manifest().items()[2]);
        assert_ne!(
            images.manifest().items()[0].sha256,
            images.manifest().items()[1].sha256
        );
        assert_eq!(prepared.bytes(), serde_json::to_vec(&body).unwrap());
        assert_eq!(
            prepared.wire_sha256,
            hex::encode(Sha256::digest(prepared.bytes()))
        );
        assert_eq!(
            prepared.accounting.input_tokens(),
            prepared.wire_bytes - images.manifest().total_encoded_bytes() as u64
                + 4096
                + images.tokens(),
        );
    }
}

#[test]
fn changed_missing_extra_and_reordered_occurrences_fail_locally() {
    let request = request("gpt-4o", vec![image(RED), image(BLUE)]);
    for data in [
        vec![RED],
        vec![BLUE, RED],
        vec![RED, RED],
        vec![RED, BLUE, RED],
    ] {
        assert_eq!(
            prepare_provider_json(
                "openai",
                "/inference",
                false,
                &body("openai", "gpt-4o", &data),
                &request
            ),
            Err(WireBudgetError::MediaMismatch),
        );
    }
}

#[test]
fn nested_anthropic_results_preserve_order_and_duplicates() {
    let request = request(
        "claude-sonnet-4-6",
        vec![
            image(RED),
            ContentBlock::ToolResult {
                tool_use_id: "read".into(),
                content: ToolResultContent::Blocks(vec![image(BLUE), image(RED)]),
                is_error: None,
            },
        ],
    );
    let body = json!({"model":request.model,"messages":[{"role":"user","content":[
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":RED}},
        {"type":"tool_result","tool_use_id":"read","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":BLUE}},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":RED}}
        ]}
    ]}]});
    assert!(prepare_provider_json("anthropic", "/inference", false, &body, &request).is_ok());
}

#[test]
fn provider_options_cannot_add_media_to_a_text_selection() {
    let request = request(
        "gpt-4o",
        vec![ContentBlock::Text {
            text: "hello".into(),
        }],
    );
    assert_eq!(
        prepare_provider_json(
            "openai",
            "/inference",
            false,
            &body("openai", "gpt-4o", &[RED]),
            &request
        ),
        Err(WireBudgetError::UnsupportedMedia),
    );
}

#[test]
fn unknown_target_and_body_model_override_cannot_borrow_a_policy() {
    let request = request("gpt-4o", vec![image(RED)]);
    assert_eq!(
        prepare_provider_json(
            "openai",
            "/inference",
            false,
            &body("openai", "gpt-4o-mini", &[RED]),
            &request
        ),
        Err(WireBudgetError::MediaMismatch),
    );
    for (provider, model) in [
        ("openai", "gpt-future"),
        ("copilot", "gpt-4o"),
        ("openrouter", "openai/gpt-4o"),
        ("bedrock", "unmapped-profile"),
        ("google", "gemini-3-future"),
    ] {
        assert_eq!(
            MediaAccounting::from_messages(provider, model, &request.messages),
            Err(WireBudgetError::UnsupportedMedia),
        );
    }
}

#[test]
fn persisted_token_revision_and_manifest_claims_are_recomputed() {
    let mut request = request("gpt-4o", vec![image(RED)]);
    let original =
        MediaAccounting::from_messages("openai", &request.model, &request.messages).unwrap();
    for (field, value) in [
        ("tokens", json!(0)),
        ("policy_revision", json!("forged")),
        ("model", json!("gpt-4o-mini")),
        ("provider", json!("anthropic")),
        (
            "manifest",
            json!({"items":[],"total_encoded_bytes":0,"total_decoded_bytes":0}),
        ),
    ] {
        let mut persisted = serde_json::to_value(&original).unwrap();
        persisted["Images"][field] = value;
        request.wire_budget.as_mut().unwrap().media = serde_json::from_value(persisted).unwrap();
        assert_eq!(
            prepare_provider_json(
                "openai",
                "/inference",
                false,
                &body("openai", "gpt-4o", &[RED]),
                &request
            ),
            Err(WireBudgetError::MediaMismatch),
        );
    }
}

#[test]
fn direct_anthropic_transport_requires_the_selected_manifest() {
    let request = request("claude-sonnet-4-6", vec![image(RED)]);
    let body = body("anthropic", &request.model, &[RED]);
    let mut budget = request.wire_budget.unwrap();
    assert_eq!(
        prepare_json_checked("anthropic", "/inference", false, &body, Some(&budget)),
        Err(WireBudgetError::UnsupportedMedia),
    );
    budget.media =
        MediaAccounting::from_messages("anthropic", &request.model, &request.messages).unwrap();
    assert!(prepare_json_checked("anthropic", "/inference", false, &body, Some(&budget)).is_ok());
    assert_eq!(
        prepare_bytes(
            "anthropic",
            "/inference",
            false,
            serde_json::to_vec(&body).unwrap(),
            &budget
        ),
        Err(WireBudgetError::UnsupportedMedia),
    );
    budget.media = MediaAccounting::None;
    assert_eq!(
        prepare_bytes(
            "anthropic",
            "/inference",
            false,
            serde_json::to_vec(&body).unwrap(),
            &budget
        ),
        Err(WireBudgetError::UnsupportedMedia),
    );
}

#[test]
fn encoded_bytes_remain_in_wire_cap_after_token_exclusion() {
    let mut request = request("gpt-4o", vec![image(RED)]);
    let body = body("openai", &request.model, &[RED]);
    request.wire_budget.as_mut().unwrap().max_wire_bytes = 50;
    let error = prepare_provider_json("openai", "/inference", false, &body, &request).unwrap_err();
    assert!(matches!(error, WireBudgetError::WireOverflow { .. }));
    assert!(error.request_budget_fields().unwrap().2 > 50);
}

#[test]
fn malformed_mime_base64_urls_and_documents_are_not_sendable() {
    let request = request("gpt-4o", vec![image(RED)]);
    for part in [
        json!({"type":"image_url","image_url":{"url":"https://example.invalid/image.png"}}),
        json!({"type":"image_url","image_url":{"url":format!("data:image/jpeg;base64,{RED}")}}),
        json!({"type":"image_url","image_url":{"url":"data:image/png;base64,broken"}}),
        json!({"type":"input_file","file_url":"https://example.invalid/file.pdf"}),
    ] {
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":[part]}]});
        assert!(prepare_provider_json("openai", "/inference", false, &body, &request).is_err());
    }
}

#[test]
fn mixed_shapes_transformations_and_unaccounted_locations_are_rejected() {
    for (provider, model) in [
        ("anthropic", "claude-sonnet-4-6"),
        ("openai", "gpt-4o"),
        ("google", "gemini-2.5-flash"),
    ] {
        let request = request(model, vec![image(RED)]);
        let mut body = body(provider, model, &[RED]);
        let field = if provider == "google" {
            "contents"
        } else {
            "messages"
        };
        let content = if provider == "google" {
            "parts"
        } else {
            "content"
        };
        body[field][0][content][0]["inline_data"] = json!({"data":BLUE,"mime_type":"image/png"});
        assert!(prepare_provider_json(provider, "/inference", false, &body, &request).is_err());
    }
    let request = request("gpt-4o", vec![image(RED)]);
    let mut extra = body("openai", &request.model, &[RED]);
    extra["input"] =
        json!([{"type":"input_image","image_url":format!("data:image/png;base64,{BLUE}")}]);
    assert_eq!(
        prepare_provider_json("openai", "/inference", false, &extra, &request),
        Err(WireBudgetError::UnsupportedMedia),
    );
}

#[test]
fn policy_counts_and_arithmetic_are_finite() {
    let request = request("gpt-4o", vec![image(RED); MAX_IMAGES + 1]);
    assert_eq!(
        MediaAccounting::from_messages("openai", &request.model, &request.messages),
        Err(WireBudgetError::UnsupportedMedia),
    );
    assert_eq!(
        estimate_image(
            Estimate::OpenAiTiles {
                base: 85,
                tile: 170
            },
            1024,
            1024
        )
        .unwrap(),
        765
    );
    assert_eq!(
        estimate_image(
            Estimate::OpenAiTiles {
                base: 85,
                tile: 170
            },
            4096,
            8192
        )
        .unwrap(),
        1105
    );
    assert_eq!(
        estimate_image(Estimate::Claude { visual_cap: 1568 }, 1000, 1000).unwrap(),
        1296
    );
    assert_eq!(
        estimate_image(Estimate::Claude { visual_cap: 1568 }, 3840, 2160).unwrap(),
        1568
    );
    assert_eq!(estimate_image(Estimate::Gemini2, 960, 540).unwrap(), 1548);
    assert!(estimate_image(Estimate::Gemini2, u64::MAX, 500).is_err());
}

#[test]
fn gemini_native_function_images_require_an_explicit_gemini3_profile() {
    let mut request = request(
        "gemini-3.5-flash",
        vec![ContentBlock::ToolResult {
            tool_use_id: "read".into(),
            content: ToolResultContent::Blocks(vec![image(RED)]),
            is_error: None,
        }],
    );
    let body = json!({"contents":[{"role":"user","parts":[{
        "functionResponse":{"name":"Read","response":{"content":"image"},
            "parts":[{"inlineData":{"mimeType":"image/png","data":RED}}]}
    }]}]});
    let prepared = prepare_provider_json("google", "/inference", false, &body, &request).unwrap();
    assert!(matches!(prepared.media, MediaAccounting::Images(ref images) if images.tokens()==1120));
    request.model = "gemini-2.5-flash".into();
    assert_eq!(
        prepare_provider_json("google", "/inference", false, &body, &request),
        Err(WireBudgetError::UnsupportedMedia),
    );
}

#[test]
fn bedrock_namespace_aliases_share_only_explicit_model_policies() {
    let request = request("us.anthropic.claude-sonnet-4-6", vec![image(RED)]);
    let canonical =
        MediaAccounting::from_messages("amazon-bedrock", &request.model, &request.messages)
            .unwrap();
    assert_eq!(
        canonical,
        MediaAccounting::from_messages("bedrock", &request.model, &request.messages).unwrap()
    );
    for model in [
        "custom-anthropic.claude-sonnet-4-6",
        "us.us.anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:us-east-1:123:inference-profile/anthropic.claude-sonnet-4-6",
        "anthropic.claude-future",
    ] {
        assert_eq!(
            MediaAccounting::from_messages("amazon-bedrock", model, &request.messages),
            Err(WireBudgetError::UnsupportedMedia)
        );
    }
    let MediaAccounting::Images(images) = canonical else {
        panic!("image accounting");
    };
    let mut manifest = serde_json::to_value(images.manifest()).unwrap();
    for (decoded, expected_ok) in [
        (MAX_BEDROCK_IMAGE_DECODED_BYTES, true),
        (MAX_BEDROCK_IMAGE_DECODED_BYTES + 1, false),
    ] {
        manifest["items"][0]["raw_bytes"] = json!(decoded);
        manifest["total_decoded_bytes"] = json!(decoded);
        let manifest = serde_json::from_value(manifest.clone()).unwrap();
        assert_eq!(
            account_manifest("amazon-bedrock", &request.model, manifest).is_ok(),
            expected_ok
        );
    }
}

#[test]
fn configured_exact_models_have_policies_and_assistant_images_fail_preflight() {
    for (provider, model) in [
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("google", "gemini-2.0-flash"),
    ] {
        let request = request(model, vec![image(RED)]);
        assert!(MediaAccounting::from_messages(provider, model, &request.messages).is_ok());
    }
    for (provider, model) in [
        ("anthropic", "claude-sonnet-4-6"),
        ("openai", "gpt-4o"),
        ("amazon-bedrock", "anthropic.claude-sonnet-4-6"),
        ("google", "gemini-3.5-flash"),
    ] {
        let mut request = request(model, vec![image(RED)]);
        request.messages = vec![Message::assistant_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "invalid-role".into(),
            content: ToolResultContent::Blocks(vec![image(RED)]),
            is_error: None,
        }])];
        assert!(matches!(
            crate::provider_wire::preflight_provider_request(provider, &request),
            Err(WireBudgetError::UnsupportedMedia)
        ));
    }
}
