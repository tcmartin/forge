//! Validate durable fork boundaries without repairing or dropping tool history.

use crate::types::{ContentBlock, Message, MessageContent, Role};
use std::collections::HashSet;

/// Borrow an exact prefix only when all included tool calls have their results.
pub fn validated_fork_prefix(messages: &[Message], end: usize) -> anyhow::Result<&[Message]> {
    let prefix = messages
        .get(..end)
        .ok_or_else(|| anyhow::anyhow!("Fork point is outside the conversation."))?;
    let mut pending = HashSet::new();
    let mut seen_tool_ids = HashSet::new();
    for message in prefix {
        if !pending.is_empty() {
            if message.role != Role::User {
                anyhow::bail!("Fork point interrupts an unfinished tool call.");
            }
            let MessageContent::Blocks(blocks) = &message.content else {
                anyhow::bail!("Fork point interrupts an unfinished tool call.");
            };
            for block in blocks {
                match block {
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        if !pending.remove(tool_use_id.as_str()) {
                            anyhow::bail!("Conversation contains an unmatched tool result.");
                        }
                    }
                    ContentBlock::ToolUse { .. } => {
                        anyhow::bail!("Conversation contains a tool block with an invalid role.");
                    }
                    _ => {}
                }
            }
            if !pending.is_empty() {
                anyhow::bail!("Fork point interrupts an unfinished tool call.");
            }
            continue;
        }

        let MessageContent::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            match block {
                ContentBlock::ToolUse { id, .. } if message.role == Role::Assistant => {
                    if id.is_empty()
                        || !seen_tool_ids.insert(id.as_str())
                        || !pending.insert(id.as_str())
                    {
                        anyhow::bail!("Conversation contains an invalid tool call identity.");
                    }
                }
                ContentBlock::ToolResult { .. } if message.role == Role::User => {
                    anyhow::bail!("Conversation contains an unmatched tool result.");
                }
                ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {
                    anyhow::bail!("Conversation contains a tool block with an invalid role.");
                }
                _ => {}
            }
        }
    }
    if !pending.is_empty() {
        anyhow::bail!("Fork point interrupts an unfinished tool call.");
    }
    Ok(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn transcript() -> Vec<Message> {
        serde_json::from_value(json!([
            {"role":"user","content":"write once"},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"write","name":"Write","input":{"content":"once"}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"write","content":"committed"}
            ]},
            {"role":"assistant","content":"finished"}
        ]))
        .unwrap()
    }

    #[test]
    fn exact_completed_prefixes_borrow_the_original_history() {
        let messages = transcript();
        for end in [0, 1, 3, 4] {
            let prefix = validated_fork_prefix(&messages, end).unwrap();
            assert_eq!(prefix.len(), end);
            assert_eq!(prefix.as_ptr(), messages.as_ptr());
        }
        assert!(validated_fork_prefix(&messages, 2).is_err());
        assert!(validated_fork_prefix(&messages, 5).is_err());
        assert_eq!(messages.len(), 4);
    }

    #[test]
    fn pending_calls_require_results_in_the_immediate_next_user_message() {
        let mut messages = transcript();
        messages.insert(2, Message::user("intervening text"));
        assert!(validated_fork_prefix(&messages, 3).is_err());
        assert!(validated_fork_prefix(&messages, messages.len()).is_err());

        let mut empty_blocks = transcript();
        empty_blocks[2].content = MessageContent::Blocks(Vec::new());
        assert!(validated_fork_prefix(&empty_blocks, 3).is_err());

        let mut non_result_blocks = transcript();
        non_result_blocks[2].content = MessageContent::Blocks(vec![ContentBlock::Text {
            text: "still waiting".into(),
        }]);
        assert!(validated_fork_prefix(&non_result_blocks, 3).is_err());

        let mut text_content = transcript();
        text_content[2].content = MessageContent::Text("still waiting".into());
        assert!(validated_fork_prefix(&text_content, 3).is_err());
    }

    #[test]
    fn parallel_results_must_share_the_immediate_next_user_message() {
        let split: Vec<Message> = serde_json::from_value(json!([
            {"role":"assistant","content":[
                {"type":"tool_use","id":"one","name":"One","input":{}},
                {"type":"tool_use","id":"two","name":"Two","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"one","content":"done"}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"two","content":"done"}
            ]}
        ]))
        .unwrap();
        assert!(validated_fork_prefix(&split, 2).is_err());
        assert!(validated_fork_prefix(&split, 3).is_err());
    }

    #[test]
    fn parallel_results_with_text_and_a_later_unique_call_are_valid() {
        let messages: Vec<Message> = serde_json::from_value(json!([
            {"role":"user","content":"start"},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"one","name":"One","input":{}},
                {"type":"tool_use","id":"two","name":"Two","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"text","text":"results:"},
                {"type":"tool_result","tool_use_id":"one","content":"done"},
                {"type":"tool_result","tool_use_id":"two","content":"done"}
            ]},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"three","name":"Three","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"three","content":"done"}
            ]},
            {"role":"assistant","content":"finished"}
        ]))
        .unwrap();
        let prefix = validated_fork_prefix(&messages, messages.len()).unwrap();
        assert_eq!(prefix.as_ptr(), messages.as_ptr());
    }

    #[test]
    fn completed_tool_call_ids_cannot_be_reused() {
        let reused: Vec<Message> = serde_json::from_value(json!([
            {"role":"assistant","content":[
                {"type":"tool_use","id":"same","name":"One","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"same","content":"done"}
            ]},
            {"role":"assistant","content":[
                {"type":"tool_use","id":"same","name":"Two","input":{}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"same","content":"done again"}
            ]}
        ]))
        .unwrap();
        assert!(validated_fork_prefix(&reused, reused.len()).is_err());
    }

    #[test]
    fn unmatched_results_and_tool_blocks_in_wrong_roles_are_rejected() {
        let messages = transcript();
        assert!(validated_fork_prefix(&messages[2..], 2).is_err());
        let mut wrong_role = transcript();
        wrong_role[1].role = Role::User;
        assert!(validated_fork_prefix(&wrong_role, 3).is_err());
        let mut unmatched = transcript();
        if let MessageContent::Blocks(blocks) = &mut unmatched[2].content {
            if let ContentBlock::ToolResult { tool_use_id, .. } = &mut blocks[0] {
                *tool_use_id = "missing".into();
            }
        }
        assert!(validated_fork_prefix(&unmatched, 3).is_err());
    }

    #[test]
    fn duplicate_pending_id_is_rejected() {
        let mut messages = transcript();
        if let MessageContent::Blocks(blocks) = &mut messages[1].content {
            blocks.push(blocks[0].clone());
        }
        assert!(validated_fork_prefix(&messages, 3).is_err());
    }
}
