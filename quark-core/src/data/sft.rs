//! Chat / tool-use fine-tuning data (supervised fine-tuning, "SFT").
//!
//! Input is JSONL with one conversation per line:
//!
//! ```json
//! {"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}
//! ```
//!
//! Roles are `system`, `user`, `assistant` and `tool` (a tool result). Each
//! conversation is rendered with the shared chat template (`crate::chat`) and
//! the model is trained only on the assistant turns.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::chat::{render_training_segments, ChatMessage};
use crate::data::batch::DataBatch;
use crate::tokenizer::bpe::{QuarkTokenizer, BOS_ID};

#[derive(serde::Deserialize)]
struct Conversation {
    messages: Vec<ChatMessage>,
}

/// A tokenized conversation. `trainable[i]` says whether token `i` is a
/// training target (i.e. part of an assistant turn).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftExample {
    pub ids: Vec<u32>,
    pub trainable: Vec<bool>,
}

/// Read conversations from JSONL files. Returns the conversations and the
/// number of lines that could not be parsed.
pub fn load_conversations(files: &[PathBuf]) -> Result<(Vec<Vec<ChatMessage>>, usize)> {
    let mut conversations = Vec::new();
    let mut skipped = 0;
    for path in files {
        let file = std::fs::File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Conversation>(&line) {
                Ok(c) if !c.messages.is_empty() => conversations.push(c.messages),
                _ => skipped += 1,
            }
        }
    }
    Ok((conversations, skipped))
}

/// Tokenize one conversation (with a leading BOS), truncated to `max_len`
/// tokens. Returns `None` if no assistant tokens survive truncation.
pub fn tokenize_conversation(
    tokenizer: &QuarkTokenizer,
    messages: &[ChatMessage],
    max_len: usize,
) -> Result<Option<SftExample>> {
    // Encode the whole transcript at once (as inference does), then map each
    // token back to its segment through its byte offsets.
    let segments = render_training_segments(messages);
    let mut text = String::new();
    let mut trainable_ranges = Vec::new();
    for (segment, trainable) in &segments {
        if *trainable {
            trainable_ranges.push(text.len()..text.len() + segment.len());
        }
        text.push_str(segment);
    }

    let (ids, offsets) = tokenizer.encode_with_offsets(&text)?;
    let mut example = SftExample { ids: vec![BOS_ID], trainable: vec![false] };
    for (id, (start, end)) in ids.into_iter().zip(offsets) {
        let trainable = trainable_ranges.iter().any(|r| start < r.end && end > r.start);
        example.ids.push(id);
        example.trainable.push(trainable);
    }

    example.ids.truncate(max_len);
    example.trainable.truncate(max_len);
    // Position 0 is never a target (nothing precedes it).
    if !example.trainable.iter().skip(1).any(|&t| t) {
        return Ok(None);
    }
    Ok(Some(example))
}

/// Collate examples into a batch, padding to the longest. Labels are the next
/// token where it is trainable and `pad_id` (ignored by the loss) elsewhere.
pub fn collate_sft(examples: &[SftExample], pad_id: u32) -> DataBatch {
    let max_len = examples.iter().map(|e| e.ids.len()).max().unwrap_or(0);
    let mut batch = DataBatch { input_ids: vec![], labels: vec![], attention_mask: vec![] };
    for ex in examples {
        let len = ex.ids.len();
        let mut input = ex.ids.clone();
        input.resize(max_len, pad_id);

        let mut labels: Vec<u32> = (1..len)
            .map(|i| if ex.trainable[i] { ex.ids[i] } else { pad_id })
            .collect();
        labels.resize(max_len, pad_id);

        let mut mask = vec![0u8; max_len];
        mask[..len].fill(1);

        batch.input_ids.push(input);
        batch.labels.push(labels);
        batch.attention_mask.push(mask);
    }
    batch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_cover_only_assistant_tokens() {
        let ex = SftExample {
            ids: vec![1, 10, 11, 20, 21, 12],
            trainable: vec![false, false, false, true, true, false],
        };
        let batch = collate_sft(&[ex], 0);
        // position i predicts token i+1; only tokens 20 and 21 are targets
        assert_eq!(batch.labels[0], vec![0, 0, 20, 21, 0, 0]);
        assert_eq!(batch.input_ids[0], vec![1, 10, 11, 20, 21, 12]);
    }

    #[test]
    fn collate_pads_to_longest() {
        let a = SftExample { ids: vec![1, 5, 6], trainable: vec![false, true, true] };
        let b = SftExample { ids: vec![1, 7], trainable: vec![false, true] };
        let batch = collate_sft(&[a, b], 0);
        assert_eq!(batch.input_ids[1], vec![1, 7, 0]);
        assert_eq!(batch.labels[1], vec![7, 0, 0]);
        assert_eq!(batch.attention_mask[1], vec![1, 1, 0]);
    }

    #[test]
    fn sample_dataset_is_valid() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/chat-sft-sample.jsonl");
        let (conversations, skipped) = load_conversations(&[path]).unwrap();
        assert_eq!(skipped, 0);
        assert!(conversations.len() >= 30);
        for conv in &conversations {
            assert!(conv.iter().any(|m| m.role == crate::chat::ChatRole::Assistant));
            let assistant = conv.iter().filter(|m| m.role == crate::chat::ChatRole::Assistant);
            for msg in assistant.filter(|m| m.content.contains("<tool_call>")) {
                let calls = crate::mcp::parse_tool_calls(&msg.content);
                assert_eq!(calls.len(), msg.content.matches("<tool_call>").count(), "{}", msg.content);
            }
        }
    }
}
