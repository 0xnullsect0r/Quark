//! The chat template shared by quark-chat, quark-code, the GUI chat panel and
//! chat fine-tuning, so models see the same format in training and use.
//!
//! ```text
//! <system>
//! You are…
//! </system>
//!
//! <user>
//! Hi
//! </user>
//!
//! <assistant>
//! Let me look. <tool_call>{"tool":"read_file","path":"a.rs"}</tool_call>
//! </assistant>
//!
//! <tool_result>
//! read_file (ok):
//! fn main() {}
//! </tool_result>
//!
//! <assistant>
//! …
//! ```

use serde::{Deserialize, Serialize};

use crate::mcp::ToolResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    /// A tool result, as fed back to the model after a `<tool_call>`.
    pub fn tool_result(result: &ToolResult) -> Self {
        let status = if result.ok { "ok" } else { "error" };
        Self::new(ChatRole::Tool, format!("{} ({status}):\n{}", result.tool, result.content))
    }
}

/// Where the model's turn ends. Generation stops at any of these, which also
/// keeps the model from writing the user's next turn or its own tool results.
pub const STOP_STRINGS: [&str; 3] = ["</assistant>", "<user>", "<tool_result>"];

fn tag(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool_result",
    }
}

/// Render `messages` and open the assistant's next turn.
pub fn render_prompt(messages: &[ChatMessage]) -> String {
    let mut out: String = render_training_segments(messages).into_iter().map(|(s, _)| s).collect();
    out.push_str("<assistant>\n");
    out
}

/// Render `messages` as text segments flagged with whether the model should
/// learn to produce them: assistant content plus its closing tag (so the model
/// learns to stop). Concatenating the segments gives the rendered transcript.
pub fn render_training_segments(messages: &[ChatMessage]) -> Vec<(String, bool)> {
    let mut segments = Vec::with_capacity(messages.len() * 3);
    for msg in messages {
        let tag = tag(msg.role);
        let content = msg.content.trim_end_matches('\n');
        if msg.role == ChatRole::Assistant {
            segments.push((format!("<{tag}>\n"), false));
            segments.push((format!("{content}\n</{tag}>"), true));
            segments.push(("\n\n".to_owned(), false));
        } else {
            segments.push((format!("<{tag}>\n{content}\n</{tag}>\n\n"), false));
        }
    }
    segments
}

/// Byte offset of the earliest stop string in `text`, if any.
pub fn find_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min()
}

/// Length of the longest suffix of `text` that is a proper prefix of a stop
/// string. Streaming holds this back until it's clear it isn't a stop.
pub fn partial_stop_len(text: &str, stops: &[String]) -> usize {
    stops
        .iter()
        .flat_map(|stop| (1..stop.len()).filter(move |&k| text.ends_with(&stop[..k])))
        .max()
        .unwrap_or(0)
}

/// The default stop strings as owned values, for `SamplingParams`.
pub fn default_stop_strings() -> Vec<String> {
    STOP_STRINGS.iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_matches_template() {
        let msgs = [
            ChatMessage::system("Be brief."),
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::new(ChatRole::Tool, "read_file (ok):\nx"),
            ChatMessage::user("Bye"),
        ];
        assert_eq!(
            render_prompt(&msgs),
            "<system>\nBe brief.\n</system>\n\n<user>\nHi\n</user>\n\n\
             <assistant>\nHello!\n</assistant>\n\n<tool_result>\nread_file (ok):\nx\n\
             </tool_result>\n\n<user>\nBye\n</user>\n\n<assistant>\n"
        );
    }

    #[test]
    fn only_assistant_content_is_trainable() {
        let segs = render_training_segments(&[ChatMessage::user("Q"), ChatMessage::assistant("A")]);
        let trainable: Vec<&str> =
            segs.iter().filter(|(_, t)| *t).map(|(s, _)| s.as_str()).collect();
        assert_eq!(trainable, ["A\n</assistant>"]);
    }

    #[test]
    fn stop_helpers() {
        let stops = default_stop_strings();
        assert_eq!(find_stop("done</assistant>\n<user>", &stops), Some(4));
        assert_eq!(find_stop("no stop here", &stops), None);
        assert_eq!(partial_stop_len("text </assis", &stops), "</assis".len());
        assert_eq!(partial_stop_len("text <", &stops), 1);
        assert_eq!(partial_stop_len("text", &stops), 0);
    }
}
