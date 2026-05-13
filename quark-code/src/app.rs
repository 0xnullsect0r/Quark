//! Application state for Quark Code.

use std::path::{Path, PathBuf};
use quark_core::mcp::McpConfig;

// ─── Mode ─────────────────────────────────────────────────────────────────────

/// Plan mode: model suggests changes but cannot apply them.
/// Build mode: model applies changes to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Plan,
    Build,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Plan  => write!(f, "PLAN"),
            Mode::Build => write!(f, "BUILD"),
        }
    }
}

// ─── Message ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role { User, Assistant, Tool, System }

#[derive(Debug, Clone)]
pub struct Message {
    pub role:    Role,
    pub content: String,
}

impl Message {
    pub fn user(s: impl Into<String>)       -> Self { Self { role: Role::User,      content: s.into() } }
    pub fn assistant(s: impl Into<String>)  -> Self { Self { role: Role::Assistant, content: s.into() } }
    pub fn tool(s: impl Into<String>)       -> Self { Self { role: Role::Tool,      content: s.into() } }
    pub fn system_msg(s: impl Into<String>) -> Self { Self { role: Role::System,    content: s.into() } }
}

// ─── File change (undo/redo) ──────────────────────────────────────────────────

/// A recorded change to a single file.
#[derive(Debug, Clone)]
pub struct FileChange {
    pub path:   PathBuf,
    /// Content before the change (`None` if the file was created).
    pub before: Option<String>,
    /// Content after the change (`None` if the file was deleted).
    pub after:  Option<String>,
}

// ─── App ──────────────────────────────────────────────────────────────────────

pub struct App {
    // ── UI state ──────────────────────────────────────────────────────────
    pub mode:        Mode,
    pub input:       String,
    /// Cursor position within `input` (byte index).
    pub cursor_pos:  usize,
    pub scroll:      usize,
    pub should_quit: bool,

    // ── Conversation ──────────────────────────────────────────────────────
    pub messages: Vec<Message>,

    // ── Model / config ────────────────────────────────────────────────────
    pub model_name:    String,
    pub system_prompt: String,
    pub mcp_cfg:       McpConfig,
    pub model_loaded:  bool,
    /// Streaming: tokens arrive here from the background generate thread.
    pub stream_buf:    String,
    pub generating:    bool,

    // ── Project ───────────────────────────────────────────────────────────
    pub project_root: PathBuf,
    /// Contents of AGENTS.md if it exists.
    pub agents_md:    Option<String>,

    // ── Undo/redo ─────────────────────────────────────────────────────────
    /// Each entry is a batch of changes that can be undone atomically.
    pub undo_stack: Vec<Vec<FileChange>>,
    pub redo_stack: Vec<Vec<FileChange>>,

    // ── Status bar ────────────────────────────────────────────────────────
    pub status_msg: String,
}

impl App {
    pub fn new(
        model_name:    String,
        system_prompt: String,
        mcp_cfg:       McpConfig,
        model_loaded:  bool,
        project_root:  PathBuf,
    ) -> Self {
        let agents_md = load_agents_md(&project_root);
        let mut app = Self {
            mode:         Mode::Build,
            input:        String::new(),
            cursor_pos:   0,
            scroll:       0,
            should_quit:  false,
            messages:     Vec::new(),
            model_name,
            system_prompt,
            mcp_cfg,
            model_loaded,
            stream_buf:   String::new(),
            generating:   false,
            project_root,
            agents_md,
            undo_stack:   Vec::new(),
            redo_stack:   Vec::new(),
            status_msg:   String::new(),
        };

        // Welcome message
        app.push_system();
        app
    }

    fn push_system(&mut self) {
        let loaded_str = if self.model_loaded { "✓ loaded" } else { "⚠ not loaded — run inference after training" };
        let agents = if self.agents_md.is_some() { "✓ AGENTS.md found" } else { "run /init to analyse project" };
        self.messages.push(Message::system_msg(format!(
            "Quark Code  ·  model: {} ({})  ·  project: {}  ·  {}",
            self.model_name,
            loaded_str,
            self.project_root.display(),
            agents,
        )));
        self.messages.push(Message::system_msg(
            "Tab → toggle Plan/Build mode   /help for commands   Ctrl+C to quit".into()
        ));
    }

    // ── Input helpers ─────────────────────────────────────────────────────────

    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor_pos, c);
        self.cursor_pos += c.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            let prev = self.input[..self.cursor_pos]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.remove(prev);
            self.cursor_pos = prev;
        }
    }

    pub fn take_input(&mut self) -> String {
        let s = std::mem::take(&mut self.input);
        self.cursor_pos = 0;
        s
    }

    // ── Undo / redo ───────────────────────────────────────────────────────────

    /// Record a batch of file changes (called after each Build-mode tool run).
    pub fn record_changes(&mut self, changes: Vec<FileChange>) {
        if !changes.is_empty() {
            self.undo_stack.push(changes);
            self.redo_stack.clear();
        }
    }

    /// Undo the last batch of file changes.
    pub fn undo(&mut self) -> Option<String> {
        let batch = self.undo_stack.pop()?;
        let mut summary = Vec::new();
        for change in &batch {
            match &change.before {
                Some(content) => {
                    if let Some(parent) = change.path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&change.path, content);
                    summary.push(format!("restored {}", change.path.display()));
                }
                None => {
                    let _ = std::fs::remove_file(&change.path);
                    summary.push(format!("deleted {}", change.path.display()));
                }
            }
        }
        self.redo_stack.push(batch);
        Some(summary.join(", "))
    }

    /// Redo the last undone batch.
    pub fn redo(&mut self) -> Option<String> {
        let batch = self.redo_stack.pop()?;
        let mut summary = Vec::new();
        for change in &batch {
            match &change.after {
                Some(content) => {
                    if let Some(parent) = change.path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&change.path, content);
                    summary.push(format!("wrote {}", change.path.display()));
                }
                None => {
                    let _ = std::fs::remove_file(&change.path);
                    summary.push(format!("deleted {}", change.path.display()));
                }
            }
        }
        self.undo_stack.push(batch);
        Some(summary.join(", "))
    }

    // ── Scroll ────────────────────────────────────────────────────────────────

    pub fn scroll_up(&mut self)   { self.scroll = self.scroll.saturating_sub(3); }
    pub fn scroll_down(&mut self) { self.scroll += 3; }
    pub fn scroll_to_bottom(&mut self) {
        self.scroll = usize::MAX; // clamped in render
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn load_agents_md(root: &Path) -> Option<String> {
    let p = root.join("AGENTS.md");
    std::fs::read_to_string(p).ok()
}
