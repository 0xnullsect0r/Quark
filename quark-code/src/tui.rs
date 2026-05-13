//! ratatui TUI for Quark Code.

use std::io;
use std::time::Duration;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};

use crate::agent::{start_turn, AgentEvent, StreamHandle};
use crate::app::{App, Message, Mode, Role};

// ─── Palette ──────────────────────────────────────────────────────────────────

const C_ACCENT:    Color = Color::Rgb(100, 180, 255);
const C_USER:      Color = Color::Rgb(180, 255, 180);
const C_ASSISTANT: Color = Color::Rgb(220, 220, 255);
const C_TOOL:      Color = Color::Rgb(255, 210, 100);
const C_SYSTEM:    Color = Color::Rgb(140, 140, 140);
const C_ERROR:     Color = Color::Rgb(255, 100, 100);
const C_BG:        Color = Color::Rgb(14, 16, 20);
const C_BG_PANEL:  Color = Color::Rgb(20, 24, 30);

// ─── Run TUI ─────────────────────────────────────────────────────────────────

pub fn run(mut app: App) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend  = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    let result = event_loop(&mut term, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    term.show_cursor()?;

    result
}

// ─── Event loop ───────────────────────────────────────────────────────────────

fn event_loop(term: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> io::Result<()> {
    let mut stream: Option<StreamHandle> = None;
    let mut stream_response = String::new();

    loop {
        // ── Drain agent stream ────────────────────────────────────────────────
        if let Some(sh) = &stream {
            let mut done = false;
            loop {
                match sh.rx.try_recv() {
                    Ok(AgentEvent::Token(t)) => {
                        stream_response.push_str(&t);
                        app.stream_buf.push_str(&t);
                        app.scroll_to_bottom();
                    }
                    Ok(AgentEvent::ToolCall { name, preview }) => {
                        app.messages.push(Message::tool(
                            format!("🔧 Calling: {} — {}", name, preview)
                        ));
                        app.scroll_to_bottom();
                    }
                    Ok(AgentEvent::ToolResult { name, ok, preview }) => {
                        let icon = if ok { "✓" } else { "✗" };
                        app.messages.push(Message::tool(
                            format!("{icon} {name}: {preview}")
                        ));
                        app.scroll_to_bottom();
                    }
                    Ok(AgentEvent::FileChanged(fc)) => {
                        app.record_changes(vec![fc]);
                    }
                    Ok(AgentEvent::Done(full)) => {
                        // Replace streaming bubble with final message
                        app.stream_buf.clear();
                        app.messages.push(Message::assistant(full));
                        app.generating = false;
                        app.scroll_to_bottom();
                        done = true;
                        break;
                    }
                    Ok(AgentEvent::Error(e)) => {
                        app.messages.push(Message::tool(format!("❌ Error: {e}")));
                        app.generating = false;
                        done = true;
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => { done = true; break; }
                }
            }
            if done {
                stream = None;
                stream_response.clear();
            }
        }

        // ── Render ────────────────────────────────────────────────────────────
        term.draw(|f| render(f, app))?;

        // ── Events (non-blocking with short timeout) ──────────────────────────
        if !event::poll(Duration::from_millis(16))? {
            continue;
        }

        match event::read()? {
            Event::Key(key) if handle_key(app, key, &mut stream) => break,
            Event::Key(_) | Event::Resize(_, _) => {}
            _ => {}
        }

        if app.should_quit { break; }
    }
    Ok(())
}

// ─── Key handler ─────────────────────────────────────────────────────────────

fn handle_key(
    app:    &mut App,
    key:    crossterm::event::KeyEvent,
    stream: &mut Option<StreamHandle>,
) -> bool {
    use KeyCode::*;
    use KeyModifiers as Mod;

    match (key.modifiers, key.code) {
        // Quit
        (Mod::CONTROL, Char('c')) | (Mod::NONE, Esc) if app.input.is_empty() => {
            app.should_quit = true;
            return true;
        }

        // Toggle Plan/Build with Tab
        (Mod::NONE, Tab) => {
            app.mode = match app.mode {
                Mode::Plan  => Mode::Build,
                Mode::Build => Mode::Plan,
            };
            let msg = format!("Switched to {} mode", app.mode);
            app.messages.push(Message::system_msg(msg));
            app.scroll_to_bottom();
        }

        // Scroll
        (Mod::NONE, PageUp)   => app.scroll_up(),
        (Mod::NONE, PageDown) => app.scroll_down(),

        // Send message / run command
        (Mod::NONE, Enter) => {
            let text = app.take_input().trim().to_owned();
            if text.is_empty() { return false; }

            if text.starts_with('/') {
                handle_slash_command(app, &text);
            } else if !app.generating {
                // Expand @file mentions before sending to model
                let expanded = crate::tools::expand_mentions(&text, &app.project_root);
                app.messages.push(Message::user(if expanded == text { text } else { expanded }));
                app.scroll_to_bottom();
                app.generating = true;
                app.stream_buf.clear();

                // In plan mode, add a reminder to the last user message
                if app.mode == Mode::Plan {
                    app.messages.push(Message::system_msg("[Plan mode — model will suggest only]".to_string()));
                }

                *stream = Some(start_turn(app));
            }
        }

        // Text input
        (Mod::NONE, Backspace)                 => app.backspace(),
        (Mod::NONE, Char(c))                   => app.insert_char(c),
        (Mod::SHIFT, Char(c))                  => app.insert_char(c),
        // Cursor movement with bounds guards
        (Mod::NONE, Left)  if app.cursor_pos > 0              => app.cursor_pos -= 1,
        (Mod::NONE, Right) if app.cursor_pos < app.input.len() => app.cursor_pos += 1,
        (Mod::NONE, Left) | (Mod::NONE, Right) => {}
        (Mod::CONTROL, Char('w'))              => delete_last_word(app),
        (Mod::CONTROL, Char('u'))              => { app.input.clear(); app.cursor_pos = 0; }

        _ => {}
    }
    false
}

fn delete_last_word(app: &mut App) {
    let trimmed = app.input[..app.cursor_pos].trim_end_matches(' ');
    let pos = trimmed.rfind(' ').map(|i| i + 1).unwrap_or(0);
    app.input.drain(pos..app.cursor_pos);
    app.cursor_pos = pos;
}

// ─── Slash commands ───────────────────────────────────────────────────────────

fn handle_slash_command(app: &mut App, cmd: &str) {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    let name  = parts[0].to_lowercase();
    let _args = parts.get(1).copied().unwrap_or("");

    match name.as_str() {
        "/exit" | "/quit" => { app.should_quit = true; }

        "/clear" => {
            app.messages.clear();
            app.messages.push(Message::system_msg("Conversation cleared.".to_string()));
        }

        "/plan" => {
            app.mode = Mode::Plan;
            app.messages.push(Message::system_msg("Switched to PLAN mode.".to_string()));
        }
        "/build" => {
            app.mode = Mode::Build;
            app.messages.push(Message::system_msg("Switched to BUILD mode.".to_string()));
        }

        "/init" => {
            app.messages.push(Message::system_msg("Scanning project…".to_string()));
            let ctx = crate::context::ProjectContext::scan(&app.project_root);
            let md  = ctx.to_agents_md();
            let path = app.project_root.join("AGENTS.md");
            match std::fs::write(&path, &md) {
                Ok(_)  => {
                    app.agents_md = Some(md);
                    app.messages.push(Message::system_msg(
                        format!("✓ Created AGENTS.md at {}", path.display())
                    ));
                }
                Err(e) => {
                    app.messages.push(Message::tool(format!("❌ Could not write AGENTS.md: {e}")));
                }
            }
        }

        "/undo" => {
            match app.undo() {
                Some(s) => app.messages.push(Message::system_msg(format!("↩ Undid: {s}"))),
                None    => app.messages.push(Message::system_msg("Nothing to undo.".to_string())),
            }
        }
        "/redo" => {
            match app.redo() {
                Some(s) => app.messages.push(Message::system_msg(format!("↪ Redid: {s}"))),
                None    => app.messages.push(Message::system_msg("Nothing to redo.".to_string())),
            }
        }

        "/diff" => {
            if app.undo_stack.is_empty() {
                app.messages.push(Message::system_msg("No pending file changes.".to_string()));
            } else {
                let count: usize = app.undo_stack.iter().map(|b| b.len()).sum();
                let files: Vec<String> = app.undo_stack.iter()
                    .flat_map(|b| b.iter())
                    .map(|fc| fc.path.display().to_string())
                    .collect();
                app.messages.push(Message::system_msg(
                    format!("{count} change(s) in: {}", files.join(", "))
                ));
            }
        }

        "/mcp" => {
            let cfg = &app.mcp_cfg;
            app.messages.push(Message::system_msg(format!(
                "MCP: read={} write={} list={} search={} cwd={} shell={}",
                cfg.read_file as u8, cfg.write_file as u8, cfg.list_dir as u8,
                cfg.search_files as u8, cfg.get_cwd as u8, cfg.run_shell as u8,
            )));
        }

        "/model" => {
            let status = if app.model_loaded { "loaded" } else { "not loaded" };
            app.messages.push(Message::system_msg(format!(
                "Model: {} ({})", app.model_name, status
            )));
        }

        "/help" => {
            app.messages.push(Message::system_msg(HELP_TEXT.to_string()));
        }

        _ => {
            app.messages.push(Message::tool(format!("Unknown command: {cmd}  (try /help)")));
        }
    }

    app.scroll_to_bottom();
}

const HELP_TEXT: &str = "\
Commands:
  /init         — scan project, create AGENTS.md
  /plan         — switch to Plan mode (suggest only, no writes)
  /build        — switch to Build mode (can apply changes)
  /undo         — undo last file change batch
  /redo         — redo last undone batch
  /diff         — show pending file changes
  /mcp          — show MCP tool status
  /model        — show model info
  /clear        — clear conversation
  /exit         — quit

Shortcuts:
  Tab           — toggle Plan/Build mode
  PageUp/Down   — scroll conversation
  Ctrl+D/U      — scroll down/up
  Ctrl+W        — delete last word
  @path/to/file — inline file contents into message";

// ─── Rendering ────────────────────────────────────────────────────────────────

fn render(f: &mut Frame, app: &App) {
    let area = f.area();

    // Background
    f.render_widget(Block::default().style(Style::default().bg(C_BG)), area);

    // Layout: title + messages + stream bubble + input + statusbar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // title bar
            Constraint::Min(5),     // messages
            Constraint::Length(3),  // input
            Constraint::Length(1),  // status bar
        ])
        .split(area);

    render_titlebar(f, app, chunks[0]);
    render_messages(f, app, chunks[1]);
    render_input(f, app, chunks[2]);
    render_statusbar(f, app, chunks[3]);
}

// ── Title bar ─────────────────────────────────────────────────────────────────

fn render_titlebar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let mode_color = match app.mode {
        Mode::Plan  => Color::Yellow,
        Mode::Build => Color::Green,
    };
    let proj = app.project_root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "project".to_string());

    let line = Line::from(vec![
        Span::styled(" ◆ Quark Code ", Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw("— "),
        Span::styled(proj, Style::default().fg(Color::White)),
        Span::raw("  "),
        Span::styled(
            format!("[ {} ]", app.mode),
            Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  model: "),
        Span::styled(&app.model_name, Style::default().fg(C_ACCENT)),
        if app.model_loaded {
            Span::styled(" ✓", Style::default().fg(Color::Green))
        } else {
            Span::styled(" (stub)", Style::default().fg(Color::DarkGray))
        },
        Span::raw("  Tab:mode  /help  Ctrl+C:quit"),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::default().bg(Color::Rgb(25, 28, 38))),
        area,
    );
}

// ── Message history ───────────────────────────────────────────────────────────

fn render_messages(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Rgb(50, 60, 80)))
        .style(Style::default().bg(C_BG_PANEL));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Build list items from messages
    let mut items: Vec<ListItem> = Vec::new();

    for msg in &app.messages {
        let (prefix, color) = match msg.role {
            Role::User      => ("  You  ▶ ", C_USER),
            Role::Assistant => (" Quark  ▶ ", C_ASSISTANT),
            Role::Tool      => ("  Tool  ▶ ", C_TOOL),
            Role::System    => ("        ℹ ", C_SYSTEM),
        };

        // Word-wrap message content into lines that fit inner width
        let max_w = inner.width.saturating_sub(prefix.len() as u16 + 2) as usize;
        let max_w = max_w.max(20);

        let content_lines = wrap_text(&msg.content, max_w);
        for (i, line_text) in content_lines.into_iter().enumerate() {
            if i == 0 {
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(prefix, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                    Span::styled(line_text, Style::default().fg(color)),
                ])));
            } else {
                let pad = " ".repeat(prefix.len());
                items.push(ListItem::new(Line::from(vec![
                    Span::raw(pad),
                    Span::styled(line_text, Style::default().fg(color)),
                ])));
            }
        }
        // Blank line between messages
        items.push(ListItem::new(Line::default()));
    }

    // Streaming bubble
    if !app.stream_buf.is_empty() {
        let max_w = inner.width.saturating_sub(12) as usize;
        let max_w = max_w.max(20);
        let lines = wrap_text(&app.stream_buf, max_w);
        for (i, l) in lines.into_iter().enumerate() {
            if i == 0 {
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(" Quark  ▶ ", Style::default().fg(C_ASSISTANT).add_modifier(Modifier::BOLD)),
                    Span::styled(l, Style::default().fg(C_ASSISTANT)),
                ])));
            } else {
                items.push(ListItem::new(Line::from(vec![
                    Span::raw("          "),
                    Span::styled(l, Style::default().fg(C_ASSISTANT)),
                ])));
            }
        }
        // Blinking cursor
        items.push(ListItem::new(Line::from(vec![
            Span::raw("          "),
            Span::styled("▋", Style::default().fg(C_ASSISTANT).add_modifier(Modifier::SLOW_BLINK)),
        ])));
    }

    let total_items = items.len();
    let visible     = inner.height as usize;

    let offset = if app.scroll == usize::MAX || app.scroll >= total_items.saturating_sub(visible) {
        total_items.saturating_sub(visible)
    } else {
        app.scroll.min(total_items.saturating_sub(visible))
    };

    let list = List::new(items.into_iter().skip(offset).collect::<Vec<_>>())
        .style(Style::default().bg(C_BG_PANEL));
    f.render_widget(list, inner);
}

// ── Input box ─────────────────────────────────────────────────────────────────

fn render_input(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let mode_color = match app.mode {
        Mode::Plan  => Color::Yellow,
        Mode::Build => Color::Green,
    };

    let title = format!(
        " {} › {} ",
        app.mode,
        if app.generating { "generating…" } else { "type your message" }
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(mode_color))
        .title(title)
        .style(Style::default().bg(C_BG_PANEL));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Show input with cursor
    let display = if app.input.is_empty() && !app.generating {
        Span::styled(
            "Ask me anything, or /init to start…",
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )
    } else {
        // Insert cursor char at cursor_pos
        let mut s = app.input.clone();
        let cursor_s = if app.generating { "…" } else { "█" };
        if app.cursor_pos <= s.len() {
            s.insert_str(app.cursor_pos, cursor_s);
        } else {
            s.push_str(cursor_s);
        }
        Span::styled(s, Style::default().fg(Color::White))
    };

    f.render_widget(Paragraph::new(display).wrap(Wrap { trim: false }), inner);
}

// ── Status bar ────────────────────────────────────────────────────────────────

fn render_statusbar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let undo_count: usize = app.undo_stack.len();
    let redo_count: usize = app.redo_stack.len();

    let tools_enabled = [
        app.mcp_cfg.read_file,
        app.mcp_cfg.write_file,
        app.mcp_cfg.list_dir,
        app.mcp_cfg.search_files,
        app.mcp_cfg.get_cwd,
        app.mcp_cfg.run_shell,
    ].iter().filter(|&&b| b).count();

    let status = if !app.status_msg.is_empty() {
        app.status_msg.clone()
    } else {
        format!(
            " undo:{undo_count}  redo:{redo_count}  tools:{tools_enabled}/6  {}",
            if app.agents_md.is_some() { "AGENTS.md ✓" } else { "no AGENTS.md (/init)" }
        )
    };

    f.render_widget(
        Paragraph::new(status)
            .style(Style::default().fg(Color::DarkGray).bg(Color::Rgb(18, 20, 28))),
        area,
    );
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Simple greedy word-wrap.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for input_line in text.lines() {
        if input_line.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut col = 0usize;
        for word in input_line.split_whitespace() {
            let wlen = word.len();
            if col > 0 && col + 1 + wlen > width {
                lines.push(current.clone());
                current.clear();
                col = 0;
            }
            if col > 0 { current.push(' '); col += 1; }
            current.push_str(word);
            col += wlen;
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    if lines.is_empty() { lines.push(String::new()); }
    lines
}
