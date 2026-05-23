#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use quark_core::mcp::McpConfig;

// ─── Background export state ──────────────────────────────────────────────────

enum ExportMessage {
    Log(String),
    Progress(f32),
    Done(PathBuf),
    Error(String),
}

struct ExportState {
    log: Vec<String>,
    progress: f32,
    is_running: bool,
    result: Option<Result<PathBuf, String>>,
    receiver: Option<Receiver<ExportMessage>>,
}

impl Default for ExportState {
    fn default() -> Self {
        Self { log: Vec::new(), progress: 0.0, is_running: false, result: None, receiver: None }
    }
}

impl ExportState {
    fn poll(&mut self) -> bool {
        let Some(rx) = &self.receiver else { return false };
        let mut changed = false;
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    changed = true;
                    match msg {
                        ExportMessage::Log(s) => {
                            self.log.push(s);
                            if self.log.len() > 500 {
                                self.log.drain(0..125);
                            }
                        }
                        ExportMessage::Progress(p) => self.progress = p,
                        ExportMessage::Done(path) => {
                            self.is_running = false;
                            self.result = Some(Ok(path));
                        }
                        ExportMessage::Error(e) => {
                            self.is_running = false;
                            self.result = Some(Err(e));
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.receiver = None;
                    self.is_running = false;
                    break;
                }
            }
        }
        changed
    }
}

/// Which export target to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportTarget {
    /// quark-chat: simple terminal chat REPL
    Chat,
    /// quark-code: full TUI coding agent (Claude Code / Copilot CLI style)
    Code,
}

pub struct ExportPanel {
    // Source
    checkpoint_path: Option<PathBuf>,
    tokenizer_path:  Option<PathBuf>,
    config_path:     Option<PathBuf>,

    // App settings
    app_name:       String,
    system_prompt:  String,
    mcp:            McpConfig,
    target:         ExportTarget,

    // Output
    output_dir: Option<PathBuf>,

    // State
    export_state: ExportState,
}

impl Default for ExportPanel {
    fn default() -> Self {
        Self {
            checkpoint_path: None,
            tokenizer_path:  None,
            config_path:     None,
            app_name:        "MyQuarkApp".into(),
            system_prompt:   "You are Quark Code, an expert AI coding assistant running locally on the user's machine.".into(),
            mcp:             McpConfig::default(),
            target:          ExportTarget::Code,
            output_dir:      None,
            export_state:    ExportState::default(),
        }
    }
}

impl ExportPanel {
    /// Poll the background export thread for new messages.  Call every frame.
    pub fn update(&mut self, ctx: &egui::Context) {
        if self.export_state.poll() {
            ctx.request_repaint();
        }
    }

    /// Called from CheckpointsPanel when user clicks "Export as App" on a checkpoint
    pub fn set_checkpoint(&mut self, path: PathBuf) {
        // Look for tokenizer.json and config.json in the same directory
        let dir = path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
        self.checkpoint_path = Some(path);
        let tok = dir.join("tokenizer.json");
        if tok.exists() {
            self.tokenizer_path = Some(tok);
        }
        let cfg = dir.join("config.json");
        if cfg.exists() {
            self.config_path = Some(cfg);
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("📦 Export as Standalone App");
        ui.separator();
        ui.label(egui::RichText::new(
            "Bundle your trained model into a self-contained executable powered by the local Quark model."
        ).weak().italics());
        ui.add_space(8.0);

        // ─── Export target selector ─────────────────────────────────────────────
        egui::CollapsingHeader::new("🎯 Export Type").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.target, ExportTarget::Code, "💻 Quark Code (TUI coding agent)");
                ui.selectable_value(&mut self.target, ExportTarget::Chat, "💬 Quark Chat (simple REPL)");
            });
            ui.add_space(4.0);
            match self.target {
                ExportTarget::Code => {
                    ui.label(egui::RichText::new(
                        "quark-code: Full terminal coding agent. Features Plan/Build modes, \
                        git integration, file tools, /init project scanning, undo/redo, \
                        @file context injection — similar to Claude Code or GitHub Copilot CLI."
                    ).weak().small());
                }
                ExportTarget::Chat => {
                    ui.label(egui::RichText::new(
                        "quark-chat: Simple terminal REPL. Type messages, get responses. \
                        MCP file tools included."
                    ).weak().small());
                }
            }
        });

        ui.add_space(4.0);

        // ─── Source files ───────────────────────────────────────────────────────
        egui::CollapsingHeader::new("📁 Source Files").default_open(true).show(ui, |ui| {
            egui::Grid::new("src_grid").num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
                ui.label("Checkpoint (.safetensors)");
                path_label(ui, &self.checkpoint_path);
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("SafeTensors", &["safetensors"])
                        .set_title("Pick checkpoint")
                        .pick_file()
                    {
                        self.checkpoint_path = Some(p);
                    }
                }
                ui.end_row();

                ui.label("Tokenizer (tokenizer.json)");
                path_label(ui, &self.tokenizer_path);
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("JSON", &["json"])
                        .set_title("Pick tokenizer.json")
                        .pick_file()
                    {
                        self.tokenizer_path = Some(p);
                    }
                }
                ui.end_row();

                ui.label("Config (config.json, optional)");
                path_label(ui, &self.config_path);
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("JSON", &["json"])
                        .set_title("Pick config.json")
                        .pick_file()
                    {
                        self.config_path = Some(p);
                    }
                }
                ui.end_row();
            });
        });

        ui.add_space(4.0);

        // ─── App identity ───────────────────────────────────────────────────────
        egui::CollapsingHeader::new("🏷 App Identity").default_open(true).show(ui, |ui| {
            egui::Grid::new("identity_grid").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                ui.label("App name");
                ui.text_edit_singleline(&mut self.app_name);
                ui.end_row();
            });
            ui.add_space(4.0);
            ui.label("System prompt:");
            ui.add(
                egui::TextEdit::multiline(&mut self.system_prompt)
                    .desired_rows(4)
                    .desired_width(f32::INFINITY)
                    .hint_text("Instructions for the assistant…"),
            );
        });

        ui.add_space(4.0);

        // ─── MCP tools ──────────────────────────────────────────────────────────
        egui::CollapsingHeader::new("🔧 MCP Tools").default_open(true).show(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    "These tools let the model read/write files on the end-user's machine.",
                )
                .weak()
                .italics(),
            );
            ui.add_space(4.0);

            egui::Grid::new("mcp_grid")
                .num_columns(2)
                .striped(true)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.checkbox(&mut self.mcp.read_file, "read_file");
                    ui.label(egui::RichText::new("Read file contents").weak());
                    ui.end_row();

                    ui.checkbox(&mut self.mcp.write_file, "write_file");
                    ui.label(egui::RichText::new("Create or overwrite files").weak());
                    ui.end_row();

                    ui.checkbox(&mut self.mcp.list_dir, "list_dir");
                    ui.label(egui::RichText::new("List directory entries").weak());
                    ui.end_row();

                    ui.checkbox(&mut self.mcp.search_files, "search_files");
                    ui.label(egui::RichText::new("Search files by name pattern").weak());
                    ui.end_row();

                    ui.checkbox(&mut self.mcp.get_cwd, "get_cwd");
                    ui.label(egui::RichText::new("Return current working directory").weak());
                    ui.end_row();

                    ui.checkbox(&mut self.mcp.run_shell, "run_shell ⚠");
                    ui.label(
                        egui::RichText::new("Execute shell commands (security risk)")
                            .color(egui::Color32::YELLOW)
                            .weak(),
                    );
                    ui.end_row();
                });
        });

        ui.add_space(4.0);

        // ─── Output directory ───────────────────────────────────────────────────
        egui::CollapsingHeader::new("📤 Output").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Output folder:");
                match &self.output_dir {
                    Some(d) => {
                        ui.label(egui::RichText::new(d.to_string_lossy()).monospace());
                    }
                    None => {
                        ui.label(egui::RichText::new("Not set").weak());
                    }
                };
                if ui.button("📁 Browse…").clicked() {
                    if let Some(d) = rfd::FileDialog::new()
                        .set_title("Choose export destination")
                        .pick_folder()
                    {
                        self.output_dir = Some(d);
                    }
                }
            });

            ui.add_space(6.0);

            let running = self.export_state.is_running;
            let can_export = !running
                && self.checkpoint_path.is_some()
                && self.tokenizer_path.is_some()
                && self.output_dir.is_some()
                && !self.app_name.trim().is_empty();

            ui.horizontal(|ui| {
                if running {
                    ui.spinner();
                    ui.label("Bundling…");
                } else if ui
                    .add_enabled(
                        can_export,
                        egui::Button::new(egui::RichText::new("📦 Bundle App").strong()),
                    )
                    .clicked()
                {
                    self.start_export();
                }
                if !can_export && !running {
                    ui.label(
                        egui::RichText::new(
                            "Set checkpoint, tokenizer, and output folder to enable export.",
                        )
                        .weak()
                        .italics(),
                    );
                }
            });

            // Progress bar while running
            if running || self.export_state.progress > 0.0 {
                ui.add_space(4.0);
                ui.add(
                    egui::ProgressBar::new(self.export_state.progress)
                        .desired_width(ui.available_width())
                        .animate(running),
                );
            }
        });

        // Result status
        match &self.export_state.result.clone() {
            Some(Ok(path)) => {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("✅  Exported to {}", path.display()))
                        .color(egui::Color32::from_rgb(80, 200, 100)),
                );
            }
            Some(Err(e)) => {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("❌  Export failed: {e}"))
                        .color(egui::Color32::RED),
                );
            }
            None => {}
        }

        // Export log
        if !self.export_state.log.is_empty() || self.export_state.is_running {
            ui.add_space(4.0);
            let log_ref = &self.export_state.log;
            egui::ScrollArea::vertical()
                .id_salt("export_log")
                .max_height(150.0)
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgb(18, 18, 20))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            for line in log_ref.iter() {
                                let color = if line.starts_with("❌") {
                                    egui::Color32::from_rgb(255, 100, 100)
                                } else if line.starts_with("✅") || line.starts_with("✔") {
                                    egui::Color32::from_rgb(100, 220, 100)
                                } else {
                                    egui::Color32::from_rgb(200, 200, 200)
                                };
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(line)
                                            .monospace()
                                            .size(11.0)
                                            .color(color),
                                    )
                                    .selectable(true),
                                );
                            }
                        });
                });
        }

        // ─── Bundle contents info ───────────────────────────────────────────────
        ui.add_space(8.0);
        ui.separator();
        let bin_name = match self.target {
            ExportTarget::Code => "quark-code",
            ExportTarget::Chat => "quark-chat",
        };
        egui::CollapsingHeader::new("ℹ Bundle layout")
            .default_open(false)
            .show(ui, |ui| {
                let name = self.app_name.trim();
                ui.label(
                    egui::RichText::new(format!(
                        "\n{name}/\n\
                        ├── {bin_name}          (or {bin_name}.exe on Windows)\n\
                        ├── model/\n\
                        │   ├── checkpoint.safetensors\n\
                        │   ├── tokenizer.json\n\
                        │   ├── config.json\n\
                        │   ├── mcp.json\n\
                        │   └── system_prompt.txt\n\
                        ├── run.sh              (Linux/macOS launcher)\n\
                        └── run.bat             (Windows launcher)\n\
                        \nRun the app:\n\
                          Linux/macOS:  ./{name}/run.sh\n\
                          Windows:      {name}\\\\run.bat\n\
                          Or directly:  ./{name}/{bin_name}\n"
                    ))
                    .monospace()
                    .weak(),
                );
            });
    }

    fn start_export(&mut self) {
        let checkpoint_path = self.checkpoint_path.clone().unwrap();
        let tokenizer_path  = self.tokenizer_path.clone().unwrap();
        let config_path     = self.config_path.clone();
        let app_name        = self.app_name.trim().to_owned();
        let system_prompt   = self.system_prompt.clone();
        let mcp             = self.mcp.clone();
        let target          = self.target;
        let output_dir      = self.output_dir.clone().unwrap();

        let (tx, rx) = std::sync::mpsc::channel();
        self.export_state.log.clear();
        self.export_state.progress = 0.0;
        self.export_state.result = None;
        self.export_state.is_running = true;
        self.export_state.receiver = Some(rx);

        std::thread::spawn(move || {
            macro_rules! log {
                ($($t:tt)*) => {{ let _ = tx.send(ExportMessage::Log(format!($($t)*))); }};
            }
            macro_rules! prog {
                ($p:expr) => {{ let _ = tx.send(ExportMessage::Progress($p)); }};
            }

            use std::fs;

            let out_root  = output_dir.join(&app_name);
            let model_dir = out_root.join("model");

            log!("Creating output directory: {}", out_root.display());
            if let Err(e) = fs::create_dir_all(&model_dir) {
                let _ = tx.send(ExportMessage::Error(format!("Cannot create output dir: {e}")));
                return;
            }
            prog!(0.1);

            // Copy checkpoint
            log!("Copying checkpoint…");
            if let Err(e) = fs::copy(&checkpoint_path, model_dir.join("checkpoint.safetensors")) {
                let _ = tx.send(ExportMessage::Error(format!("Failed to copy checkpoint: {e}")));
                return;
            }
            prog!(0.35);
            log!("✔  checkpoint.safetensors");

            // Copy tokenizer
            log!("Copying tokenizer…");
            if let Err(e) = fs::copy(&tokenizer_path, model_dir.join("tokenizer.json")) {
                let _ = tx.send(ExportMessage::Error(format!("Failed to copy tokenizer: {e}")));
                return;
            }
            prog!(0.45);
            log!("✔  tokenizer.json");

            // Copy config if present
            if let Some(cfg) = &config_path {
                let _ = fs::copy(cfg, model_dir.join("config.json"));
                log!("✔  config.json");
            }

            // Write mcp.json
            match serde_json::to_string_pretty(&mcp) {
                Ok(json) => {
                    if let Err(e) = fs::write(model_dir.join("mcp.json"), json) {
                        log!("⚠  Could not write mcp.json: {e}");
                    } else {
                        log!("✔  mcp.json");
                    }
                }
                Err(e) => log!("⚠  Could not serialize mcp config: {e}"),
            }

            // Write system prompt
            if let Err(e) = fs::write(model_dir.join("system_prompt.txt"), &system_prompt) {
                log!("⚠  Could not write system_prompt.txt: {e}");
            } else {
                log!("✔  system_prompt.txt");
            }

            // Write app metadata config
            let config_json_path = model_dir.join("config.json");
            if !config_json_path.exists() {
                let meta = serde_json::json!({
                    "name": app_name,
                    "export_type": match target {
                        ExportTarget::Code => "quark-code",
                        ExportTarget::Chat => "quark-chat",
                    }
                });
                if let Ok(s) = serde_json::to_string_pretty(&meta) {
                    let _ = fs::write(&config_json_path, s);
                }
            }
            prog!(0.60);

            // Copy binary
            let own_exe = std::env::current_exe()
                .unwrap_or_else(|_| PathBuf::from("quark-gui"));
            let exe_dir = own_exe.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));

            let src_bin_name = match target {
                ExportTarget::Code => {
                    if cfg!(windows) { "quark-code.exe" } else { "quark-code" }
                }
                ExportTarget::Chat => {
                    if cfg!(windows) { "quark-chat.exe" } else { "quark-chat" }
                }
            };

            let bin_src = exe_dir.join(src_bin_name);
            let bin_dst = out_root.join(src_bin_name);

            log!("Copying binary {}…", src_bin_name);
            if bin_src.exists() {
                if let Err(e) = fs::copy(&bin_src, &bin_dst) {
                    let _ = tx.send(ExportMessage::Error(
                        format!("Failed to copy binary: {e}")
                    ));
                    return;
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Ok(meta) = fs::metadata(&bin_dst) {
                        let mut perms = meta.permissions();
                        perms.set_mode(0o755);
                        let _ = fs::set_permissions(&bin_dst, perms);
                    }
                }
                log!("✔  {src_bin_name}");
            } else {
                let pkg = match target { ExportTarget::Code => "quark-code", ExportTarget::Chat => "quark-chat" };
                let _ = fs::write(
                    out_root.join("MISSING_BINARY.txt"),
                    format!(
                        "{src_bin_name} not found at {}\n\nBuild it with:\n  cargo build --release --package {pkg} --features backend-cpu",
                        bin_src.display()
                    ),
                );
                log!("⚠  Binary not found — see MISSING_BINARY.txt");
            }
            prog!(0.85);

            // Write launchers
            let _ = fs::write(
                out_root.join("run.sh"),
                format!("#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\"\n./{src_bin_name} \"$@\"\n"),
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(meta) = fs::metadata(out_root.join("run.sh")) {
                    let mut p = meta.permissions();
                    p.set_mode(0o755);
                    let _ = fs::set_permissions(out_root.join("run.sh"), p);
                }
            }
            let _ = fs::write(
                out_root.join("run.bat"),
                format!("@echo off\r\ncd /d \"%~dp0\"\r\n{src_bin_name} %*\r\n"),
            );
            log!("✔  run.sh / run.bat");
            prog!(1.0);

            log!("✅  Bundle complete → {}", out_root.display());
            let _ = tx.send(ExportMessage::Done(out_root));
        });
    }
}

fn path_label(ui: &mut egui::Ui, path: &Option<PathBuf>) {
    match path {
        Some(p) => ui.label(
            egui::RichText::new(p.file_name().unwrap_or_default().to_string_lossy())
                .color(egui::Color32::GREEN)
                .monospace(),
        ),
        None => ui.label(egui::RichText::new("Not set").weak()),
    };
}
