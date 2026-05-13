#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use quark_core::mcp::McpConfig;

pub struct ExportPanel {
    // Source
    checkpoint_path: Option<PathBuf>,
    tokenizer_path:  Option<PathBuf>,
    config_path:     Option<PathBuf>,

    // App settings
    app_name:       String,
    system_prompt:  String,
    mcp:            McpConfig,

    // Output
    output_dir: Option<PathBuf>,

    // State
    status: Option<(bool, String)>, // (is_ok, msg)
}

impl Default for ExportPanel {
    fn default() -> Self {
        Self {
            checkpoint_path: None,
            tokenizer_path:  None,
            config_path:     None,
            app_name:        "MyQuarkApp".into(),
            system_prompt:   "You are a helpful coding assistant with access to MCP tools for reading and writing files.".into(),
            mcp:             McpConfig::default(),
            output_dir:      None,
            status:          None,
        }
    }
}

impl ExportPanel {
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
        ui.heading("📦 Export as Standalone Chat App");
        ui.separator();
        ui.label(egui::RichText::new(
            "Bundle your trained model into a self-contained executable chat app with MCP file tools."
        ).weak().italics());
        ui.add_space(8.0);

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

            let can_export = self.checkpoint_path.is_some()
                && self.tokenizer_path.is_some()
                && self.output_dir.is_some()
                && !self.app_name.trim().is_empty();

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        can_export,
                        egui::Button::new(egui::RichText::new("📦 Bundle App").strong()),
                    )
                    .clicked()
                {
                    self.do_export();
                }
                if !can_export {
                    ui.label(
                        egui::RichText::new(
                            "Set checkpoint, tokenizer, and output folder to enable export.",
                        )
                        .weak()
                        .italics(),
                    );
                }
            });
        });

        // Status message
        if let Some((ok, msg)) = &self.status {
            ui.add_space(4.0);
            let color = if *ok {
                egui::Color32::from_rgb(80, 200, 100)
            } else {
                egui::Color32::RED
            };
            ui.label(egui::RichText::new(msg.as_str()).color(color));
        }

        // ─── Bundle contents info ───────────────────────────────────────────────
        ui.add_space(8.0);
        ui.separator();
        egui::CollapsingHeader::new("ℹ Bundle layout")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "
{name}/
├── quark-chat          (or quark-chat.exe on Windows)
├── model/
│   ├── checkpoint.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   ├── mcp.json
│   └── system_prompt.txt
├── run.sh              (Linux/macOS launcher)
└── run.bat             (Windows launcher)

Run the app:
  Linux/macOS:  ./{name}/run.sh
  Windows:      {name}\\run.bat
  Or directly:  ./{name}/quark-chat
",
                        name = self.app_name.trim()
                    ))
                    .monospace()
                    .weak(),
                );
            });
    }

    fn do_export(&mut self) {
        let result = self.try_export();
        match result {
            Ok(path) => self.status = Some((true, format!("✓ Exported to {}", path.display()))),
            Err(e) => self.status = Some((false, format!("Export failed: {e}"))),
        }
    }

    fn try_export(&self) -> anyhow::Result<PathBuf> {
        use std::fs;

        let name = self.app_name.trim();
        let out_root = self.output_dir.as_ref().unwrap().join(name);
        let model_dir = out_root.join("model");
        fs::create_dir_all(&model_dir)?;

        // Copy checkpoint
        let ckpt_src = self.checkpoint_path.as_ref().unwrap();
        fs::copy(ckpt_src, model_dir.join("checkpoint.safetensors"))
            .map_err(|e| anyhow::anyhow!("Failed to copy checkpoint: {e}"))?;

        // Copy tokenizer
        fs::copy(
            self.tokenizer_path.as_ref().unwrap(),
            model_dir.join("tokenizer.json"),
        )
        .map_err(|e| anyhow::anyhow!("Failed to copy tokenizer: {e}"))?;

        // Copy config if present
        if let Some(cfg) = &self.config_path {
            let _ = fs::copy(cfg, model_dir.join("config.json"));
        }

        // Write mcp.json
        let mcp_json = serde_json::to_string_pretty(&self.mcp)?;
        fs::write(model_dir.join("mcp.json"), mcp_json)?;

        // Write system prompt
        fs::write(model_dir.join("system_prompt.txt"), &self.system_prompt)?;

        // Write config.json with app name if not already there
        let config_json_path = model_dir.join("config.json");
        if !config_json_path.exists() {
            let meta = serde_json::json!({ "name": name });
            fs::write(&config_json_path, serde_json::to_string_pretty(&meta)?)?;
        }

        // Copy quark-chat binary from beside our own executable
        let own_exe = std::env::current_exe()
            .unwrap_or_else(|_| std::path::PathBuf::from("quark-gui"));
        let exe_dir = own_exe.parent().unwrap_or(std::path::Path::new("."));

        #[cfg(windows)]
        let chat_exe_name = "quark-chat.exe";
        #[cfg(not(windows))]
        let chat_exe_name = "quark-chat";

        let chat_src = exe_dir.join(chat_exe_name);
        let chat_dst = out_root.join(chat_exe_name);

        if chat_src.exists() {
            fs::copy(&chat_src, &chat_dst)
                .map_err(|e| anyhow::anyhow!("Failed to copy quark-chat binary: {e}"))?;
            // Make executable on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&chat_dst)?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&chat_dst, perms)?;
            }
        } else {
            // Write a note explaining how to get the binary
            fs::write(
                out_root.join("MISSING_BINARY.txt"),
                format!(
                    "quark-chat binary not found at {}\n\nBuild it with:\n  cargo build --release --package quark-chat --features backend-cpu\nThen copy target/release/quark-chat here.",
                    chat_src.display()
                ),
            )?;
        }

        // Write launcher scripts
        let app_exe = if cfg!(windows) {
            format!(".\\{chat_exe_name}")
        } else {
            format!("./{chat_exe_name}")
        };

        fs::write(
            out_root.join("run.sh"),
            format!("#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\"\n{app_exe} \"$@\"\n"),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = fs::metadata(out_root.join("run.sh"))?.permissions();
            p.set_mode(0o755);
            fs::set_permissions(out_root.join("run.sh"), p)?;
        }

        fs::write(
            out_root.join("run.bat"),
            format!("@echo off\r\ncd /d \"%~dp0\"\r\n{chat_exe_name} %*\r\n"),
        )?;

        Ok(out_root)
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
