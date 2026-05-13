#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use quark_core::data::{pile_components, start_pile_build, PileConfig, PileMessage};

// ─── File list entry ──────────────────────────────────────────────────────────

struct FileEntry {
    path: PathBuf,
    size_bytes: u64,
}

// ─── Pile download state ──────────────────────────────────────────────────────

struct PileState {
    /// Received log lines (capped at MAX_LOG_LINES).
    log: Vec<String>,
    progress: f32,
    phase: String,
    is_running: bool,
    finished: bool,
    error: Option<String>,
    receiver: Option<Receiver<PileMessage>>,
}

const MAX_LOG_LINES: usize = 2000;

impl Default for PileState {
    fn default() -> Self {
        Self {
            log: Vec::new(),
            progress: 0.0,
            phase: String::new(),
            is_running: false,
            finished: false,
            error: None,
            receiver: None,
        }
    }
}

impl PileState {
    /// Drain the channel and update state.  Returns `true` if any message
    /// arrived (so the caller knows to request a repaint).
    fn poll(&mut self) -> bool {
        let Some(rx) = &self.receiver else { return false };
        let mut changed = false;
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    changed = true;
                    match msg {
                        PileMessage::Log(s) => {
                            self.log.push(s);
                            if self.log.len() > MAX_LOG_LINES {
                                self.log.drain(0..MAX_LOG_LINES / 4);
                            }
                        }
                        PileMessage::Progress(p) => self.progress = p,
                        PileMessage::Phase(s) => self.phase = s,
                        PileMessage::Done => {
                            self.is_running = false;
                            self.finished = true;
                        }
                        PileMessage::Error(e) => {
                            self.is_running = false;
                            self.error = Some(e);
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

    fn start(&mut self, cfg: PileConfig) {
        self.log.clear();
        self.progress = 0.0;
        self.phase = "Starting…".into();
        self.error = None;
        self.finished = false;
        self.is_running = true;
        self.receiver = Some(start_pile_build(cfg));
    }
}

// ─── Main panel ───────────────────────────────────────────────────────────────

pub struct DatasetPanel {
    // ── manual file list ──────────────────────────────────────────────────
    files: Vec<FileEntry>,
    max_seq_len: usize,
    tokenizer_path: Option<PathBuf>,
    vocab_size_input: usize,
    status: String,

    // ── Pile section ──────────────────────────────────────────────────────
    pile_enabled: bool,
    pile_config: PileConfig,
    /// Selected component index into pile_components().
    pile_component_idx: usize,
    pile_state: PileState,
    /// Whether to auto-scroll the log.
    pile_log_autoscroll: bool,
}

impl Default for DatasetPanel {
    fn default() -> Self {
        Self {
            files: vec![],
            max_seq_len: 2048,
            tokenizer_path: None,
            vocab_size_input: 32000,
            status: String::new(),
            pile_enabled: false,
            pile_config: PileConfig::default(),
            pile_component_idx: 0,
            pile_state: PileState::default(),
            pile_log_autoscroll: true,
        }
    }
}

impl DatasetPanel {
    pub fn file_paths(&self) -> Vec<PathBuf> {
        self.files.iter().map(|f| f.path.clone()).collect()
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    /// Returns the Pile repo `the-pile/` directory if a successful build exists.
    pub fn pile_output_dir(&self) -> Option<PathBuf> {
        if self.pile_state.finished {
            Some(self.pile_config.target_dir.join("the-pile"))
        } else {
            None
        }
    }

    /// Must be called every frame so the live log updates.
    pub fn update(&mut self, ctx: &egui::Context) {
        if self.pile_state.poll() {
            ctx.request_repaint();
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("📂 Dataset");
        ui.separator();

        // ── Manual files ──────────────────────────────────────────────────
        egui::CollapsingHeader::new("📄 Manual Files")
            .default_open(!self.pile_enabled)
            .show(ui, |ui| {
                self.manual_files_ui(ui);
            });

        ui.add_space(8.0);

        // ── The Pile ──────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.pile_enabled, "📦 Use The Pile (EleutherAI)");
            if self.pile_enabled {
                ui.label(
                    egui::RichText::new("~825 GiB — requires Python 3 & git")
                        .weak()
                        .small(),
                );
            }
        });

        if self.pile_enabled {
            ui.add_space(4.0);
            egui::Frame::new()
                .stroke(egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    self.pile_ui(ui);
                });
        }

        ui.separator();

        // ── Sequence length ───────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Max sequence length:");
            ui.add(
                egui::Slider::new(&mut self.max_seq_len, 128usize..=8192)
                    .step_by(128.0)
                    .suffix(" tokens")
                    .logarithmic(true),
            );
        });

        ui.separator();

        // ── Tokenizer ─────────────────────────────────────────────────────
        egui::CollapsingHeader::new("🔤 Tokenizer")
            .default_open(true)
            .show(ui, |ui| {
                self.tokenizer_ui(ui);
            });
    }

    // ── Manual files sub-UI ───────────────────────────────────────────────────

    fn manual_files_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("➕ Add Files…").clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("Text / JSONL", &["txt", "jsonl"])
                    .set_title("Add training data files")
                    .pick_files()
                {
                    for p in paths {
                        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                        self.files.push(FileEntry { path: p, size_bytes: size });
                    }
                }
            }
            if ui.button("📁 Add Folder…").clicked() {
                if let Some(folder) = rfd::FileDialog::new()
                    .set_title("Add folder of training data")
                    .pick_folder()
                {
                    if let Ok(entries) = walkdir(&folder) {
                        for p in entries {
                            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                            self.files.push(FileEntry { path: p, size_bytes: size });
                        }
                    }
                }
            }
            if !self.files.is_empty() && ui.button("🗑 Clear All").clicked() {
                self.files.clear();
            }
        });

        if self.files.is_empty() {
            ui.label(
                egui::RichText::new("No files added.  Add .txt or .jsonl files.")
                    .weak()
                    .italics(),
            );
        } else {
            let total: u64 = self.files.iter().map(|f| f.size_bytes).sum();
            ui.label(format!("{} files — {}", self.files.len(), fmt_bytes(total)));

            egui::ScrollArea::vertical()
                .id_salt("file_list")
                .max_height(180.0)
                .show(ui, |ui| {
                    let mut remove: Option<usize> = None;
                    for (i, entry) in self.files.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    entry.path.file_name().unwrap_or_default().to_string_lossy(),
                                )
                                .monospace(),
                            );
                            ui.label(egui::RichText::new(fmt_bytes(entry.size_bytes)).weak());
                            if ui.small_button("✖").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove {
                        self.files.remove(i);
                    }
                });
        }
    }

    // ── The Pile sub-UI ───────────────────────────────────────────────────────

    fn pile_ui(&mut self, ui: &mut egui::Ui) {
        let running = self.pile_state.is_running;
        let components = pile_components();

        // ── Config row ────────────────────────────────────────────────────
        egui::Grid::new("pile_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                // Target directory
                ui.label("Data directory:");
                ui.horizontal(|ui| {
                    let dir_str = self.pile_config.target_dir.to_string_lossy().to_string();
                    let mut dir_edit = dir_str.clone();
                    let resp = ui.add_enabled(
                        !running,
                        egui::TextEdit::singleline(&mut dir_edit).desired_width(300.0),
                    );
                    if resp.changed() {
                        self.pile_config.target_dir = PathBuf::from(&dir_edit);
                    }
                    if ui.add_enabled(!running, egui::Button::new("📁")).clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .set_title("Select data directory")
                            .pick_folder()
                        {
                            self.pile_config.target_dir = p;
                        }
                    }
                });
                ui.end_row();

                // Python command
                ui.label("Python command:");
                ui.add_enabled(
                    !running,
                    egui::TextEdit::singleline(&mut self.pile_config.python_cmd)
                        .desired_width(160.0),
                );
                ui.end_row();

                // Component selector
                ui.label("Component:");
                egui::ComboBox::from_id_salt("pile_component")
                    .selected_text(components[self.pile_component_idx].label)
                    .width(300.0)
                    .show_ui(ui, |ui| {
                        for (i, c) in components.iter().enumerate() {
                            let label = format!(
                                "{} ({:.0} GiB)",
                                c.label, c.approx_size_gib
                            );
                            ui.selectable_value(&mut self.pile_component_idx, i, label);
                        }
                    });
                ui.end_row();

                // Interleave output
                ui.label("Interleave output files:");
                ui.add_enabled(
                    !running,
                    egui::DragValue::new(&mut self.pile_config.interleave_output)
                        .range(1u32..=128)
                        .speed(1.0),
                );
                ui.end_row();

                // Skip-if-exists checkbox
                ui.label("Skip clone if repo exists:");
                ui.add_enabled(
                    !running,
                    egui::Checkbox::without_text(&mut self.pile_config.skip_clone_if_exists),
                );
                ui.end_row();
            });

        ui.add_space(6.0);

        // Size warning
        let sel = &components[self.pile_component_idx];
        ui.label(
            egui::RichText::new(format!(
                "⚠  Approximate download size: {:.0} GiB.  Ensure sufficient disk space.",
                sel.approx_size_gib
            ))
            .color(egui::Color32::YELLOW)
            .small(),
        );

        ui.add_space(6.0);

        // ── Control buttons ───────────────────────────────────────────────
        ui.horizontal(|ui| {
            if running {
                if ui.button("⏹ Cancel").clicked() {
                    // Drop the receiver — the background thread's next send
                    // will fail and it will stop gracefully.
                    self.pile_state.receiver = None;
                    self.pile_state.is_running = false;
                    self.pile_state.log.push("⏹  Build cancelled by user.".into());
                }
            } else {
                let label = if self.pile_state.finished {
                    "🔄 Rebuild"
                } else {
                    "▶ Start Build"
                };
                if ui.button(label).clicked() {
                    let mut cfg = self.pile_config.clone();
                    cfg.component = components[self.pile_component_idx].id.to_owned();
                    self.pile_state.start(cfg);
                }
            }

            if self.pile_state.finished {
                ui.label(
                    egui::RichText::new("✅ Build complete")
                        .color(egui::Color32::GREEN)
                        .strong(),
                );
            } else if let Some(err) = &self.pile_state.error.clone() {
                ui.label(
                    egui::RichText::new(format!("❌ {err}"))
                        .color(egui::Color32::RED)
                        .small(),
                );
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.pile_log_autoscroll, "Auto-scroll");
            });
        });

        ui.add_space(4.0);

        // ── Progress bar + phase label ────────────────────────────────────
        if running || self.pile_state.progress > 0.0 {
            ui.add(
                egui::ProgressBar::new(self.pile_state.progress)
                    .desired_width(ui.available_width())
                    .animate(running)
                    .text(format!(
                        "{:.0}%  {}",
                        self.pile_state.progress * 100.0,
                        self.pile_state.phase,
                    )),
            );
            ui.add_space(4.0);
        } else {
            ui.label(
                egui::RichText::new("Press ▶ Start Build to begin downloading The Pile.")
                    .weak()
                    .italics(),
            );
            ui.add_space(4.0);
        }

        // ── Scrolling log ─────────────────────────────────────────────────
        if !self.pile_state.log.is_empty() || running {
            ui.label(
                egui::RichText::new(format!(
                    "Build log ({} lines):",
                    self.pile_state.log.len()
                ))
                .small()
                .weak(),
            );

            let log_ref = &self.pile_state.log;
            let autoscroll = self.pile_log_autoscroll;

            let scroll_area = egui::ScrollArea::vertical()
                .id_salt("pile_log")
                .max_height(280.0)
                .stick_to_bottom(autoscroll)
                .auto_shrink([false, false]);

            scroll_area.show(ui, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(18, 18, 20))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        // Render last 500 lines for performance.
                        let start = log_ref.len().saturating_sub(500);
                        for line in &log_ref[start..] {
                            let color = if line.starts_with("❌") || line.contains("[err]") {
                                egui::Color32::from_rgb(255, 100, 100)
                            } else if line.starts_with("✔") || line.starts_with("✅") {
                                egui::Color32::from_rgb(100, 220, 100)
                            } else if line.starts_with("⚠") {
                                egui::Color32::YELLOW
                            } else if line.starts_with("ℹ") {
                                egui::Color32::from_rgb(100, 180, 255)
                            } else {
                                egui::Color32::from_rgb(200, 200, 200)
                            };
                            ui.label(
                                egui::RichText::new(line)
                                    .monospace()
                                    .size(11.0)
                                    .color(color),
                            );
                        }
                        if running {
                            ui.label(
                                egui::RichText::new("▋")
                                    .monospace()
                                    .size(11.0)
                                    .color(egui::Color32::LIGHT_GRAY),
                            );
                        }
                    });
            });
        }
    }

    // ── Tokenizer sub-UI ──────────────────────────────────────────────────────

    fn tokenizer_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Tokenizer:");
            match &self.tokenizer_path {
                Some(p) => ui.label(
                    egui::RichText::new(
                        p.file_name().unwrap_or_default().to_string_lossy(),
                    )
                    .color(egui::Color32::GREEN),
                ),
                None => ui.label(egui::RichText::new("None loaded").weak()),
            };
            if ui.button("📂 Load…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Tokenizer", &["json"])
                    .set_title("Load tokenizer.json")
                    .pick_file()
                {
                    self.tokenizer_path = Some(p);
                }
            }
        });

        ui.add_space(4.0);
        ui.label("Train new tokenizer from dataset:");
        ui.horizontal(|ui| {
            ui.label("Vocab size:");
            ui.add(
                egui::DragValue::new(&mut self.vocab_size_input)
                    .range(1000..=128000)
                    .speed(100.0),
            );
            let can_train = !self.files.is_empty();
            if ui
                .add_enabled(can_train, egui::Button::new("🏋 Train Tokenizer"))
                .clicked()
            {
                self.status =
                    "Tokenizer training queued (run from CLI or start training)".into();
            }
        });
        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).weak().italics());
        }
    }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

fn walkdir(dir: &PathBuf) -> anyhow::Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            results.extend(walkdir(&path).unwrap_or_default());
        } else if let Some(ext) = path.extension() {
            if ext == "txt" || ext == "jsonl" {
                results.push(path);
            }
        }
    }
    Ok(results)
}

fn fmt_bytes(b: u64) -> String {
    if b == 0 {
        return "0 B".into();
    }
    let (val, unit) = if b >= 1 << 30 {
        (b as f64 / (1u64 << 30) as f64, "GiB")
    } else if b >= 1 << 20 {
        (b as f64 / (1u64 << 20) as f64, "MiB")
    } else {
        (b as f64 / 1024.0, "KiB")
    };
    format!("{val:.1} {unit}")
}
