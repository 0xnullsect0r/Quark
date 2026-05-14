#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use quark_core::data::{
    detect_python, hf_datasets, start_hf_build, HfConfig, HfDataset, HfDatasetCategory,
    HfMessage,
};

// ─── File list entry ──────────────────────────────────────────────────────────

struct FileEntry {
    path: PathBuf,
    size_bytes: u64,
}

// ─── HF download state ────────────────────────────────────────────────────────

struct HfState {
    /// Received log lines (capped at MAX_LOG_LINES).
    log: Vec<String>,
    progress: f32,
    phase: String,
    /// Download speed in bytes/s (from SPEED: protocol line).
    speed_bps: f32,
    /// Cumulative bytes downloaded.
    bytes_downloaded: u64,
    /// Estimated total bytes across all shards.
    bytes_total: u64,
    /// Basename of the file currently being downloaded.
    current_file: String,
    is_running: bool,
    finished: bool,
    error: Option<String>,
    receiver: Option<Receiver<HfMessage>>,
}

const MAX_LOG_LINES: usize = 2000;

impl Default for HfState {
    fn default() -> Self {
        Self {
            log: Vec::new(),
            progress: 0.0,
            phase: String::new(),
            speed_bps: 0.0,
            bytes_downloaded: 0,
            bytes_total: 0,
            current_file: String::new(),
            is_running: false,
            finished: false,
            error: None,
            receiver: None,
        }
    }
}

impl HfState {
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
                        HfMessage::Log(s) => {
                            self.log.push(s);
                            if self.log.len() > MAX_LOG_LINES {
                                self.log.drain(0..MAX_LOG_LINES / 4);
                            }
                        }
                        HfMessage::Progress(p) => self.progress = p,
                        HfMessage::Phase(s) => self.phase = s,
                        HfMessage::Speed(bps) => self.speed_bps = bps,
                        HfMessage::ByteProgress { downloaded, total } => {
                            self.bytes_downloaded = downloaded;
                            if total > 0 {
                                self.bytes_total = total;
                            }
                        }
                        HfMessage::CurrentFile(f) => self.current_file = f,
                        HfMessage::Done => {
                            self.is_running = false;
                            self.finished = true;
                            self.speed_bps = 0.0;
                        }
                        HfMessage::Error(e) => {
                            self.is_running = false;
                            self.error = Some(e);
                            self.speed_bps = 0.0;
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

    fn start(&mut self, cfg: HfConfig) {
        self.log.clear();
        self.progress = 0.0;
        self.phase = "Starting…".into();
        self.speed_bps = 0.0;
        self.bytes_downloaded = 0;
        self.bytes_total = 0;
        self.current_file = String::new();
        self.error = None;
        self.finished = false;
        self.is_running = true;
        self.receiver = Some(start_hf_build(cfg));
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

    // ── HuggingFace datasets section ──────────────────────────────────────
    hf_enabled: bool,
    hf_config: HfConfig,
    /// Indices into `hf_datasets()` that are currently checked.
    hf_selected: std::collections::HashSet<usize>,
    hf_state: HfState,
    /// Whether to auto-scroll the log.
    hf_log_autoscroll: bool,
}

impl Default for DatasetPanel {
    fn default() -> Self {
        // Pre-select a reasonable starter set.
        let defaults: std::collections::HashSet<usize> = hf_datasets()
            .iter()
            .enumerate()
            .filter(|(_, d)| {
                matches!(
                    d.id,
                    "github_code"
                        | "wikipedia_en"
                        | "scientific_papers"
                        | "ultrachat"
                        | "openhermes"
                )
            })
            .map(|(i, _)| i)
            .collect();
        Self {
            files: vec![],
            max_seq_len: 2048,
            tokenizer_path: None,
            vocab_size_input: 32000,
            status: String::new(),
            hf_enabled: false,
            hf_config: HfConfig::default(),
            hf_selected: defaults,
            hf_state: HfState::default(),
            hf_log_autoscroll: true,
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
    /// Returns the JSONL datasets directory if a successful download exists.
    pub fn hf_output_dir(&self) -> Option<PathBuf> {
        if self.hf_state.finished {
            Some(self.hf_config.target_dir.join("datasets"))
        } else {
            None
        }
    }

    /// Must be called every frame so the live log updates.
    pub fn update(&mut self, ctx: &egui::Context) {
        if self.hf_state.poll() {
            ctx.request_repaint();
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("📂 Dataset");
        ui.separator();

        // ── Manual files ──────────────────────────────────────────────────
        egui::CollapsingHeader::new("📄 Manual Files")
            .default_open(!self.hf_enabled)
            .show(ui, |ui| {
                self.manual_files_ui(ui);
            });

        ui.add_space(8.0);

        // ── HuggingFace Datasets ──────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.hf_enabled, "🤗 HuggingFace Datasets");
            if self.hf_enabled {
                ui.label(
                    egui::RichText::new("streams directly to JSONL — no full pre-download required")
                        .weak()
                        .small(),
                );
            }
        });

        if self.hf_enabled {
            ui.add_space(4.0);
            egui::Frame::new()
                .stroke(egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    self.hf_ui(ui);
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

    // ── HuggingFace datasets sub-UI ───────────────────────────────────────────

    fn hf_ui(&mut self, ui: &mut egui::Ui) {
        let running = self.hf_state.is_running;
        let datasets = hf_datasets();

        // ── Config grid ───────────────────────────────────────────────────
        egui::Grid::new("hf_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                // Target directory
                ui.label("Data directory:");
                ui.horizontal(|ui| {
                    let dir_str = self.hf_config.target_dir.to_string_lossy().to_string();
                    let mut dir_edit = dir_str.clone();
                    let resp = ui.add_enabled(
                        !running,
                        egui::TextEdit::singleline(&mut dir_edit).desired_width(300.0),
                    );
                    if resp.changed() {
                        self.hf_config.target_dir = std::path::PathBuf::from(&dir_edit);
                    }
                    if ui.add_enabled(!running, egui::Button::new("📁")).clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .set_title("Select data directory")
                            .pick_folder()
                        {
                            self.hf_config.target_dir = p;
                        }
                    }
                });
                ui.end_row();

                // Python command
                ui.label("Python command:");
                ui.add_enabled(
                    !running,
                    egui::TextEdit::singleline(&mut self.hf_config.python_cmd)
                        .desired_width(160.0),
                );
                ui.end_row();

                // Max GB per dataset
                ui.label("Max GB per dataset:");
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        !running,
                        egui::Slider::new(&mut self.hf_config.max_gb_per_dataset, 0.0f32..=200.0)
                            .suffix(" GB")
                            .step_by(1.0),
                    );
                    if self.hf_config.max_gb_per_dataset == 0.0 {
                        ui.label(egui::RichText::new("(unlimited)").weak().small());
                    }
                });
                ui.end_row();

                // HuggingFace token
                ui.label("HF Token (optional):");
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        !running,
                        egui::TextEdit::singleline(&mut self.hf_config.hf_token)
                            .password(true)
                            .hint_text("hf_xxxxxxxxxxxxxxxx — required for 🔑 datasets")
                            .desired_width(300.0),
                    );
                    ui.label(
                        egui::RichText::new("not saved to disk")
                            .weak()
                            .small()
                            .italics(),
                    );
                });
                ui.end_row();

                // Parallel workers (download accelerator)
                ui.label("Parallel connections:");
                ui.horizontal(|ui| {
                    let mut w = self.hf_config.parallel_workers as u32;
                    ui.add_enabled(
                        !running,
                        egui::Slider::new(&mut w, 1u32..=20)
                            .suffix(" workers")
                            .step_by(1.0),
                    );
                    self.hf_config.parallel_workers = w as u8;
                    ui.label(
                        egui::RichText::new(format!(
                            "splits each file into {} chunks downloaded simultaneously",
                            self.hf_config.parallel_workers
                        ))
                        .weak()
                        .small(),
                    );
                });
                ui.end_row();
            });

        ui.add_space(8.0);

        // ── Dataset checkboxes grouped by category ────────────────────────
        let categories = [
            HfDatasetCategory::Code,
            HfDatasetCategory::Knowledge,
            HfDatasetCategory::Instructions,
        ];

        for category in &categories {
            let group: Vec<(usize, &HfDataset)> = datasets
                .iter()
                .enumerate()
                .filter(|(_, d)| &d.category == category)
                .collect();

            let group_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let all_in_group = group_indices.iter().all(|i| self.hf_selected.contains(i));
            let none_in_group = group_indices.iter().all(|i| !self.hf_selected.contains(i));

            egui::CollapsingHeader::new(
                egui::RichText::new(category.label()).strong(),
            )
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!running && !all_in_group, egui::Button::new("Select All"))
                        .clicked()
                    {
                        for i in &group_indices {
                            self.hf_selected.insert(*i);
                        }
                    }
                    if ui
                        .add_enabled(!running && !none_in_group, egui::Button::new("Select None"))
                        .clicked()
                    {
                        for i in &group_indices {
                            self.hf_selected.remove(i);
                        }
                    }
                    let group_gib: f32 = group_indices
                        .iter()
                        .filter(|i| self.hf_selected.contains(i))
                        .map(|&i| datasets[i].approx_size_gib)
                        .sum();
                    if group_gib > 0.0 {
                        let limit = self.hf_config.max_gb_per_dataset;
                        let effective = if limit > 0.0 {
                            (limit * group_indices
                                .iter()
                                .filter(|i| self.hf_selected.contains(i))
                                .count() as f32)
                                .min(group_gib)
                        } else {
                            group_gib
                        };
                        let label = if limit > 0.0 {
                            format!("≈ {effective:.0} GB (capped at {limit:.0} GB/dataset)")
                        } else {
                            format!("≈ {group_gib:.0} GB total")
                        };
                        ui.label(egui::RichText::new(label).weak().small());
                    }
                });

                ui.add_space(2.0);

                for (i, ds) in &group {
                    let mut checked = self.hf_selected.contains(i);
                    let label = format!(
                        "{}  (~{:.0} GB)",
                        ds.label,
                        ds.approx_size_gib
                    );
                    let resp = ui.add_enabled(!running, egui::Checkbox::new(&mut checked, label));
                    if resp.changed() {
                        if checked {
                            self.hf_selected.insert(*i);
                        } else {
                            self.hf_selected.remove(i);
                        }
                    }
                }
            });

            ui.add_space(4.0);
        }

        // ── Warnings ──────────────────────────────────────────────────────
        let needs_token = self
            .hf_selected
            .iter()
            .any(|&i| datasets[i].hf_token_required);
        if needs_token && self.hf_config.hf_token.is_empty() {
            ui.label(
                egui::RichText::new(
                    "⚠  One or more selected datasets require a HuggingFace token.  \
                     Enter your token above or those datasets will be skipped.",
                )
                .color(egui::Color32::YELLOW)
                .small(),
            );
        }

        if self.hf_selected.is_empty() {
            ui.label(
                egui::RichText::new("⚠  No datasets selected.")
                    .color(egui::Color32::YELLOW)
                    .small(),
            );
        }

        ui.add_space(6.0);

        // ── Control buttons ───────────────────────────────────────────────
        ui.horizontal(|ui| {
            if running {
                if ui.button("⏹ Cancel").clicked() {
                    self.hf_state.receiver = None;
                    self.hf_state.is_running = false;
                    self.hf_state.log.push("⏹  Download cancelled by user.".into());
                }
            } else {
                let label = if self.hf_state.finished { "🔄 Re-download" } else { "▶ Start Download" };
                let can_start = !self.hf_selected.is_empty();
                if ui.add_enabled(can_start, egui::Button::new(label)).clicked() {
                    let mut cfg = self.hf_config.clone();
                    let mut sorted: Vec<usize> = self.hf_selected.iter().copied().collect();
                    sorted.sort_unstable();
                    cfg.selected_ids = sorted
                        .iter()
                        .map(|&i| datasets[i].id.to_owned())
                        .collect();
                    self.hf_state.start(cfg);
                }
            }

            if self.hf_state.finished {
                ui.label(
                    egui::RichText::new("✅ Download complete")
                        .color(egui::Color32::GREEN)
                        .strong(),
                );
            } else if let Some(err) = &self.hf_state.error.clone() {
                ui.label(
                    egui::RichText::new(format!("❌ {err}"))
                        .color(egui::Color32::RED)
                        .small(),
                );
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.hf_log_autoscroll, "Auto-scroll");
            });
        });

        ui.add_space(4.0);

        // ── Progress bar + phase label ────────────────────────────────────
        if running || self.hf_state.progress > 0.0 {
            ui.add(
                egui::ProgressBar::new(self.hf_state.progress)
                    .desired_width(ui.available_width())
                    .animate(running)
                    .text(format!(
                        "{:.0}%  {}",
                        self.hf_state.progress * 100.0,
                        self.hf_state.phase,
                    )),
            );

            // ── Speed / bytes / ETA / current file ────────────────────────
            if running || self.hf_state.bytes_downloaded > 0 {
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    // Speed badge
                    let speed = self.hf_state.speed_bps;
                    let speed_color = if speed >= 5.0 * 1024.0 * 1024.0 {
                        egui::Color32::from_rgb(80, 220, 80)
                    } else if speed >= 1024.0 * 1024.0 {
                        egui::Color32::YELLOW
                    } else {
                        egui::Color32::from_rgb(255, 140, 0)
                    };
                    ui.label(
                        egui::RichText::new(format!("⬇ {}", fmt_speed(speed)))
                            .monospace()
                            .strong()
                            .color(speed_color),
                    );

                    ui.separator();

                    // Bytes counter
                    let dl = self.hf_state.bytes_downloaded;
                    let tot = self.hf_state.bytes_total;
                    if tot > 0 {
                        ui.label(
                            egui::RichText::new(format!(
                                "{} / {}",
                                fmt_bytes(dl),
                                fmt_bytes(tot)
                            ))
                            .monospace(),
                        );

                        // ETA
                        if speed > 0.0 && dl < tot {
                            let remaining = (tot - dl) as f32;
                            let eta_secs = remaining / speed;
                            ui.separator();
                            ui.label(
                                egui::RichText::new(format!("ETA: {}", fmt_eta(eta_secs)))
                                    .weak(),
                            );
                        }
                    } else if dl > 0 {
                        ui.label(
                            egui::RichText::new(format!("{} downloaded", fmt_bytes(dl)))
                                .monospace(),
                        );
                    }

                    // Current file
                    if running && !self.hf_state.current_file.is_empty() {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("📄 {}", self.hf_state.current_file))
                                .small()
                                .weak(),
                        );
                    }
                });
            }

            ui.add_space(4.0);
        } else {
            ui.label(
                egui::RichText::new("Press ▶ Start Download to begin.")
                    .weak()
                    .italics(),
            );
            ui.add_space(4.0);
        }

        // ── Scrolling log ─────────────────────────────────────────────────
        if !self.hf_state.log.is_empty() || running {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "Download log ({} lines):",
                        self.hf_state.log.len()
                    ))
                    .small()
                    .weak(),
                );
                if ui.small_button("📋 Copy All").on_hover_text("Copy entire log to clipboard").clicked() {
                    let full = self.hf_state.log.join("\n");
                    ui.ctx().copy_text(full);
                }
            });

            let log_ref = &self.hf_state.log;
            let autoscroll = self.hf_log_autoscroll;

            egui::ScrollArea::vertical()
                .id_salt("hf_log")
                .max_height(280.0)
                .stick_to_bottom(autoscroll)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgb(18, 18, 20))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
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
                            if running {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("▋")
                                            .monospace()
                                            .size(11.0)
                                            .color(egui::Color32::LIGHT_GRAY),
                                    )
                                    .selectable(false),
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

fn fmt_speed(bps: f32) -> String {
    if bps >= 1024.0 * 1024.0 {
        format!("{:.1} MB/s", bps / (1024.0 * 1024.0))
    } else if bps >= 1024.0 {
        format!("{:.0} KB/s", bps / 1024.0)
    } else {
        format!("{bps:.0} B/s")
    }
}

fn fmt_eta(secs: f32) -> String {
    if secs > 3600.0 {
        format!("~{:.0}h {:.0}m", secs / 3600.0, (secs % 3600.0) / 60.0)
    } else if secs > 60.0 {
        format!("~{:.0}m {:.0}s", secs / 60.0, secs % 60.0)
    } else {
        format!("~{secs:.0}s")
    }
}
