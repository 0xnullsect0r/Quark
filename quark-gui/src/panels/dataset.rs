#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;

struct FileEntry {
    path: PathBuf,
    size_bytes: u64,
}

pub struct DatasetPanel {
    files: Vec<FileEntry>,
    max_seq_len: usize,
    tokenizer_path: Option<PathBuf>,
    vocab_size_input: usize,
    status: String,
}

impl Default for DatasetPanel {
    fn default() -> Self {
        Self {
            files: vec![],
            max_seq_len: 2048,
            tokenizer_path: None,
            vocab_size_input: 32000,
            status: String::new(),
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

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("📂 Dataset");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("➕ Add Files…").clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("Text / JSONL", &["txt", "jsonl"])
                    .set_title("Add training data files")
                    .pick_files()
                {
                    for p in paths {
                        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                        self.files.push(FileEntry {
                            path: p,
                            size_bytes: size,
                        });
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
                            let size =
                                std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                            self.files.push(FileEntry {
                                path: p,
                                size_bytes: size,
                            });
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
                egui::RichText::new("No files added. Add .txt or .jsonl files.")
                    .weak()
                    .italics(),
            );
        } else {
            let total_bytes: u64 = self.files.iter().map(|f| f.size_bytes).sum();
            ui.label(format!(
                "{} files  —  {} total",
                self.files.len(),
                fmt_bytes(total_bytes)
            ));

            egui::ScrollArea::vertical()
                .id_salt("file_list")
                .max_height(200.0)
                .show(ui, |ui| {
                    let mut to_remove: Option<usize> = None;
                    for (i, entry) in self.files.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    entry
                                        .path
                                        .file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy(),
                                )
                                .monospace(),
                            );
                            ui.label(egui::RichText::new(fmt_bytes(entry.size_bytes)).weak());
                            if ui.small_button("✖").clicked() {
                                to_remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = to_remove {
                        self.files.remove(i);
                    }
                });
        }

        ui.separator();

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

        egui::CollapsingHeader::new("🔤 Tokenizer")
            .default_open(true)
            .show(ui, |ui| {
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
            });
    }
}

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
    let gib = b as f64 / (1u64 << 30) as f64;
    let mib = b as f64 / (1u64 << 20) as f64;
    let kib = b as f64 / 1024.0;
    if gib >= 1.0 {
        format!("{gib:.2} GiB")
    } else if mib >= 1.0 {
        format!("{mib:.1} MiB")
    } else {
        format!("{kib:.0} KiB")
    }
}
