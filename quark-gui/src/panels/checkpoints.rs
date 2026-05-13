#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use std::time::SystemTime;

struct CkptEntry {
    path: PathBuf,
    size_bytes: u64,
    modified: Option<SystemTime>,
}

pub struct CheckpointsPanel {
    dir: Option<PathBuf>,
    entries: Vec<CkptEntry>,
    loaded: Option<PathBuf>,
    confirm_delete: Option<usize>,
    status: String,
}

impl Default for CheckpointsPanel {
    fn default() -> Self {
        Self {
            dir: None,
            entries: vec![],
            loaded: None,
            confirm_delete: None,
            status: String::new(),
        }
    }
}

impl CheckpointsPanel {
    pub fn loaded_path(&self) -> Option<&PathBuf> {
        self.loaded.as_ref()
    }

    fn scan(&mut self) {
        self.entries.clear();
        if let Some(dir) = &self.dir {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for entry in rd.flatten() {
                    let p = entry.path();
                    if p.extension().map_or(false, |e| e == "safetensors") {
                        let meta = std::fs::metadata(&p).ok();
                        self.entries.push(CkptEntry {
                            size_bytes: meta.as_ref().map(|m| m.len()).unwrap_or(0),
                            modified: meta.and_then(|m| m.modified().ok()),
                            path: p,
                        });
                    }
                }
                self.entries.sort_by(|a, b| b.modified.cmp(&a.modified));
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("💾 Checkpoints");
        ui.separator();

        // Directory picker
        ui.horizontal(|ui| {
            ui.label("Directory:");
            match &self.dir {
                Some(d) => {
                    ui.label(egui::RichText::new(d.to_string_lossy()).monospace());
                }
                None => {
                    ui.label(egui::RichText::new("Not set").weak());
                }
            };
            if ui.button("📁 Browse…").clicked() {
                if let Some(d) = rfd::FileDialog::new()
                    .set_title("Checkpoint directory")
                    .pick_folder()
                {
                    self.dir = Some(d);
                    self.scan();
                }
            }
            if ui.button("🔄 Scan").clicked() {
                self.scan();
            }
        });

        if let Some(loaded) = &self.loaded {
            ui.label(
                egui::RichText::new(format!(
                    "✓ Loaded: {}",
                    loaded.file_name().unwrap_or_default().to_string_lossy()
                ))
                .color(egui::Color32::GREEN),
            );
        }

        if !self.status.is_empty() {
            ui.label(egui::RichText::new(&self.status).weak().italics());
        }

        ui.separator();

        if self.entries.is_empty() {
            ui.label(
                egui::RichText::new("No .safetensors files found.")
                    .weak()
                    .italics(),
            );
            return;
        }

        let mut to_delete: Option<usize> = None;
        let mut to_load: Option<usize> = None;

        egui::ScrollArea::vertical()
            .id_salt("ckpt_scroll")
            .show(ui, |ui| {
                for (i, entry) in self.entries.iter().enumerate() {
                    let name = entry.path.file_name().unwrap_or_default().to_string_lossy();
                    let is_loaded = self.loaded.as_ref() == Some(&entry.path);

                    egui::Frame::new()
                        .inner_margin(egui::Margin::same(6))
                        .corner_radius(4)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if is_loaded {
                                    ui.label(
                                        egui::RichText::new("●").color(egui::Color32::GREEN),
                                    );
                                }
                                ui.label(
                                    egui::RichText::new(name.as_ref()).strong().monospace(),
                                );
                                ui.label(egui::RichText::new(fmt_bytes(entry.size_bytes)).weak());

                                if let Some(modified) = entry.modified {
                                    if let Ok(dur) = SystemTime::now().duration_since(modified) {
                                        let secs = dur.as_secs();
                                        let age = if secs < 60 {
                                            format!("{secs}s ago")
                                        } else if secs < 3600 {
                                            format!("{}m ago", secs / 60)
                                        } else {
                                            format!("{}h ago", secs / 3600)
                                        };
                                        ui.label(egui::RichText::new(age).weak());
                                    }
                                }

                                if ui.button("📥 Load").clicked() {
                                    to_load = Some(i);
                                }

                                if ui.button("📤 Export…").clicked() {
                                    if let Some(dst) = rfd::FileDialog::new()
                                        .add_filter("safetensors", &["safetensors"])
                                        .set_file_name(name.as_ref())
                                        .save_file()
                                    {
                                        let _ = std::fs::copy(&entry.path, &dst);
                                        self.status =
                                            format!("Exported to {}", dst.display());
                                    }
                                }

                                if self.confirm_delete == Some(i) {
                                    ui.label(
                                        egui::RichText::new("Delete?")
                                            .color(egui::Color32::RED),
                                    );
                                    if ui.button("✓ Yes").clicked() {
                                        to_delete = Some(i);
                                        self.confirm_delete = None;
                                    }
                                    if ui.button("✗ No").clicked() {
                                        self.confirm_delete = None;
                                    }
                                } else if ui.button("🗑").clicked() {
                                    self.confirm_delete = Some(i);
                                }
                            });
                        });
                }
            });

        if let Some(i) = to_load {
            self.loaded = Some(self.entries[i].path.clone());
            self.status = format!(
                "Loaded {}",
                self.entries[i]
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
        }
        if let Some(i) = to_delete {
            let p = self.entries[i].path.clone();
            let _ = std::fs::remove_file(&p);
            self.status = format!(
                "Deleted {}",
                p.file_name().unwrap_or_default().to_string_lossy()
            );
            self.entries.remove(i);
        }
    }
}

fn fmt_bytes(b: u64) -> String {
    if b == 0 {
        return "0 B".into();
    }
    let gib = b as f64 / (1u64 << 30) as f64;
    let mib = b as f64 / (1u64 << 20) as f64;
    if gib >= 1.0 {
        format!("{gib:.2} GiB")
    } else {
        format!("{mib:.0} MiB")
    }
}
