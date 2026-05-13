#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use quark_core::memory::budget::HardwareBudget;
use quark_core::memory::tier::TierConfig;

pub struct SettingsPanel {
    budget: HardwareBudget,
    tier: TierConfig,
    settings_path: PathBuf,
    status_msg: Option<(std::time::Instant, String, bool)>,
}

impl Default for SettingsPanel {
    fn default() -> Self {
        let settings_path = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("quark")
            .join("settings.toml");
        let tier = if settings_path.exists() {
            TierConfig::load(&settings_path).unwrap_or_default()
        } else {
            TierConfig::default()
        };
        Self {
            budget: HardwareBudget::detect(),
            tier,
            settings_path,
            status_msg: None,
        }
    }
}

impl SettingsPanel {
    pub fn tier_config(&self) -> &TierConfig {
        &self.tier
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("⚙ Settings");
        ui.separator();

        egui::CollapsingHeader::new("🖥 Detected Hardware")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("hw_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("VRAM total");
                        ui.label(if self.budget.vram_total_bytes == 0 {
                            "N/A".into()
                        } else {
                            fmt_bytes(self.budget.vram_total_bytes)
                        });
                        ui.end_row();

                        ui.label("VRAM free");
                        ui.label(if self.budget.vram_free_bytes == 0 {
                            "N/A".into()
                        } else {
                            fmt_bytes(self.budget.vram_free_bytes)
                        });
                        ui.end_row();

                        ui.label("RAM total");
                        ui.label(fmt_bytes(self.budget.ram_total_bytes));
                        ui.end_row();

                        ui.label("RAM free");
                        ui.label(fmt_bytes(self.budget.ram_free_bytes));
                        ui.end_row();

                        ui.label("CPU cores");
                        ui.label(self.budget.cpu_logical_cores.to_string());
                        ui.end_row();

                        ui.label("Disk free");
                        ui.label(fmt_bytes(self.budget.disk_free_bytes));
                        ui.end_row();
                    });

                if ui.button("🔄 Refresh").clicked() {
                    self.budget = HardwareBudget::detect();
                }
            });

        ui.add_space(8.0);

        egui::CollapsingHeader::new("📊 Resource Limits")
            .default_open(true)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Quark will not exceed these fractions of available resources.",
                    )
                    .weak()
                    .italics(),
                );
                ui.add_space(6.0);

                let vram_abs = self.tier.vram_limit_bytes(&self.budget);
                let ram_abs = self.tier.ram_limit_bytes(&self.budget);
                let cpu_cnt = self.tier.cpu_thread_count(&self.budget);

                egui::Grid::new("limits_grid")
                    .num_columns(3)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("🟦 VRAM limit");
                        ui.add(
                            egui::Slider::new(&mut self.tier.vram_limit_frac, 0.10f32..=0.95f32)
                                .step_by(0.05)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                        );
                        ui.label(egui::RichText::new(if self.budget.vram_total_bytes == 0 {
                            "N/A".into()
                        } else {
                            format!("= {}", fmt_bytes(vram_abs))
                        }).weak());
                        ui.end_row();

                        ui.label("🟩 RAM limit");
                        ui.add(
                            egui::Slider::new(&mut self.tier.ram_limit_frac, 0.10f32..=0.95f32)
                                .step_by(0.05)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                        );
                        ui.label(egui::RichText::new(format!("= {}", fmt_bytes(ram_abs))).weak());
                        ui.end_row();

                        ui.label("🟨 CPU threads");
                        ui.add(
                            egui::Slider::new(&mut self.tier.cpu_thread_frac, 0.05f32..=1.0f32)
                                .step_by(0.05)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                        );
                        ui.label(
                            egui::RichText::new(format!("= {} cores", cpu_cnt)).weak(),
                        );
                        ui.end_row();

                        ui.label("🟥 GPU compute");
                        ui.add(
                            egui::Slider::new(&mut self.tier.gpu_compute_frac, 0.10f32..=1.0f32)
                                .step_by(0.05)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                        );
                        ui.label(
                            egui::RichText::new("(enforced by backend)").weak(),
                        );
                        ui.end_row();
                    });

                ui.add_space(6.0);
                ui.label("💾 Disk offload path:");
                ui.horizontal(|ui| {
                    let mut path_str =
                        self.tier.disk_offload_path.to_string_lossy().to_string();
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut path_str).desired_width(300.0),
                    );
                    if resp.changed() {
                        self.tier.disk_offload_path = PathBuf::from(&path_str);
                    }
                    if ui.button("📂").clicked() {
                        if let Some(folder) = rfd::FileDialog::new()
                            .set_title("Disk offload folder")
                            .pick_folder()
                        {
                            self.tier.disk_offload_path = folder;
                        }
                    }
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("💾 Save Settings").clicked() {
                        let result: anyhow::Result<()> = (|| {
                            if let Some(p) = self.settings_path.parent() {
                                std::fs::create_dir_all(p)?;
                            }
                            self.tier.save(&self.settings_path)
                        })();
                        match result {
                            Ok(()) => {
                                self.status_msg = Some((
                                    std::time::Instant::now(),
                                    "Settings saved.".into(),
                                    true,
                                ))
                            }
                            Err(e) => {
                                self.status_msg = Some((
                                    std::time::Instant::now(),
                                    format!("Save failed: {e}"),
                                    false,
                                ))
                            }
                        }
                    }
                    if ui.button("↺ Reset to Defaults").clicked() {
                        self.tier = TierConfig::default();
                    }
                });

                if let Some((t, msg, ok)) = &self.status_msg {
                    if t.elapsed().as_secs() < 3 {
                        let color = if *ok {
                            egui::Color32::from_rgb(80, 200, 100)
                        } else {
                            egui::Color32::RED
                        };
                        ui.label(egui::RichText::new(msg.as_str()).color(color));
                    } else {
                        self.status_msg = None;
                    }
                }
            });

        ui.add_space(8.0);

        egui::CollapsingHeader::new("🔧 Compute Backend")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Backend is selected at build time via Cargo features:",
                    )
                    .weak(),
                );
                ui.add_space(4.0);
                egui::Grid::new("backend_grid")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("backend-cpu");
                        ui.label("CPU (ndarray) — all platforms");
                        ui.end_row();
                        ui.label("backend-wgpu");
                        ui.label("GPU via WGPU (Metal / Vulkan)");
                        ui.end_row();
                        ui.label("backend-cuda");
                        ui.label("NVIDIA CUDA GPU");
                        ui.end_row();
                    });
                #[cfg(feature = "backend-cuda")]
                ui.label(
                    egui::RichText::new("✓ CUDA backend active")
                        .color(egui::Color32::GREEN)
                        .strong(),
                );
                #[cfg(feature = "backend-wgpu")]
                ui.label(
                    egui::RichText::new("✓ WGPU backend active")
                        .color(egui::Color32::from_rgb(100, 180, 255))
                        .strong(),
                );
                #[cfg(feature = "backend-cpu")]
                ui.label(
                    egui::RichText::new("✓ CPU backend active")
                        .color(egui::Color32::GRAY)
                        .strong(),
                );
            });
    }
}

fn fmt_bytes(b: u64) -> String {
    if b == 0 {
        return "0 B".to_string();
    }
    let gib = b as f64 / (1u64 << 30) as f64;
    let mib = b as f64 / (1u64 << 20) as f64;
    if gib >= 1.0 {
        format!("{gib:.2} GiB")
    } else {
        format!("{mib:.0} MiB")
    }
}
