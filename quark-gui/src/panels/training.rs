#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use egui_plot::{Line, Plot, PlotPoints};
use quark_core::memory::budget::HardwareBudget;
use quark_core::memory::tier::TierConfig;
use quark_core::training::metrics::{MetricsReceiver, TrainingMetrics};
use quark_core::training::trainer::TrainerConfig;

pub struct TrainingPanel {
    pub trainer_config: TrainerConfig,
    metrics_rx: Option<MetricsReceiver>,
    latest: Option<TrainingMetrics>,
    loss_history: Vec<[f64; 2]>,
    lr_history: Vec<[f64; 2]>,
    is_running: bool,
    budget: HardwareBudget,
}

impl Default for TrainingPanel {
    fn default() -> Self {
        Self {
            trainer_config: TrainerConfig::default(),
            metrics_rx: None,
            latest: None,
            loss_history: Vec::new(),
            lr_history: Vec::new(),
            is_running: false,
            budget: HardwareBudget::detect(),
        }
    }
}

impl TrainingPanel {
    pub fn set_metrics_receiver(&mut self, rx: MetricsReceiver) {
        self.metrics_rx = Some(rx);
        self.is_running = true;
    }

    pub fn stop(&mut self) {
        self.is_running = false;
    }

    fn drain_metrics(&mut self) {
        if let Some(rx) = &mut self.metrics_rx {
            while let Ok(m) = rx.try_recv() {
                if self.loss_history.len() > 2000 {
                    self.loss_history.remove(0);
                }
                if self.lr_history.len() > 2000 {
                    self.lr_history.remove(0);
                }
                self.loss_history.push([m.step as f64, m.loss as f64]);
                self.lr_history
                    .push([m.step as f64, m.learning_rate as f64]);
                self.latest = Some(m);
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.drain_metrics();
        if self.is_running {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }

        ui.heading("🏋 Training");
        ui.separator();

        // Control buttons
        ui.horizontal(|ui| {
            if self.is_running {
                if ui
                    .button(egui::RichText::new("⏹ Stop").color(egui::Color32::RED))
                    .clicked()
                {
                    self.is_running = false;
                }
                ui.spinner();
                ui.label("Training…");
            } else {
                if ui
                    .button(
                        egui::RichText::new("▶ Start Training")
                            .color(egui::Color32::GREEN)
                            .strong(),
                    )
                    .clicked()
                {
                    self.is_running = true;
                    self.loss_history.clear();
                    self.lr_history.clear();
                }
            }
        });

        // Latest metrics banner
        if let Some(m) = &self.latest {
            ui.separator();
            egui::Grid::new("metrics_banner")
                .num_columns(6)
                .spacing([16.0, 2.0])
                .show(ui, |ui| {
                    ui.label("Step");
                    ui.label(egui::RichText::new(m.step.to_string()).strong());
                    ui.label("Loss");
                    ui.label(
                        egui::RichText::new(format!("{:.4}", m.loss))
                            .strong()
                            .color(egui::Color32::from_rgb(255, 160, 50)),
                    );
                    ui.label("LR");
                    ui.label(
                        egui::RichText::new(format!("{:.2e}", m.learning_rate)).strong(),
                    );
                    ui.end_row();
                    ui.label("tok/s");
                    ui.label(format!("{:.0}", m.tokens_per_sec));
                    ui.label("Epoch");
                    ui.label(m.epoch.to_string());
                    ui.label("ETA");
                    let eta = m.eta_secs;
                    ui.label(format!("{}h {}m", eta / 3600, (eta % 3600) / 60));
                    ui.end_row();
                });
        }

        // Memory tier bars
        if let Some(m) = &self.latest {
            ui.separator();
            ui.label(egui::RichText::new("Memory Tiers").strong());
            let vram_lim = TierConfig::default().vram_limit_bytes(&self.budget).max(1);
            let ram_lim = TierConfig::default().ram_limit_bytes(&self.budget).max(1);

            let vram_used = m.vram_used_bytes;
            let ram_used = m.ram_used_bytes;
            let disk_used = m.disk_used_bytes;

            ui.horizontal(|ui| {
                ui.label("🟦 VRAM");
                let frac = (vram_used as f64 / vram_lim as f64).min(1.0) as f32;
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(180.0)
                        .fill(tier_color(frac))
                        .text(format!(
                            "{} / {}",
                            fmt_bytes(vram_used),
                            fmt_bytes(vram_lim)
                        )),
                );
            });
            ui.horizontal(|ui| {
                ui.label("🟩 RAM  ");
                let frac = (ram_used as f64 / ram_lim as f64).min(1.0) as f32;
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(180.0)
                        .fill(tier_color(frac))
                        .text(format!(
                            "{} / {}",
                            fmt_bytes(ram_used),
                            fmt_bytes(ram_lim)
                        )),
                );
            });
            ui.horizontal(|ui| {
                ui.label("💾 Disk ");
                ui.label(format!("{} used", fmt_bytes(disk_used)));
            });
        }

        // Loss chart
        if !self.loss_history.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Loss").strong());
            Plot::new("loss_chart")
                .height(160.0)
                .allow_drag(false)
                .allow_zoom(false)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.loss_history.iter().copied().collect();
                    plot_ui.line(
                        Line::new(pts)
                            .name("loss")
                            .color(egui::Color32::from_rgb(255, 140, 50))
                            .width(1.5),
                    );
                });

            ui.label(egui::RichText::new("Learning Rate").strong());
            Plot::new("lr_chart")
                .height(100.0)
                .allow_drag(false)
                .allow_zoom(false)
                .show(ui, |plot_ui| {
                    let pts: PlotPoints = self.lr_history.iter().copied().collect();
                    plot_ui.line(
                        Line::new(pts)
                            .name("lr")
                            .color(egui::Color32::from_rgb(80, 180, 255))
                            .width(1.5),
                    );
                });
        }

        ui.separator();

        // Training config form
        egui::CollapsingHeader::new("⚙ Training Configuration")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("trainer_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Output dir");
                        let mut dir_str = self
                            .trainer_config
                            .output_dir
                            .to_string_lossy()
                            .to_string();
                        if ui.text_edit_singleline(&mut dir_str).changed() {
                            self.trainer_config.output_dir = PathBuf::from(dir_str);
                        }
                        ui.end_row();

                        ui.label("Max steps");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.max_steps)
                                .range(1..=1_000_000u64)
                                .speed(100.0),
                        );
                        ui.end_row();

                        ui.label("Batch size");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.batch_size)
                                .range(1..=256usize),
                        );
                        ui.end_row();

                        ui.label("Grad accum steps");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.grad_accum_steps)
                                .range(1..=128usize),
                        );
                        ui.end_row();

                        ui.label("Save every N steps");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.save_every_steps)
                                .range(1..=10000u64)
                                .speed(50.0),
                        );
                        ui.end_row();

                        ui.label("Mixed precision");
                        ui.checkbox(&mut self.trainer_config.mixed_precision, "bf16/f16");
                        ui.end_row();

                        ui.label("Max grad norm");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.max_grad_norm)
                                .range(0.1f32..=10.0f32)
                                .speed(0.1),
                        );
                        ui.end_row();

                        ui.label("Seed");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.seed)
                                .range(0..=u64::MAX),
                        );
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.label(egui::RichText::new("AdamW Optimizer").strong());
                egui::Grid::new("adamw_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Learning rate");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.adamw.lr)
                                .range(1e-6..=1e-2f64)
                                .speed(1e-5),
                        );
                        ui.end_row();
                        ui.label("β₁");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.adamw.beta1)
                                .range(0.5..=0.999f64)
                                .speed(0.001),
                        );
                        ui.end_row();
                        ui.label("β₂");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.adamw.beta2)
                                .range(0.5..=0.9999f64)
                                .speed(0.0001),
                        );
                        ui.end_row();
                        ui.label("Weight decay");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.adamw.weight_decay)
                                .range(0.0..=1.0f64)
                                .speed(0.01),
                        );
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.label(egui::RichText::new("LR Schedule (Cosine)").strong());
                egui::Grid::new("sched_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Warmup steps");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.schedule.warmup_steps)
                                .range(0..=10000u64)
                                .speed(10.0),
                        );
                        ui.end_row();
                        ui.label("Min LR");
                        ui.add(
                            egui::DragValue::new(&mut self.trainer_config.schedule.min_lr)
                                .range(1e-7..=1e-3f64)
                                .speed(1e-6),
                        );
                        ui.end_row();
                    });
            });
    }
}

fn tier_color(frac: f32) -> egui::Color32 {
    if frac < 0.7 {
        egui::Color32::from_rgb(50, 190, 80)
    } else if frac < 0.9 {
        egui::Color32::from_rgb(240, 180, 40)
    } else {
        egui::Color32::from_rgb(220, 50, 50)
    }
}

fn fmt_bytes(b: u64) -> String {
    if b == 0 {
        return "0 B".into();
    }
    let gib = b as f64 / (1u64 << 30) as f64;
    let mib = b as f64 / (1u64 << 20) as f64;
    if gib >= 1.0 {
        format!("{gib:.1} GiB")
    } else {
        format!("{mib:.0} MiB")
    }
}
