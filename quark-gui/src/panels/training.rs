#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use egui_plot::{Line, Plot, PlotPoints};
use quark_core::memory::budget::HardwareBudget;
use quark_core::memory::tier::TierConfig;
use quark_core::model::config::QuarkConfig;
use quark_core::training::metrics::{MetricsReceiver, TrainingEvent, TrainingMetrics};
use quark_core::training::trainer::{
    estimate_memory, start_training, Precision, TrainerConfig, TrainingHandle,
};

use super::config::ConfigPanel;
use super::dataset::DatasetPanel;

const MAX_LOG_LINES: usize = 2000;

pub struct TrainingPanel {
    pub trainer_config: TrainerConfig,
    metrics_rx: Option<MetricsReceiver>,
    stop_flag: Option<Arc<AtomicBool>>,
    latest: Option<TrainingMetrics>,
    loss_history: Vec<[f64; 2]>,
    eval_history: Vec<[f64; 2]>,
    lr_history: Vec<[f64; 2]>,
    is_running: bool,
    phase: String,
    log: Vec<String>,
    budget: HardwareBudget,
}

impl Default for TrainingPanel {
    fn default() -> Self {
        Self {
            trainer_config: TrainerConfig::default(),
            metrics_rx: None,
            stop_flag: None,
            latest: None,
            loss_history: Vec::new(),
            eval_history: Vec::new(),
            lr_history: Vec::new(),
            is_running: false,
            phase: String::new(),
            log: Vec::new(),
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
        if let Some(flag) = &self.stop_flag {
            flag.store(true, Ordering::SeqCst);
        }
        self.is_running = false;
    }

    fn drain_events(&mut self) {
        let Some(rx) = &mut self.metrics_rx else { return };
        while let Ok(event) = rx.try_recv() {
            match event {
                TrainingEvent::Metrics(m) => {
                    if self.loss_history.len() > MAX_LOG_LINES {
                        self.loss_history.drain(0..MAX_LOG_LINES / 4);
                    }
                    if self.lr_history.len() > MAX_LOG_LINES {
                        self.lr_history.drain(0..MAX_LOG_LINES / 4);
                    }
                    self.loss_history.push([m.step as f64, m.loss as f64]);
                    self.lr_history.push([m.step as f64, m.learning_rate as f64]);
                    self.latest = Some(m);
                }
                TrainingEvent::Eval { step, loss } => {
                    self.eval_history.push([step as f64, loss as f64]);
                }
                TrainingEvent::Log(s) => {
                    self.log.push(s);
                    if self.log.len() > MAX_LOG_LINES {
                        self.log.drain(0..MAX_LOG_LINES / 4);
                    }
                }
                TrainingEvent::Phase(s) => {
                    self.phase = s;
                }
                TrainingEvent::Done => {
                    self.is_running = false;
                    self.stop_flag = None;
                }
                TrainingEvent::Error(e) => {
                    self.log.push(format!("❌  {e}"));
                    self.is_running = false;
                    self.stop_flag = None;
                }
            }
        }
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        config_panel: &ConfigPanel,
        dataset_panel: &DatasetPanel,
    ) {
        self.drain_events();
        if self.is_running {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }

        egui::ScrollArea::vertical()
            .id_salt("training_panel")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.heading("🏋 Training");
                ui.separator();

                // Memory estimate for the current model + batch settings
                let estimate =
                    estimate_memory(config_panel.config(), &self.trainer_config, &self.budget);
                let color = if estimate.fits() {
                    egui::Color32::from_rgb(120, 200, 120)
                } else {
                    egui::Color32::from_rgb(255, 110, 90)
                };
                ui.label(
                    egui::RichText::new(format!(
                        "≈{} needed to train · {} {} free{}",
                        super::config::fmt_bytes(estimate.needed_bytes),
                        super::config::fmt_bytes(estimate.available_bytes),
                        estimate.device,
                        if estimate.fits() {
                            ""
                        } else {
                            " — likely too big: pick a smaller preset, batch size or context"
                        },
                    ))
                    .color(color),
                );

                // Control buttons
                ui.horizontal(|ui| {
                    if self.is_running {
                        if ui
                            .button(egui::RichText::new("⏹ Stop").color(egui::Color32::RED))
                            .clicked()
                        {
                            self.stop();
                        }
                        ui.spinner();
                        if self.phase.is_empty() {
                            ui.label("Training…");
                        } else {
                            ui.label(&self.phase);
                        }
                    } else {
                        if ui
                            .button(
                                egui::RichText::new("▶ Start Training")
                                    .color(egui::Color32::GREEN)
                                    .strong(),
                            )
                            .clicked()
                        {
                            self.loss_history.clear();
                            self.eval_history.clear();
                            self.lr_history.clear();
                            self.log.clear();
                            self.phase.clear();
                            self.latest = None;
                            let model_config = config_panel.config().clone();
                            let corpus_files = dataset_panel.corpus_files();
                            let tokenizer_path = dataset_panel.active_tokenizer_path();
                            let (handle, rx) = start_training(
                                model_config,
                                self.trainer_config.clone(),
                                corpus_files,
                                tokenizer_path,
                            );
                            self.stop_flag = Some(Arc::clone(&handle.stop_flag));
                            self.metrics_rx = Some(rx);
                            self.is_running = true;
                        }
                        if self.latest.is_some() && !self.is_running {
                            ui.label(
                                egui::RichText::new("✅ Finished")
                                    .color(egui::Color32::GREEN)
                                    .strong(),
                            );
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
                            ui.label("Grad norm");
                            ui.label(format!("{:.3}", m.grad_norm));
                            ui.label("Eval loss");
                            ui.label(match self.eval_history.last() {
                                Some([_, loss]) => format!("{loss:.4}"),
                                None => "—".to_owned(),
                            });
                            ui.end_row();
                        });
                }

                // Process memory (spilling to disk is not implemented, so
                // everything must fit in RAM/VRAM)
                if let Some(m) = &self.latest {
                    ui.separator();
                    ui.label(egui::RichText::new("Memory").strong());
                    let ram_total = self.budget.ram_total_bytes.max(1);
                    let ram_used = m.ram_used_bytes;
                    ui.horizontal(|ui| {
                        ui.label("🟩 RAM ");
                        let frac = (ram_used as f64 / ram_total as f64).min(1.0) as f32;
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .desired_width(180.0)
                                .fill(tier_color(frac))
                                .text(format!(
                                    "{} / {}",
                                    fmt_bytes(ram_used),
                                    fmt_bytes(ram_total)
                                )),
                        );
                    });
                    if m.vram_used_bytes > 0 && self.budget.vram_total_bytes > 0 {
                        let vram_total = self.budget.vram_total_bytes;
                        ui.horizontal(|ui| {
                            ui.label("🟦 VRAM");
                            let frac =
                                (m.vram_used_bytes as f64 / vram_total as f64).min(1.0) as f32;
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width(180.0)
                                    .fill(tier_color(frac))
                                    .text(format!(
                                        "{} / {}",
                                        fmt_bytes(m.vram_used_bytes),
                                        fmt_bytes(vram_total)
                                    )),
                            );
                        });
                    }
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
                            let pts: PlotPoints =
                                self.loss_history.iter().copied().collect();
                            plot_ui.line(
                                Line::new(pts)
                                    .name("loss")
                                    .color(egui::Color32::from_rgb(255, 140, 50))
                                    .width(1.5),
                            );
                            if !self.eval_history.is_empty() {
                                let pts: PlotPoints =
                                    self.eval_history.iter().copied().collect();
                                plot_ui.line(
                                    Line::new(pts)
                                        .name("eval loss")
                                        .color(egui::Color32::from_rgb(120, 220, 120))
                                        .width(2.0),
                                );
                            }
                        });

                    ui.label(egui::RichText::new("Learning Rate").strong());
                    Plot::new("lr_chart")
                        .height(100.0)
                        .allow_drag(false)
                        .allow_zoom(false)
                        .show(ui, |plot_ui| {
                            let pts: PlotPoints =
                                self.lr_history.iter().copied().collect();
                            plot_ui.line(
                                Line::new(pts)
                                    .name("lr")
                                    .color(egui::Color32::from_rgb(80, 180, 255))
                                    .width(1.5),
                            );
                        });
                }

                ui.separator();

                // Live training log
                egui::CollapsingHeader::new("📋 Training Log")
                    .default_open(true)
                    .show(ui, |ui| {
                        if self.log.is_empty() && !self.is_running {
                            ui.label(
                                egui::RichText::new("Start training to see live logs here.")
                                    .weak()
                                    .italics(),
                            );
                        } else {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} lines",
                                        self.log.len()
                                    ))
                                    .small()
                                    .weak(),
                                );
                                if ui
                                    .small_button("📋 Copy")
                                    .on_hover_text("Copy log to clipboard")
                                    .clicked()
                                {
                                    ui.ctx().copy_text(self.log.join("\n"));
                                }
                            });
                            egui::ScrollArea::vertical()
                                .id_salt("training_log")
                                .max_height(200.0)
                                .stick_to_bottom(true)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    egui::Frame::new()
                                        .fill(egui::Color32::from_rgb(18, 18, 20))
                                        .corner_radius(4.0)
                                        .inner_margin(egui::Margin::same(6))
                                        .show(ui, |ui| {
                                            let start = self.log.len().saturating_sub(300);
                                            for line in &self.log[start..] {
                                                let color =
                                                    if line.starts_with("❌") {
                                                        egui::Color32::from_rgb(255, 100, 100)
                                                    } else if line.starts_with("✅")
                                                        || line.starts_with("✔")
                                                    {
                                                        egui::Color32::from_rgb(100, 220, 100)
                                                    } else if line.starts_with("⚠") {
                                                        egui::Color32::YELLOW
                                                    } else if line.starts_with("💾") {
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
                                            if self.is_running {
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
                    });

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
                                    egui::DragValue::new(
                                        &mut self.trainer_config.grad_accum_steps,
                                    )
                                    .range(1..=128usize),
                                );
                                ui.end_row();

                                ui.label("Save every N steps");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.trainer_config.save_every_steps,
                                    )
                                    .range(1..=10000u64)
                                    .speed(50.0),
                                );
                                ui.end_row();

                                ui.label("Precision");
                                ui.horizontal(|ui| {
                                    ui.radio_value(
                                        &mut self.trainer_config.precision,
                                        Precision::F32,
                                        "f32",
                                    );
                                    ui.add_enabled_ui(Precision::Bf16.is_supported(), |ui| {
                                        ui.radio_value(
                                            &mut self.trainer_config.precision,
                                            Precision::Bf16,
                                            "bf16",
                                        )
                                        .on_disabled_hover_text(
                                            "bf16 training needs a CUDA build \
                                             (--features backend-cuda)",
                                        );
                                    });
                                });
                                ui.end_row();

                                ui.label("Max grad norm");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.trainer_config.max_grad_norm,
                                    )
                                    .range(0.1f32..=10.0f32)
                                    .speed(0.1),
                                );
                                ui.end_row();

                                ui.label("Resume");
                                ui.checkbox(
                                    &mut self.trainer_config.resume,
                                    "Continue from latest checkpoint",
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
                                    egui::DragValue::new(
                                        &mut self.trainer_config.adamw.lr,
                                    )
                                    .range(1e-6..=1e-2f64)
                                    .speed(1e-5),
                                );
                                ui.end_row();
                                ui.label("β₁");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.trainer_config.adamw.beta1,
                                    )
                                    .range(0.5..=0.999f64)
                                    .speed(0.001),
                                );
                                ui.end_row();
                                ui.label("β₂");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.trainer_config.adamw.beta2,
                                    )
                                    .range(0.5..=0.9999f64)
                                    .speed(0.0001),
                                );
                                ui.end_row();
                                ui.label("Weight decay");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.trainer_config.adamw.weight_decay,
                                    )
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
                                    egui::DragValue::new(
                                        &mut self.trainer_config.schedule.warmup_steps,
                                    )
                                    .range(0..=10000u64)
                                    .speed(10.0),
                                );
                                ui.end_row();
                                ui.label("Min LR");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.trainer_config.schedule.min_lr,
                                    )
                                    .range(1e-7..=1e-3f64)
                                    .speed(1e-6),
                                );
                                ui.end_row();
                            });
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
