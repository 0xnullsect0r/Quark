#![allow(dead_code, unused_imports, unused_variables)]

use quark_core::model::config::{ModelPreset, QuarkConfig};

pub struct ConfigPanel {
    pub config: QuarkConfig,
    preset: ModelPreset,
}

impl Default for ConfigPanel {
    fn default() -> Self {
        Self {
            config: QuarkConfig::quark_tiny(),
            preset: ModelPreset::QuarkTiny,
        }
    }
}

impl ConfigPanel {
    pub fn config(&self) -> &QuarkConfig {
        &self.config
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("⚙ Model Configuration");
        ui.separator();

        // Preset picker
        ui.horizontal(|ui| {
            ui.label("Preset:");
            let old_preset = self.preset;
            egui::ComboBox::from_id_salt("preset_combo")
                .selected_text(preset_name(self.preset))
                .show_ui(ui, |ui| {
                    for preset in PRESETS {
                        let label = match QuarkConfig::from_preset(preset) {
                            Some(cfg) => format!(
                                "{:<11} {} params · ≈{} to train (batch 1, f32)",
                                preset_name(preset),
                                fmt_count(cfg.param_count()),
                                fmt_bytes(cfg.training_memory_bytes(
                                    1,
                                    cfg.max_position_embeddings,
                                    4,
                                    true
                                )),
                            ),
                            None => preset_name(preset).to_owned(),
                        };
                        ui.selectable_value(&mut self.preset, preset, label);
                    }
                });
            if self.preset != old_preset {
                if let Some(cfg) = QuarkConfig::from_preset(self.preset) {
                    self.config = cfg;
                }
            }
        });

        ui.add_space(4.0);

        // Parameter count banner
        let params = self.config.param_count();
        ui.label(
            egui::RichText::new(format!("{} parameters", fmt_count(params)))
                .strong()
                .color(egui::Color32::from_rgb(120, 200, 255)),
        );
        ui.label(
            egui::RichText::new(
                "Training memory depends on batch size too — see the estimate in the Training tab.",
            )
            .small()
            .weak(),
        );

        ui.separator();

        let mut changed = false;

        egui::CollapsingHeader::new("🏗 Architecture")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("arch_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Vocabulary size");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.vocab_size)
                                    .range(1000..=128000)
                                    .speed(100.0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Hidden size");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.hidden_size)
                                    .range(64..=16384)
                                    .speed(64.0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Num layers");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.num_hidden_layers)
                                    .range(1..=128),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Attention heads");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.num_attention_heads)
                                    .range(1..=128),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("KV heads (GQA)");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.num_key_value_heads)
                                    .range(1..=128),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Intermediate (FFN)");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.intermediate_size)
                                    .range(64..=65536)
                                    .speed(64.0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Max seq length");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.max_position_embeddings)
                                    .range(128..=131072)
                                    .speed(128.0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Tie embeddings");
                        changed |=
                            ui.checkbox(&mut self.config.tie_word_embeddings, "").changed();
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("🧩 Mixture-of-Experts")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("moe_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Total experts");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.num_experts)
                                    .range(1..=64),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Active experts (Top-K)");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.num_experts_per_tok)
                                    .range(1..=16),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("MoE layers");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.num_moe_layers)
                                    .range(0..=64),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("MoE layer frequency");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.config.moe_layer_freq)
                                    .range(1..=32),
                            )
                            .changed();
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("🔩 Hyperparameters")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("hyp_grid")
                    .num_columns(2)
                    .striped(true)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("RMSNorm ε");
                        let mut rms_eps = self.config.rms_norm_eps as f32;
                        if ui
                            .add(
                                egui::DragValue::new(&mut rms_eps)
                                    .range(1e-8f32..=1e-3f32)
                                    .speed(1e-7),
                            )
                            .changed()
                        {
                            self.config.rms_norm_eps = rms_eps as f64;
                            changed = true;
                        }
                        ui.end_row();

                        ui.label("RoPE θ");
                        let mut theta = self.config.rope_theta as f32;
                        if ui
                            .add(
                                egui::DragValue::new(&mut theta)
                                    .range(1000.0f32..=1_000_000.0f32)
                                    .speed(1000.0),
                            )
                            .changed()
                        {
                            self.config.rope_theta = theta as f64;
                            changed = true;
                        }
                        ui.end_row();
                    });
            });

        if changed && self.preset != ModelPreset::Custom {
            self.preset = ModelPreset::Custom;
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("📋 Export config as JSON").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .set_file_name("quark_config.json")
                    .save_file()
                {
                    if let Ok(json) = serde_json::to_string_pretty(&self.config) {
                        let _ = std::fs::write(&path, json);
                    }
                }
            }
            if ui.button("📂 Load config from JSON").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON", &["json"])
                    .pick_file()
                {
                    if let Ok(text) = std::fs::read_to_string(&path) {
                        if let Ok(cfg) = serde_json::from_str::<QuarkConfig>(&text) {
                            self.config = cfg;
                            self.preset = ModelPreset::Custom;
                        }
                    }
                }
            }
        });
    }
}

const PRESETS: [ModelPreset; 15] = [
    ModelPreset::QuarkTiny,
    ModelPreset::QuarkSmall,
    ModelPreset::Quark10BA2B,
    ModelPreset::Quark1B,
    ModelPreset::Quark3B,
    ModelPreset::Quark7B,
    ModelPreset::Quark20B,
    ModelPreset::Quark30B,
    ModelPreset::Quark48B,
    ModelPreset::Quark74B,
    ModelPreset::Quark120B,
    ModelPreset::Quark249B,
    ModelPreset::Quark300B,
    ModelPreset::Quark400B,
    ModelPreset::Custom,
];

fn preset_name(preset: ModelPreset) -> &'static str {
    match preset {
        ModelPreset::QuarkTiny => "Quark Tiny",
        ModelPreset::QuarkSmall => "Quark Small",
        ModelPreset::Quark10BA2B => "Quark 10B-A2B",
        ModelPreset::Quark1B => "Quark 1B",
        ModelPreset::Quark3B => "Quark 3B",
        ModelPreset::Quark7B => "Quark 7B",
        ModelPreset::Quark20B => "Quark 20B",
        ModelPreset::Quark30B => "Quark 30B",
        ModelPreset::Quark48B => "Quark 48B",
        ModelPreset::Quark74B => "Quark 74B",
        ModelPreset::Quark120B => "Quark 120B",
        ModelPreset::Quark249B => "Quark 249B",
        ModelPreset::Quark300B => "Quark 300B",
        ModelPreset::Quark400B => "Quark 400B",
        ModelPreset::Custom => "Custom",
    }
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else {
        format!("{:.0}M", n as f64 / 1e6)
    }
}

pub fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.1} TB", bytes as f64 / 1e12)
    } else {
        format!("{:.1} GB", bytes as f64 / 1e9)
    }
}
