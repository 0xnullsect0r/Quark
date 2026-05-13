#![allow(dead_code, unused_imports, unused_variables)]

use quark_core::model::config::{ModelPreset, QuarkConfig};

pub struct ConfigPanel {
    pub config: QuarkConfig,
    preset: ModelPreset,
}

impl Default for ConfigPanel {
    fn default() -> Self {
        Self {
            config: QuarkConfig::quark_1b(),
            preset: ModelPreset::Quark1B,
        }
    }
}

impl ConfigPanel {
    pub fn config(&self) -> &QuarkConfig {
        &self.config
    }

    fn estimated_params(&self) -> u64 {
        let c = &self.config;
        let head_dim = c.hidden_size / c.num_attention_heads.max(1);
        let attn = c.hidden_size
            * (c.num_attention_heads + c.num_key_value_heads * 2)
            * head_dim
            + c.hidden_size * c.hidden_size;
        let ffn_dense = c.hidden_size * c.intermediate_size * 3;
        let ffn_moe = ffn_dense * c.num_experts;
        let moe_layers = c.num_hidden_layers / c.moe_layer_freq.max(1);
        let dense_layers = c.num_hidden_layers.saturating_sub(moe_layers);
        let layer = attn
            + if moe_layers > 0 {
                (dense_layers * ffn_dense + moe_layers * ffn_moe)
                    / c.num_hidden_layers.max(1)
            } else {
                ffn_dense
            };
        let embed = c.vocab_size * c.hidden_size;
        (embed + c.num_hidden_layers * layer) as u64
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("⚙ Model Configuration");
        ui.separator();

        // Preset picker
        ui.horizontal(|ui| {
            ui.label("Preset:");
            let old_preset = self.preset;
            egui::ComboBox::from_id_salt("preset_combo")
                .selected_text(match self.preset {
                    ModelPreset::Quark1B => "Quark 1B",
                    ModelPreset::Quark3B => "Quark 3B",
                    ModelPreset::Custom => "Custom",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.preset,
                        ModelPreset::Quark1B,
                        "Quark 1B (~1B params, 8–16 GB RAM)",
                    );
                    ui.selectable_value(
                        &mut self.preset,
                        ModelPreset::Quark3B,
                        "Quark 3B (~3B params, 16–32 GB RAM)",
                    );
                    ui.selectable_value(&mut self.preset, ModelPreset::Custom, "Custom");
                });
            if self.preset != old_preset {
                match self.preset {
                    ModelPreset::Quark1B => self.config = QuarkConfig::quark_1b(),
                    ModelPreset::Quark3B => self.config = QuarkConfig::quark_3b(),
                    ModelPreset::Custom => {}
                }
            }
        });

        ui.add_space(4.0);

        // Parameter count banner
        let params = self.estimated_params();
        let param_str = if params >= 1_000_000_000 {
            format!("~{:.2}B parameters", params as f64 / 1e9)
        } else {
            format!("~{:.0}M parameters", params as f64 / 1e6)
        };
        ui.label(
            egui::RichText::new(param_str)
                .strong()
                .color(egui::Color32::from_rgb(120, 200, 255)),
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
