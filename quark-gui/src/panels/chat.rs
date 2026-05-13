#![allow(dead_code, unused_imports, unused_variables)]

use quark_core::inference::sampling::SamplingParams;

#[derive(Clone, PartialEq)]
enum Role {
    User,
    Assistant,
}

#[derive(Clone)]
struct Message {
    role: Role,
    content: String,
}

pub struct ChatPanel {
    messages: Vec<Message>,
    input: String,
    system_prompt: String,
    sampling: SamplingParams,
    loaded_model: Option<String>,
    is_generating: bool,
}

impl Default for ChatPanel {
    fn default() -> Self {
        Self {
            messages: vec![],
            input: String::new(),
            system_prompt: "You are a helpful coding assistant.".into(),
            sampling: SamplingParams::default(),
            loaded_model: None,
            is_generating: false,
        }
    }
}

impl ChatPanel {
    pub fn set_model(&mut self, name: impl Into<String>) {
        self.loaded_model = Some(name.into());
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("💬 Chat");
        ui.separator();

        // Model status bar
        ui.horizontal(|ui| {
            match &self.loaded_model {
                Some(name) => {
                    ui.label(
                        egui::RichText::new(format!("Model: {name}"))
                            .color(egui::Color32::GREEN)
                            .strong(),
                    );
                }
                None => {
                    ui.label(
                        egui::RichText::new(
                            "⚠ No model loaded — go to Checkpoints tab to load one.",
                        )
                        .color(egui::Color32::YELLOW),
                    );
                }
            }
            if ui.button("🗑 Clear Chat").clicked() {
                self.messages.clear();
            }
        });

        ui.separator();

        // System prompt
        egui::CollapsingHeader::new("System Prompt")
            .default_open(false)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.system_prompt)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .hint_text("System instructions…"),
                );
            });

        // Sampling params
        egui::CollapsingHeader::new("Sampling Parameters")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("sampling_grid")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Temperature");
                        ui.add(
                            egui::Slider::new(&mut self.sampling.temperature, 0.01f32..=2.0f32)
                                .step_by(0.05),
                        );
                        ui.end_row();

                        ui.label("Top-K");
                        ui.add(
                            egui::DragValue::new(&mut self.sampling.top_k).range(1..=500usize),
                        );
                        ui.end_row();

                        ui.label("Top-P");
                        ui.add(
                            egui::Slider::new(&mut self.sampling.top_p, 0.01f32..=1.0f32)
                                .step_by(0.01),
                        );
                        ui.end_row();

                        ui.label("Max new tokens");
                        ui.add(
                            egui::DragValue::new(&mut self.sampling.max_new_tokens)
                                .range(1..=8192usize)
                                .speed(10.0),
                        );
                        ui.end_row();
                    });
            });

        ui.separator();

        // Chat history
        let avail_height = ui.available_height() - 60.0;
        egui::ScrollArea::vertical()
            .id_salt("chat_history")
            .max_height(avail_height)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for msg in &self.messages {
                    match msg.role {
                        Role::User => {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("You")
                                        .strong()
                                        .color(egui::Color32::from_rgb(100, 180, 255)),
                                );
                                ui.label(&msg.content);
                            });
                        }
                        Role::Assistant => {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Quark")
                                        .strong()
                                        .color(egui::Color32::from_rgb(80, 220, 120)),
                                );
                                ui.label(&msg.content);
                            });
                        }
                    }
                    ui.separator();
                }
                if self.is_generating {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Quark")
                                .strong()
                                .color(egui::Color32::from_rgb(80, 220, 120)),
                        );
                        ui.spinner();
                        ui.label(egui::RichText::new("generating…").weak().italics());
                    });
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
            });

        // Input bar
        ui.separator();
        ui.horizontal(|ui| {
            let send_shortcut =
                ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);

            let text_edit = egui::TextEdit::singleline(&mut self.input)
                .desired_width(ui.available_width() - 80.0)
                .hint_text("Ask Quark something…");
            let response = ui.add(text_edit);

            let can_send = !self.input.trim().is_empty()
                && !self.is_generating
                && self.loaded_model.is_some();

            if (send_shortcut && response.has_focus() || ui.add_enabled(can_send, egui::Button::new("Send")).clicked()) && can_send {
                let user_msg = std::mem::take(&mut self.input);
                self.messages.push(Message {
                    role: Role::User,
                    content: user_msg,
                });
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: "(Model inference not yet wired — checkpoint is loaded but generation requires the training loop to finish)".into(),
                });
            }

            if self.loaded_model.is_none() {
                ui.label(egui::RichText::new("Load a checkpoint first").weak());
            }
        });
    }
}
