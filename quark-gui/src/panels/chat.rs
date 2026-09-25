#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;
use std::sync::Arc;

use quark_core::chat::{default_stop_strings, render_prompt, ChatMessage};
use quark_core::inference::sampling::SamplingParams;
use quark_core::inference::InferenceEngine;
use quark_core::model::config::QuarkConfig;

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

type LoadReceiver = std::sync::mpsc::Receiver<Result<Arc<InferenceEngine>, String>>;

enum EngineState {
    None,
    Loading,
    Ready(Arc<InferenceEngine>),
    Error(String),
}

pub struct ChatPanel {
    messages: Vec<Message>,
    input: String,
    system_prompt: String,
    sampling: SamplingParams,
    engine_state: EngineState,
    loaded_model_name: Option<String>,
    is_generating: bool,
    /// Streams decoded tokens from the inference thread.
    response_rx: Option<std::sync::mpsc::Receiver<String>>,
    /// Receives the result of background model loading.
    load_rx: Option<LoadReceiver>,
}

impl Default for ChatPanel {
    fn default() -> Self {
        Self {
            messages: vec![],
            input: String::new(),
            system_prompt: "You are a helpful coding assistant.".into(),
            sampling: SamplingParams::default(),
            engine_state: EngineState::None,
            loaded_model_name: None,
            is_generating: false,
            response_rx: None,
            load_rx: None,
        }
    }
}

impl ChatPanel {
    /// Begin loading a checkpoint in a background thread.
    pub fn start_load(&mut self, checkpoint: PathBuf, config: QuarkConfig, tokenizer: PathBuf) {
        self.engine_state = EngineState::Loading;
        self.loaded_model_name = checkpoint
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());

        let (tx, rx) = std::sync::mpsc::channel::<Result<Arc<InferenceEngine>, String>>();
        self.load_rx = Some(rx);

        std::thread::spawn(move || {
            let result = InferenceEngine::load(&checkpoint, &config, &tokenizer)
                .map(Arc::new)
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
    }

    fn poll_load(&mut self) {
        if let Some(rx) = &self.load_rx {
            match rx.try_recv() {
                Ok(Ok(engine)) => {
                    self.engine_state = EngineState::Ready(engine);
                    self.load_rx = None;
                }
                Ok(Err(e)) => {
                    self.engine_state = EngineState::Error(e);
                    self.load_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.engine_state = EngineState::Error("Load thread crashed".into());
                    self.load_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    fn drain_response(&mut self) {
        if let Some(rx) = &self.response_rx {
            let last = self.messages.iter_mut().rev().find(|m| m.role == Role::Assistant);
            while let Ok(token) = rx.try_recv() {
                if let Some(msg) = self.messages.iter_mut().rev().find(|m| m.role == Role::Assistant) {
                    msg.content.push_str(&token);
                }
            }
            if matches!(rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Disconnected)) {
                self.response_rx = None;
                self.is_generating = false;
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.poll_load();
        self.drain_response();

        if self.is_generating || matches!(&self.engine_state, EngineState::Loading) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        ui.heading("💬 Chat");
        ui.separator();

        // Model status bar
        ui.horizontal(|ui| {
            match &self.engine_state {
                EngineState::None => {
                    ui.label(
                        egui::RichText::new(
                            "⚠ No model loaded — go to Checkpoints tab to load one.",
                        )
                        .color(egui::Color32::YELLOW),
                    );
                }
                EngineState::Loading => {
                    ui.spinner();
                    ui.label(egui::RichText::new("Loading model…").weak());
                }
                EngineState::Ready(_) => {
                    let name = self.loaded_model_name.as_deref().unwrap_or("Model");
                    ui.label(
                        egui::RichText::new(format!("Model: {name}"))
                            .color(egui::Color32::GREEN)
                            .strong(),
                    );
                }
                EngineState::Error(e) => {
                    ui.label(
                        egui::RichText::new(format!("❌ Load failed: {e}"))
                            .color(egui::Color32::RED),
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
            let response_widget = ui.add(text_edit);

            let model_ready = matches!(&self.engine_state, EngineState::Ready(_));
            let can_send =
                !self.input.trim().is_empty() && !self.is_generating && model_ready;

            let send = (send_shortcut && response_widget.has_focus())
                || ui.add_enabled(can_send, egui::Button::new("Send")).clicked();

            if send && can_send {
                let user_text = std::mem::take(&mut self.input);
                self.messages.push(Message { role: Role::User, content: user_text.clone() });
                self.messages.push(Message { role: Role::Assistant, content: String::new() });

                if let EngineState::Ready(engine) = &self.engine_state {
                    let engine = Arc::clone(engine);
                    let sampling = self.sampling.clone();
                    let system_prompt = self.system_prompt.clone();
                    let (token_tx, token_rx) = std::sync::mpsc::channel::<String>();
                    self.response_rx = Some(token_rx);
                    self.is_generating = true;

                    // Build full prompt from conversation history (skipping the
                    // empty assistant placeholder being streamed into)
                    let mut history = vec![ChatMessage::system(system_prompt)];
                    history.extend(self.messages.iter().filter(|m| !m.content.is_empty()).map(
                        |msg| match msg.role {
                            Role::User => ChatMessage::user(msg.content.as_str()),
                            Role::Assistant => ChatMessage::assistant(msg.content.as_str()),
                        },
                    ));
                    let full_prompt = render_prompt(&history);
                    let sampling = SamplingParams {
                        stop_strings: default_stop_strings(),
                        ..sampling
                    };

                    std::thread::spawn(move || {
                        let _ = engine.generate_streaming(&full_prompt, sampling, token_tx);
                    });
                }
            }

            if !model_ready && !matches!(&self.engine_state, EngineState::Loading) {
                ui.label(egui::RichText::new("Load a checkpoint first").weak());
            }
        });
    }
}
