#![allow(dead_code, unused_imports, unused_variables)]

use crate::panels::{
    ChatPanel, CheckpointsPanel, ConfigPanel, DatasetPanel, GettingStartedPanel, SettingsPanel,
    TrainingPanel,
};

#[derive(Debug, Default, PartialEq, Eq)]
enum ActivePanel {
    #[default]
    Config,
    Dataset,
    Training,
    Checkpoints,
    Chat,
    Settings,
    Help,
}

pub struct QuarkApp {
    active: ActivePanel,
    config_panel: ConfigPanel,
    dataset_panel: DatasetPanel,
    training_panel: TrainingPanel,
    checkpoints_panel: CheckpointsPanel,
    chat_panel: ChatPanel,
    settings_panel: SettingsPanel,
    getting_started: GettingStartedPanel,
    update_info: Option<quark_core::updater::UpdateInfo>,
    update_rx: Option<std::sync::mpsc::Receiver<Option<quark_core::updater::UpdateInfo>>>,
}

impl QuarkApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let update_rx = Some(quark_core::updater::spawn_update_check());
        Self {
            active: ActivePanel::default(),
            config_panel: ConfigPanel::default(),
            dataset_panel: DatasetPanel::default(),
            training_panel: TrainingPanel::default(),
            checkpoints_panel: CheckpointsPanel::default(),
            chat_panel: ChatPanel::default(),
            settings_panel: SettingsPanel::default(),
            getting_started: GettingStartedPanel::default(),
            update_info: None,
            update_rx,
        }
    }
}

impl eframe::App for QuarkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll update checker
        if let Some(rx) = &self.update_rx {
            if let Ok(msg) = rx.try_recv() {
                self.update_info = msg;
                self.update_rx = None;
            }
        }

        // Update banner
        if let Some(info) = &self.update_info.clone() {
            egui::TopBottomPanel::top("update_banner").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(40, 40, 0),
                        format!("⬆ Quark {} is available!", info.version),
                    );
                    if ui.button("🌐 View Release").clicked() {
                        let _ = open::that(&info.html_url);
                    }
                    if ui.button("✖ Dismiss").clicked() {
                        self.update_info = None;
                    }
                });
            });
        }

        egui::TopBottomPanel::top("nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active, ActivePanel::Config, "⚙ Config");
                ui.selectable_value(&mut self.active, ActivePanel::Dataset, "📂 Dataset");
                ui.selectable_value(&mut self.active, ActivePanel::Training, "🏋 Training");
                ui.selectable_value(&mut self.active, ActivePanel::Checkpoints, "💾 Checkpoints");
                ui.selectable_value(&mut self.active, ActivePanel::Chat, "💬 Chat");
                ui.selectable_value(&mut self.active, ActivePanel::Settings, "🛠 Settings");
                ui.selectable_value(&mut self.active, ActivePanel::Help, "❓ Help");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.active {
            ActivePanel::Config => self.config_panel.ui(ui),
            ActivePanel::Dataset => self.dataset_panel.ui(ui),
            ActivePanel::Training => self.training_panel.ui(ui),
            ActivePanel::Checkpoints => self.checkpoints_panel.ui(ui),
            ActivePanel::Chat => self.chat_panel.ui(ui),
            ActivePanel::Settings => self.settings_panel.ui(ui),
            ActivePanel::Help => self.getting_started.ui(ui),
        });
    }
}
