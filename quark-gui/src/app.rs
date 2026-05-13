#![allow(dead_code, unused_imports, unused_variables)]

use crate::panels::{
    ChatPanel, CheckpointsPanel, ConfigPanel, DatasetPanel, SettingsPanel, TrainingPanel,
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
}

pub struct QuarkApp {
    active: ActivePanel,
    config_panel: ConfigPanel,
    dataset_panel: DatasetPanel,
    training_panel: TrainingPanel,
    checkpoints_panel: CheckpointsPanel,
    chat_panel: ChatPanel,
    settings_panel: SettingsPanel,
}

impl QuarkApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            active: ActivePanel::default(),
            config_panel: ConfigPanel::default(),
            dataset_panel: DatasetPanel::default(),
            training_panel: TrainingPanel::default(),
            checkpoints_panel: CheckpointsPanel::default(),
            chat_panel: ChatPanel::default(),
            settings_panel: SettingsPanel::default(),
        }
    }
}

impl eframe::App for QuarkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active, ActivePanel::Config, "⚙ Config");
                ui.selectable_value(&mut self.active, ActivePanel::Dataset, "📂 Dataset");
                ui.selectable_value(&mut self.active, ActivePanel::Training, "🏋 Training");
                ui.selectable_value(&mut self.active, ActivePanel::Checkpoints, "💾 Checkpoints");
                ui.selectable_value(&mut self.active, ActivePanel::Chat, "💬 Chat");
                ui.selectable_value(&mut self.active, ActivePanel::Settings, "🛠 Settings");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.active {
            ActivePanel::Config => self.config_panel.ui(ui),
            ActivePanel::Dataset => self.dataset_panel.ui(ui),
            ActivePanel::Training => self.training_panel.ui(ui),
            ActivePanel::Checkpoints => self.checkpoints_panel.ui(ui),
            ActivePanel::Chat => self.chat_panel.ui(ui),
            ActivePanel::Settings => self.settings_panel.ui(ui),
        });
    }
}
