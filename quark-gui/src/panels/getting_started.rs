//! Getting Started help panel — shown on first launch and accessible from the Help tab.

pub struct GettingStartedPanel {
    pub show_on_startup: bool,
}

impl Default for GettingStartedPanel {
    fn default() -> Self {
        Self { show_on_startup: true }
    }
}

impl GettingStartedPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("🚀 Getting Started with Quark");
        ui.separator();

        egui::ScrollArea::vertical().id_salt("gs_scroll").show(ui, |ui| {
            step(ui, "1", "Configure your model",
                "Go to the **Config** tab. Choose a preset:\n\
                 • **Quark 1B** — ~1B parameters; trains on 8–16 GB RAM (no GPU needed)\n\
                 • **Quark 3B** — ~3B parameters; trains best with 16+ GB RAM or a GPU\n\
                 • **Custom** — tune every architecture parameter yourself\n\n\
                 The live parameter count updates as you drag sliders.");

            step(ui, "2", "Set resource limits",
                "Go to the **Settings** tab before training. Set the sliders to match your hardware:\n\
                 • **VRAM limit** — how much GPU memory Quark may use (default 60%)\n\
                 • **RAM limit** — system memory cap (default 75%)\n\
                 • **CPU threads** — how many cores the training loop uses (default 80%)\n\
                 • **GPU compute** — fraction of GPU cycles reserved for Quark\n\n\
                 Quark automatically tiers model layers across VRAM → RAM → Disk, so you can \
                 train models larger than your GPU VRAM.");

            step(ui, "3", "Add training data",
                "Go to the **Dataset** tab.\n\
                 • Click **Add Files…** to pick `.txt` or `.jsonl` files\n\
                 • Click **Add Folder…** to recursively add a whole directory\n\
                 • `.jsonl` files should have one JSON object per line with a `\"text\"` field\n\n\
                 Then either load an existing `tokenizer.json` or train a new BPE tokenizer from \
                 your dataset by entering a vocab size and clicking **Train Tokenizer**.");

            step(ui, "4", "Start training",
                "Go to the **Training** tab.\n\
                 • Adjust batch size, learning rate, max steps, and schedule in **Training Configuration**\n\
                 • Click **▶ Start Training**\n\
                 • Watch the live loss and LR charts update in real time\n\
                 • Color-coded VRAM/RAM progress bars show memory tier usage\n\n\
                 Training saves a checkpoint every N steps (configurable). You can stop and resume \
                 at any time — Quark saves the full optimizer state.");

            step(ui, "5", "Browse checkpoints",
                "Go to the **Checkpoints** tab.\n\
                 • Click **Browse…** to point to your output directory\n\
                 • Select any `.safetensors` checkpoint and click **📥 Load**\n\
                 • Use **📤 Export…** to copy a checkpoint to share it\n\n\
                 Checkpoints are HuggingFace-compatible `.safetensors` files and can be loaded \
                 in tools like `transformers` or converted to GGUF for llama.cpp.");

            step(ui, "6", "Chat with your model",
                "Go to the **Chat** tab after loading a checkpoint.\n\
                 • Edit the system prompt to set the assistant's persona\n\
                 • Adjust **Temperature**, **Top-K**, and **Top-P** in Sampling Parameters\n\
                 • Type your message and press **Enter** or **Send**\n\n\
                 The chat panel supports streaming output — tokens appear as they are generated.");

            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Tip: fine-tune an existing model").strong());
            ui.label(
                "Load a checkpoint in the Checkpoints tab, then go back to Training and enable \
                 LoRA in the config. Only a small number of adapter parameters will be trained, \
                 making fine-tuning much faster and memory-efficient."
            );

            ui.add_space(12.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_on_startup, "Show this panel on startup");
                if ui.button("🌐 Open GitHub").clicked() {
                    let _ = open::that("https://github.com/0xnullsect0r/Quark");
                }
            });
        });
    }
}

fn step(ui: &mut egui::Ui, num: &str, title: &str, body: &str) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("Step {num}")).strong()
            .color(egui::Color32::from_rgb(100, 180, 255))
            .size(15.0));
        ui.label(egui::RichText::new(title).strong().size(15.0));
    });
    ui.add_space(2.0);
    // Render simple **bold** markers manually
    for line in body.lines() {
        if line.is_empty() {
            ui.add_space(4.0);
        } else {
            ui.label(line);
        }
    }
}
