#![allow(dead_code, unused_imports)]

mod app;
mod panels;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "quark=info".into()),
        )
        .init();

    tracing::info!("Starting Quark GUI");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Quark LLM Builder")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Quark",
        options,
        Box::new(|cc| Ok(Box::new(app::QuarkApp::new(cc)))),
    )
}
