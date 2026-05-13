#![allow(dead_code, unused_imports, unused_variables)]

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "quark_cli=info".into()),
        )
        .init();

    tracing::info!("Quark CLI");

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => {
            tracing::info!("Training not yet implemented");
        }
        Some("infer") => {
            tracing::info!("Inference not yet implemented");
        }
        Some("tokenize") => {
            tracing::info!("Tokenisation not yet implemented");
        }
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            eprintln!("Usage: quark-cli <train|infer|tokenize>");
            std::process::exit(1);
        }
        None => {
            println!("Quark LLM CLI\nUsage: quark-cli <train|infer|tokenize>");
        }
    }

    Ok(())
}
