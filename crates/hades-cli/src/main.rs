use clap::Parser;
use hades_core::config;

/// HADES-Burn — Rust rewrite of the HADES knowledge graph system.
///
/// AI model interface for semantic search over academic papers,
/// backed by ArangoDB with vector similarity and graph traversal.
#[derive(Parser)]
#[command(name = "hades-burn", version, about)]
struct Cli {
    /// Target ArangoDB database name (overrides config/env)
    #[arg(long = "database", alias = "db", global = true)]
    database: Option<String>,

    /// GPU device index for embedding commands
    #[arg(short = 'g', long = "gpu", global = true)]
    gpu: Option<u32>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut config = config::load_config()?;
    config.apply_cli_overrides(cli.database.as_deref(), cli.gpu);

    eprintln!(
        "hades-burn {} — database: {}, device: {}",
        env!("CARGO_PKG_VERSION"),
        config.effective_database(),
        config.effective_device(),
    );

    Ok(())
}
