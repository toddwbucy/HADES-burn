use clap::Parser;

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

fn main() {
    let cli = Cli::parse();

    if let Some(ref db) = cli.database {
        eprintln!("database: {db}");
    }
    if let Some(gpu) = cli.gpu {
        eprintln!("gpu: {gpu}");
    }

    println!(
        "hades-burn {} — workspace scaffold operational",
        env!("CARGO_PKG_VERSION")
    );
}
