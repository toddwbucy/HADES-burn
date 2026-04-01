//! HADES-Burn CLI — Rust rewrite of the HADES knowledge graph system.
//!
//! During the strangler-fig migration, every command dispatches to the
//! Python `hades` CLI via subprocess.  As Rust modules mature, commands
//! switch to native implementations one at a time.

mod commands;
mod dispatch;

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};
use hades_core::config;

use commands::{
    arxiv::ArxivCmd, codebase::CodebaseCmd, db::DbCmd, embed::EmbedCmd,
    graph_embed::GraphEmbedCmd, smell::SmellCmd, task::TaskCmd,
};

/// HADES-Burn — AI model interface for semantic search over academic papers,
/// backed by ArangoDB with vector similarity and graph traversal.
#[derive(Parser)]
#[command(name = "hades-burn", version, about)]
struct Cli {
    /// Target ArangoDB database name (overrides config/env).
    #[arg(long = "database", alias = "db", global = true)]
    database: Option<String>,

    /// GPU device index for embedding commands.
    #[arg(short = 'g', long = "gpu", global = true)]
    gpu: Option<u32>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// System status for workspace discovery.
    Status {
        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// Verbose output.
        #[arg(short = 'V', long)]
        verbose: bool,
    },

    /// Metadata-first context orientation for a database.
    Orient {
        /// Collection to orient on.
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },

    /// Extract text from documents (PDF, LaTeX, etc).
    Extract {
        /// File path to extract from.
        file: PathBuf,

        /// Output format (text, json).
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// Output file path.
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },

    /// Ingest documents into the knowledge base.
    Ingest {
        /// Input arXiv IDs or file paths (can mix both).
        inputs: Vec<String>,

        /// Custom document ID (single input only).
        #[arg(long)]
        id: Option<String>,

        /// Run in batch mode.
        #[arg(short = 'b', long)]
        batch: bool,

        /// Resume a previously interrupted batch.
        #[arg(short = 'r', long)]
        resume: bool,

        /// Custom metadata as JSON.
        #[arg(short = 'm', long)]
        metadata: Option<String>,

        /// Embedding task type (e.g. "code" for Jina Code LoRA).
        #[arg(short = 't', long)]
        task: Option<String>,

        /// Compliance claims (CS-NN format).
        #[arg(long)]
        claims: Vec<String>,

        /// Collection profile (arxiv, sync, default).
        #[arg(short = 'c', long)]
        collection: Option<String>,

        /// Force re-processing of existing documents.
        #[arg(short = 'f', long)]
        force: bool,

        /// Reset batch state (clear previous checkpoint).
        #[arg(long)]
        reset: bool,

        /// Maximum concurrent items (overrides config).
        #[arg(long)]
        concurrency: Option<usize>,
    },

    /// Create a compliance edge linking a document to a smell node.
    Link {
        /// Source document ID.
        source_id: String,

        /// Compliance claims (CS-NN format).
        #[arg(long)]
        claims: Vec<String>,

        /// Skip confirmation.
        #[arg(short = 'y', long)]
        force: bool,
    },

    /// arXiv paper sync and status.
    #[command(subcommand)]
    Arxiv(ArxivCmd),

    /// Database operations — query, CRUD, indexes, and graph traversal.
    #[command(subcommand)]
    Db(DbCmd),

    /// Embedding generation and service management.
    #[command(subcommand)]
    Embed(EmbedCmd),

    /// Code ingestion and graph operations.
    #[command(subcommand)]
    Codebase(CodebaseCmd),

    /// Persephone task management — create, track, review, and hand off tasks.
    #[command(subcommand)]
    Task(TaskCmd),

    /// Code smell and compliance checking.
    #[command(subcommand)]
    Smell(SmellCmd),

    /// Graph embedding operations — train and query structural embeddings.
    #[command(subcommand)]
    GraphEmbed(GraphEmbedCmd),
}

/// Initialize tracing (structured logging to stderr).
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
}

fn main() -> anyhow::Result<()> {
    // Capture raw args before clap consumes them — used for pass-through dispatch.
    let raw_args: Vec<String> = std::env::args().collect();

    let cli = Cli::parse();

    let mut config = config::load_config()?;
    config.apply_cli_overrides(cli.database.as_deref(), cli.gpu);

    // Save global opts before cli.command is consumed by the match.
    let database = cli.database;
    let gpu = cli.gpu;

    // ── Native command dispatch ──────────────────────────────────────────
    // Commands with native Rust implementations are handled here.
    // All other commands fall through to the Python CLI dispatch below.
    match cli.command {
        Commands::Ingest {
            inputs, id, batch, resume, metadata, task, claims, collection, force,
            reset, concurrency,
        } => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            let result = rt.block_on(commands::ingest::run(
                &config,
                inputs.into_iter().map(PathBuf::from).collect(),
                batch,
                metadata.as_deref(),
                &claims,
                collection.as_deref(),
                force,
                task.as_deref(),
                id.as_deref(),
                resume,
                reset,
                concurrency,
            ));
            return match result {
                Ok(()) => Ok(()),
                Err(e) => {
                    if e.downcast_ref::<commands::ingest::IngestFailure>().is_some() {
                        process::exit(1);
                    }
                    Err(e)
                }
            };
        }
        Commands::Arxiv(ArxivCmd::Sync {
            from_date, categories, max_results, batch_size, incremental,
        }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::arxiv_sync::run(
                &config,
                from_date.as_deref(),
                categories.as_deref(),
                max_results,
                batch_size,
                incremental,
            ));
        }
        Commands::Arxiv(ArxivCmd::SyncStatus { limit }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(commands::arxiv_sync::status(&config, limit));
        }
        Commands::Codebase(CodebaseCmd::Ingest { path, language, batch }) => {
            init_tracing();
            let rt = tokio::runtime::Runtime::new()?;
            let result = rt.block_on(commands::codebase_ingest::run(
                &config,
                path,
                language.as_deref(),
                batch,
            ));
            return match result {
                Ok(()) => Ok(()),
                Err(e) => {
                    if e.downcast_ref::<commands::codebase_ingest::CodebaseIngestFailure>()
                        .is_some()
                    {
                        process::exit(1);
                    }
                    Err(e)
                }
            };
        }
        _ => {} // Fall through to Python passthrough.
    }

    // ── Python passthrough (strangler-fig) ───────────────────────────────
    // Commands not yet ported to Rust dispatch to the Python `hades` CLI.
    let passthrough = strip_global_opts(&raw_args[1..]);
    let status = dispatch::dispatch_with_globals(
        database.as_deref(),
        gpu,
        &passthrough.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    )?;

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}

/// Remove global options (--database/--db, --gpu/-g) from the raw arg list,
/// since `dispatch_with_globals` re-inserts them in the format the Python CLI expects.
fn strip_global_opts(args: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut skip_next = false;

    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        match arg.as_str() {
            "--database" | "--db" | "--gpu" => {
                // These flags take a value in the next position
                skip_next = true;
                continue;
            }
            "-g" => {
                skip_next = true;
                continue;
            }
            _ => {}
        }
        // Handle --database=value and --db=value forms
        if arg.starts_with("--database=") || arg.starts_with("--db=") || arg.starts_with("--gpu=")
        {
            continue;
        }
        // Handle -g<value> (no space) form
        if arg.starts_with("-g") && arg.len() > 2 {
            let rest = &arg[2..];
            if rest.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
        }
        result.push(arg.clone());
    }
    result
}

// NOTE: The build_dispatch_args() functions that previously translated
// parsed clap args back into strings have been removed.  The passthrough
// approach (strip_global_opts + raw args) is more reliable during the
// strangler-fig phase because it avoids flag-name mismatches between
// the Rust clap definitions and the Python Typer CLI.
