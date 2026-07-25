//! `hades-viewer` — a local, WebGL graph-visualization frontend for HADES graphs.
//!
//! Loosely coupled by design: this binary depends on no other HADES crate. It
//! consumes the `hades` CLI's JSON/jsonl output contract (see [`assemble`]) and
//! renders the result with a WebGL viewer. The backend we know works stays
//! untouched — the only change to the workspace is this crate's membership.
//!
//! Two modes:
//!   * `hades-viewer dump  --db <db> --graph <g>`  -> print the graph-JSON to stdout
//!     (proves the coupling contract with no frontend involved).
//!   * `hades-viewer serve --db <db> [--bind ...]` -> serve the viewer + graph API.

mod assemble;
mod contract;
mod server;

use anyhow::Result;
use assemble::Backend;
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hades-viewer", version, about = "WebGL graph viewer for HADES graphs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Assemble a graph snapshot and print it as JSON (no server, no frontend).
    Dump(DumpArgs),
    /// Serve the interactive viewer over loopback.
    Serve(ServeArgs),
}

/// Shared backend-invocation options.
#[derive(Args, Clone)]
struct BackendArgs {
    /// Source database (passed to `hades --db`).
    #[arg(long)]
    db: String,
    /// Path to the `hades` binary (defaults to `hades` on PATH).
    #[arg(long, default_value = "hades")]
    hades_bin: String,
    /// Cap documents per collection; marks the snapshot partial when hit.
    #[arg(long)]
    limit: Option<u32>,
}

impl BackendArgs {
    fn backend(&self) -> Backend {
        Backend {
            bin: self.hades_bin.clone(),
            database: self.db.clone(),
            limit: self.limit,
        }
    }
}

#[derive(Args)]
struct DumpArgs {
    #[command(flatten)]
    backend: BackendArgs,
    /// Named graph to assemble.
    #[arg(long)]
    graph: String,
    /// Pretty-print the JSON.
    #[arg(long)]
    pretty: bool,
}

#[derive(Args)]
struct ServeArgs {
    /// Databases the viewer may serve — repeat `--db` to restrict to a specific
    /// set (first is the default). Omit entirely to serve EVERY database the
    /// hades user can access (ArangoDB ACLs are the gate).
    #[arg(long = "db")]
    databases: Vec<String>,
    /// Path to the `hades` binary (defaults to `hades` on PATH).
    #[arg(long, default_value = "hades")]
    hades_bin: String,
    /// Cap documents per collection; marks the snapshot partial when hit.
    #[arg(long)]
    limit: Option<u32>,
    /// Require HTTP Basic auth with this password (any username). Mandatory when
    /// binding to a non-loopback address, since the viewer reads whole databases.
    #[arg(long)]
    password: Option<String>,
    /// Address to bind (loopback or a private LAN IP only).
    #[arg(long, default_value = "127.0.0.1:8088")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hades_viewer=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Dump(args) => {
            let snapshot = args.backend.backend().assemble(&args.graph).await?;
            let json = if args.pretty {
                serde_json::to_string_pretty(&snapshot)?
            } else {
                serde_json::to_string(&snapshot)?
            };
            println!("{json}");
            Ok(())
        }
        Command::Serve(args) => {
            server::serve(
                args.hades_bin,
                args.limit,
                args.databases,
                args.password,
                &args.bind,
            )
            .await
        }
    }
}
