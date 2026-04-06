//! Native Rust handlers for `hades status` and `hades orient` commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout via the shared output formatter
//! ([`OutputFormat`] supports JSON envelope, JSONL, and table).

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand, OrientParams, StatusParams};

use super::output::{self, OutputFormat};

/// Connect, dispatch a command, and print the result with envelope.
async fn dispatch_and_print(
    config: &HadesConfig,
    cmd: DaemonCommand,
    command_name: &str,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    output::print_output(command_name, result, &fmt);
    Ok(())
}

/// `hades status [--verbose] [--format F]`
pub async fn run_status(config: &HadesConfig, verbose: bool, format: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::Status(StatusParams { verbose }),
        "status",
        format,
    )
    .await
}

/// `hades orient [--collection C] [--format F]`
pub async fn run_orient(config: &HadesConfig, collection: Option<&str>, format: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::Orient(OrientParams {
            collection: collection.map(String::from),
        }),
        "orient",
        format,
    )
    .await
}
