//! Native Rust handlers for `hades status` and `hades orient` commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout.

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand, OrientParams, StatusParams};

/// Connect, dispatch a command, and pretty-print the JSON result.
async fn dispatch_and_print(config: &HadesConfig, cmd: DaemonCommand) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades status [--verbose]`
pub async fn run_status(config: &HadesConfig, verbose: bool) -> Result<()> {
    dispatch_and_print(config, DaemonCommand::Status(StatusParams { verbose })).await
}

/// `hades orient [--collection C]`
pub async fn run_orient(config: &HadesConfig, collection: Option<&str>) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::Orient(OrientParams {
            collection: collection.map(String::from),
        }),
    )
    .await
}
