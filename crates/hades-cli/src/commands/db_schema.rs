//! Native Rust handlers for `hades db schema` commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout using the standard HADES JSON envelope.

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

use super::output::{self, OutputFormat};

/// Connect, dispatch a command, and print the result.
async fn dispatch_and_print(config: &HadesConfig, cmd: DaemonCommand, command_name: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    output::print_output(command_name, result, &OutputFormat::Json);
    Ok(())
}

/// `hades db schema init --seed SEED`
pub async fn run_init(config: &HadesConfig, seed: &str) -> Result<()> {
    config.require_writable_database()?;

    dispatch_and_print(
        config,
        DaemonCommand::DbSchemaInit(dispatch::DbSchemaInitParams {
            seed: seed.to_string(),
        }),
        "db.schema.init",
    )
    .await
}

/// `hades db schema list`
pub async fn run_list(config: &HadesConfig) -> Result<()> {
    dispatch_and_print(config, DaemonCommand::DbSchemaList {}, "db.schema.list").await
}

/// `hades db schema show NAME`
pub async fn run_show(config: &HadesConfig, name: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::DbSchemaShow(dispatch::DbSchemaShowParams {
            name: name.to_string(),
        }),
        "db.schema.show",
    )
    .await
}

/// `hades db schema version`
pub async fn run_version(config: &HadesConfig) -> Result<()> {
    dispatch_and_print(config, DaemonCommand::DbSchemaVersion {}, "db.schema.version").await
}
