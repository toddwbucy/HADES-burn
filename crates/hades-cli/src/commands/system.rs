//! Native Rust handlers for `hades status` and `hades orient` commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout.  The `format` parameter is accepted for
//! CLI compatibility; currently only JSON output is implemented.

use anyhow::{Context, Result};
use serde_json::Value;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand, OrientParams, StatusParams};

/// Connect, dispatch a command, and print the result in the requested format.
async fn dispatch_and_print(config: &HadesConfig, cmd: DaemonCommand, format: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&result)?),
        _ => print_text(&result),
    }
    Ok(())
}

/// Simple text rendering of a JSON value — one key: value per line.
fn print_text(value: &Value) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                match v {
                    Value::Object(_) | Value::Array(_) => {
                        println!("{k}:");
                        println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
                    }
                    _ => println!("{k}: {v}"),
                }
            }
        }
        _ => println!("{}", serde_json::to_string_pretty(value).unwrap_or_default()),
    }
}

/// `hades status [--verbose] [--format F]`
pub async fn run_status(config: &HadesConfig, verbose: bool, format: &str) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::Status(StatusParams { verbose }),
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
        format,
    )
    .await
}
