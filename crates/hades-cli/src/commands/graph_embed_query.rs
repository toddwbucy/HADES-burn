//! Native Rust implementation of `hades graph-embed embed` and
//! `hades graph-embed neighbors` commands.
//!
//! Thin CLI adapter: creates an ArangoPool and delegates to the shared
//! dispatch layer in [`hades_core::dispatch`].

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, GraphEmbedEmbedParams, GraphEmbedNeighborsParams};

use super::output::{self, OutputFormat};

// ---------------------------------------------------------------------------
// graph-embed embed
// ---------------------------------------------------------------------------

/// Run the `graph-embed embed <node_id>` command.
pub async fn run_embed(config: &HadesConfig, node_id: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    let data = dispatch::dispatch(
        &pool,
        config,
        dispatch::DaemonCommand::GraphEmbedEmbed(GraphEmbedEmbedParams {
            node_id: node_id.to_string(),
        }),
    )
    .await?;

    let result_data = json!({ "status": "success", "data": data });
    output::print_output("graph-embed.query", result_data, &OutputFormat::Json);
    Ok(())
}

// ---------------------------------------------------------------------------
// graph-embed neighbors
// ---------------------------------------------------------------------------

/// Run the `graph-embed neighbors <node_id> --limit N` command.
pub async fn run_neighbors(config: &HadesConfig, node_id: &str, limit: u32) -> Result<()> {
    let pool = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    let data = dispatch::dispatch(
        &pool,
        config,
        dispatch::DaemonCommand::GraphEmbedNeighbors(GraphEmbedNeighborsParams {
            node_id: node_id.to_string(),
            limit: Some(limit),
        }),
    )
    .await?;

    let result_data = json!({
        "status": "success",
        "data": data,
        "metadata": {
            "count": data.get("neighbors").and_then(|n| n.as_array()).map(|a| a.len()).unwrap_or(0),
        },
    });
    output::print_output("graph-embed.similar", result_data, &OutputFormat::Json);
    Ok(())
}
