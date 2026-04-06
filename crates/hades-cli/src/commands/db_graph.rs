//! Native Rust handlers for `hades db graph` commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout using the standard HADES JSON envelope.
//! Graph create/drop enforce the database safety guard (production
//! databases are read-only).

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

use super::output::{self, OutputFormat};

/// Connect, dispatch a command, and print the result in the requested format.
async fn dispatch_and_print(config: &HadesConfig, cmd: DaemonCommand, command_name: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, cmd).await?;
    output::print_output(command_name, result, &OutputFormat::Json);
    Ok(())
}

/// `hades db graph traverse START [--direction D] [--min-depth N] [--max-depth N] [--graph G]`
pub async fn run_traverse(
    config: &HadesConfig,
    start: &str,
    direction: &str,
    min_depth: u32,
    max_depth: u32,
    graph: Option<&str>,
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::DbGraphTraverse(dispatch::DbGraphTraverseParams {
            start: start.to_string(),
            direction: direction.to_string(),
            min_depth,
            max_depth,
            limit: None,
            graph: graph.map(String::from),
        }),
        "db.graph.traverse",
    )
    .await
}

/// `hades db graph shortest-path FROM TO [--graph G]`
pub async fn run_shortest_path(
    config: &HadesConfig,
    source: &str,
    target: &str,
    graph: Option<&str>,
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::DbGraphShortestPath(dispatch::DbGraphShortestPathParams {
            source: source.to_string(),
            target: target.to_string(),
            direction: "any".to_string(),
            graph: graph.map(String::from),
        }),
        "db.graph.shortest-path",
    )
    .await
}

/// `hades db graph neighbors VERTEX [--direction D] [--limit N] [--graph G]`
pub async fn run_neighbors(
    config: &HadesConfig,
    vertex: &str,
    direction: &str,
    limit: u32,
    graph: Option<&str>,
) -> Result<()> {
    dispatch_and_print(
        config,
        DaemonCommand::DbGraphNeighbors(dispatch::DbGraphNeighborsParams {
            vertex: vertex.to_string(),
            direction: direction.to_string(),
            limit: Some(limit),
            graph: graph.map(String::from),
        }),
        "db.graph.neighbors",
    )
    .await
}

/// `hades db graph list`
pub async fn run_list(config: &HadesConfig) -> Result<()> {
    dispatch_and_print(config, DaemonCommand::DbGraphList {}, "db.graph.list").await
}

/// `hades db graph create NAME [--edge-definitions JSON]`
pub async fn run_create(
    config: &HadesConfig,
    name: &str,
    edge_definitions: Option<&str>,
) -> Result<()> {
    config.require_writable_database()?;
    let edge_defs = edge_definitions
        .map(|s| serde_json::from_str(s).context("invalid --edge-definitions JSON"))
        .transpose()?;

    dispatch_and_print(
        config,
        DaemonCommand::DbGraphCreate(dispatch::DbGraphCreateParams {
            name: name.to_string(),
            edge_definitions: edge_defs,
        }),
        "db.graph.create",
    )
    .await
}

/// `hades db graph drop NAME [--drop-collections] [--force]`
pub async fn run_drop(
    config: &HadesConfig,
    name: &str,
    drop_collections: bool,
    force: bool,
) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to drop graph '{name}' without --force (-y). \
             This operation is irreversible."
        );
    }
    config.require_writable_database()?;

    dispatch_and_print(
        config,
        DaemonCommand::DbGraphDrop(dispatch::DbGraphDropParams {
            name: name.to_string(),
            drop_collections,
            force,
        }),
        "db.graph.drop",
    )
    .await
}
