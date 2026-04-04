//! Native Rust handlers for `hades db graph` commands.
//!
//! Each function constructs a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout.  Graph create/drop enforce the database
//! safety guard (production databases are read-only).

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

/// `hades db graph traverse START [--direction D] [--min-depth N] [--max-depth N] [--graph G]`
pub async fn run_traverse(
    config: &HadesConfig,
    start: &str,
    direction: &str,
    min_depth: u32,
    max_depth: u32,
    graph: Option<&str>,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGraphTraverse {
            start: start.to_string(),
            direction: direction.to_string(),
            min_depth,
            max_depth,
            limit: None,
            graph: graph.map(String::from),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db graph shortest-path FROM TO [--graph G]`
pub async fn run_shortest_path(
    config: &HadesConfig,
    source: &str,
    target: &str,
    graph: Option<&str>,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGraphShortestPath {
            source: source.to_string(),
            target: target.to_string(),
            direction: "any".to_string(),
            graph: graph.map(String::from),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db graph neighbors VERTEX [--direction D] [--limit N] [--graph G]`
pub async fn run_neighbors(
    config: &HadesConfig,
    vertex: &str,
    direction: &str,
    limit: u32,
    graph: Option<&str>,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGraphNeighbors {
            vertex: vertex.to_string(),
            direction: direction.to_string(),
            limit: Some(limit),
            graph: graph.map(String::from),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db graph list`
pub async fn run_list(config: &HadesConfig) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(&pool, config, DaemonCommand::DbGraphList {}).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
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

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGraphCreate {
            name: name.to_string(),
            edge_definitions: edge_defs,
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
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

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGraphDrop {
            name: name.to_string(),
            drop_collections,
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
