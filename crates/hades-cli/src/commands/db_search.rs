//! Native Rust handler for `hades db query` — semantic search.
//!
//! Thin CLI adapter: creates an ArangoPool and delegates to the shared
//! dispatch layer in [`hades_core::dispatch`], which is the same code path
//! the daemon and the MCP `db_query` tool use. The search pipeline itself
//! (embed, cosine top-K, detail fetch, hybrid/structural reranking) lives
//! in `dispatch::handlers::db_query` so the CLI and the remote surfaces
//! cannot drift apart. See #187.

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

use super::output::{self, OutputFormat};

/// `hades db query TEXT [--limit N] [--collection C] [--hybrid] [--structural] [--format F]`
#[allow(clippy::too_many_arguments)]
pub async fn run_query(
    config: &HadesConfig,
    search_text: &str,
    limit: u32,
    collection: Option<&str>,
    hybrid: bool,
    structural: bool,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbQuery {
            text: search_text.to_string(),
            limit: Some(limit),
            collection: collection.map(str::to_string),
            hybrid,
            // `--rerank` is rejected in main.rs before reaching this point.
            rerank: false,
            structural,
        },
    )
    .await?;

    output::print_output("db.query", data, &fmt);
    Ok(())
}
