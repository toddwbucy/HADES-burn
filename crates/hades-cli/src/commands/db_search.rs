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

    // `HADES_DEFAULT_COLLECTION` is resolved here, in the CLI, and never inside
    // the shared handler. It is a per-shell convenience belonging to whoever
    // typed the command; the same handler also runs inside a long-lived daemon
    // serving several databases to several agents, where reading it would make
    // one operator's ambient environment a process-wide default that crosses
    // database boundaries (#191 review).
    let collection = collection
        .map(str::to_string)
        .or_else(default_collection_from_env);

    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbQuery {
            text: search_text.to_string(),
            limit: Some(limit),
            collection,
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

/// The `HADES_DEFAULT_COLLECTION` profile name, if the environment names one.
///
/// Blank or whitespace-only is treated as unset, matching the old CLI. An
/// unknown name is deliberately NOT filtered here: `resolve_query_profile`
/// rejects it, so a typo in the variable produces a loud error rather than a
/// silent sweep of the wrong collections.
fn default_collection_from_env() -> Option<String> {
    std::env::var("HADES_DEFAULT_COLLECTION")
        .ok()
        .filter(|v| !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    /// The env var must not be read anywhere the daemon can reach.
    ///
    /// `db.query` runs in-process for the CLI *and* inside the shared daemon and
    /// MCP server. Reading the variable in the handler made one operator's shell
    /// setting a process-global default that applied to every database and every
    /// agent, turning a wrong-profile search into a `success` with zero results.
    /// The CLI resolves it before dispatch instead, so this test guards the
    /// boundary rather than the behaviour.
    #[test]
    fn env_default_is_not_read_by_the_shared_dispatch_layer() {
        let dispatch_src = include_str!("../../../hades-core/src/dispatch.rs");
        let handler_reads_env = dispatch_src
            .lines()
            .filter(|l| l.contains("HADES_DEFAULT_COLLECTION"))
            .any(|l| l.contains("env::var") || l.contains("env :: var"));
        assert!(
            !handler_reads_env,
            "dispatch.rs must not read HADES_DEFAULT_COLLECTION: it runs inside the \
             daemon, where the variable is a process-wide global that crosses \
             database boundaries. Resolve it in the CLI and pass it in."
        );
    }
}
