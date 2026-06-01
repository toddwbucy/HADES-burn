//! `hades codebase prune-orphans` — remove orphan symbols and dangling edges.
//!
//! Maintenance command that cleans up the codebase graph after file deletions
//! that predate cascade-aware purging (see [`hades_core::dispatch`] `db_purge`,
//! which now cascades `codebase_files` deletes to symbols and edges). It removes:
//! - `codebase_symbols` whose `file_key` no longer resolves to a document in
//!   `codebase_files` (orphan symbols), and
//! - edges in every codebase edge collection whose `_from`/`_to` endpoint no
//!   longer resolves to an existing document (dangling edges).
//!
//! Symbols are pruned before edges so that edges left dangling by the symbol
//! removal are cleaned in the same run. Missing collections are treated as a
//! no-op (nothing to prune), so the command is safe on databases that have no
//! codebase graph.
//!
//! Writes only where the `hades` user holds rw grants — ArangoDB ACLs are the
//! authoritative gate. Pass `--dry-run` to report counts without modifying the
//! graph.
//!
//! Output: JSON summary to stdout; human-readable summary + logs to stderr.

use std::io::Write;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::info;

use hades_core::config::HadesConfig;
use hades_core::db::collections::CODEBASE;
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::{ArangoError, ArangoPool};

use super::output::{self, OutputFormat};

/// `hades codebase prune-orphans [--dry-run]`
pub async fn run_prune(config: &HadesConfig, dry_run: bool) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // 1. Orphan symbols: file_key no longer present in codebase_files.
    let orphan_symbols = prune_orphan_symbols(&pool, dry_run).await?;

    // 2. Dangling edges: either endpoint no longer resolves. In a real run this
    //    follows the symbol prune, so edges to just-removed orphan symbols are
    //    counted as dangling too. In a dry-run nothing is removed, so those
    //    edges still resolve and the count is a *lower bound* — see the
    //    `dry_run_note` below.
    let mut dangling = serde_json::Map::new();
    for (label, col) in [
        ("defines_edges", CODEBASE.defines_edges),
        ("calls_edges", CODEBASE.calls_edges),
        ("implements_edges", CODEBASE.implements_edges),
        ("imports_edges", CODEBASE.imports_edges),
    ] {
        let n = prune_dangling_edges(&pool, col, dry_run).await?;
        dangling.insert(label.to_string(), json!(n));
    }
    let total_edges: u64 = dangling.values().filter_map(Value::as_u64).sum();

    let mut report = json!({
        "dry_run": dry_run,
        "orphan_symbols": orphan_symbols,
        "dangling_edges": dangling,
    });
    if dry_run {
        // The dangling-edge count excludes edges that only become dangling once
        // the orphan symbols are removed (nothing is deleted in a dry-run), so
        // a real run may remove at least this many edges.
        report["dangling_edges_note"] = json!(
            "lower bound: excludes edges that become dangling after orphan \
             symbols are removed; a real run may remove more"
        );
    }
    output::print_output("codebase.prune_orphans", report, &OutputFormat::Json);

    // Human-readable summary to stderr.
    let mut err = std::io::stderr().lock();
    if dry_run {
        let _ = writeln!(
            err,
            "Prune (dry-run): would remove {orphan_symbols} orphan symbols, \
             ≥{total_edges} dangling edges",
        );
    } else {
        let _ = writeln!(
            err,
            "Prune: removed {orphan_symbols} orphan symbols, {total_edges} dangling edges",
        );
    }

    if dry_run {
        info!(orphan_symbols, total_edges, "prune dry-run complete");
    } else {
        info!(orphan_symbols, total_edges, "prune complete");
    }
    Ok(())
}

/// Remove (or, in dry-run, count) symbols whose `file_key` is absent from
/// `codebase_files`.
async fn prune_orphan_symbols(pool: &ArangoPool, dry_run: bool) -> Result<u64> {
    let aql = if dry_run {
        "FOR s IN @@symbols \
         FILTER DOCUMENT(CONCAT(@files_name, '/', s.file_key)) == null \
         COLLECT WITH COUNT INTO n RETURN n"
    } else {
        "LET removed = (FOR s IN @@symbols \
            FILTER DOCUMENT(CONCAT(@files_name, '/', s.file_key)) == null \
            REMOVE s IN @@symbols RETURN 1) \
         RETURN LENGTH(removed)"
    };
    let bind = json!({
        "@symbols": CODEBASE.symbols,
        "files_name": CODEBASE.files,
    });
    let n = run_count(pool, aql, &bind, target(dry_run)).await?;
    info!(orphan_symbols = n, dry_run, "pruned orphan symbols");
    Ok(n)
}

/// Remove (or, in dry-run, count) edges in `col` whose `_from` or `_to`
/// endpoint no longer resolves to an existing document.
async fn prune_dangling_edges(pool: &ArangoPool, col: &str, dry_run: bool) -> Result<u64> {
    let aql = if dry_run {
        "FOR e IN @@edges \
         FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null \
         COLLECT WITH COUNT INTO n RETURN n"
    } else {
        "LET removed = (FOR e IN @@edges \
            FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null \
            REMOVE e IN @@edges RETURN 1) \
         RETURN LENGTH(removed)"
    };
    let bind = json!({ "@edges": col });
    let n = run_count(pool, aql, &bind, target(dry_run)).await?;
    info!(
        collection = col,
        dangling_edges = n,
        dry_run,
        "pruned dangling edges"
    );
    Ok(n)
}

/// Route through the reader for dry-runs, the writer for mutating prunes.
fn target(dry_run: bool) -> ExecutionTarget {
    if dry_run {
        ExecutionTarget::Reader
    } else {
        ExecutionTarget::Writer
    }
}

/// Execute a count-returning AQL query. A missing collection (ArangoDB error
/// 1203) is treated as zero — there is nothing to prune.
async fn run_count(
    pool: &ArangoPool,
    aql: &str,
    bind: &Value,
    target: ExecutionTarget,
) -> Result<u64> {
    match query::query_single(pool, aql, Some(bind), target).await {
        Ok(v) => Ok(v.and_then(|v| v.as_u64()).unwrap_or(0)),
        Err(ArangoError::Api {
            error_num: 1203, ..
        }) => Ok(0),
        Err(e) => Err(e).context("prune AQL query failed"),
    }
}
