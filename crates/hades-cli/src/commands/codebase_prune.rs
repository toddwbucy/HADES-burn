//! `hades codebase prune-orphans` — remove orphaned children and dangling edges.
//!
//! Maintenance command that cleans up the codebase graph after file deletions
//! that predate cascade-aware purging (see [`hades_core::dispatch`] `db_purge`,
//! which now cascades `codebase_files` deletes to symbols and edges). It removes
//! every child record whose owning document no longer exists:
//! - `codebase_symbols` whose `file_key` no longer resolves to a document in
//!   `codebase_files` (orphan symbols),
//! - `codebase_chunks` whose `file_key` no longer resolves (orphan chunks),
//! - `codebase_embeddings` whose `file_key` **or** `chunk_key` no longer
//!   resolves (orphan embeddings), and
//! - edges in every codebase edge collection whose `_from`/`_to` endpoint no
//!   longer resolves to an existing document (dangling edges).
//!
//! Those four map onto `codebase validate` invariants #6/#7, #1, #2/#3, and the
//! edge-endpoint checks respectively. Until #157 this command covered only the
//! symbols and edges, so `validate` could report orphan chunks and embeddings
//! that the obvious remedy silently declined to touch — a no-op that looked like
//! a fix.
//!
//! Sweep order is cascade-aware: symbols before edges (so edges left dangling by
//! the symbol removal are caught in the same run) and chunks before embeddings
//! (so embeddings orphaned by the chunk removal are caught too). Missing
//! collections are treated as a no-op, so the command is safe on databases that
//! have no codebase graph.
//!
//! This is the sweep for records whose owning file node is **already gone**.
//! To retire a file node that still exists because its source file was deleted,
//! use [`super::codebase_retire`]; to find such files, use
//! [`super::codebase_drift`].
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

    // 2. Orphan chunks: file_key no longer present in codebase_files.
    //    Swept before embeddings so embeddings left orphaned by this removal are
    //    caught in the same run (see the embedding step's chunk_key clause).
    let orphan_chunks = prune_orphan_chunks(&pool, dry_run).await?;

    // 3. Orphan embeddings: file_key gone, or chunk_key no longer resolves.
    let orphan_embeddings = prune_orphan_embeddings(&pool, dry_run).await?;

    // 4. Dangling edges: either endpoint no longer resolves. In a real run this
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
        "orphan_chunks": orphan_chunks,
        "orphan_embeddings": orphan_embeddings,
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
        // Same shape for embeddings: a dry-run leaves the orphan chunks in
        // place, so embeddings that would be orphaned by their removal are not
        // counted yet.
        report["orphan_embeddings_note"] = json!(
            "lower bound: excludes embeddings that become orphaned after orphan \
             chunks are removed; a real run may remove more"
        );
    }
    output::print_output("codebase.prune_orphans", report, &OutputFormat::Json);

    // Human-readable summary to stderr.
    let mut err = std::io::stderr().lock();
    if dry_run {
        let _ = writeln!(
            err,
            "Prune (dry-run): would remove {orphan_symbols} orphan symbols, \
             {orphan_chunks} orphan chunks, ≥{orphan_embeddings} orphan embeddings, \
             ≥{total_edges} dangling edges",
        );
    } else {
        let _ = writeln!(
            err,
            "Prune: removed {orphan_symbols} orphan symbols, {orphan_chunks} orphan chunks, \
             {orphan_embeddings} orphan embeddings, {total_edges} dangling edges",
        );
    }

    if dry_run {
        info!(
            orphan_symbols,
            orphan_chunks, orphan_embeddings, total_edges, "prune dry-run complete"
        );
    } else {
        info!(
            orphan_symbols,
            orphan_chunks, orphan_embeddings, total_edges, "prune complete"
        );
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

/// Remove (or, in dry-run, count) chunks whose `file_key` is absent from
/// `codebase_files`.
///
/// These are the leftovers of a file node deleted without its children —
/// `codebase validate` invariant #1 (`chunk_file_ref`).
async fn prune_orphan_chunks(pool: &ArangoPool, dry_run: bool) -> Result<u64> {
    let aql = if dry_run {
        "FOR c IN @@chunks \
         FILTER DOCUMENT(CONCAT(@files_name, '/', c.file_key)) == null \
         COLLECT WITH COUNT INTO n RETURN n"
    } else {
        "LET removed = (FOR c IN @@chunks \
            FILTER DOCUMENT(CONCAT(@files_name, '/', c.file_key)) == null \
            REMOVE c IN @@chunks RETURN 1) \
         RETURN LENGTH(removed)"
    };
    let bind = json!({
        "@chunks": CODEBASE.chunks,
        "files_name": CODEBASE.files,
    });
    let n = run_count(pool, aql, &bind, target(dry_run)).await?;
    info!(orphan_chunks = n, dry_run, "pruned orphan chunks");
    Ok(n)
}

/// Remove (or, in dry-run, count) embeddings whose owning file **or** chunk no
/// longer exists.
///
/// Covers both `codebase validate` invariant #3 (`embedding_file_ref`) and #2
/// (`embedding_chunk_ref`). The `chunk_key` clause is what makes the sweep
/// cascade: chunks are pruned immediately before this step, so embeddings
/// orphaned by that removal are cleaned in the same run rather than surviving
/// as a fresh #2 violation.
async fn prune_orphan_embeddings(pool: &ArangoPool, dry_run: bool) -> Result<u64> {
    let aql = if dry_run {
        "FOR e IN @@embeddings \
         FILTER DOCUMENT(CONCAT(@files_name, '/', e.file_key)) == null \
             OR DOCUMENT(CONCAT(@chunks_name, '/', e.chunk_key)) == null \
         COLLECT WITH COUNT INTO n RETURN n"
    } else {
        "LET removed = (FOR e IN @@embeddings \
            FILTER DOCUMENT(CONCAT(@files_name, '/', e.file_key)) == null \
                OR DOCUMENT(CONCAT(@chunks_name, '/', e.chunk_key)) == null \
            REMOVE e IN @@embeddings RETURN 1) \
         RETURN LENGTH(removed)"
    };
    let bind = json!({
        "@embeddings": CODEBASE.embeddings,
        "files_name": CODEBASE.files,
        "chunks_name": CODEBASE.chunks,
    });
    let n = run_count(pool, aql, &bind, target(dry_run)).await?;
    info!(orphan_embeddings = n, dry_run, "pruned orphan embeddings");
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

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hades_core::db::crud;
    use serde_json::Value;

    /// Shared prefix for every fixture these tests write.
    const FIXTURE_PREFIX: &str = "__hades_test157_";

    fn test_pool() -> Option<ArangoPool> {
        let socket = std::path::PathBuf::from(
            std::env::var("ARANGO_SOCKET")
                .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
        );
        if !socket.exists() {
            if std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true") {
                panic!(
                    "ARANGO_TESTS is set but socket not found at {}",
                    socket.display()
                );
            }
            eprintln!(
                "skipping: ArangoDB socket not found at {}",
                socket.display()
            );
            return None;
        }
        let Ok(password) = std::env::var("ARANGO_PASSWORD") else {
            eprintln!("skipping: ARANGO_PASSWORD not set");
            return None;
        };
        let client =
            hades_core::db::ArangoClient::with_socket(socket, "bident_burn", "root", &password);
        Some(ArangoPool::new(client.clone(), client))
    }

    async fn collections_present(pool: &ArangoPool) -> bool {
        for col in [CODEBASE.files, CODEBASE.chunks, CODEBASE.embeddings] {
            let aql = "RETURN LENGTH(FOR d IN @@col LIMIT 1 RETURN 1)";
            let bind = json!({ "@col": col });
            if query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
                .await
                .is_err()
            {
                return false;
            }
        }
        true
    }

    /// Remove every fixture this test family may have left behind, any run.
    async fn cleanup(pool: &ArangoPool) {
        for col in [CODEBASE.chunks, CODEBASE.embeddings, CODEBASE.symbols] {
            let aql = "FOR d IN @@col FILTER STARTS_WITH(d.file_key, @p) REMOVE d IN @@col";
            let bind = json!({ "@col": col, "p": FIXTURE_PREFIX });
            let _ =
                query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Writer).await;
        }
        let aql = "FOR d IN @@files FILTER STARTS_WITH(d._key, @p) REMOVE d IN @@files";
        let bind = json!({ "@files": CODEBASE.files, "p": FIXTURE_PREFIX });
        let _ = query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Writer).await;
    }

    async fn count_with_prefix(pool: &ArangoPool, col: &str) -> u64 {
        let aql = "FOR d IN @@col FILTER STARTS_WITH(d.file_key, @p) COLLECT WITH COUNT INTO n \
                   RETURN n";
        let bind = json!({ "@col": col, "p": FIXTURE_PREFIX });
        query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    /// Orphan chunks and embeddings must be swept, and the embedding sweep must
    /// cascade off the chunk removal within a single run (#157).
    ///
    /// Fixture shape, all with **no** owning `codebase_files` document:
    /// - `orphan-file`  — 2 chunks + 2 embeddings, file node absent
    /// - `live-file`    — a real file node, 1 chunk, and 1 embedding pointing at
    ///   a `chunk_key` that does not exist (invariant #2 only)
    ///
    /// The live file proves the sweep is targeted: its node and its valid chunk
    /// must survive, and only the dangling embedding is removed.
    #[tokio::test]
    async fn prune_removes_orphan_chunks_and_cascades_to_embeddings() {
        let Some(pool) = test_pool() else { return };
        if !collections_present(&pool).await {
            eprintln!("skipping: target database has no codebase collections");
            return;
        }
        cleanup(&pool).await;

        let pid = std::process::id();
        let orphan_fk = format!("{FIXTURE_PREFIX}{pid}_orphan");
        let live_fk = format!("{FIXTURE_PREFIX}{pid}_live");

        // A live file node — this one must survive untouched.
        let files: Vec<Value> = vec![json!({ "_key": live_fk, "kind": "file" })];
        crud::insert_documents(&pool, CODEBASE.files, &files, true)
            .await
            .expect("insert file node");

        // Chunks: two orphaned (no file node), one live.
        let chunks: Vec<Value> = vec![
            json!({ "_key": format!("{orphan_fk}_chunk_0"), "file_key": orphan_fk }),
            json!({ "_key": format!("{orphan_fk}_chunk_1"), "file_key": orphan_fk }),
            json!({ "_key": format!("{live_fk}_chunk_0"),   "file_key": live_fk }),
        ];
        crud::insert_documents(&pool, CODEBASE.chunks, &chunks, true)
            .await
            .expect("insert chunks");

        // Embeddings: two on the orphaned chunks (file_key AND chunk_key gone
        // once chunks are swept), one on the live chunk (valid, must survive),
        // one on the live file but pointing at a chunk that never existed.
        let embeddings: Vec<Value> = vec![
            json!({ "_key": format!("{orphan_fk}_chunk_0_emb"), "file_key": orphan_fk,
                    "chunk_key": format!("{orphan_fk}_chunk_0") }),
            json!({ "_key": format!("{orphan_fk}_chunk_1_emb"), "file_key": orphan_fk,
                    "chunk_key": format!("{orphan_fk}_chunk_1") }),
            json!({ "_key": format!("{live_fk}_chunk_0_emb"),   "file_key": live_fk,
                    "chunk_key": format!("{live_fk}_chunk_0") }),
            json!({ "_key": format!("{live_fk}_chunk_9_emb"),   "file_key": live_fk,
                    "chunk_key": format!("{live_fk}_chunk_9") }),
        ];
        crud::insert_documents(&pool, CODEBASE.embeddings, &embeddings, true)
            .await
            .expect("insert embeddings");

        assert_eq!(count_with_prefix(&pool, CODEBASE.chunks).await, 3);
        assert_eq!(count_with_prefix(&pool, CODEBASE.embeddings).await, 4);

        // Dry-run must not modify anything.
        prune_orphan_chunks(&pool, true).await.unwrap();
        prune_orphan_embeddings(&pool, true).await.unwrap();
        assert_eq!(
            count_with_prefix(&pool, CODEBASE.chunks).await,
            3,
            "dry-run must not delete chunks"
        );
        assert_eq!(
            count_with_prefix(&pool, CODEBASE.embeddings).await,
            4,
            "dry-run must not delete embeddings"
        );

        // Real run, in the same order run_prune uses.
        let removed_chunks = prune_orphan_chunks(&pool, false).await.unwrap();
        let removed_embs = prune_orphan_embeddings(&pool, false).await.unwrap();

        assert_eq!(removed_chunks, 2, "both orphan chunks must be removed");
        // 2 orphaned by file_key + 1 dangling chunk_key on the live file. The
        // two orphan-file embeddings would also be caught by the chunk_key
        // clause now that their chunks are gone — that cascade is the point.
        assert_eq!(
            removed_embs, 3,
            "orphan-file embeddings plus the dangling-chunk_key embedding"
        );

        assert_eq!(
            count_with_prefix(&pool, CODEBASE.chunks).await,
            1,
            "the live file's chunk must survive"
        );
        assert_eq!(
            count_with_prefix(&pool, CODEBASE.embeddings).await,
            1,
            "the live file's valid embedding must survive"
        );

        cleanup(&pool).await;
    }
}
