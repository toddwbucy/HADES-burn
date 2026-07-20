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

/// Counts from one full sweep, in the order they were performed.
#[derive(Debug, Default)]
pub(crate) struct PruneCounts {
    pub orphan_symbols: u64,
    pub orphan_chunks: u64,
    pub orphan_embeddings: u64,
    pub dangling: serde_json::Map<String, Value>,
}

/// Run every sweep, in the order the cascades require.
///
/// Separated from [`run_prune`] so the **ordering itself** is testable. Calling
/// the individual sweeps directly in a test would pass even if this sequence
/// were reordered, which is precisely the failure that would break the cascade.
pub(crate) async fn sweep_orphans(pool: &ArangoPool, dry_run: bool) -> Result<PruneCounts> {
    // 1. Orphan symbols: file_key no longer present in codebase_files.
    let orphan_symbols = prune_orphan_symbols(pool, dry_run).await?;

    // 2. Orphan chunks: file_key no longer present in codebase_files.
    //    MUST precede the embedding sweep: an embedding whose own file_key is
    //    fine but whose chunk_key points at one of these chunks only becomes an
    //    orphan once the chunk is gone.
    let orphan_chunks = prune_orphan_chunks(pool, dry_run).await?;

    // 3. Orphan embeddings: file_key gone, or chunk_key no longer resolves.
    let orphan_embeddings = prune_orphan_embeddings(pool, dry_run).await?;

    // 4. Dangling edges: either endpoint no longer resolves. In a real run this
    //    follows the symbol prune, so edges to just-removed orphan symbols are
    //    counted as dangling too. In a dry-run nothing is removed, so those
    //    edges still resolve and the count is a *lower bound*.
    let mut dangling = serde_json::Map::new();
    for (label, col) in [
        ("defines_edges", CODEBASE.defines_edges),
        ("calls_edges", CODEBASE.calls_edges),
        ("implements_edges", CODEBASE.implements_edges),
        ("imports_edges", CODEBASE.imports_edges),
    ] {
        let n = prune_dangling_edges(pool, col, dry_run).await?;
        dangling.insert(label.to_string(), json!(n));
    }

    Ok(PruneCounts {
        orphan_symbols,
        orphan_chunks,
        orphan_embeddings,
        dangling,
    })
}

/// `hades codebase prune-orphans [--dry-run]`
pub async fn run_prune(config: &HadesConfig, dry_run: bool) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let PruneCounts {
        orphan_symbols,
        orphan_chunks,
        orphan_embeddings,
        dangling,
    } = sweep_orphans(&pool, dry_run).await?;
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

/// Remove (or, in dry-run, count) documents in `col` whose `file_key` no longer
/// resolves to a document in `codebase_files`.
///
/// Shared by the symbol and chunk sweeps, which differ only by collection. The
/// embedding sweep is deliberately *not* folded in here: since #157 it builds
/// its filter dynamically (the `chunk_key` clause, and its suppression when
/// `codebase_chunks` is absent), so collapsing all three would mean a helper
/// whose body is mostly branching on which caller it has.
async fn prune_orphans_by_file_key(pool: &ArangoPool, col: &str, dry_run: bool) -> Result<u64> {
    let aql = if dry_run {
        "FOR d IN @@col \
         FILTER DOCUMENT(CONCAT(@files_name, '/', d.file_key)) == null \
         COLLECT WITH COUNT INTO n RETURN n"
    } else {
        "LET removed = (FOR d IN @@col \
            FILTER DOCUMENT(CONCAT(@files_name, '/', d.file_key)) == null \
            REMOVE d IN @@col RETURN 1) \
         RETURN LENGTH(removed)"
    };
    let bind = json!({ "@col": col, "files_name": CODEBASE.files });
    run_count(pool, aql, &bind, target(dry_run)).await
}

/// Remove (or, in dry-run, count) symbols whose `file_key` is absent from
/// `codebase_files`.
async fn prune_orphan_symbols(pool: &ArangoPool, dry_run: bool) -> Result<u64> {
    let n = prune_orphans_by_file_key(pool, CODEBASE.symbols, dry_run).await?;
    info!(orphan_symbols = n, dry_run, "pruned orphan symbols");
    Ok(n)
}

/// Remove (or, in dry-run, count) chunks whose `file_key` is absent from
/// `codebase_files`.
///
/// These are the leftovers of a file node deleted without its children —
/// `codebase validate` invariant #1 (`chunk_file_ref`).
async fn prune_orphan_chunks(pool: &ArangoPool, dry_run: bool) -> Result<u64> {
    let n = prune_orphans_by_file_key(pool, CODEBASE.chunks, dry_run).await?;
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
    // Whether the chunk_key clause is safe to apply at all.
    //
    // `DOCUMENT()` returns null for a *missing collection* just as it does for a
    // missing document, so if `codebase_chunks` does not exist the clause would
    // classify every embedding as an orphan and delete the lot — contradicting
    // this module's "missing collections are a no-op" guarantee, in the most
    // destructive possible direction. Check first, and fall back to the
    // file_key-only filter.
    let chunks_exist = collection_exists(pool, CODEBASE.chunks).await;

    // The `e.chunk_key != null` guard matters for the same reason: with a null
    // or absent field, `CONCAT('codebase_chunks/', null)` yields the bare
    // collection prefix, `DOCUMENT()` returns null, and a record that simply
    // never carried a chunk reference would be deleted as an orphan.
    let filter = if chunks_exist {
        "FILTER DOCUMENT(CONCAT(@files_name, '/', e.file_key)) == null \
             OR (e.chunk_key != null \
                 AND DOCUMENT(CONCAT(@chunks_name, '/', e.chunk_key)) == null)"
    } else {
        "FILTER DOCUMENT(CONCAT(@files_name, '/', e.file_key)) == null"
    };
    let aql = if dry_run {
        format!("FOR e IN @@embeddings {filter} COLLECT WITH COUNT INTO n RETURN n")
    } else {
        format!(
            "LET removed = (FOR e IN @@embeddings {filter} \
                REMOVE e IN @@embeddings RETURN 1) \
             RETURN LENGTH(removed)"
        )
    };
    let mut bind = json!({
        "@embeddings": CODEBASE.embeddings,
        "files_name": CODEBASE.files,
    });
    if chunks_exist {
        bind["chunks_name"] = json!(CODEBASE.chunks);
    }
    let n = run_count(pool, &aql, &bind, target(dry_run)).await?;
    info!(
        orphan_embeddings = n,
        dry_run, chunks_exist, "pruned orphan embeddings"
    );
    Ok(n)
}

/// Whether `col` exists. A missing collection (1203) is `false`; any other error
/// is treated as "exists" so a transient failure cannot widen the delete filter.
async fn collection_exists(pool: &ArangoPool, col: &str) -> bool {
    let aql = "RETURN LENGTH(FOR d IN @@col LIMIT 1 RETURN 1)";
    let bind = json!({ "@col": col });
    !matches!(
        query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader).await,
        Err(ArangoError::Api {
            error_num: 1203,
            ..
        })
    )
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
    use hades_core::db::{ArangoClient, crud};

    /// Create a throwaway database, run `f` against it, then drop it.
    ///
    /// These sweeps are **collection-global** — they delete every orphan in the
    /// database, not just fixture rows. Running them against a shared database
    /// would make exact-count assertions depend on whatever else happens to be
    /// there, *and* would delete unrelated records as a side effect. An isolated
    /// database makes the counts exact and the blast radius nil, and lets
    /// concurrent runs coexist.
    async fn with_temp_db<F, Fut>(f: F)
    where
        F: FnOnce(ArangoPool) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
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
            return;
        }
        let Ok(password) = std::env::var("ARANGO_PASSWORD") else {
            eprintln!("skipping: ARANGO_PASSWORD not set");
            return;
        };

        // Unique per process so parallel runs never collide.
        let db_name = format!("hades_test157_{}", std::process::id());
        let sys = ArangoClient::with_socket(socket.clone(), "_system", "root", &password);
        let _ = sys.delete(&format!("database/{db_name}")).await;
        if let Err(e) = sys.post("database", &json!({ "name": db_name })).await {
            eprintln!("skipping: cannot create test database ({e})");
            return;
        }

        let client = ArangoClient::with_socket(socket, &db_name, "root", &password);
        let pool = ArangoPool::new(client.clone(), client);
        for col in [
            CODEBASE.files,
            CODEBASE.chunks,
            CODEBASE.embeddings,
            CODEBASE.symbols,
        ] {
            crud::create_collection(&pool, col, None)
                .await
                .expect("create fixture collection");
        }

        // Catch a panicking assertion so the database is always dropped, then
        // resume it. Without this, a failing test leaks its database — teardown
        // sits after the await and never runs. (Observed while verifying that
        // the cascade test catches a reordered sweep: the deliberate failure
        // left `hades_test157_<pid>` behind.)
        // `JoinHandle` captures a panic instead of unwinding through us, so the
        // database is dropped either way and the failure is re-raised after.
        let outcome = tokio::task::spawn(async move { f(pool).await }).await;

        let _ = sys.delete(&format!("database/{db_name}")).await;

        if let Err(join_err) = outcome {
            std::panic::resume_unwind(join_err.into_panic());
        }
    }

    async fn count(pool: &ArangoPool, col: &str) -> u64 {
        let aql = "RETURN LENGTH(FOR d IN @@col RETURN 1)";
        let bind = json!({ "@col": col });
        query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    async fn exists(pool: &ArangoPool, col: &str, key: &str) -> bool {
        crud::get_document(pool, col, key).await.is_ok()
    }

    /// The sweep must remove orphan chunks and embeddings, and the embedding
    /// sweep must **cascade off the chunk removal within the same run** (#157).
    ///
    /// Fixture (one live file node, `live`):
    ///
    /// | doc | file_key | chunk_key | orphaned by |
    /// |---|---|---|---|
    /// | `c_orphan` | gone | – | its own file_key (#1) |
    /// | `c_live` | live | – | nothing — must survive |
    /// | `e_file_orphan` | gone | c_orphan | its own file_key (#3) |
    /// | `e_cascade` | **live** | c_orphan | **only** the in-run chunk sweep |
    /// | `e_never` | live | never-existed | dangling chunk_key (#2) |
    /// | `e_valid` | live | c_live | nothing — must survive |
    ///
    /// `e_cascade` is the load-bearing row: its own `file_key` is fine and its
    /// `chunk_key` resolves *at the start of the run*. It only becomes an orphan
    /// because chunks are swept first. Reorder the sweep and it survives, so the
    /// ordering — not merely the presence of the `chunk_key` clause — is what
    /// this asserts. That is why the test drives [`sweep_orphans`] rather than
    /// calling the individual sweeps in a hardcoded order.
    #[tokio::test]
    async fn sweep_cascades_chunk_removal_into_embeddings() {
        with_temp_db(|pool| async move {
            crud::insert_documents(&pool, CODEBASE.files, &[json!({ "_key": "live" })], true)
                .await
                .expect("insert file");
            crud::insert_documents(
                &pool,
                CODEBASE.chunks,
                &[
                    json!({ "_key": "c_orphan", "file_key": "gone" }),
                    json!({ "_key": "c_live", "file_key": "live" }),
                ],
                true,
            )
            .await
            .expect("insert chunks");
            crud::insert_documents(
                &pool,
                CODEBASE.embeddings,
                &[
                    json!({ "_key": "e_file_orphan", "file_key": "gone",  "chunk_key": "c_orphan" }),
                    json!({ "_key": "e_cascade",     "file_key": "live",  "chunk_key": "c_orphan" }),
                    json!({ "_key": "e_never",       "file_key": "live",  "chunk_key": "c_never" }),
                    json!({ "_key": "e_valid",       "file_key": "live",  "chunk_key": "c_live" }),
                ],
                true,
            )
            .await
            .expect("insert embeddings");

            // Dry-run changes nothing, and under-counts embeddings because the
            // orphan chunk is still present (the documented lower bound).
            let dry = sweep_orphans(&pool, true).await.expect("dry sweep");
            assert_eq!(dry.orphan_chunks, 1);
            assert_eq!(count(&pool, CODEBASE.chunks).await, 2, "dry-run deleted chunks");
            assert_eq!(
                count(&pool, CODEBASE.embeddings).await,
                4,
                "dry-run deleted embeddings"
            );
            assert_eq!(
                dry.orphan_embeddings, 2,
                "dry-run sees only e_file_orphan and e_never; e_cascade still resolves"
            );

            let real = sweep_orphans(&pool, false).await.expect("real sweep");
            assert_eq!(real.orphan_chunks, 1);
            assert_eq!(
                real.orphan_embeddings, 3,
                "e_file_orphan + e_never + e_cascade (the cascade)"
            );

            assert!(exists(&pool, CODEBASE.chunks, "c_live").await, "live chunk removed");
            assert!(
                exists(&pool, CODEBASE.embeddings, "e_valid").await,
                "valid embedding removed"
            );
            assert!(
                !exists(&pool, CODEBASE.embeddings, "e_cascade").await,
                "cascade embedding survived — chunks must be swept before embeddings"
            );
            assert_eq!(count(&pool, CODEBASE.chunks).await, 1);
            assert_eq!(count(&pool, CODEBASE.embeddings).await, 1);
        })
        .await;
    }

    /// An embedding with no `chunk_key` at all is not an orphan (#157 review).
    ///
    /// `CONCAT('codebase_chunks/', null)` yields the bare collection prefix and
    /// `DOCUMENT()` returns null for it, so an unguarded filter would delete
    /// every record that simply never carried a chunk reference.
    #[tokio::test]
    async fn null_chunk_key_is_not_an_orphan() {
        with_temp_db(|pool| async move {
            crud::insert_documents(&pool, CODEBASE.files, &[json!({ "_key": "live" })], true)
                .await
                .expect("insert file");
            crud::insert_documents(
                &pool,
                CODEBASE.embeddings,
                &[
                    json!({ "_key": "e_no_chunk_field", "file_key": "live" }),
                    json!({ "_key": "e_null_chunk", "file_key": "live", "chunk_key": null }),
                ],
                true,
            )
            .await
            .expect("insert embeddings");

            let counts = sweep_orphans(&pool, false).await.expect("sweep");
            assert_eq!(
                counts.orphan_embeddings, 0,
                "embeddings without a chunk reference must not be treated as orphans"
            );
            assert_eq!(count(&pool, CODEBASE.embeddings).await, 2);
        })
        .await;
    }
}
