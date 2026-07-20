//! `hades codebase retire` — retire graph nodes for deleted source files.
//!
//! The complement of `codebase ingest --force`. `--force` refreshes a file that
//! still **exists**, rebuilding its subtree in place while preserving inbound
//! authored edges. This retires a file that is **gone**: there is nothing to
//! re-ingest, so the node and everything it owns must come out.
//!
//! Before this existed the only options were `db purge` (cascades the codebase
//! collections but leaves authored bridge edges dangling at a deleted endpoint)
//! or hand-written AQL. Hand-written AQL is how this goes wrong — see the
//! symbol-anchored edge note below.
//!
//! # Targets are explicit
//!
//! This command never discovers its own targets. Callers pass keys (`--file`,
//! repeatable) or a newline-separated list (`--from <path>`, or `-` for stdin).
//! Discovery is [`super::codebase_drift`]'s job, and keeping the two apart means
//! the destructive command can never act on a bad root: a wrong-root drift
//! report is a harmless wrong answer, whereas a wrong-root retire would delete
//! the graph.
//!
//! # What gets removed
//!
//! For each target file node:
//!
//! - the `codebase_files` document, its chunks, embeddings, and symbols
//! - every codebase edge incident on the file **or on one of its symbols**
//! - every *other* edge, in any edge collection, incident on the file node
//!
//! That last category is reported separately and requires `--yes`, because it is
//! where hand-authored records live (conformance verdicts, spec bridges). Those
//! edges are **void** rather than lost — a verdict of the form "file X complies
//! with smell Y" has no referent once X is deleted — but they are irreplaceable
//! if the judgement was wrong about the file being gone, so they are never
//! removed silently.
//!
//! # The symbol-anchored edge trap
//!
//! `calls` and `implements` edges connect **symbols**, not files. Filtering them
//! on a file key returns zero — confidently and wrongly. The endpoint id list
//! below is therefore built from the file id **plus every symbol id owned by the
//! file**, snapshotted in-query so concurrent inserts cannot slip an edge past
//! the filter. Getting this wrong leaves dangling edges and trades one set of
//! validate failures for another.
//!
//! Output: JSON summary to stdout; human-readable summary + logs to stderr.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tracing::{info, warn};

use hades_core::config::HadesConfig;
use hades_core::db::collections::CODEBASE;
use hades_core::db::crud;
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::{ArangoError, ArangoPool};

use super::output::{self, OutputFormat};

/// ArangoDB collection type discriminator for edge collections.
const EDGE_COLLECTION_TYPE: u32 = 3;

/// How many edge keys to list per collection in the JSON payload.
const SAMPLE_LIMIT: usize = 200;

/// `hades codebase retire [--file KEY]... [--from PATH] [--dry-run] [-y]`
pub async fn run_retire(
    config: &HadesConfig,
    files: Vec<String>,
    from: Option<PathBuf>,
    dry_run: bool,
    yes: bool,
) -> Result<()> {
    let targets = collect_targets(files, from)?;
    if targets.is_empty() {
        bail!("no target keys given: pass --file <key> (repeatable) or --from <path>");
    }

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // Which targets actually exist? Retiring an unknown key is a no-op, but
    // silently doing nothing hides a typo or a stale target list.
    let (present, unknown) = partition_existing(&pool, &targets).await?;
    if present.is_empty() {
        bail!(
            "none of the {} requested key(s) exist in {}",
            targets.len(),
            CODEBASE.files
        );
    }

    // Non-codebase edges incident on these file nodes. Surfaced before any
    // deletion so the operator sees exactly which authored records are at
    // stake, in both dry-run and real runs.
    let other_edges = scan_other_edges(&pool, &present).await?;
    let other_total: u64 = other_edges.values().map(|(n, _)| *n).sum();

    if other_total > 0 && !dry_run && !yes {
        bail!(
            "{} edge(s) outside the codebase collections are incident on these file \
             nodes (authored records such as conformance verdicts). Re-run with \
             --dry-run to inspect them, then --yes to remove them.",
            other_total
        );
    }

    let codebase_counts = sweep_codebase(&pool, &present, dry_run).await?;
    let other_removed = if dry_run {
        BTreeMap::new()
    } else {
        remove_other_edges(&pool, &present, &other_edges).await?
    };

    let mut authored = serde_json::Map::new();
    for (col, (count, keys)) in &other_edges {
        authored.insert(
            col.clone(),
            json!({
                "count": count,
                "keys": keys.iter().take(SAMPLE_LIMIT).collect::<Vec<_>>(),
                "truncated": keys.len() > SAMPLE_LIMIT,
                "removed": other_removed.get(col).copied().unwrap_or(0),
            }),
        );
    }

    let report = json!({
        "dry_run": dry_run,
        "requested": targets.len(),
        "retired": present.len(),
        "unknown_keys": unknown,
        "codebase": codebase_counts,
        "other_edges": Value::Object(authored),
        "other_edges_total": other_total,
    });
    output::print_output("codebase.retire", report, &OutputFormat::Json);

    let mut err = std::io::stderr().lock();
    let verb = if dry_run { "would retire" } else { "retired" };
    let _ = writeln!(
        err,
        "Retire: {verb} {} file node(s); {} other-collection edge(s) affected",
        present.len(),
        other_total,
    );
    if !unknown.is_empty() {
        let _ = writeln!(
            err,
            "warning: {} key(s) not found in the graph",
            unknown.len()
        );
    }

    info!(
        dry_run,
        retired = present.len(),
        unknown = unknown.len(),
        other_edges = other_total,
        "retire complete"
    );
    Ok(())
}

/// Gather target keys from repeated `--file` flags and/or a `--from` list.
fn collect_targets(mut files: Vec<String>, from: Option<PathBuf>) -> Result<Vec<String>> {
    if let Some(path) = from {
        let raw = if path.as_os_str() == "-" {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .context("failed to read target keys from stdin")?;
            buf
        } else {
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read target list {}", path.display()))?
        };
        files.extend(
            raw.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string),
        );
    }
    files.sort();
    files.dedup();
    Ok(files)
}

/// Split requested keys into those present in `codebase_files` and those not.
async fn partition_existing(
    pool: &ArangoPool,
    targets: &[String],
) -> Result<(Vec<String>, Vec<String>)> {
    let aql = "FOR k IN @keys FILTER DOCUMENT(CONCAT(@files_name, '/', k)) != null RETURN k";
    let bind = json!({ "keys": targets, "files_name": CODEBASE.files });
    let rows =
        match query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader).await {
            Ok(rows) => rows.results,
            Err(ArangoError::Api {
                error_num: 1203, ..
            }) => Vec::new(),
            Err(e) => return Err(e).context("failed to check which file nodes exist"),
        };
    let present: Vec<String> = rows
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    // Set lookup, not `Vec::contains` — the intended input is a whole drift
    // report piped in, so a linear scan per target is quadratic on exactly the
    // case this command is built for.
    let present_set: std::collections::HashSet<&str> = present.iter().map(String::as_str).collect();
    let unknown: Vec<String> = targets
        .iter()
        .filter(|k| !present_set.contains(k.as_str()))
        .cloned()
        .collect();
    Ok((present, unknown))
}

/// Find edges outside the codebase collections that touch these file nodes **or
/// any symbol they own**.
///
/// Scans every edge collection in the database rather than a hardcoded list, so
/// domain-specific bridge collections (whatever a given graph happens to call
/// them) are surfaced without HADES knowing their names.
///
/// The endpoint list spans symbols for the same reason `sweep_codebase`'s does:
/// an authored edge may anchor to a *symbol* rather than the file. Scanning file
/// ids alone would leave such an edge unreported, un-gated by `--yes`, and
/// un-removed — dangling the moment the sweep deletes its symbol endpoint. That
/// is the symbol-anchored trap in the module header, and it applies with more
/// force here than to the codebase collections, because these edges are the
/// irreplaceable ones.
async fn scan_other_edges(
    pool: &ArangoPool,
    keys: &[String],
) -> Result<BTreeMap<String, (u64, Vec<String>)>> {
    let codebase_edges = [
        CODEBASE.defines_edges,
        CODEBASE.calls_edges,
        CODEBASE.implements_edges,
        CODEBASE.imports_edges,
    ];

    let collections = crud::list_collections(pool, true)
        .await
        .context("failed to list collections")?;

    let mut found = BTreeMap::new();
    for info in collections {
        if info.collection_type != EDGE_COLLECTION_TYPE
            || codebase_edges.contains(&info.name.as_str())
        {
            continue;
        }
        let aql = "LET ids = APPEND( \
                       (FOR s IN @@symbols FILTER s.file_key IN @keys RETURN s._id), \
                       (FOR k IN @keys RETURN CONCAT(@files_name, '/', k))) \
                   FOR e IN @@edges FILTER e._from IN ids OR e._to IN ids \
                   SORT e._key RETURN e._key";
        let bind = json!({
            "@edges": info.name,
            "@symbols": CODEBASE.symbols,
            "keys": keys,
            "files_name": CODEBASE.files,
        });
        let rows = match query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader)
            .await
        {
            Ok(rows) => rows.results,
            // A collection that vanished between listing and scanning has
            // nothing to report.
            Err(ArangoError::Api {
                error_num: 1203, ..
            }) => continue,
            // Anything else is fatal. This scan is the pre-flight for a
            // destructive operation: if it fails and we continue, the collection
            // reports zero authored edges, the `--yes` gate is never triggered
            // for it, and the sweep proceeds to delete the endpoints those edges
            // point at — silently dangling records the operator was never shown.
            // Refusing to act on an incomplete picture is the only safe answer.
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "failed to scan edge collection '{}' for edges incident on the target \
                         file nodes; refusing to retire on an incomplete picture",
                        info.name
                    )
                });
            }
        };
        if rows.is_empty() {
            continue;
        }
        let edge_keys: Vec<String> = rows
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        found.insert(info.name, (edge_keys.len() as u64, edge_keys));
    }
    Ok(found)
}

/// Delete (or count) the codebase subtree for every target key.
///
/// `ids` spans the file nodes *and* every symbol they own, so symbol-anchored
/// `calls`/`implements` edges are caught. See the module docs.
async fn sweep_codebase(pool: &ArangoPool, keys: &[String], dry_run: bool) -> Result<Value> {
    let aql = if dry_run {
        "LET ids = APPEND( \
            (FOR s IN @@symbols FILTER s.file_key IN @keys RETURN s._id), \
            (FOR k IN @keys RETURN CONCAT(@files_name, '/', k))) \
         RETURN { \
           files: LENGTH(FOR d IN @@files FILTER d._key IN @keys RETURN 1), \
           chunks: LENGTH(FOR d IN @@chunks FILTER d.file_key IN @keys RETURN 1), \
           embeddings: LENGTH(FOR d IN @@embs FILTER d.file_key IN @keys RETURN 1), \
           symbols: LENGTH(FOR d IN @@symbols FILTER d.file_key IN @keys RETURN 1), \
           defines_edges: LENGTH(FOR e IN @@defines FILTER e._from IN ids OR e._to IN ids RETURN 1), \
           calls_edges: LENGTH(FOR e IN @@calls FILTER e._from IN ids OR e._to IN ids RETURN 1), \
           implements_edges: LENGTH(FOR e IN @@implements FILTER e._from IN ids OR e._to IN ids RETURN 1), \
           imports_edges: LENGTH(FOR e IN @@imports FILTER e._from IN ids OR e._to IN ids RETURN 1) }"
    } else {
        "LET ids = APPEND( \
            (FOR s IN @@symbols FILTER s.file_key IN @keys RETURN s._id), \
            (FOR k IN @keys RETURN CONCAT(@files_name, '/', k))) \
         LET defs = (FOR e IN @@defines FILTER e._from IN ids OR e._to IN ids REMOVE e IN @@defines RETURN 1) \
         LET calls = (FOR e IN @@calls FILTER e._from IN ids OR e._to IN ids REMOVE e IN @@calls RETURN 1) \
         LET impls = (FOR e IN @@implements FILTER e._from IN ids OR e._to IN ids REMOVE e IN @@implements RETURN 1) \
         LET imps = (FOR e IN @@imports FILTER e._from IN ids OR e._to IN ids REMOVE e IN @@imports RETURN 1) \
         LET chunks = (FOR d IN @@chunks FILTER d.file_key IN @keys REMOVE d IN @@chunks RETURN 1) \
         LET embs = (FOR d IN @@embs FILTER d.file_key IN @keys REMOVE d IN @@embs RETURN 1) \
         LET syms = (FOR d IN @@symbols FILTER d.file_key IN @keys REMOVE d IN @@symbols RETURN 1) \
         LET meta = (FOR d IN @@files FILTER d._key IN @keys REMOVE d IN @@files RETURN 1) \
         RETURN { files: LENGTH(meta), chunks: LENGTH(chunks), embeddings: LENGTH(embs), \
                  symbols: LENGTH(syms), defines_edges: LENGTH(defs), calls_edges: LENGTH(calls), \
                  implements_edges: LENGTH(impls), imports_edges: LENGTH(imps) }"
    };
    let bind = json!({
        "@files": CODEBASE.files,
        "@chunks": CODEBASE.chunks,
        "@embs": CODEBASE.embeddings,
        "@symbols": CODEBASE.symbols,
        "@defines": CODEBASE.defines_edges,
        "@calls": CODEBASE.calls_edges,
        "@implements": CODEBASE.implements_edges,
        "@imports": CODEBASE.imports_edges,
        "files_name": CODEBASE.files,
        "keys": keys,
    });
    let target = if dry_run {
        ExecutionTarget::Reader
    } else {
        ExecutionTarget::Writer
    };
    match query::query_single(pool, aql, Some(&bind), target).await {
        Ok(v) => Ok(v.unwrap_or_else(|| json!({}))),
        Err(ArangoError::Api {
            error_num: 1203, ..
        }) => Ok(json!({})),
        Err(e) => Err(e).context("codebase retire sweep failed"),
    }
}

/// Remove the previously-scanned non-codebase edges, by explicit key.
///
/// Deleting by the keys captured during the scan (rather than re-running the
/// endpoint filter) guarantees the reported listing and what actually gets
/// deleted are the same set — the operator cannot be shown one set and have a
/// different one removed.
///
/// The scan and this removal are separate statements, so the set is *not*
/// transactionally snapshotted: an edge added between them is not removed (it
/// will surface on the next run), and one deleted between them is tolerated via
/// `ignoreErrors`. Closing that window would need a real transaction; the
/// property that matters here — never delete something that was not reported —
/// holds regardless.
async fn remove_other_edges(
    pool: &ArangoPool,
    _keys: &[String],
    scanned: &BTreeMap<String, (u64, Vec<String>)>,
) -> Result<BTreeMap<String, u64>> {
    let mut removed = BTreeMap::new();
    for (col, (_, edge_keys)) in scanned {
        // Direct key removal — no nested scan per key. `ignoreErrors` tolerates
        // an edge that vanished between the scan and now (see the TOCTOU note on
        // this function).
        let aql = "LET gone = (FOR k IN @keys \
                       REMOVE k IN @@edges OPTIONS { ignoreErrors: true } RETURN 1) \
                   RETURN LENGTH(gone)";
        let bind = json!({ "@edges": col, "keys": edge_keys });
        match query::query_single(pool, aql, Some(&bind), ExecutionTarget::Writer).await {
            Ok(v) => {
                removed.insert(col.clone(), v.and_then(|v| v.as_u64()).unwrap_or(0));
            }
            Err(e) => {
                warn!(collection = %col, error = %e, "failed to remove edges");
                removed.insert(col.clone(), 0);
            }
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    use hades_core::db::crud;

    const FIXTURE_PREFIX: &str = "__hades_test158_";
    /// Edge collection created for the test to stand in for an authored bridge.
    const AUTHORED_EDGES: &str = "hades_test158_authored_edges";

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
        for col in [CODEBASE.files, CODEBASE.symbols, CODEBASE.chunks] {
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

    async fn count(pool: &ArangoPool, col: &str, field: &str) -> u64 {
        let aql = format!(
            "FOR d IN @@col FILTER STARTS_WITH(d.{field}, @p) COLLECT WITH COUNT INTO n RETURN n"
        );
        let bind = json!({ "@col": col, "p": FIXTURE_PREFIX });
        query::query_single(pool, &aql, Some(&bind), ExecutionTarget::Reader)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

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
        let _ = crud::drop_collection(pool, AUTHORED_EDGES, true).await;
    }

    /// The destructive sweep must actually delete, and must surface authored
    /// edges anchored to a **symbol** rather than the file node.
    ///
    /// The symbol-anchored authored edge is the regression guard: scanning file
    /// ids alone leaves it unreported and unremoved, dangling once the sweep
    /// deletes its symbol endpoint.
    #[tokio::test]
    async fn retire_sweeps_subtree_and_finds_symbol_anchored_authored_edges() {
        let Some(pool) = test_pool() else { return };
        if !collections_present(&pool).await {
            eprintln!("skipping: target database has no codebase collections");
            return;
        }
        cleanup(&pool).await;

        let pid = std::process::id();
        let fkey = format!("{FIXTURE_PREFIX}{pid}_doomed");
        let skey = format!("{fkey}__sym__abcd");
        let live_fkey = format!("{FIXTURE_PREFIX}{pid}_live");

        for (col, docs) in [
            (
                CODEBASE.files,
                vec![
                    json!({ "_key": fkey, "kind": "file" }),
                    json!({ "_key": live_fkey, "kind": "file" }),
                ],
            ),
            (
                CODEBASE.symbols,
                vec![json!({ "_key": skey, "file_key": fkey, "kind": "function" })],
            ),
            (
                CODEBASE.chunks,
                vec![json!({ "_key": format!("{fkey}_chunk_0"), "file_key": fkey })],
            ),
        ] {
            crud::insert_documents(&pool, col, &docs, true)
                .await
                .expect("insert fixture");
        }

        // A stand-in authored bridge collection with two edges: one anchored to
        // the file node, one anchored to the file's SYMBOL.
        crud::create_collection(&pool, AUTHORED_EDGES, Some(EDGE_COLLECTION_TYPE))
            .await
            .expect("create authored edge collection");
        let authored = vec![
            json!({ "_key": "on_file", "_from": format!("{}/{}", CODEBASE.files, fkey),
                    "_to": format!("{}/{}", CODEBASE.files, live_fkey) }),
            json!({ "_key": "on_symbol", "_from": format!("{}/{}", CODEBASE.symbols, skey),
                    "_to": format!("{}/{}", CODEBASE.files, live_fkey) }),
        ];
        crud::insert_documents(&pool, AUTHORED_EDGES, &authored, true)
            .await
            .expect("insert authored edges");

        // The scan must find BOTH — the symbol-anchored one is the regression.
        let targets = vec![fkey.clone()];
        let scanned = scan_other_edges(&pool, &targets).await.unwrap();
        let found = scanned
            .get(AUTHORED_EDGES)
            .map(|(_, keys)| keys.clone())
            .unwrap_or_default();
        assert!(
            found.contains(&"on_file".to_string()),
            "file-anchored authored edge must be surfaced, got {found:?}"
        );
        assert!(
            found.contains(&"on_symbol".to_string()),
            "SYMBOL-anchored authored edge must be surfaced — scanning file ids \
             alone leaves it dangling after the sweep. got {found:?}"
        );

        // Dry-run must not delete.
        sweep_codebase(&pool, &targets, true).await.unwrap();
        assert_eq!(
            count(&pool, CODEBASE.symbols, "file_key").await,
            1,
            "dry-run deleted symbols"
        );

        // Real sweep.
        let counts = sweep_codebase(&pool, &targets, false).await.unwrap();
        assert_eq!(counts["files"].as_u64().unwrap(), 1);
        assert_eq!(counts["symbols"].as_u64().unwrap(), 1);
        assert_eq!(counts["chunks"].as_u64().unwrap(), 1);
        assert_eq!(
            count(&pool, CODEBASE.symbols, "file_key").await,
            0,
            "symbols must be gone after the real sweep"
        );

        let removed = remove_other_edges(&pool, &targets, &scanned).await.unwrap();
        assert_eq!(
            removed.get(AUTHORED_EDGES).copied().unwrap_or(0),
            2,
            "both authored edges must be removed, not just the file-anchored one"
        );

        // The live file node is untouched.
        let live = crud::get_document(&pool, CODEBASE.files, &live_fkey).await;
        assert!(live.is_ok(), "unrelated file node must survive");

        cleanup(&pool).await;
    }

    #[test]
    fn collect_targets_dedupes_and_sorts_flag_input() {
        let got = collect_targets(vec!["b".into(), "a".into(), "b".into()], None).unwrap();
        assert_eq!(got, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn collect_targets_reads_list_file_ignoring_blanks_and_comments() {
        let dir = tempfile::TempDir::new().unwrap();
        let list = dir.path().join("keys.txt");
        let mut f = std::fs::File::create(&list).unwrap();
        writeln!(f, "# a comment").unwrap();
        writeln!(f, "alpha").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "   beta   ").unwrap();
        writeln!(f, "# trailing comment").unwrap();
        drop(f);

        let got = collect_targets(Vec::new(), Some(list)).unwrap();
        assert_eq!(got, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn collect_targets_merges_flags_and_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let list = dir.path().join("keys.txt");
        std::fs::write(&list, "beta\ngamma\n").unwrap();

        let got = collect_targets(vec!["alpha".into(), "beta".into()], Some(list)).unwrap();
        // "beta" appears in both sources and must not be retired twice.
        assert_eq!(
            got,
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[test]
    fn collect_targets_missing_file_is_an_error_not_an_empty_run() {
        // A typo'd path must not silently degrade into "retire nothing" — or
        // worse, be indistinguishable from a legitimately empty list.
        let err = collect_targets(Vec::new(), Some(PathBuf::from("/nonexistent/keys.txt")));
        assert!(err.is_err());
    }
}
