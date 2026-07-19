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
    let unknown: Vec<String> = targets
        .iter()
        .filter(|k| !present.contains(k))
        .cloned()
        .collect();
    Ok((present, unknown))
}

/// Find edges outside the codebase collections that touch these file nodes.
///
/// Scans every edge collection in the database rather than a hardcoded list, so
/// domain-specific bridge collections (whatever a given graph happens to call
/// them) are surfaced without HADES knowing their names.
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
        let aql = "LET ids = (FOR k IN @keys RETURN CONCAT(@files_name, '/', k)) \
                   FOR e IN @@edges FILTER e._from IN ids OR e._to IN ids \
                   SORT e._key RETURN e._key";
        let bind = json!({
            "@edges": info.name,
            "keys": keys,
            "files_name": CODEBASE.files,
        });
        let rows = match query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader)
            .await
        {
            Ok(rows) => rows.results,
            Err(ArangoError::Api {
                error_num: 1203, ..
            }) => continue,
            Err(e) => {
                warn!(collection = %info.name, error = %e, "edge scan failed, skipping");
                continue;
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
/// endpoint filter) guarantees the operator's dry-run listing and what actually
/// gets deleted are the same set.
async fn remove_other_edges(
    pool: &ArangoPool,
    _keys: &[String],
    scanned: &BTreeMap<String, (u64, Vec<String>)>,
) -> Result<BTreeMap<String, u64>> {
    let mut removed = BTreeMap::new();
    for (col, (_, edge_keys)) in scanned {
        let aql = "LET gone = (FOR k IN @keys \
                       FOR e IN @@edges FILTER e._key == k REMOVE e IN @@edges RETURN 1) \
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
