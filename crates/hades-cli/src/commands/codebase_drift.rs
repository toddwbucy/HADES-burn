//! `hades codebase drift` — compare the code graph against the source tree.
//!
//! `codebase validate` checks only *internal* consistency: chunk→file refs,
//! edge endpoints, key determinism. Nothing there notices that the tree the
//! graph describes has moved on. A graph can pass almost every invariant while
//! a fifth of its file nodes point at deleted files and dozens of source files
//! have no node at all.
//!
//! This command closes that gap. It reports, in both directions:
//!
//! - **stale** — a `codebase_files` node with no source file under this root
//! - **uningested** — a source file with no node in the graph
//! - **changed** — a matched file whose content differs from what was ingested
//! - **unhandled** — a file under the root that ingest has no handler for
//!
//! "Under this root" is load-bearing. File keys are relative to the ingest root,
//! so they carry no evidence of which tree produced them; a comparison that read
//! the whole `codebase_files` collection reported every node of a second tree as
//! stale while those files sat untouched on disk (#192). Ingest now records an
//! `ingest_root` on each node and this command compares only its own. Nodes
//! belonging to another root are excluded and counted under `other_roots`;
//! nodes predating attribution are kept (dropping them would report an entire
//! existing graph as uningested) and counted under `stale.unattributed`, since
//! those are the ones a `--full` run would hand to `codebase retire` without
//! being able to prove they belong here.
//!
//! The last two exist because their absence made a clean report a false green
//! (#183). A file ingest cannot handle used to fall outside drift's notion of
//! source entirely — neither ingested nor reportable — so a tree that was only
//! partially covered reported `0/0`. And `symbol_hash` is name-only for Python
//! and Rust at tier `semantic` (tier `structural` and C++ hash the serialized
//! symbol list; tier `text` hashes full content), so an edited body, signature,
//! or comment left the stored chunks stale while every counter read zero.
//! `changed` compares a full-source digest
//! instead. A gate that cannot cover something must still say so.
//!
//! It is strictly read-only. Acting on the result is [`super::codebase_retire`]
//! (for stale nodes) and `codebase ingest` (uningested, and `--force` for
//! changed). Read `stale` before retiring from it: nodes predating `ingest_root`
//! cannot be attributed to this tree, and `stale.unattributed` counts exactly
//! those. One re-ingest per root drives it to zero.
//!
//! Discovery uses the same `discover_files` walk and the same key derivation as
//! ingest, so a file counts as "present" exactly when ingest would have picked
//! it up. Pass the same `--language` / `--unparsed-ext` flags used at ingest
//! time, or the comparison measures flag differences rather than drift.
//!
//! **The root matters.** File keys are relative to the ingest root, so pointing
//! this at the wrong directory reports ~100% drift in both directions rather
//! than a small honest number. A near-total mismatch means "wrong root", not
//! "rebuild the graph".
//!
//! Output: JSON summary to stdout; human-readable summary + logs to stderr.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;
use tracing::info;

use hades_core::config::HadesConfig;
use hades_core::db::collections::CODEBASE;
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::{ArangoError, ArangoPool};

use hades_core::code::compute_content_hash;

use super::codebase_ingest::{
    INGEST_ROOT_FIELD, discover_files_detailed, file_key_for, ingest_base_path,
    normalize_unparsed_ext, parse_language_arg,
};
use super::output::{self, OutputFormat};

/// How many sample keys to include per category before truncating.
///
/// Truncation keeps a pathological run (e.g. a wrong root against a large
/// repository) from emitting a wall of keys. Pass `--full` to disable it — that
/// is what makes the output feed `codebase retire --from -`.
const SAMPLE_LIMIT: usize = 200;

/// `hades codebase drift <path> [--language L] [--unparsed-ext e1,e2] [--full]`
pub async fn run_drift(
    config: &HadesConfig,
    path: PathBuf,
    language: Option<&str>,
    unparsed_ext: &[String],
    full: bool,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // Parsed and normalized by the *same* helpers ingest uses. Drift documents
    // that its flags must match the ingest invocation, so it has to accept and
    // normalize exactly what ingest does — a divergence here reports live files
    // as stale, and those feed `codebase retire`.
    let lang_override = parse_language_arg(language)?;
    let unparsed_set = normalize_unparsed_ext(unparsed_ext);

    // Disk side: exactly what ingest would discover, keyed exactly as ingest keys it.
    // The same walk also reports what ingest would NOT pick up, so files with no
    // handler are counted rather than falling outside the comparison entirely.
    let base = ingest_base_path(&path);
    let discovery = discover_files_detailed(&path, lang_override, &unparsed_set)
        .context("failed to discover source files")?;
    let files = discovery.files;
    let disk: BTreeSet<String> = files.iter().map(|f| file_key_for(&base, f)).collect();

    // Graph side, restricted to nodes belonging to this ingest root (#192).
    let graph_side = graph_file_side(&pool, &base).await?;
    let graph_hashes = graph_side.in_scope;
    let graph: BTreeSet<String> = graph_hashes.keys().cloned().collect();

    let stale: Vec<&String> = graph.difference(&disk).collect();
    // Of the nodes we are about to call stale, how many could not be attributed
    // to this root? Those are the ones that might belong to another tree, and
    // they are exactly the ones `retire` would delete on a `--full` run.
    let stale_unattributed = stale
        .iter()
        .filter(|k| graph_side.unattributed.contains(**k))
        .count();
    let uningested: Vec<&String> = disk.difference(&graph).collect();
    let matched = graph.intersection(&disk).count();

    // Content drift over matched files: hash what is on disk now and compare to
    // what was stored at ingest. Files ingested before `content_hash` existed
    // are reported separately rather than guessed at in either direction.
    let mut changed: Vec<String> = Vec::new();
    let mut unverifiable = 0usize;
    for file in &files {
        let key = file_key_for(&base, file);
        let Some(stored) = graph_hashes.get(&key) else {
            continue; // uningested; already counted above
        };
        match stored {
            Some(stored_hash) => match std::fs::read_to_string(file) {
                Ok(source) => {
                    if &compute_content_hash(&source) != stored_hash {
                        changed.push(key);
                    }
                }
                // Unreadable or non-UTF8 now: can't compare, don't claim clean.
                Err(_) => unverifiable += 1,
            },
            None => unverifiable += 1,
        }
    }
    changed.sort();

    let limit = if full { usize::MAX } else { SAMPLE_LIMIT };
    let sample =
        |v: &[&String]| -> Vec<String> { v.iter().take(limit).map(|s| (*s).clone()).collect() };

    let owned_sample = |v: &[String]| -> Vec<String> { v.iter().take(limit).cloned().collect() };
    let unhandled = &discovery.unhandled;

    let mut report = json!({
        "root": base.display().to_string(),
        "graph_nodes": graph.len(),
        // File nodes in this database that belong to a different ingest root.
        // Excluded from every bucket above — they are another tree's, and their
        // source files exist. Non-zero means this database holds more than one
        // code graph, which is worth knowing before acting on `stale`.
        "other_roots": graph_side.other_roots,
        "source_files": disk.len(),
        "matched": matched,
        "stale": {
            "count": stale.len(),
            "keys": sample(&stale),
            "truncated": stale.len() > limit,
            // Stale keys on nodes with no recorded ingest root. They predate
            // attribution, so they cannot be confirmed to belong to this tree —
            // re-ingest stamps them and the number drops to zero.
            "unattributed": stale_unattributed,
        },
        "uningested": {
            "count": uningested.len(),
            "keys": sample(&uningested),
            "truncated": uningested.len() > limit,
        },
        "changed": {
            "count": changed.len(),
            "keys": owned_sample(&changed),
            "truncated": changed.len() > limit,
            // Matched files whose stored digest predates `content_hash`, or whose
            // source is no longer readable. Neither clean nor changed — unknown.
            "unverifiable": unverifiable,
        },
        "unhandled": {
            "count": unhandled.len(),
            "files": unhandled.iter().take(limit).collect::<Vec<_>>(),
            "truncated": unhandled.len() > limit,
        },
    });

    // `clean` answers one question: does the graph match the tree for everything
    // drift was able to check?
    //
    // `unverifiable` counts, because a file whose stored digest predates
    // `content_hash` was never compared — reporting a clean sweep over
    // uncompared files is the false green this command exists to remove.
    //
    // `unhandled` deliberately does NOT count. Every repository contains files
    // no analyzer handles — README.md, Cargo.toml, LICENSE — so gating on it
    // would pin `clean` to false forever and trade a false green for a
    // permanent false red, which carries just as little information. The count
    // and per-file reasons are reported either way, so a caller that wants to
    // treat unhandled files as a gap can read the bucket and decide.
    report["clean"] =
        json!(stale.is_empty() && uningested.is_empty() && changed.is_empty() && unverifiable == 0);

    // A near-total mismatch in both directions almost always means the root is
    // wrong, not that the graph is worthless. Say so rather than let an
    // operator act on it.
    let total = graph.len().max(disk.len());
    if total > 0 && matched * 10 < total {
        report["warning"] = json!(
            "fewer than 10% of nodes matched: this usually means the wrong ingest \
             root was given, not that the graph should be rebuilt. File keys are \
             relative to the ingest root."
        );
    }

    output::print_output("codebase.drift", report, &OutputFormat::Json);

    let mut err = std::io::stderr().lock();
    let _ = writeln!(
        err,
        "Drift vs {}: {} matched, {} stale (no counterpart under this root), \
         {} uningested (no node), \
         {} changed (content differs), {} unhandled (no ingest handler)",
        base.display(),
        matched,
        stale.len(),
        uningested.len(),
        changed.len(),
        unhandled.len(),
    );
    if graph_side.other_roots > 0 {
        let _ = writeln!(
            err,
            "note: {} file node(s) belong to a different ingest root and were \
             excluded — this database holds more than one code graph",
            graph_side.other_roots,
        );
    }
    if stale_unattributed > 0 {
        let _ = writeln!(
            err,
            "warning: {stale_unattributed} of {} stale key(s) sit on nodes with no \
             recorded ingest root, so they cannot be confirmed to belong to this \
             tree — re-ingest each root once to attribute them before feeding \
             `--full` output to `codebase retire`",
            stale.len(),
        );
    }
    if unverifiable > 0 {
        let _ = writeln!(
            err,
            "note: {unverifiable} matched file(s) have no stored content digest \
             (ingested before content_hash) — re-ingest to make them checkable",
        );
    }
    if total > 0 && matched * 10 < total {
        let _ = writeln!(
            err,
            "warning: <10% matched — check the ingest root before acting on this",
        );
    }

    info!(
        matched,
        stale = stale.len(),
        uningested = uningested.len(),
        "drift check complete"
    );
    Ok(())
}

/// The `codebase_files` nodes that belong to this ingest root.
///
/// Splitting the collection by root is what keeps a second tree in the same
/// database out of this one's `stale` bucket (#192). File keys are relative to
/// the ingest root, so they carry no evidence of which tree produced them —
/// without the stored `ingest_root` a comparison against one root reports every
/// node of the other as stale, while those files sit untouched on disk. That
/// matters more than a wrong count because `--full` exists to feed
/// `codebase retire`, which deletes.
///
/// Three groups come back, because "not this root" and "root unknown" are
/// different answers and only one of them is safe to act on:
///
/// - `in_scope`: nodes stamped with this root, plus nodes with no stamp at all.
///   The unstamped ones predate `ingest_root` and cannot be attributed either
///   way, so they are kept rather than silently dropped — excluding them would
///   report a whole existing graph as uningested.
/// - `unattributed`: which of those carry no stamp, so the caller can say how
///   much of its answer rests on an assumption.
/// - `other_roots`: how many nodes were excluded as belonging elsewhere.
struct GraphSide {
    /// key -> stored content digest (`None` for pre-`content_hash` ingests).
    in_scope: BTreeMap<String, Option<String>>,
    /// Keys in `in_scope` with no `ingest_root` recorded.
    unattributed: BTreeSet<String>,
    /// Nodes excluded because they carry a different `ingest_root`.
    other_roots: usize,
}

async fn graph_file_side(pool: &ArangoPool, base: &std::path::Path) -> Result<GraphSide> {
    let root = base.display().to_string();
    let aql = format!(
        "FOR f IN @@files \
           FILTER f.{field} == null OR f.{field} == @root \
           RETURN {{ key: f._key, hash: f.content_hash, attributed: f.{field} != null }}",
        field = INGEST_ROOT_FIELD
    );
    let bind = json!({ "@files": CODEBASE.files, "root": root });
    let counted = count_other_roots(pool, &root).await?;
    let entries = graph_file_hashes(pool, &aql, &bind).await?;
    Ok(classify(entries, counted))
}

/// Split fetched rows into the in-scope map and the unattributed key set.
///
/// Separated from the query so the classification — the part that decides what
/// a `--full` run will hand to `codebase retire` — is testable without a
/// database.
fn classify(entries: Vec<(String, Option<String>, bool)>, other_roots: usize) -> GraphSide {
    let mut in_scope = BTreeMap::new();
    let mut unattributed = BTreeSet::new();
    for (key, hash, attributed) in entries {
        if !attributed {
            unattributed.insert(key.clone());
        }
        in_scope.insert(key, hash);
    }
    GraphSide {
        in_scope,
        unattributed,
        other_roots,
    }
}

/// How many file nodes carry a different `ingest_root` than this one.
///
/// Reported rather than merely skipped: an operator who expected one tree and
/// finds several should learn it from drift, not from a surprising `retire`.
async fn count_other_roots(pool: &ArangoPool, root: &str) -> Result<usize> {
    let aql = format!(
        "RETURN LENGTH(FOR f IN @@files \
           FILTER f.{field} != null AND f.{field} != @root RETURN 1)",
        field = INGEST_ROOT_FIELD
    );
    let bind = json!({ "@files": CODEBASE.files, "root": root });
    match query::query_single(pool, &aql, Some(&bind), ExecutionTarget::Reader).await {
        Ok(v) => Ok(v.and_then(|v| v.as_u64()).unwrap_or(0) as usize),
        Err(e) if is_missing_collection(&e) => Ok(0),
        Err(e) => Err(e.into()),
    }
}

/// Run the prepared query and flatten it to `(key, digest, attributed)`.
///
/// A missing collection is an empty graph.
async fn graph_file_hashes(
    pool: &ArangoPool,
    aql: &str,
    bind: &serde_json::Value,
) -> Result<Vec<(String, Option<String>, bool)>> {
    let bind = bind.clone();
    match query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader).await {
        Ok(rows) => Ok(rows
            .results
            .iter()
            .filter_map(|v| {
                let key = v.get("key")?.as_str()?.to_string();
                let hash = v.get("hash").and_then(|h| h.as_str()).map(str::to_string);
                let attributed = v
                    .get("attributed")
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false);
                Some((key, hash, attributed))
            })
            .collect()),
        Err(e) if is_missing_collection(&e) => Ok(Vec::new()),
        Err(e) => Err(e).context("failed to read codebase_files keys"),
    }
}

/// ArangoDB's "collection or view not found". A database with no code graph is
/// an empty graph, not an error.
fn is_missing_collection(e: &ArangoError) -> bool {
    matches!(
        e,
        ArangoError::Api {
            error_num: 1203,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(key: &str, attributed: bool) -> (String, Option<String>, bool) {
        (key.to_string(), Some("digest".to_string()), attributed)
    }

    /// Nodes with no recorded root are kept in scope, not dropped (#192).
    ///
    /// Dropping them would be the safer-looking choice and is the wrong one: on
    /// a graph built before attribution existed, *every* node is unattributed,
    /// so excluding them would report the whole tree as `uningested` and invite
    /// a full re-ingest of a graph that was fine.
    #[test]
    fn unattributed_nodes_stay_in_scope_and_are_flagged() {
        let side = classify(vec![row("a_rs", true), row("b_rs", false)], 0);

        assert_eq!(side.in_scope.len(), 2, "both nodes must be comparable");
        assert!(side.in_scope.contains_key("b_rs"));
        assert_eq!(
            side.unattributed.len(),
            1,
            "only the unstamped node is unattributed"
        );
        assert!(side.unattributed.contains("b_rs"));
        assert!(!side.unattributed.contains("a_rs"));
    }

    /// The `other_roots` count is carried through untouched — it is reported so
    /// an operator learns this database holds more than one code graph from
    /// drift rather than from a surprising `retire`.
    #[test]
    fn other_roots_count_is_reported() {
        let side = classify(vec![row("a_rs", true)], 7);
        assert_eq!(side.other_roots, 7);
        assert_eq!(
            side.in_scope.len(),
            1,
            "nodes from other roots never reach the comparison"
        );
    }

    /// A fully-attributed graph reports nothing unattributed, so the warning
    /// that gates `--full` into `codebase retire` stays quiet once every root
    /// has been ingested once by a version that records attribution.
    #[test]
    fn a_fully_attributed_graph_has_nothing_unverifiable_by_root() {
        let side = classify(vec![row("a_rs", true), row("b_rs", true)], 0);
        assert!(side.unattributed.is_empty());
    }

    /// Pre-`content_hash` digests survive classification as `None` rather than
    /// being confused with "no node" — `changed`/`unverifiable` still depends on
    /// telling those apart.
    #[test]
    fn missing_digest_is_preserved_distinctly_from_a_missing_node() {
        let side = classify(vec![("a_rs".to_string(), None, true)], 0);
        assert_eq!(side.in_scope.get("a_rs"), Some(&None));
        assert!(side.in_scope.contains_key("a_rs"));
    }
}
