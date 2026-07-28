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
//! - **collisions** — a source file whose key is held by a node from a different
//!   ingest root, so this tree has no node of its own for it
//! - **changed** — a matched file whose content differs from what was ingested
//! - **unhandled** — a file under the root that ingest has no handler for
//!
//! "Under this root" is load-bearing. File keys are relative to the ingest root,
//! so they carry no evidence of which tree produced them; a comparison that read
//! the whole `codebase_files` collection reported every node of a second tree as
//! stale while those files sat untouched on disk (#192). Ingest now records an
//! `ingest_root` on each node and this command compares only its own.
//!
//! That scopes the comparison, and it is all it scopes. `keys::file_key` is
//! still purely root-relative, so two trees sharing a relative path (`src/main.py`
//! in both) share one document, and the later ingest overwrites the earlier. Root
//! attribution makes drift stop mislabelling the *other* tree's nodes; it cannot
//! separate nodes that were never distinct.
//!
//! What it can do is stop recommending the action that makes it worse. A source
//! file whose key is already held by another root is reported under
//! **collisions**, not `uningested` (#196): the documented remedy for
//! `uningested` is `codebase ingest`, which on a collision overwrites the other
//! tree's node, and that tree's next drift then reports the same file missing —
//! the two roots trade one document back and forth indefinitely. Separating the
//! buckets makes the situation legible; keeping colliding trees in separate
//! databases is the answer until the root is folded into the key, which is a
//! migration. Nodes
//! belonging to another root are excluded and reported under `other_roots`;
//! nodes predating attribution are kept (dropping them would report an entire
//! existing graph as uningested) and listed under `stale.unattributed_keys`,
//! held out of `stale.keys` so a `--full` pipe into `codebase retire` cannot
//! delete a node this command could not prove belongs here.
//!
//! Attribution is recorded over the files ingest *discovers*, so re-ingesting a
//! root attributes every node whose file still exists. A node whose file is
//! already gone is never discovered and can never be attributed by any number of
//! re-ingests — which is precisely the population `stale` exists to surface. On
//! a graph built before attribution, expect a residual `unattributed` count that
//! only a reviewed `codebase retire` clears. It does not grow: everything
//! ingested from now on carries its root.
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
//! changed). `stale.keys` is attributed and safe to feed `retire`;
//! `stale.unattributed_keys` is the reviewed pile.
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
    // Split rather than counted. The documented pipeline is
    // `drift --full | ... | codebase retire --from -`, so the keys that cannot
    // be proven to belong to this tree travel on stdout straight into a delete
    // while the only caveat went to stderr, which the pipe drops. Even a reader
    // who saw "3 of 40 unattributed" had no way to tell which three. Listing
    // them separately makes the warning actionable: pipe `stale.keys`, review
    // `stale.unattributed_keys`.
    let (stale_unattributed, stale_attributed): (Vec<&String>, Vec<&String>) = stale
        .iter()
        .partition(|k| graph_side.unattributed.contains(**k));
    // A disk file with no node here is usually just uningested. But it can also
    // be a file whose node was claimed by a different tree: `file_key` is purely
    // root-relative, so `/repo-a/src/main.py` and `/repo-b/src/main.py` derive
    // the same key and only one document can exist (#196). Those two want
    // opposite responses, and telling them apart matters because the documented
    // remedy for `uningested` is `codebase ingest`, which on a collision
    // overwrites the other tree's node and moves the problem rather than fixing
    // it — then the other tree's drift reports the same file uningested, and the
    // two roots trade the node back and forth forever.
    let (colliding, uningested): (Vec<&String>, Vec<&String>) = disk
        .difference(&graph)
        .partition(|k| graph_side.other_root_keys.contains_key(k.as_str()));
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
        // File nodes belonging to a different ingest root, excluded from every
        // bucket above. The roots are listed because the count alone cannot tell
        // a second repository from a mis-rooted ingest of this one.
        "other_roots": {
            "count": graph_side.other_roots,
            "roots": graph_side.other_root_paths,
        },
        "source_files": disk.len(),
        "matched": matched,
        "stale": {
            "count": stale.len(),
            // Attributed only: every key here is on a node this ingest root
            // owns, so it is safe to hand to `codebase retire`.
            "keys": sample(&stale_attributed),
            "truncated": stale_attributed.len() > limit,
            // Stale keys on nodes with no recorded ingest root, listed rather
            // than counted so they can be reviewed instead of piped. They
            // predate attribution and cannot be confirmed to belong to this
            // tree. Re-ingest stamps any whose file still exists; one whose file
            // is already gone can never be attributed by any re-ingest, and is
            // pre-#192 backlog to clear with a reviewed `retire`.
            "unattributed": stale_unattributed.len(),
            "unattributed_keys": sample(&stale_unattributed),
            "unattributed_truncated": stale_unattributed.len() > limit,
        },
        "uningested": {
            "count": uningested.len(),
            "keys": sample(&uningested),
            "truncated": uningested.len() > limit,
        },
        // Source files under this root whose key is already taken by a node
        // belonging to a different ingest root. NOT uningested: re-ingesting
        // them overwrites the other tree's node rather than adding one (#196).
        "collisions": {
            "count": colliding.len(),
            "keys": colliding
                .iter()
                .take(limit)
                .map(|k| json!({
                    "key": k,
                    "held_by": graph_side.other_root_keys.get(k.as_str()),
                }))
                .collect::<Vec<_>>(),
            "truncated": colliding.len() > limit,
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
    report["clean"] = json!(
        stale.is_empty()
            && uningested.is_empty()
            && changed.is_empty()
            && unverifiable == 0
            && colliding.is_empty()
    );

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
             excluded: {}. A root under this one is usually a mis-rooted ingest \
             of this tree rather than a second graph — `codebase ingest` on a \
             single file bases its keys at that file's parent.",
            graph_side.other_roots,
            graph_side.other_root_paths.join(", "),
        );
    }
    if !stale_unattributed.is_empty() {
        let _ = writeln!(
            err,
            "warning: {} of {} stale key(s) sit on nodes with no recorded ingest \
             root, so they cannot be confirmed to belong to this tree. They are \
             listed separately as `stale.unattributed_keys` and are NOT in \
             `stale.keys`, so a `--full` pipe into `codebase retire` will not \
             touch them. Re-ingesting each root attributes any whose file still \
             exists; one whose file is already gone can never be attributed, and \
             is pre-attribution backlog to clear with a reviewed retire.",
            stale_unattributed.len(),
            stale.len(),
        );
    }
    if !colliding.is_empty() {
        let _ = writeln!(
            err,
            "warning: {} source file(s) under this root have keys already held by \
             nodes from another ingest root, so this tree has no node of its own \
             for them. Do NOT re-ingest to fix it: `file_key` is root-relative, so \
             the write would overwrite the other tree's node and the two roots \
             would trade it back and forth (#196). Keep colliding trees in \
             separate databases.",
            colliding.len(),
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
/// - `other_roots`: nodes excluded as belonging elsewhere, with the roots they
///   name. The roots matter as much as the count: a root that is a descendant of
///   the drift root is usually not a second repository but a mis-rooted ingest of
///   this one (`codebase ingest <a single file>` bases keys at the file's
///   parent), and an operator can only tell the two apart by reading the paths.
struct GraphSide {
    /// key -> stored content digest (`None` for pre-`content_hash` ingests).
    in_scope: BTreeMap<String, Option<String>>,
    /// Keys in `in_scope` with no `ingest_root` recorded.
    unattributed: BTreeSet<String>,
    /// Nodes excluded because they carry a different `ingest_root`.
    other_roots: usize,
    /// The distinct roots those nodes name, sorted.
    other_root_paths: Vec<String>,
    /// Excluded key -> the root it belongs to. Lets the caller tell a source
    /// file with no node from one whose node was claimed by another tree
    /// (#196): the keys are root-relative, so both trees derive the same key
    /// and only one document can exist.
    other_root_keys: BTreeMap<String, String>,
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
    let (counted, other_root_paths, other_root_keys) = other_roots(pool, &root).await?;
    let entries = graph_file_hashes(pool, &aql, &bind).await?;
    Ok(classify(
        entries,
        counted,
        other_root_paths,
        other_root_keys,
    ))
}

/// Split fetched rows into the in-scope map and the unattributed key set.
///
/// Separated from the query so the classification — the part that decides what
/// a `--full` run will hand to `codebase retire` — is testable without a
/// database.
fn classify(
    entries: Vec<(String, Option<String>, bool)>,
    other_roots: usize,
    other_root_paths: Vec<String>,
    other_root_keys: BTreeMap<String, String>,
) -> GraphSide {
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
        other_root_paths,
        other_root_keys,
    }
}

/// File nodes carrying a different `ingest_root`, as a count and the distinct
/// roots they name.
///
/// Reported rather than merely skipped: an operator who expected one tree and
/// finds several should learn it from drift, not from a surprising `retire`.
/// The paths are reported too, because "another root" covers two very different
/// situations — a genuinely separate repository, and a mis-rooted ingest of this
/// one — and only the path distinguishes them.
#[allow(clippy::type_complexity)]
async fn other_roots(
    pool: &ArangoPool,
    root: &str,
) -> Result<(usize, Vec<String>, BTreeMap<String, String>)> {
    let aql = format!(
        "FOR f IN @@files \
           FILTER f.{field} != null AND f.{field} != @root \
           RETURN {{ key: f._key, root: f.{field} }}",
        field = INGEST_ROOT_FIELD
    );
    let bind = json!({ "@files": CODEBASE.files, "root": root });
    match query::query(
        pool,
        &aql,
        Some(&bind),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    {
        Ok(rows) => {
            let mut keys = BTreeMap::new();
            let mut roots = BTreeSet::new();
            for row in &rows.results {
                let (Some(k), Some(r)) = (
                    row.get("key").and_then(|k| k.as_str()),
                    row.get("root").and_then(|r| r.as_str()),
                ) else {
                    continue;
                };
                roots.insert(r.to_string());
                keys.insert(k.to_string(), r.to_string());
            }
            Ok((keys.len(), roots.into_iter().collect(), keys))
        }
        Err(e) if is_missing_collection(&e) => Ok((0, Vec::new(), BTreeMap::new())),
        Err(e) => Err(e).context("failed to read other ingest roots"),
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
        let side = classify(
            vec![row("a_rs", true), row("b_rs", false)],
            0,
            Vec::new(),
            BTreeMap::new(),
        );

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
    fn other_roots_count_and_paths_are_reported() {
        let side = classify(
            vec![row("a_rs", true)],
            7,
            vec!["/other/tree".to_string()],
            BTreeMap::new(),
        );
        assert_eq!(side.other_roots, 7);
        assert_eq!(side.other_root_paths, vec!["/other/tree".to_string()]);
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
        let side = classify(
            vec![row("a_rs", true), row("b_rs", true)],
            0,
            Vec::new(),
            BTreeMap::new(),
        );
        assert!(side.unattributed.is_empty());
    }

    /// A disk file whose key is held by another root is a collision, not a
    /// missing node (#196).
    ///
    /// The distinction is the whole point: `uningested` tells an operator to run
    /// `codebase ingest`, which on a collision overwrites the other tree's node
    /// and hands the same complaint to that tree's next drift run.
    #[test]
    fn a_key_held_by_another_root_is_a_collision_not_uningested() {
        let mut held = BTreeMap::new();
        held.insert("src_main_py".to_string(), "/repo-b".to_string());
        let side = classify(Vec::new(), 1, vec!["/repo-b".to_string()], held);

        // Disk has the file; the graph side (this root) does not.
        let disk: BTreeSet<String> = ["src_main_py".to_string(), "src_util_py".to_string()].into();
        let graph: BTreeSet<String> = BTreeSet::new();
        let (colliding, uningested): (Vec<&String>, Vec<&String>) = disk
            .difference(&graph)
            .partition(|k| side.other_root_keys.contains_key(k.as_str()));

        assert_eq!(colliding, vec!["src_main_py"], "held by /repo-b");
        assert_eq!(uningested, vec!["src_util_py"], "genuinely absent");
    }

    /// Pre-`content_hash` digests survive classification as `None` rather than
    /// being confused with "no node" — `changed`/`unverifiable` still depends on
    /// telling those apart.
    #[test]
    fn missing_digest_is_preserved_distinctly_from_a_missing_node() {
        let side = classify(
            vec![("a_rs".to_string(), None, true)],
            0,
            Vec::new(),
            BTreeMap::new(),
        );
        assert_eq!(side.in_scope.get("a_rs"), Some(&None));
        assert!(side.in_scope.contains_key("a_rs"));
    }
}
