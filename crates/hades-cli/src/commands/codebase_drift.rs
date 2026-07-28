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
//! - **stale** — a `codebase_files` node whose source file no longer exists
//! - **uningested** — a source file with no node in the graph
//! - **changed** — a matched file whose content differs from what was ingested
//! - **unhandled** — a file under the root that ingest has no handler for
//!
//! The last two exist because their absence made a clean report a false green
//! (#183). A file ingest cannot handle used to fall outside drift's notion of
//! source entirely — neither ingested nor reportable — so a tree that was only
//! partially covered reported `0/0`. And `symbol_hash` is deliberately
//! name-only, so an edited body, signature, or comment left the stored chunks
//! stale while every counter read zero; `changed` compares a full-source digest
//! instead. A gate that cannot cover something must still say so.
//!
//! It is strictly read-only. Acting on the result is [`super::codebase_retire`]
//! (for stale nodes) and `codebase ingest` (uningested, and `--force` for
//! changed).
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
    discover_files_detailed, file_key_for, ingest_base_path, normalize_unparsed_ext,
    parse_language_arg,
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

    // Graph side: key -> stored content digest (absent for pre-#183 ingests).
    let graph_hashes = graph_file_hashes(&pool).await?;
    let graph: BTreeSet<String> = graph_hashes.keys().cloned().collect();

    let stale: Vec<&String> = graph.difference(&disk).collect();
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
        "source_files": disk.len(),
        "matched": matched,
        "stale": {
            "count": stale.len(),
            "keys": sample(&stale),
            "truncated": stale.len() > limit,
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

/// Every `_key` in `codebase_files` with its stored content digest.
///
/// The digest is `None` for nodes written before `content_hash` existed, which
/// the caller reports as unverifiable rather than assuming either way. A missing
/// collection is an empty graph.
async fn graph_file_hashes(pool: &ArangoPool) -> Result<BTreeMap<String, Option<String>>> {
    let aql = "FOR f IN @@files RETURN { key: f._key, hash: f.content_hash }";
    let bind = json!({ "@files": CODEBASE.files });
    match query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader).await {
        Ok(rows) => Ok(rows
            .results
            .iter()
            .filter_map(|v| {
                let key = v.get("key")?.as_str()?.to_string();
                let hash = v.get("hash").and_then(|h| h.as_str()).map(str::to_string);
                Some((key, hash))
            })
            .collect()),
        Err(ArangoError::Api {
            error_num: 1203, ..
        }) => Ok(BTreeMap::new()),
        Err(e) => Err(e).context("failed to read codebase_files keys"),
    }
}
