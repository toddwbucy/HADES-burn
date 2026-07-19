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
//!
//! It is strictly read-only. Acting on the result is [`super::codebase_retire`]
//! (for stale nodes) and `codebase ingest` (for uningested files).
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

use std::collections::{BTreeSet, HashSet};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;
use tracing::info;

use hades_core::code::Language;
use hades_core::config::HadesConfig;
use hades_core::db::collections::CODEBASE;
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::{ArangoError, ArangoPool};

use super::codebase_ingest::{discover_files, file_key_for, ingest_base_path};
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

    let lang_override = match language {
        Some(name) => Some(
            Language::from_extension(name)
                .or_else(|| Language::from_path(&format!("x.{name}")))
                .ok_or_else(|| anyhow::anyhow!("unknown language: {name}"))?,
        ),
        None => None,
    };
    let unparsed_set: HashSet<String> = unparsed_ext.iter().map(|e| e.to_lowercase()).collect();

    // Disk side: exactly what ingest would discover, keyed exactly as ingest keys it.
    let base = ingest_base_path(&path);
    let files = discover_files(&path, lang_override, &unparsed_set)
        .context("failed to discover source files")?;
    let disk: BTreeSet<String> = files.iter().map(|f| file_key_for(&base, f)).collect();

    // Graph side.
    let graph = graph_file_keys(&pool).await?;

    let stale: Vec<&String> = graph.difference(&disk).collect();
    let uningested: Vec<&String> = disk.difference(&graph).collect();
    let matched = graph.intersection(&disk).count();

    let limit = if full { usize::MAX } else { SAMPLE_LIMIT };
    let sample =
        |v: &[&String]| -> Vec<String> { v.iter().take(limit).map(|s| (*s).clone()).collect() };

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
    });

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
        "Drift vs {}: {} matched, {} stale (file deleted), {} uningested (no node)",
        base.display(),
        matched,
        stale.len(),
        uningested.len(),
    );
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

/// Every `_key` in `codebase_files`. A missing collection is an empty graph.
async fn graph_file_keys(pool: &ArangoPool) -> Result<BTreeSet<String>> {
    let aql = "FOR f IN @@files RETURN f._key";
    let bind = json!({ "@files": CODEBASE.files });
    match query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader).await {
        Ok(rows) => Ok(rows
            .results
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()),
        Err(ArangoError::Api {
            error_num: 1203, ..
        }) => Ok(BTreeSet::new()),
        Err(e) => Err(e).context("failed to read codebase_files keys"),
    }
}
