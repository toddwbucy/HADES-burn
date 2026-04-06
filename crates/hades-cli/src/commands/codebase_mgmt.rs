//! Native Rust handlers for `hades codebase` management commands.
//!
//! `codebase stats` routes through dispatch (AQL queries against ArangoDB).
//! `codebase update` calls `codebase_ingest::run()` directly (file I/O).
//!
//! Convention: JSON to stdout, human-readable diagnostics to stderr.

use std::io::Write;

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, CodebaseStatsParams, DaemonCommand};

use super::output::{self, OutputFormat};

// ── codebase stats ──────────────────────────────────────────────────

/// `hades codebase stats`
pub async fn run_stats(config: &HadesConfig) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let cmd = DaemonCommand::CodebaseStats(CodebaseStatsParams {});
    let result = dispatch::dispatch(&pool, config, cmd)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    output::print_output("codebase.stats", result.clone(), &OutputFormat::Json);

    // Human-readable summary to stderr.
    if let Some(cols) = result["collections"].as_object() {
        let mut err = std::io::stderr().lock();
        writeln!(err, "Codebase collections:")?;
        for (label, count) in cols {
            writeln!(err, "  {label}: {}", count.as_u64().unwrap_or(0))?;
        }
    }

    Ok(())
}
