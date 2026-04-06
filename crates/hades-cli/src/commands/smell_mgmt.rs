//! Native Rust handlers for `hades smell` and `hades link` commands.
//!
//! All commands route through dispatch (AQL queries against ArangoDB).
//! `link` also requires a writable database.
//!
//! Convention: JSON to stdout, human-readable diagnostics to stderr.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand, SmellCheckParams, SmellVerifyParams, SmellReportParams, LinkCodeSmellParams};

use super::output::{self, OutputFormat};

// ── smell check ─────────────────────────────────────────────────────

/// `hades smell check PATH [--format F] [--verbose]`
pub async fn run_smell_check(
    config: &HadesConfig,
    path: &Path,
    format: &str,
    verbose: bool,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let cmd = DaemonCommand::SmellCheck(SmellCheckParams {
        path: path.display().to_string(),
        verbose,
    });
    let result = dispatch::dispatch(&pool, config, cmd)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Output to stdout with envelope.
    output::print_output("smell.check", result.clone(), &fmt);

    // Human-readable summary to stderr when using table format.
    if matches!(fmt, OutputFormat::Table) {
        let mut err = std::io::stderr().lock();
        let passed = result["passed"].as_bool().unwrap_or(true);
        let count = result["violation_count"].as_u64().unwrap_or(0);
        let files = result["files_checked"].as_u64().unwrap_or(0);
        let smells = result["smells_loaded"].as_u64().unwrap_or(0);

        if passed {
            writeln!(err, "PASSED — {files} files checked, {smells} smells loaded, 0 violations")?;
        } else {
            writeln!(err, "FAILED — {files} files checked, {count} violations")?;
        }

        if let Some(violations) = result["violations"].as_array() {
            for v in violations {
                let tier = v["tier"].as_str().unwrap_or("unknown");
                let marker = if tier == "static" { "BLOCK" } else { "WARN" };
                writeln!(
                    err,
                    "  [{marker}] {file}:{line} — {name} ({pattern})",
                    file = v["file"].as_str().unwrap_or("?"),
                    line = v["line"],
                    name = v["smell_name"].as_str().unwrap_or("?"),
                    pattern = v["pattern"].as_str().unwrap_or("?"),
                )?;
                if verbose {
                    writeln!(err, "         {}", v["content"].as_str().unwrap_or(""))?;
                }
            }
        }
    }
    Ok(())
}

// ── smell verify ────────────────────────────────────────────────────

/// `hades smell verify PATH [--claims CS-NN ...]`
pub async fn run_smell_verify(
    config: &HadesConfig,
    path: &Path,
    claims: &[String],
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let cmd = DaemonCommand::SmellVerify(SmellVerifyParams {
        path: path.display().to_string(),
        claims: claims.to_vec(),
    });
    let result = dispatch::dispatch(&pool, config, cmd)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    output::print_output("smell.verify", result, &OutputFormat::Json);
    Ok(())
}

// ── smell report ────────────────────────────────────────────────────

/// `hades smell report PATH [--output O] [--format F]`
pub async fn run_smell_report(
    config: &HadesConfig,
    path: &Path,
    output: Option<&Path>,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let cmd = DaemonCommand::SmellReport(SmellReportParams {
        path: path.display().to_string(),
    });
    let result = dispatch::dispatch(&pool, config, cmd)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Write full report to file if --output specified.
    if let Some(out_path) = output {
        let output_json = serde_json::to_string_pretty(&result)?;
        std::fs::write(out_path, &output_json)?;
        eprintln!("Report written to {}", out_path.display());
    }

    // Output to stdout with envelope.
    output::print_output("smell.report", result.clone(), &fmt);

    // Human-readable summary to stderr when using table format.
    if matches!(fmt, OutputFormat::Table) {
        let mut err = std::io::stderr().lock();
        let passed = result["passed"].as_bool().unwrap_or(true);
        let verdict = if passed { "PASSED" } else { "FAILED" };
        writeln!(err, "Overall: {verdict}")?;

        let check = &result["static_check"];
        let vcount = check["violation_count"].as_u64().unwrap_or(0);
        let fcount = check["files_checked"].as_u64().unwrap_or(0);
        writeln!(err, "Static check: {fcount} files, {vcount} violations")?;

        let verify = &result["ref_verification"];
        let refs = verify["refs_found"].as_u64().unwrap_or(0);
        let verified = result.get("ref_verification")
            .and_then(|v| v["verified_refs"].as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let missing = verify["missing_from_graph"].as_array().map(|a| a.len()).unwrap_or(0);
        let unlinked = verify["unlinked_claims"].as_array().map(|a| a.len()).unwrap_or(0);
        writeln!(err, "Refs: {refs} found, {verified} verified, {missing} missing, {unlinked} unlinked")?;

        if let Some(probes) = result["embedding_probe"].as_array()
            && !probes.is_empty()
        {
            writeln!(err, "Embedding probe: {} items", probes.len())?;
            for p in probes {
                let cs = p["cs_id"].as_str().unwrap_or("?");
                if let Some(sim) = p["cosine_similarity"].as_f64() {
                    let pass = p["pass"].as_bool().unwrap_or(false);
                    let mark = if pass { "OK" } else { "LOW" };
                    writeln!(err, "  [{mark}] {cs}: {sim:.4}")?;
                } else {
                    writeln!(err, "  [ERR] {cs}: {}", p["error"].as_str().unwrap_or("?"))?;
                }
            }
        }
    }
    Ok(())
}

// ── link ─────────────────────────────────────────────────────────────

/// `hades link SOURCE_ID --claims CS-NN [--force]`
pub async fn run_link(
    config: &HadesConfig,
    source_id: &str,
    claims: &[String],
    force: bool,
) -> Result<()> {
    if force {
        anyhow::bail!("--force is not yet implemented for link");
    }
    config.require_writable_database()?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let mut results = Vec::with_capacity(claims.len());
    for claim in claims {
        let cmd = DaemonCommand::LinkCodeSmell(LinkCodeSmellParams {
            source_id: source_id.to_string(),
            smell_id: claim.clone(),
            enforcement: "static".to_string(),
            methods: Vec::new(),
            summary: None,
        });
        let result = dispatch::dispatch(&pool, config, cmd)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        results.push(result);
    }

    output::print_output("link", serde_json::Value::Array(results), &OutputFormat::Json);
    Ok(())
}
