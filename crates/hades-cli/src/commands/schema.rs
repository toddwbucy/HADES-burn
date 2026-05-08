//! `hades schema apply` and friends — declarative schema commands.
//!
//! Distinct from `hades db schema {init, list, show, version}` (which
//! inspect/initialize the `hades_schema` collection itself). The
//! commands here operate on YAML schema files for project bootstrap;
//! see `docs/declarative-schema.md`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Subcommand;
use serde_json::json;

use hades_core::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::schema_apply::{self, ApplyOptions};

use super::output::{self, OutputFormat};

#[derive(Debug, Subcommand)]
pub enum SchemaCmd {
    /// Apply a YAML schema file to the database.
    ///
    /// Bootstrap-only by default: refuses to run if the database
    /// already has user-data in any of the file's declared
    /// collections. Pass `--force` to override (dangerous).
    Apply {
        /// Path to the YAML schema file.
        file: PathBuf,

        /// Build the plan and print it without executing any writes.
        #[arg(long)]
        dry_run: bool,

        /// Skip the in-use guard. Allows applying to a database that
        /// already has data in declared collections; existing
        /// documents are overwritten by `_key`.
        #[arg(short = 'y', long)]
        force: bool,
    },
}

/// `hades --db <name> schema apply <file>`
pub async fn run_apply(
    config: &HadesConfig,
    file: &PathBuf,
    dry_run: bool,
    force: bool,
) -> Result<()> {
    if !dry_run {
        config.require_writable_database()?;
    }

    let yaml = std::fs::read_to_string(file)
        .with_context(|| format!("failed to read schema file {}", file.display()))?;

    let schema = schema_apply::parse(&yaml)
        .with_context(|| format!("failed to parse {}", file.display()))?;

    if dry_run {
        // Dry-run: validate + plan, print plan as JSON, return.
        schema_apply::validate(&schema).context("schema validation failed")?;
        let ops = schema_apply::plan(&schema);
        output::print_output(
            "schema.apply",
            json!({
                "dry_run": true,
                "operations": ops,
                "operation_count": ops.len(),
            }),
            &OutputFormat::Json,
        );
        return Ok(());
    }

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = schema_apply::apply(&pool, &schema, ApplyOptions { dry_run: false, force })
        .await
        .context("schema apply failed")?;

    output::print_output(
        "schema.apply",
        json!({
            "applied": true,
            "result": result,
        }),
        &OutputFormat::Json,
    );
    Ok(())
}
