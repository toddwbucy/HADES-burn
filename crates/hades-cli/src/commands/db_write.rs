//! Native Rust handlers for `hades db` write commands.
//!
//! Each function validates inputs, constructs a [`DaemonCommand`], calls
//! [`dispatch`], and prints the result to stdout.  Write commands enforce
//! the database safety guard (production databases are read-only).
//!
//! `create-database` is CLI-only and bypasses dispatch.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;
use hades_core::db::{ArangoPool, collections::CollectionProfile, crud};
use hades_core::dispatch::{self, DaemonCommand};
use tracing::warn;

use super::output::{self, OutputFormat};

/// Read JSON data from `--data` arg, `--input` file, or stdin.
fn resolve_json_input(data: Option<&str>, input: Option<&Path>) -> Result<serde_json::Value> {
    if let Some(s) = data {
        return serde_json::from_str(s).context("invalid --data JSON");
    }
    if let Some(path) = input {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        return serde_json::from_str(&contents)
            .with_context(|| format!("invalid JSON in {}", path.display()));
    }
    // Read from stdin.
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin()
        .lock()
        .read_to_string(&mut buf)
        .context("failed to read JSON from stdin")?;
    if buf.trim().is_empty() {
        anyhow::bail!("no input data — provide --data, --input, or pipe JSON to stdin");
    }
    serde_json::from_str(&buf).context("invalid JSON from stdin")
}

/// `hades db insert <collection> [--data JSON] [--input FILE]`
pub async fn run_insert(
    config: &HadesConfig,
    collection: &str,
    data: Option<&str>,
    input: Option<&Path>,
) -> Result<()> {
    let json_data = resolve_json_input(data, input)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbInsert(dispatch::DbInsertParams {
            collection: collection.to_string(),
            data: json_data,
        }),
    )
    .await?;
    output::print_output("db.insert", result, &OutputFormat::Json);
    Ok(())
}

/// `hades db update <collection> <key> [--data JSON]`
pub async fn run_update(
    config: &HadesConfig,
    collection: &str,
    key: &str,
    data: Option<&str>,
) -> Result<()> {
    let json_data = resolve_json_input(data, None)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbUpdate(dispatch::DbUpdateParams {
            collection: collection.to_string(),
            key: key.to_string(),
            data: json_data,
        }),
    )
    .await?;
    output::print_output("db.update", result, &OutputFormat::Json);
    Ok(())
}

/// `hades db delete <collection> <key> [--force]`
pub async fn run_delete(
    config: &HadesConfig,
    collection: &str,
    key: &str,
    force: bool,
) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to delete {collection}/{key} without --force (-y). \
             This operation is irreversible."
        );
    }
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbDelete(dispatch::DbDeleteParams {
            collection: collection.to_string(),
            key: key.to_string(),
        }),
    )
    .await?;
    output::print_output("db.delete", result, &OutputFormat::Json);
    Ok(())
}

/// `hades db purge <document_id> [--force]`
pub async fn run_purge(config: &HadesConfig, document_id: &str, force: bool) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to purge '{document_id}' without --force (-y). \
             This deletes the document AND all related chunks/embeddings."
        );
    }
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbPurge(dispatch::DbPurgeParams {
            document_id: document_id.to_string(),
        }),
    )
    .await?;
    output::print_output("db.purge", result, &OutputFormat::Json);
    Ok(())
}

/// `hades db create <name> [--type document|edge]`
pub async fn run_create_collection(
    config: &HadesConfig,
    name: &str,
    collection_type: &str,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbCreateCollection(dispatch::DbCreateCollectionParams {
            name: name.to_string(),
            collection_type: Some(collection_type.to_string()),
        }),
    )
    .await?;
    output::print_output("db.create", result, &OutputFormat::Json);
    Ok(())
}

/// `hades db drop-database <name> --force`
///
/// CLI-only — refuses non-writable target databases (production data is
/// sacrosanct), requires explicit `--force` because the operation is
/// irreversible. Connects to `_system` like `create-database`.
pub async fn run_drop_database(config: &HadesConfig, name: &str, force: bool) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to drop database '{name}' without --force (-y). \
             This operation is irreversible — all collections and documents \
             will be deleted permanently."
        );
    }

    use hades_core::db::ArangoClient;

    let mut system_config = config.clone();
    system_config.database.name = Some("_system".to_string());
    let client = ArangoClient::from_config(&system_config, false)
        .context("failed to connect to ArangoDB for database deletion")?;

    let path = format!("database/{name}");
    let resp = client
        .delete(&path)
        .await
        .with_context(|| format!("failed to drop database '{name}'"))?;

    output::print_output(
        "db.drop-database",
        json!({
            "dropped": true,
            "name": name,
            "response": resp,
        }),
        &OutputFormat::Json,
    );
    Ok(())
}

/// `hades db truncate <collection> -y`
///
/// Empties a collection in place — every document is removed, the collection
/// and its indexes remain. Requires explicit `-y`. If the collection is still
/// referenced by the active schema it warns loudly but proceeds: retiring a
/// superseded-but-still-referenced collection is a legitimate migration step.
///
/// Collection-scoped by design — there is deliberately no database-wide
/// equivalent in the CLI. ArangoDB ACLs remain the authoritative gate: a
/// read-only database rejects the write regardless of `-y`.
pub async fn run_truncate(config: &HadesConfig, collection: &str, force: bool) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to truncate '{collection}' without --force (-y). \
             This permanently deletes every document in the collection."
        );
    }
    let schema_referenced = CollectionProfile::is_schema_referenced(collection);
    if schema_referenced {
        warn!(
            collection,
            "truncating a collection the active schema still references — \
             all of its documents will be deleted"
        );
    }
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let response = crud::truncate_collection(&pool, collection, false)
        .await
        .with_context(|| format!("failed to truncate collection '{collection}'"))?;
    output::print_output(
        "db.truncate",
        json!({
            "truncated": true,
            "collection": collection,
            "schema_referenced": schema_referenced,
            "response": response,
        }),
        &OutputFormat::Json,
    );
    Ok(())
}

/// `hades db drop-collection <collection> -y`
///
/// Removes a collection entirely (documents, indexes, and the collection
/// itself). Requires explicit `-y`. Warns loudly — but proceeds — if the
/// collection is still referenced by the active schema.
///
/// Collection-scoped by design — there is deliberately no database-wide
/// equivalent in the CLI. ArangoDB ACLs remain the authoritative gate.
pub async fn run_drop_collection(
    config: &HadesConfig,
    collection: &str,
    force: bool,
) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to drop collection '{collection}' without --force (-y). \
             This permanently removes the collection, its documents, and its indexes."
        );
    }
    let schema_referenced = CollectionProfile::is_schema_referenced(collection);
    if schema_referenced {
        warn!(
            collection,
            "dropping a collection the active schema still references — \
             it will have to be recreated before re-ingest"
        );
    }
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let response = crud::drop_collection(&pool, collection, false)
        .await
        .with_context(|| format!("failed to drop collection '{collection}'"))?;
    output::print_output(
        "db.drop-collection",
        json!({
            "dropped": true,
            "collection": collection,
            "schema_referenced": schema_referenced,
            "response": response,
        }),
        &OutputFormat::Json,
    );
    Ok(())
}

/// `hades db create-database <name>`
///
/// CLI-only — creates a separate ArangoClient targeting `_system` because
/// ArangoDB's `POST /_api/database` requires the system database context.
///
/// Uses `ArangoClient::from_config` with the database overridden to `_system`
/// so that socket/TCP fallback and auth behavior match all other commands.
pub async fn run_create_database(config: &HadesConfig, name: &str) -> Result<()> {
    use hades_core::db::ArangoClient;

    let mut system_config = config.clone();
    system_config.database.name = Some("_system".to_string());
    let client = ArangoClient::from_config(&system_config, false)
        .context("failed to connect to ArangoDB for database creation")?;

    let body = json!({ "name": name });
    let resp = client
        .post("database", &body)
        .await
        .with_context(|| format!("failed to create database '{name}'"))?;

    output::print_output(
        "db.create-database",
        json!({
            "created": true,
            "name": name,
            "response": resp,
        }),
        &OutputFormat::Json,
    );
    Ok(())
}

/// `hades db create-index [--collection C] [--dimension D] [--metric M]`
pub async fn run_create_index(
    config: &HadesConfig,
    collection: Option<&str>,
    dimension: Option<u32>,
    metric: Option<&str>,
) -> Result<()> {
    let collection = collection.unwrap_or_else(|| {
        hades_core::db::collections::CollectionProfile::default_profile().embeddings
    });
    let dimension = dimension.context("--dimension is required for create-index")?;
    let metric = metric.unwrap_or("cosine");

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbCreateIndex(dispatch::DbCreateIndexParams {
            collection: collection.to_string(),
            dimension,
            metric: metric.to_string(),
        }),
    )
    .await?;
    output::print_output("db.create-index", result, &OutputFormat::Json);
    Ok(())
}
