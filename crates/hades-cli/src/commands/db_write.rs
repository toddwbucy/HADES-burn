//! Native Rust handlers for `hades db` write commands.
//!
//! Each function validates inputs, constructs a [`DaemonCommand`], calls
//! [`dispatch`], and prints the result to stdout.  Write commands enforce
//! the database safety guard (production databases are read-only).
//!
//! `create-database` and `backfill-text` are CLI-only and bypass dispatch.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

/// Read JSON data from `--data` arg, `--input` file, or stdin.
fn resolve_json_input(data: Option<&str>, input: Option<&Path>) -> Result<serde_json::Value> {
    if let Some(s) = data {
        return serde_json::from_str(s).context("invalid --data JSON");
    }
    if let Some(path) = input {
        let contents =
            std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
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
    config.require_writable_database()?;
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
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db update <collection> <key> [--data JSON]`
pub async fn run_update(
    config: &HadesConfig,
    collection: &str,
    key: &str,
    data: Option<&str>,
) -> Result<()> {
    config.require_writable_database()?;
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
    println!("{}", serde_json::to_string_pretty(&result)?);
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
    config.require_writable_database()?;
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
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db purge <document_id> [--force]`
pub async fn run_purge(
    config: &HadesConfig,
    document_id: &str,
    force: bool,
) -> Result<()> {
    if !force {
        anyhow::bail!(
            "refusing to purge '{document_id}' without --force (-y). \
             This deletes the document AND all related chunks/embeddings."
        );
    }
    config.require_writable_database()?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let result = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbPurge(dispatch::DbPurgeParams {
            document_id: document_id.to_string(),
        }),
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db create <name> [--type document|edge]`
pub async fn run_create_collection(
    config: &HadesConfig,
    name: &str,
    collection_type: &str,
) -> Result<()> {
    config.require_writable_database()?;
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
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db create-database <name>`
///
/// CLI-only — creates a separate ArangoClient targeting `_system` because
/// ArangoDB's `POST /_api/database` requires the system database context.
pub async fn run_create_database(config: &HadesConfig, name: &str) -> Result<()> {
    use hades_core::db::ArangoClient;

    let socket_path = config
        .effective_socket(false)
        .context("no ArangoDB socket configured for write operations")?;

    let password = config.database.password.as_deref().unwrap_or("");
    let client = ArangoClient::with_socket(
        socket_path.into(),
        "_system",
        &config.database.username,
        password,
    );

    let body = json!({ "name": name });
    let resp = client
        .post("database", &body)
        .await
        .with_context(|| format!("failed to create database '{name}'"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "created": true,
            "name": name,
            "response": resp,
        }))?
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
    config.require_writable_database()?;

    let collection = collection.unwrap_or_else(|| {
        hades_core::db::collections::CollectionProfile::default_profile().embeddings
    });
    let dimension =
        dimension.context("--dimension is required for create-index")?;
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
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

/// `hades db backfill-text [--collection C] [--dry-run] [--batch-size N]`
///
/// CLI-only — batch AQL UPDATE to populate text fields for embedding.
/// Uses cursor pagination with progress output to stderr.
pub async fn run_backfill_text(
    config: &HadesConfig,
    collection: Option<&str>,
    dry_run: bool,
    batch_size: u32,
) -> Result<()> {
    if !dry_run {
        config.require_writable_database()?;
    }

    use hades_core::db::collections::CollectionProfile;

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let profiles: Vec<(&str, &CollectionProfile)> = if let Some(name) = collection {
        let p = CollectionProfile::get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown collection profile '{name}' — valid: {}",
                CollectionProfile::all()
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        vec![(name, p)]
    } else {
        CollectionProfile::all().to_vec()
    };

    let mut total_updated: u64 = 0;

    for (name, profile) in &profiles {
        // Find documents missing a non-empty text field.
        let count_aql = "FOR d IN @@col \
                          FILTER d.text == null || LENGTH(TRIM(TO_STRING(d.text))) == 0 \
                          COLLECT WITH COUNT INTO c \
                          RETURN c";
        let bind = json!({ "@col": profile.metadata });

        let count_result = hades_core::db::query::query_single(
            &pool,
            count_aql,
            Some(&bind),
            hades_core::db::query::ExecutionTarget::Reader,
        )
        .await;

        let missing_count = match count_result {
            Ok(Some(v)) => v.as_u64().unwrap_or(0),
            Ok(None) => 0,
            Err(e) if e.is_not_found() => {
                eprintln!("[{name}] collection '{}' not found, skipping", profile.metadata);
                continue;
            }
            Err(e) => {
                return Err(e).context(format!("failed to count missing text in '{}'", profile.metadata));
            }
        };

        if missing_count == 0 {
            eprintln!("[{name}] {} — all documents have text, nothing to do", profile.metadata);
            continue;
        }

        eprintln!(
            "[{name}] {} — {missing_count} documents missing text field",
            profile.metadata
        );

        if dry_run {
            // Show sample documents that would be updated.
            let sample_aql = "FOR d IN @@col \
                              FILTER d.text == null || LENGTH(TRIM(TO_STRING(d.text))) == 0 \
                              LIMIT 3 \
                              RETURN { _key: d._key, title: d.title, name: d.name }";
            let sample = hades_core::db::query::query(
                &pool,
                sample_aql,
                Some(&bind),
                None,
                false,
                hades_core::db::query::ExecutionTarget::Reader,
            )
            .await;
            if let Ok(r) = sample {
                for doc in &r.results {
                    eprintln!("  sample: {}", serde_json::to_string(doc).unwrap_or_default());
                }
            }
            continue;
        }

        // Batch update: build text from title + abstract/content fields.
        let update_aql = "FOR d IN @@col \
             FILTER d.text == null || LENGTH(TRIM(TO_STRING(d.text))) == 0 \
             LIMIT @batch \
             LET t = CONCAT_SEPARATOR(' ', \
                 NOT_NULL(d.title, ''), \
                 NOT_NULL(d.abstract, d.description, d.content, '') \
             ) \
             UPDATE d WITH { text: t } IN @@col \
             RETURN 1";

        let mut batch_num = 0u64;
        loop {
            let update_bind = json!({
                "@col": profile.metadata,
                "batch": batch_size,
            });
            let result = hades_core::db::query::query(
                &pool,
                update_aql,
                Some(&update_bind),
                None,
                false,
                hades_core::db::query::ExecutionTarget::Writer,
            )
            .await
            .with_context(|| format!("backfill update failed on '{}'", profile.metadata))?;

            let updated = result.results.len() as u64;
            if updated == 0 {
                break;
            }
            total_updated += updated;
            batch_num += 1;
            eprintln!(
                "  [{name}] batch {batch_num}: updated {updated} documents ({total_updated} total)"
            );
        }
    }

    let output = json!({
        "total_updated": total_updated,
        "dry_run": dry_run,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
