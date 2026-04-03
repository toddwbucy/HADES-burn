//! Native Rust handlers for `hades db` read commands.
//!
//! Each function builds a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result as JSON to stdout.

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

/// `hades db get <collection> <key>`
pub async fn run_get(config: &HadesConfig, collection: &str, key: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGet {
            collection: collection.to_string(),
            key: key.to_string(),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db count <collection>`
pub async fn run_count(config: &HadesConfig, collection: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbCount {
            collection: collection.to_string(),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db collections`
pub async fn run_collections(config: &HadesConfig) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(&pool, config, DaemonCommand::DbCollections {}).await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db check <document_id>`
pub async fn run_check(config: &HadesConfig, document_id: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbCheck {
            document_id: document_id.to_string(),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db recent [--limit N]`
pub async fn run_recent(config: &HadesConfig, limit: u32) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbRecent { limit: Some(limit) },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db list [--collection PROFILE] [--limit N] [--paper ID]`
pub async fn run_list(
    config: &HadesConfig,
    collection: Option<&str>,
    limit: u32,
    paper: Option<&str>,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbList {
            collection: collection.map(String::from),
            limit: Some(limit),
            paper: paper.map(String::from),
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db aql <AQL> [--bind JSON] [--limit N]`
pub async fn run_aql(
    config: &HadesConfig,
    aql: &str,
    bind: Option<&str>,
    limit: Option<u32>,
) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let bind_value = match bind {
        Some(s) => Some(serde_json::from_str(s).context("invalid --bind JSON")?),
        None => None,
    };
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbAql {
            aql: aql.to_string(),
            bind: bind_value,
            limit,
        },
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db health [--verbose]`
pub async fn run_health(config: &HadesConfig, verbose: bool) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbHealth { verbose },
    )
    .await?;

    // Print structured JSON output.
    let output = json!({ "status": "success", "data": data });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// `hades db stats`
pub async fn run_stats(config: &HadesConfig) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(&pool, config, DaemonCommand::DbStats {}).await?;
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}

/// `hades db export <collection> [--limit N]`
///
/// Streams all documents as JSONL to stdout (or to a file via --output).
/// This is CLI-only — not routed through the daemon dispatch.
pub async fn run_export(
    config: &HadesConfig,
    collection: &str,
    output: Option<&std::path::Path>,
    limit: Option<u32>,
) -> Result<()> {
    use hades_core::db::query::{self, ExecutionTarget};
    use std::io::Write;

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let aql = if let Some(lim) = limit {
        format!("FOR d IN @@col LIMIT {lim} RETURN d")
    } else {
        "FOR d IN @@col RETURN d".to_string()
    };

    let result = query::query(
        &pool,
        &aql,
        Some(&json!({ "@col": collection })),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    .context("export query failed")?;

    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(
            std::fs::File::create(path)
                .with_context(|| format!("failed to create {}", path.display()))?,
        ),
        None => Box::new(std::io::stdout().lock()),
    };

    for doc in &result.results {
        writeln!(writer, "{}", serde_json::to_string(doc)?)?;
    }

    eprintln!("exported {} documents from '{collection}'", result.results.len());
    Ok(())
}

/// `hades db index-status [--collection C]`
///
/// CLI-only — shows vector index info for embedding collections.
pub async fn run_index_status(config: &HadesConfig, collection: Option<&str>) -> Result<()> {
    use hades_core::db::collections::CollectionProfile;
    use hades_core::db::index;

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let collections: Vec<&str> = if let Some(c) = collection {
        vec![c]
    } else {
        CollectionProfile::all()
            .iter()
            .map(|(_, p)| p.embeddings)
            .collect()
    };

    let mut results = Vec::new();
    for col in &collections {
        let indexes = index::list_indexes(&pool, col).await.unwrap_or_default();
        let vector_idx = indexes.iter().find(|i| i.index_type == "vector");
        results.push(json!({
            "collection": col,
            "has_vector_index": vector_idx.is_some(),
            "vector_index": vector_idx.map(|i| json!({
                "id": i.id,
                "fields": i.fields,
                "params": i.params,
            })),
            "total_indexes": indexes.len(),
        }));
    }

    println!("{}", serde_json::to_string_pretty(&json!({ "indexes": results }))?);
    Ok(())
}

/// `hades db databases`
///
/// CLI-only — lists accessible databases.
pub async fn run_databases(config: &HadesConfig) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let resp = pool
        .reader()
        .get("database/user")
        .await
        .context("failed to list databases")?;

    let databases = resp
        .get("result")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "databases": databases,
            "count": databases.len(),
        }))?
    );
    Ok(())
}
