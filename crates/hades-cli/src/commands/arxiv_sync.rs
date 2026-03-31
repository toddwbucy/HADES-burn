//! Native implementation of `hades arxiv sync` and `hades arxiv sync-status`.

use anyhow::{Context, Result};
use chrono::{Duration, NaiveDate, Utc};
use serde_json::json;
use tracing::{info, warn};

use hades_core::arxiv::sync::SyncConfig;
use hades_core::arxiv::ArxivClient;
use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::persephone::embedding::EmbeddingClient;

/// Run the `hades arxiv sync` command.
pub async fn run(
    config: &HadesConfig,
    from_date: Option<&str>,
    categories: Option<&str>,
    max_results: u32,
    batch_size: u32,
    incremental: bool,
) -> Result<()> {
    let pool = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    // Determine start date.
    let start_date = resolve_start_date(&pool, from_date, incremental).await?;

    // Parse categories.
    let cat_list: Vec<String> = categories
        .map(|s| s.split(',').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect())
        .unwrap_or_default();

    info!(
        start_date = %start_date,
        categories = ?cat_list,
        max_results,
        batch_size,
        incremental,
        "arxiv sync starting"
    );

    // Connect to services.
    let arxiv_client = ArxivClient::for_sync()
        .context("failed to create arXiv client")?;
    let embed_client = EmbeddingClient::connect_default().await
        .context("failed to connect to embedding service")?;

    let sync_config = SyncConfig {
        start_date,
        categories: cat_list,
        max_results,
        batch_size,
    };

    // Run the sync pipeline.
    let result = hades_core::arxiv::sync::run_sync(
        &pool,
        &arxiv_client,
        &embed_client,
        &sync_config,
    )
    .await?;

    // Update watermark.
    if let Err(e) = hades_core::arxiv::sync_metadata::update_sync_watermark(
        &pool,
        result.stored as u64,
        0,
    )
    .await
    {
        warn!(error = %e, "failed to update sync watermark");
    }

    // Print JSON summary to stdout.
    let summary = json!({
        "status": "complete",
        "fetched": result.fetched,
        "duplicates": result.duplicates,
        "stored": result.stored,
        "errors": result.errors,
        "start_date": start_date.to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);

    if result.errors > 0 {
        warn!(errors = result.errors, "some papers failed to sync");
    }

    Ok(())
}

/// Run the `hades arxiv sync-status` command.
pub async fn status(config: &HadesConfig, limit: u32) -> Result<()> {
    let pool = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    match hades_core::arxiv::sync_metadata::get_sync_status(&pool).await? {
        Some(watermark) => {
            let history_len = watermark.sync_history.len();
            let start = history_len.saturating_sub(limit as usize);

            let output = json!({
                "last_sync": watermark.last_sync,
                "total_synced": watermark.total_synced,
                "history": &watermark.sync_history[start..],
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        None => {
            let output = json!({
                "status": "no_sync_history",
                "message": "No sync has been performed yet.",
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}

/// Resolve the start date from explicit flag, incremental watermark, or 7-day default.
async fn resolve_start_date(
    pool: &ArangoPool,
    from_date: Option<&str>,
    incremental: bool,
) -> Result<NaiveDate> {
    if let Some(date_str) = from_date {
        // Explicit --from flag takes priority.
        NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .with_context(|| format!("invalid date format '{date_str}', expected YYYY-MM-DD"))
    } else if incremental {
        // Incremental mode: use last sync watermark, fall back to 7 days ago.
        match hades_core::arxiv::sync_metadata::get_last_sync_date(pool).await {
            Some(dt) => {
                info!(last_sync = %dt, "incremental sync from watermark");
                Ok(dt.date_naive())
            }
            None => {
                info!("no prior sync found, falling back to 7 days ago");
                Ok(default_start_date())
            }
        }
    } else {
        Ok(default_start_date())
    }
}

/// Default start date: 7 days ago.
fn default_start_date() -> NaiveDate {
    (Utc::now() - Duration::days(7)).date_naive()
}
