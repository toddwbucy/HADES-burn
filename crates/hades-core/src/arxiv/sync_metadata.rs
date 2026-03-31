//! Sync watermark persistence — tracks last sync timestamp and history.
//!
//! The watermark lives in `sync_metadata/abstracts` in ArangoDB.
//! It records the last sync time, cumulative count, and a rolling
//! history of sync runs (capped at 100 entries).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, warn};

use crate::db::{ArangoError, ArangoErrorKind, ArangoPool};
use crate::db::crud;

/// Collection storing sync state.
const SYNC_METADATA_COLLECTION: &str = "sync_metadata";
/// Document key for the abstracts sync watermark.
const SYNC_WATERMARK_KEY: &str = "abstracts";
/// Maximum history entries to retain.
const MAX_HISTORY: usize = 100;

/// A single sync run entry in the history log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncHistoryEntry {
    /// Date string (YYYY-MM-DD).
    pub date: String,
    /// Number of new papers added.
    pub added: u64,
    /// Number of papers updated (currently always 0).
    pub updated: u64,
    /// ISO 8601 timestamp of this sync run.
    pub timestamp: String,
}

/// The full sync watermark document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncWatermark {
    /// Last sync timestamp (ISO 8601 UTC).
    pub last_sync: String,
    /// Cumulative total of papers synced.
    pub total_synced: u64,
    /// Rolling history of recent sync runs.
    pub sync_history: Vec<SyncHistoryEntry>,
}

/// Read the last sync date from the watermark document.
///
/// Returns `None` if the watermark doesn't exist or can't be parsed.
pub async fn get_last_sync_date(pool: &ArangoPool) -> Option<DateTime<Utc>> {
    match crud::get_document(pool, SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY).await {
        Ok(doc) => {
            let last_sync = doc["last_sync"].as_str()?;
            let normalized = last_sync.replace('Z', "+00:00");
            DateTime::parse_from_rfc3339(&normalized)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        }
        Err(e) if e.is_not_found() => {
            debug!("no sync watermark found, first sync");
            None
        }
        Err(e) => {
            warn!(error = %e, "failed to read sync watermark");
            None
        }
    }
}

/// Read the full sync watermark document.
///
/// Returns `None` if it doesn't exist.
pub async fn get_sync_status(pool: &ArangoPool) -> Result<Option<SyncWatermark>, ArangoError> {
    match crud::get_document(pool, SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY).await {
        Ok(doc) => {
            let watermark: SyncWatermark = serde_json::from_value(doc)
                .map_err(|e| ArangoError::Request(format!("failed to parse watermark: {e}")))?;
            Ok(Some(watermark))
        }
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e),
    }
}

/// Update the sync watermark after a successful sync run.
///
/// Creates the `sync_metadata` collection and document if they don't exist.
pub async fn update_sync_watermark(
    pool: &ArangoPool,
    added: u64,
    updated: u64,
) -> Result<(), ArangoError> {
    let now = Utc::now();
    let timestamp = now.to_rfc3339();
    let date = now.format("%Y-%m-%d").to_string();

    let entry = SyncHistoryEntry {
        date,
        added,
        updated,
        timestamp: timestamp.clone(),
    };

    // Ensure collection exists (ignore 409 Conflict if already there).
    ensure_collection(pool).await?;

    match crud::get_document(pool, SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY).await {
        Ok(existing) => {
            // Update existing watermark.
            let mut watermark: SyncWatermark = serde_json::from_value(existing)
                .map_err(|e| ArangoError::Request(format!("failed to parse watermark: {e}")))?;

            watermark.last_sync = timestamp;
            watermark.total_synced += added;
            watermark.sync_history.push(entry);

            // Trim history to last MAX_HISTORY entries.
            if watermark.sync_history.len() > MAX_HISTORY {
                let start = watermark.sync_history.len() - MAX_HISTORY;
                watermark.sync_history = watermark.sync_history[start..].to_vec();
            }

            let doc = serde_json::to_value(&watermark)
                .map_err(|e| ArangoError::Request(format!("failed to serialize watermark: {e}")))?;
            crud::replace_document(pool, SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY, &doc).await?;
            debug!(total_synced = watermark.total_synced, "sync watermark updated");
        }
        Err(e) if e.is_not_found() => {
            // Create new watermark.
            let watermark = SyncWatermark {
                last_sync: timestamp,
                total_synced: added,
                sync_history: vec![entry],
            };
            let mut doc = serde_json::to_value(&watermark)
                .map_err(|e| ArangoError::Request(format!("failed to serialize watermark: {e}")))?;
            doc["_key"] = json!(SYNC_WATERMARK_KEY);
            crud::insert_documents(pool, SYNC_METADATA_COLLECTION, &[doc], false).await?;
            info!(total_synced = added, "sync watermark created");
        }
        Err(e) => return Err(e),
    }

    Ok(())
}

/// Ensure the sync_metadata collection exists.
async fn ensure_collection(pool: &ArangoPool) -> Result<(), ArangoError> {
    match crud::create_collection(pool, SYNC_METADATA_COLLECTION, None).await {
        Ok(_) => {
            debug!(collection = SYNC_METADATA_COLLECTION, "created collection");
            Ok(())
        }
        Err(e) if matches!(e.kind(), ArangoErrorKind::Conflict) => {
            // Already exists — fine.
            Ok(())
        }
        Err(e) => Err(e),
    }
}
