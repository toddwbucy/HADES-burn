//! Error types for batch processing.

use serde::Serialize;

/// Batch-level error (not per-item).
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    /// State file I/O error.
    #[error("state file error: {0}")]
    State(#[from] std::io::Error),

    /// State file contains invalid JSON.
    #[error("state file corrupt: {0}")]
    StateCorrupt(#[from] serde_json::Error),
}

/// Per-item error record with provenance.
#[derive(Debug, Clone, Serialize)]
pub struct ItemError {
    /// The item identifier.
    pub item_id: String,
    /// Which processing stage failed (e.g. "extraction", "embedding", "storage").
    pub stage: String,
    /// The error message.
    pub message: String,
    /// Wall-clock time before failure, in milliseconds.
    pub duration_ms: u64,
}
