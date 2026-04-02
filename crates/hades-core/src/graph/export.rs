//! Export structural embeddings back to ArangoDB.
//!
//! After RGCN training completes, the model produces a structural
//! embedding for every node.  This module writes those embeddings
//! back as a `structural_embedding` field on each document, grouped
//! by collection and chunked into batch AQL UPDATEs.
//!
//! Mirrors Python `RGCNTrainer.export_embeddings()`.
//!
//! ## Cross-database export
//!
//! The `pool` parameter can point to a different database than the one
//! used for training.  This supports the "train on snapshot, export to
//! live" workflow.  The [`IDMap`] keys are collection-qualified
//! (`"collection/key"`), so they're portable across databases with the
//! same schema.

use std::collections::HashMap;

use serde_json::{Value, json};
use tracing::{info, warn};

use crate::db::query::{self, ExecutionTarget};
use crate::db::{ArangoError, ArangoPool};

use super::types::IDMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from embedding export.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// ArangoDB query or connection error.
    #[error("ArangoDB error: {0}")]
    Arango(#[from] ArangoError),

    /// Embedding data doesn't match the expected dimensions.
    #[error(
        "embedding dimension mismatch: got {actual} floats, \
         expected {expected} (num_nodes={num_nodes} × embed_dim={embed_dim})"
    )]
    DimensionMismatch {
        actual: usize,
        expected: usize,
        num_nodes: usize,
        embed_dim: usize,
    },

    /// Raw bytes length is not a multiple of 4 (not valid F32 data).
    #[error("embedding bytes length {len} is not a multiple of 4")]
    InvalidBytes { len: usize },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for embedding export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Number of documents per AQL UPDATE batch.
    /// Default: 100 (matches Python).
    pub chunk_size: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self { chunk_size: 100 }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Summary of an embedding export operation.
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Total number of documents updated across all collections.
    pub total_exported: usize,
    /// Per-collection update counts.
    pub by_collection: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Byte decoding
// ---------------------------------------------------------------------------

/// Decode raw little-endian F32 bytes into a `Vec<f32>`.
///
/// Use this to convert the inline `embeddings` bytes from
/// [`EmbeddingsResult`](crate::persephone::training::EmbeddingsResult)
/// before passing to [`export_embeddings`].
pub fn decode_f32_embeddings(bytes: &[u8]) -> Result<Vec<f32>, ExportError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ExportError::InvalidBytes { len: bytes.len() });
    }
    let result: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Write structural embeddings back to ArangoDB documents.
///
/// For each node in `id_map`, writes the corresponding embedding slice
/// from `embeddings` as the `structural_embedding` field on the document.
/// Documents are grouped by collection and updated in batches via AQL.
///
/// # Arguments
///
/// * `pool` — ArangoDB connection pool (may target a different database
///   than the one used for training).
/// * `id_map` — node index → ArangoDB `_id` mapping from graph loading.
/// * `embeddings` — flat `[N × embed_dim]` embedding matrix (row-major).
/// * `embed_dim` — embedding dimension per node (e.g. 128).
/// * `config` — chunk size for batch updates.
///
/// # Errors
///
/// Returns [`ExportError::DimensionMismatch`] if `embeddings.len()`
/// doesn't equal `id_map.len() × embed_dim`.  Individual collection
/// errors are logged as warnings but don't abort the export — the
/// function continues with remaining collections.
pub async fn export_embeddings(
    pool: &ArangoPool,
    id_map: &IDMap,
    embeddings: &[f32],
    embed_dim: usize,
    config: &ExportConfig,
) -> Result<ExportResult, ExportError> {
    let num_nodes = id_map.len();
    let expected = num_nodes * embed_dim;
    if embeddings.len() != expected {
        return Err(ExportError::DimensionMismatch {
            actual: embeddings.len(),
            expected,
            num_nodes,
            embed_dim,
        });
    }

    // Group nodes by collection
    let groups = id_map.nodes_by_collection();

    let mut total_exported: usize = 0;
    let mut by_collection: HashMap<String, usize> = HashMap::new();

    info!(
        num_nodes,
        embed_dim,
        collections = groups.len(),
        db = %pool.database(),
        "exporting structural embeddings"
    );

    for (col_name, nodes) in &groups {
        let mut col_count: usize = 0;

        for chunk in nodes.chunks(config.chunk_size) {
            let updates: Vec<Value> = chunk
                .iter()
                .map(|&(arango_id, idx)| {
                    // Extract document _key from "collection/key"
                    let key = arango_id
                        .split('/')
                        .nth(1)
                        .unwrap_or(arango_id);

                    let start = idx * embed_dim;
                    let emb = &embeddings[start..start + embed_dim];

                    json!({
                        "_key": key,
                        "structural_embedding": emb,
                    })
                })
                .collect();

            let num_updates = updates.len();

            // Backtick-quote the collection name to prevent AQL injection
            let aql = format!(
                "FOR u IN @updates \
                 UPDATE u._key WITH {{ structural_embedding: u.structural_embedding }} \
                 IN `{}` \
                 OPTIONS {{ ignoreErrors: true }} \
                 RETURN 1",
                col_name,
            );

            match query::query(
                pool,
                &aql,
                Some(&json!({ "updates": updates })),
                None,
                false,
                ExecutionTarget::Writer,
            )
            .await
            {
                Ok(result) => {
                    col_count += result.results.len();
                }
                Err(e) => {
                    warn!(
                        collection = col_name,
                        attempted = num_updates,
                        error = %e,
                        "batch update failed, continuing"
                    );
                }
            }
        }

        if col_count > 0 {
            info!(collection = col_name, count = col_count, "exported");
        }

        total_exported += col_count;
        by_collection.insert(col_name.to_string(), col_count);
    }

    info!(
        total = total_exported,
        db = %pool.database(),
        "embedding export complete"
    );

    Ok(ExportResult {
        total_exported,
        by_collection,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_f32_embeddings() {
        // Encode two f32 values as little-endian bytes
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-0.5f32).to_le_bytes());

        let result = decode_f32_embeddings(&bytes).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - 2.5).abs() < f32::EPSILON);
        assert!((result[2] - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_decode_f32_embeddings_empty() {
        let result = decode_f32_embeddings(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_f32_embeddings_invalid_length() {
        let err = decode_f32_embeddings(&[0, 1, 2]).unwrap_err();
        assert!(matches!(err, ExportError::InvalidBytes { len: 3 }));
    }

    #[test]
    fn test_export_config_default() {
        let cfg = ExportConfig::default();
        assert_eq!(cfg.chunk_size, 100);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = ExportError::DimensionMismatch {
            actual: 10,
            expected: 12,
            num_nodes: 3,
            embed_dim: 4,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("12"));
        assert!(msg.contains("num_nodes=3"));
        assert!(msg.contains("embed_dim=4"));
    }

    #[test]
    fn test_export_result_construction() {
        let mut by_col = HashMap::new();
        by_col.insert("hope_axioms".to_string(), 50);
        by_col.insert("atlas_definitions".to_string(), 30);

        let result = ExportResult {
            total_exported: 80,
            by_collection: by_col,
        };
        assert_eq!(result.total_exported, 80);
        assert_eq!(result.by_collection.len(), 2);
        assert_eq!(result.by_collection["hope_axioms"], 50);
    }

    #[test]
    fn test_export_error_display() {
        let err = ExportError::DimensionMismatch {
            actual: 100,
            expected: 128,
            num_nodes: 1,
            embed_dim: 128,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("128"));
        assert!(msg.contains("num_nodes=1"));

        let err2 = ExportError::InvalidBytes { len: 7 };
        assert!(err2.to_string().contains("7"));
    }
}
