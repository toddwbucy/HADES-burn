//! ArangoDB index management.
//!
//! Provides listing, creation, and deletion of indexes — including the
//! `vector` index type used for server-side ANN search via
//! APPROX_NEAR_COSINE / APPROX_NEAR_L2 / APPROX_NEAR_INNER_PRODUCT.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, instrument};

use super::error::ArangoError;
use super::pool::ArangoPool;
use super::query;

/// Percent-encode a string for use as a URL query parameter value.
fn encode_query_value(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

/// Distance metric for a vector index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VectorMetric {
    Cosine,
    L2,
    InnerProduct,
}

impl VectorMetric {
    /// String value for ArangoDB index params.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::L2 => "l2",
            Self::InnerProduct => "innerProduct",
        }
    }
}

/// Summary of an ArangoDB index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    /// Full index ID (e.g. `"collection/12345"`).
    pub id: String,
    /// Index type (e.g. `"primary"`, `"hash"`, `"vector"`).
    #[serde(rename = "type")]
    pub index_type: String,
    /// Indexed field names.
    pub fields: Vec<String>,
    /// Additional params (metric, dimension, nLists for vector indexes).
    #[serde(default)]
    pub params: Option<Value>,
}

/// List all indexes on a collection.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn list_indexes(
    pool: &ArangoPool,
    collection: &str,
) -> Result<Vec<IndexInfo>, ArangoError> {
    let encoded = encode_query_value(collection);
    let path = format!("index?collection={encoded}");
    let resp = pool.reader().get(&path).await?;

    let indexes = resp
        .get("indexes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ArangoError::Request(
                "index list response missing 'indexes' array".to_string(),
            )
        })?;

    let mut result = Vec::with_capacity(indexes.len());
    for idx in indexes {
        let info: IndexInfo = serde_json::from_value(idx.clone())?;
        result.push(info);
    }

    debug!(collection, count = result.len(), "listed indexes");
    Ok(result)
}

/// Create a vector index on a collection (ArangoDB 3.12+).
///
/// When `n_lists` is `None`, auto-calculates as `max(1, doc_count / 15)`
/// matching the Python HADES behavior.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn create_vector_index(
    pool: &ArangoPool,
    collection: &str,
    field: &str,
    dimension: u32,
    n_lists: Option<u32>,
    n_probe: u32,
    metric: VectorMetric,
) -> Result<IndexInfo, ArangoError> {
    let n_lists = match n_lists {
        Some(n) => n,
        None => {
            // Auto-calculate from collection size
            let aql = format!("RETURN LENGTH({collection})");
            let result = query::query_single(
                pool,
                &aql,
                None,
                query::ExecutionTarget::Reader,
            )
            .await?;
            let doc_count = result
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            (doc_count / 15).max(1) as u32
        }
    };

    let encoded = encode_query_value(collection);
    let path = format!("index?collection={encoded}");
    let body = serde_json::json!({
        "type": "vector",
        "fields": [field],
        "params": {
            "metric": metric.as_str(),
            "dimension": dimension,
            "nLists": n_lists,
            "defaultNProbe": n_probe,
        },
    });

    debug!(
        collection,
        field,
        dimension,
        n_lists,
        n_probe,
        metric = metric.as_str(),
        "creating vector index"
    );

    let resp = pool.writer().post(&path, &body).await?;
    let info: IndexInfo = serde_json::from_value(resp)?;

    debug!(id = %info.id, "vector index created");
    Ok(info)
}

/// Drop an index by its full ID (e.g. `"collection/12345"`).
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn drop_index(
    pool: &ArangoPool,
    index_id: &str,
) -> Result<(), ArangoError> {
    let path = format!("index/{index_id}");
    pool.writer().delete(&path).await?;
    debug!(index_id, "index dropped");
    Ok(())
}

/// Find the first vector index on a collection and return its metric.
///
/// Returns `None` if no vector index exists.
pub async fn detect_vector_index(
    pool: &ArangoPool,
    collection: &str,
) -> Result<Option<VectorMetric>, ArangoError> {
    let indexes = list_indexes(pool, collection).await?;
    for idx in &indexes {
        if idx.index_type == "vector" {
            let metric = idx
                .params
                .as_ref()
                .and_then(|p| p.get("metric"))
                .and_then(|m| m.as_str())
                .unwrap_or("cosine");
            let vm = match metric {
                "l2" => VectorMetric::L2,
                "innerProduct" => VectorMetric::InnerProduct,
                _ => VectorMetric::Cosine,
            };
            return Ok(Some(vm));
        }
    }
    Ok(None)
}
