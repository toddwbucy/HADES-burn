//! Graph loader — fetches edges and node embeddings from ArangoDB.
//!
//! Port of `~/git/HADES/core/graph/loader.py::GraphLoader.load()`.
//!
//! Two-phase algorithm:
//!   1. Scan all 22 edge collections → build IDMap + raw edge lists.
//!   2. Group nodes by collection → batch-fetch Jina embeddings → fill GraphData.

use std::collections::HashMap;

use serde_json::json;
use tracing::{debug, info, instrument, warn};

use crate::db::{ArangoError, ArangoPool};
use crate::db::query::{self, ExecutionTarget};

use super::schema::{EDGE_COLLECTION_NAMES, JINA_DIM};
use super::types::{GraphData, GraphDataError, IDMap};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the graph loader.
#[derive(Debug, thiserror::Error)]
pub enum GraphLoaderError {
    #[error("ArangoDB query failed: {0}")]
    Arango(#[from] ArangoError),

    #[error("graph validation failed: {0}")]
    Validation(#[from] GraphDataError),

    #[error("edge result is not a [src, dst] pair at index {index}")]
    MalformedEdge { index: usize },

    #[error("embedding result is not a [key, embedding] pair at index {index}")]
    MalformedEmbedding { index: usize },

    #[error("node count {num_nodes} exceeds u32::MAX")]
    TooManyNodes { num_nodes: usize },
}

// ---------------------------------------------------------------------------
// Result type alias
// ---------------------------------------------------------------------------

/// Loaded graph plus the bidirectional ID map.
pub type LoadResult = (GraphData, IDMap);

// ---------------------------------------------------------------------------
// AQL batch size for cursor pagination
// ---------------------------------------------------------------------------

/// Batch size for edge scans and embedding fetches.
/// Matches the Python default of 10 000.
const LOADER_BATCH_SIZE: u32 = 10_000;

/// Maximum number of keys per embedding fetch query.
/// Keeps AQL bind variable sizes reasonable.
const EMBEDDING_BATCH_KEYS: usize = 5_000;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load the full NL knowledge graph from ArangoDB.
///
/// Scans all 22 RGCN edge collections to discover nodes, then batch-fetches
/// Jina V4 embeddings per vertex collection. Returns a validated [`GraphData`]
/// and the [`IDMap`] used to translate between ArangoDB `_id` and integer
/// node indices.
///
/// Collections that do not exist (future relation slots) are silently skipped.
#[instrument(skip(pool), fields(db = %pool.database()))]
pub async fn load(pool: &ArangoPool) -> Result<LoadResult, GraphLoaderError> {
    // ------------------------------------------------------------------
    // Phase 1: Edge loading — discover nodes, accumulate raw edges
    // ------------------------------------------------------------------
    let mut id_map = IDMap::new();
    let mut raw_edges: Vec<(u32, u32, u32)> = Vec::new();

    info!(
        num_collections = EDGE_COLLECTION_NAMES.len(),
        "loading edges"
    );

    for (rel_idx, &col_name) in EDGE_COLLECTION_NAMES.iter().enumerate() {
        match load_edge_collection(pool, &mut id_map, col_name, rel_idx, &mut raw_edges).await {
            Ok(count) => {
                if count > 0 {
                    info!(collection = col_name, edges = count, "loaded edges");
                } else {
                    debug!(collection = col_name, "empty (future slot)");
                }
            }
            Err(GraphLoaderError::Arango(ref e)) if e.is_not_found() => {
                debug!(collection = col_name, "not found (future slot)");
            }
            Err(e) => return Err(e),
        }
    }

    let num_nodes = id_map.len();
    let num_edges = raw_edges.len();

    info!(num_nodes, num_edges, "edge loading complete");

    // ------------------------------------------------------------------
    // Phase 2: Allocate GraphData, set collection indices, load embeddings
    // ------------------------------------------------------------------
    let mut graph = GraphData::with_capacity(num_nodes, num_edges);

    // Add all edges
    for &(src, dst, rel) in &raw_edges {
        graph.add_edge(src, dst, rel);
    }
    drop(raw_edges); // free memory before loading embeddings

    // Group nodes by collection and build the sorted collection_names index
    let nodes_by_col = id_map.nodes_by_collection();
    let mut collection_names: Vec<String> = nodes_by_col.keys().map(|s| s.to_string()).collect();
    collection_names.sort();
    let col_to_idx: HashMap<&str, u32> = collection_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect();

    // Assign collection type index for every node
    for (col_name, node_list) in &nodes_by_col {
        let col_idx = col_to_idx[*col_name];
        for &(_arango_id, node_idx) in node_list {
            graph.node_collections[node_idx] = col_idx;
        }
    }
    graph.collection_names = collection_names;

    info!(
        num_collections = graph.collection_names.len(),
        "loading node embeddings"
    );

    // Fetch embeddings per vertex collection
    for (col_name, node_list) in &nodes_by_col {
        match load_collection_embeddings(pool, &mut graph, col_name, node_list).await {
            Ok(embedded) => {
                info!(
                    collection = *col_name,
                    embedded,
                    total = node_list.len(),
                    "fetched embeddings"
                );
            }
            Err(GraphLoaderError::Arango(ref e)) if e.is_not_found() => {
                debug!(
                    collection = *col_name,
                    "not queryable for embeddings, skipping"
                );
            }
            Err(e) => return Err(e),
        }
    }

    info!(
        embedded = graph.embedded_count(),
        total = graph.num_nodes,
        coverage_pct = format_args!("{:.1}", graph.embedding_coverage() * 100.0),
        "embedding loading complete"
    );

    graph.validate()?;

    Ok((graph, id_map))
}

// ---------------------------------------------------------------------------
// Phase 1 helpers
// ---------------------------------------------------------------------------

/// Scan a single edge collection and accumulate edges into `raw_edges`.
///
/// Returns the number of edges loaded from this collection.
async fn load_edge_collection(
    pool: &ArangoPool,
    id_map: &mut IDMap,
    col_name: &str,
    rel_idx: usize,
    raw_edges: &mut Vec<(u32, u32, u32)>,
) -> Result<usize, GraphLoaderError> {
    let aql = "FOR e IN @@col RETURN [e._from, e._to]";
    let bind_vars = json!({"@col": col_name});

    let result = query::query(
        pool,
        aql,
        Some(&bind_vars),
        Some(LOADER_BATCH_SIZE),
        false,
        ExecutionTarget::Reader,
    )
    .await?;

    let mut count = 0;
    for (i, doc) in result.results.iter().enumerate() {
        let arr = doc
            .as_array()
            .ok_or(GraphLoaderError::MalformedEdge { index: i })?;
        if arr.len() < 2 {
            return Err(GraphLoaderError::MalformedEdge { index: i });
        }
        let src_id = arr[0]
            .as_str()
            .ok_or(GraphLoaderError::MalformedEdge { index: i })?;
        let dst_id = arr[1]
            .as_str()
            .ok_or(GraphLoaderError::MalformedEdge { index: i })?;

        let src_idx = u32::try_from(id_map.get_or_create(src_id))
            .map_err(|_| GraphLoaderError::TooManyNodes { num_nodes: id_map.len() })?;
        let dst_idx = u32::try_from(id_map.get_or_create(dst_id))
            .map_err(|_| GraphLoaderError::TooManyNodes { num_nodes: id_map.len() })?;
        raw_edges.push((src_idx, dst_idx, rel_idx as u32));
        count += 1;
    }

    Ok(count)
}

// ---------------------------------------------------------------------------
// Phase 2 helpers
// ---------------------------------------------------------------------------

/// Fetch embeddings for all nodes in a single vertex collection.
///
/// Batches the key list into chunks of `EMBEDDING_BATCH_KEYS` to avoid
/// oversized AQL bind variables.
///
/// Returns the number of nodes that received an embedding.
async fn load_collection_embeddings(
    pool: &ArangoPool,
    graph: &mut GraphData,
    col_name: &str,
    node_list: &[(&str, usize)],
) -> Result<usize, GraphLoaderError> {
    // Build (key, node_idx) pairs — extract _key from "collection/_key"
    let keyed_nodes: Vec<(&str, usize)> = node_list
        .iter()
        .filter_map(|(arango_id, idx)| {
            arango_id
                .split('/')
                .nth(1)
                .map(|key| (key, *idx))
        })
        .collect();

    let mut total_embedded = 0;
    let mut skipped = 0usize;

    // Process in batches to keep AQL bind variable size reasonable
    for batch in keyed_nodes.chunks(EMBEDDING_BATCH_KEYS) {
        let keys: Vec<&str> = batch.iter().map(|(k, _)| *k).collect();
        let key_to_idx: HashMap<&str, usize> = batch.iter().map(|&(k, idx)| (k, idx)).collect();

        let aql = "FOR d IN @@col FILTER d._key IN @keys RETURN [d._key, d.embedding]";
        let bind_vars = json!({
            "@col": col_name,
            "keys": keys,
        });

        let result = query::query(
            pool,
            aql,
            Some(&bind_vars),
            Some(LOADER_BATCH_SIZE),
            false,
            ExecutionTarget::Reader,
        )
        .await?;

        for (i, doc) in result.results.iter().enumerate() {
            let arr = doc
                .as_array()
                .ok_or(GraphLoaderError::MalformedEmbedding { index: i })?;
            if arr.len() < 2 {
                return Err(GraphLoaderError::MalformedEmbedding { index: i });
            }

            let key = match arr[0].as_str() {
                Some(k) => k,
                None => continue, // null key — skip
            };

            // Skip null / wrong-dimension embeddings
            let emb_arr = match arr[1].as_array() {
                Some(a) if a.len() == JINA_DIM => a,
                _ => {
                    skipped += 1;
                    continue;
                }
            };

            let node_idx = match key_to_idx.get(key) {
                Some(&idx) => idx,
                None => {
                    warn!(key, collection = col_name, "unexpected key in results");
                    continue;
                }
            };

            // Parse f64 JSON numbers → f32 feature vector; skip if any element is non-numeric
            let embedding: Option<Vec<f32>> = emb_arr
                .iter()
                .map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            let Some(embedding) = embedding else {
                warn!(key, collection = col_name, "embedding contains non-numeric values, skipping");
                continue;
            };

            graph.set_node_features(node_idx, &embedding);
            total_embedded += 1;
        }
    }

    if skipped > 0 {
        warn!(
            collection = col_name,
            skipped,
            "embeddings skipped (null or wrong dimension)"
        );
    }

    Ok(total_embedded)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = GraphLoaderError::MalformedEdge { index: 42 };
        assert!(e.to_string().contains("42"));

        let e = GraphLoaderError::TooManyNodes {
            num_nodes: usize::MAX,
        };
        assert!(e.to_string().contains("u32::MAX"));
    }

    #[test]
    fn test_constants() {
        assert_eq!(LOADER_BATCH_SIZE, 10_000);
        const { assert!(EMBEDDING_BATCH_KEYS > 0) };
    }
}
