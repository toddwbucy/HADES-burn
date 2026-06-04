//! Graph loader — fetches edges and node embeddings from ArangoDB.
//!
//! Port of `~/git/HADES/core/graph/loader.py::GraphLoader.load()`.
//!
//! Two-phase algorithm:
//!   1. Scan all 22 edge collections → build IDMap + raw edge lists.
//!   2. Group nodes by collection → batch-fetch Jina embeddings → fill GraphData.

use std::collections::{HashMap, HashSet};

use serde_json::json;
use tracing::{debug, info, instrument, warn};

use crate::db::collections::CODEBASE;
use crate::db::query::{self, ExecutionTarget};
use crate::db::{ArangoError, ArangoPool};

use super::runtime_schema::RuntimeSchema;
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

/// Load the full knowledge graph from ArangoDB using `schema` to enumerate
/// edge collections and the feature dimension.
///
/// Scans every edge collection in `schema.meta.relation_order` to discover
/// nodes, then batch-fetches embeddings per vertex collection. Returns a
/// validated [`GraphData`] and the [`IDMap`] used to translate between
/// ArangoDB `_id` and integer node indices.
///
/// Collections that do not exist (future relation slots) are silently skipped.
#[instrument(skip(pool, schema), fields(db = %pool.database()))]
pub async fn load(
    pool: &ArangoPool,
    schema: &RuntimeSchema,
) -> Result<LoadResult, GraphLoaderError> {
    // ------------------------------------------------------------------
    // Phase 1: Edge loading — discover nodes, accumulate raw edges
    // ------------------------------------------------------------------
    let mut id_map = IDMap::new();
    let mut raw_edges: Vec<(u32, u32, u32)> = Vec::new();

    let relation_order = &schema.meta.relation_order;
    let feature_dim = schema.meta.feature_dim;

    info!(
        num_collections = relation_order.len(),
        feature_dim, "loading edges"
    );

    for (rel_idx, col_name) in relation_order.iter().enumerate() {
        match load_edge_collection(pool, &mut id_map, col_name, rel_idx, &mut raw_edges).await {
            Ok(count) => {
                if count > 0 {
                    info!(collection = %col_name, edges = count, "loaded edges");
                } else {
                    debug!(collection = %col_name, "empty (future slot)");
                }
            }
            Err(GraphLoaderError::Arango(ref e)) if e.is_not_found() => {
                debug!(collection = %col_name, "not found (future slot)");
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
    let mut graph =
        GraphData::with_schema_capacity(num_nodes, num_edges, relation_order.len(), feature_dim);

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

    // Fetch embeddings per vertex collection. Code collections carry no
    // node-level `embedding` (their embeddings live per-chunk in
    // `codebase_embeddings`), so their node features are mean-pooled from the
    // file's chunk embeddings instead (#132).
    for (col_name, node_list) in &nodes_by_col {
        let result = if *col_name == CODEBASE.files || *col_name == CODEBASE.symbols {
            load_codebase_node_features(pool, &mut graph, col_name, node_list, feature_dim).await
        } else {
            load_collection_embeddings(pool, &mut graph, col_name, node_list, feature_dim).await
        };
        match result {
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

        let src_idx = u32::try_from(id_map.get_or_create(src_id)).map_err(|_| {
            GraphLoaderError::TooManyNodes {
                num_nodes: id_map.len(),
            }
        })?;
        let dst_idx = u32::try_from(id_map.get_or_create(dst_id)).map_err(|_| {
            GraphLoaderError::TooManyNodes {
                num_nodes: id_map.len(),
            }
        })?;
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
    feature_dim: usize,
) -> Result<usize, GraphLoaderError> {
    // Build (key, node_idx) pairs — extract _key from "collection/_key"
    let keyed_nodes: Vec<(&str, usize)> = node_list
        .iter()
        .filter_map(|(arango_id, idx)| arango_id.split('/').nth(1).map(|key| (key, *idx)))
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
                Some(a) if a.len() == feature_dim => a,
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
                warn!(
                    key,
                    collection = col_name,
                    "embedding contains non-numeric values, skipping"
                );
                continue;
            };

            graph.set_node_features(node_idx, &embedding);
            total_embedded += 1;
        }
    }

    if skipped > 0 {
        warn!(
            collection = col_name,
            skipped, "embeddings skipped (null or wrong dimension)"
        );
    }

    Ok(total_embedded)
}

/// Load features for a codebase node collection (`codebase_files` /
/// `codebase_symbols`), which carry no node-level `embedding`.
///
/// Each node's feature is the **mean-pool of its file's chunk embeddings** in
/// `codebase_embeddings` (keyed by `file_key`):
/// - `codebase_files` pool by their own `_key` (which *is* the file_key),
/// - `codebase_symbols` inherit their parent file's pooled embedding, mapped
///   via the symbol's `file_key` field — so all symbols in a file share that
///   file's pooled vector.
///
/// Derived on read (no node-level denorm, no re-ingest); see #132.
async fn load_codebase_node_features(
    pool: &ArangoPool,
    graph: &mut GraphData,
    col_name: &str,
    node_list: &[(&str, usize)],
    feature_dim: usize,
) -> Result<usize, GraphLoaderError> {
    // 1. Map each node index → the file_key whose chunks feed its feature.
    let mut node_file_key: Vec<(usize, String)> = Vec::with_capacity(node_list.len());
    if col_name == CODEBASE.symbols {
        // Symbols carry the parent file_key in a field — fetch it.
        let keyed: Vec<(&str, usize)> = node_list
            .iter()
            .filter_map(|(id, idx)| id.split('/').nth(1).map(|k| (k, *idx)))
            .collect();
        for batch in keyed.chunks(EMBEDDING_BATCH_KEYS) {
            let keys: Vec<&str> = batch.iter().map(|(k, _)| *k).collect();
            let key_to_idx: HashMap<&str, usize> = batch.iter().map(|&(k, i)| (k, i)).collect();
            let aql = "FOR d IN @@col FILTER d._key IN @keys RETURN [d._key, d.file_key]";
            let bind = json!({ "@col": col_name, "keys": keys });
            let res = query::query(
                pool,
                aql,
                Some(&bind),
                Some(LOADER_BATCH_SIZE),
                false,
                ExecutionTarget::Reader,
            )
            .await?;
            for doc in &res.results {
                let arr = match doc.as_array() {
                    Some(a) if a.len() >= 2 => a,
                    _ => continue,
                };
                if let (Some(k), Some(fk)) = (arr[0].as_str(), arr[1].as_str())
                    && let Some(&idx) = key_to_idx.get(k)
                {
                    node_file_key.push((idx, fk.to_string()));
                }
            }
        }
    } else {
        // codebase_files: the node `_key` is the file_key.
        for (id, idx) in node_list {
            if let Some(k) = id.split('/').nth(1) {
                node_file_key.push((*idx, k.to_string()));
            }
        }
    }

    if node_file_key.is_empty() {
        return Ok(0);
    }

    // 2. Mean-pool chunk embeddings for the distinct file_keys.
    let unique: Vec<String> = node_file_key
        .iter()
        .map(|(_, fk)| fk.as_str())
        .collect::<HashSet<_>>()
        .into_iter()
        .map(String::from)
        .collect();
    let pooled = pool_chunk_embeddings(pool, &unique, feature_dim).await?;

    // 3. Assign each node its file's pooled vector.
    let mut embedded = 0;
    for (idx, fk) in &node_file_key {
        if let Some(vec) = pooled.get(fk.as_str()) {
            graph.set_node_features(*idx, vec);
            embedded += 1;
        }
    }
    Ok(embedded)
}

/// Mean-pool `codebase_embeddings` per `file_key`. Returns `file_key → pooled
/// feature vector` for keys that have at least one valid chunk embedding.
async fn pool_chunk_embeddings(
    pool: &ArangoPool,
    file_keys: &[String],
    feature_dim: usize,
) -> Result<HashMap<String, Vec<f32>>, GraphLoaderError> {
    // Accumulate (sum, count) per file_key, then divide.
    let mut acc: HashMap<String, (Vec<f32>, usize)> = HashMap::new();

    for batch in file_keys.chunks(EMBEDDING_BATCH_KEYS) {
        let aql = "FOR e IN @@col FILTER e.file_key IN @fkeys RETURN [e.file_key, e.embedding]";
        let bind = json!({ "@col": CODEBASE.embeddings, "fkeys": batch });
        let res = query::query(
            pool,
            aql,
            Some(&bind),
            Some(LOADER_BATCH_SIZE),
            false,
            ExecutionTarget::Reader,
        )
        .await?;
        for doc in &res.results {
            let arr = match doc.as_array() {
                Some(a) if a.len() >= 2 => a,
                _ => continue,
            };
            let fk = match arr[0].as_str() {
                Some(s) => s,
                None => continue,
            };
            let emb = match arr[1].as_array() {
                Some(a) if a.len() == feature_dim => a,
                _ => continue,
            };
            let vec: Option<Vec<f32>> = emb.iter().map(|v| v.as_f64().map(|f| f as f32)).collect();
            let Some(vec) = vec else { continue };
            let entry = acc
                .entry(fk.to_string())
                .or_insert_with(|| (vec![0.0f32; feature_dim], 0));
            for (s, x) in entry.0.iter_mut().zip(vec.iter()) {
                *s += *x;
            }
            entry.1 += 1;
        }
    }

    Ok(acc
        .into_iter()
        .map(|(fk, (mut sum, n))| {
            if n > 0 {
                let inv = 1.0 / n as f32;
                for x in sum.iter_mut() {
                    *x *= inv;
                }
            }
            (fk, sum)
        })
        .collect())
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
