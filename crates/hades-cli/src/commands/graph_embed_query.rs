//! Native Rust implementation of `hades graph-embed embed` and
//! `hades graph-embed neighbors` commands.
//!
//! Both commands operate on pre-computed `structural_embedding` fields
//! stored on ArangoDB documents (written by the export step after
//! RGCN training).  No model inference is involved.

use anyhow::{Context, Result};
use serde_json::json;
use tracing::info;

use hades_core::config::HadesConfig;
use hades_core::db::crud::{list_collections, CollectionInfo};
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::ArangoPool;

// ---------------------------------------------------------------------------
// graph-embed embed
// ---------------------------------------------------------------------------

/// Run the `graph-embed embed <node_id>` command.
///
/// Looks up the pre-computed `structural_embedding` field for a single
/// node and prints it as JSON.
pub async fn run_embed(config: &HadesConfig, node_id: &str) -> Result<()> {
    let (col, key) = parse_node_id(node_id)?;

    let pool = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    let aql = "FOR d IN @@col FILTER d._key == @key \
         RETURN { _id: d._id, structural_embedding: d.structural_embedding, \
         title: d.title, name: d.name }";

    let result = query::query_single(
        &pool,
        aql,
        Some(&json!({ "@col": col, "key": key })),
        ExecutionTarget::Reader,
    )
    .await
    .context("failed to query node")?;

    let node = result.ok_or_else(|| anyhow::anyhow!("node not found: {node_id}"))?;

    let embedding = node
        .get("structural_embedding")
        .filter(|v| !v.is_null())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no structural embedding for {node_id} — run 'hades graph-embed train' first"
            )
        })?;

    let embed_dim = embedding
        .as_array()
        .map(|a| a.len())
        .ok_or_else(|| anyhow::anyhow!("structural embedding is not an array for {node_id}"))?;

    if embed_dim == 0 {
        anyhow::bail!("structural embedding is empty (0 dimensions) for {node_id}");
    }

    let label = node
        .get("title")
        .and_then(|v| v.as_str())
        .or_else(|| node.get("name").and_then(|v| v.as_str()))
        .unwrap_or(key);

    let output = json!({
        "status": "success",
        "data": {
            "node_id": node_id,
            "label": label,
            "embedding_dim": embed_dim,
            "embedding": embedding,
        },
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// graph-embed neighbors
// ---------------------------------------------------------------------------

/// Run the `graph-embed neighbors <node_id> --limit N` command.
///
/// Finds the k-nearest structural neighbors by dot-product similarity,
/// searching all document collections that contain `structural_embedding`
/// fields.
pub async fn run_neighbors(config: &HadesConfig, node_id: &str, limit: u32) -> Result<()> {
    let (col, key) = parse_node_id(node_id)?;

    let pool = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    // Step 1: Fetch the target node's embedding.
    let emb_aql = "FOR d IN @@col FILTER d._key == @key RETURN d.structural_embedding";

    let result = query::query_single(
        &pool,
        emb_aql,
        Some(&json!({ "@col": col, "key": key })),
        ExecutionTarget::Reader,
    )
    .await
    .context("failed to query node embedding")?;

    // Distinguish "node not found" from "node has no embedding".
    let emb_value = result.ok_or_else(|| {
        anyhow::anyhow!("node not found: {node_id}")
    })?;

    let target_emb = if emb_value.is_null() {
        anyhow::bail!(
            "no structural embedding for {node_id} — run 'hades graph-embed train' first"
        );
    } else {
        emb_value
    };

    let embed_dim = target_emb
        .as_array()
        .map(|a| a.len())
        .ok_or_else(|| anyhow::anyhow!("structural embedding is not an array for {node_id}"))?;

    if embed_dim == 0 {
        anyhow::bail!("structural embedding is empty (0 dimensions) for {node_id}");
    }

    // Step 2: Discover document collections (type 2, non-system).
    let collections: Vec<CollectionInfo> = list_collections(&pool, true)
        .await
        .context("failed to list collections")?
        .into_iter()
        .filter(|c| c.collection_type == 2)
        .collect();

    info!(
        num_collections = collections.len(),
        embed_dim,
        "searching for structural neighbors"
    );

    // Step 3: Search each collection for nearest neighbors.
    let mut all_neighbors: Vec<serde_json::Value> = Vec::new();
    let mut collections_succeeded: usize = 0;
    let mut last_error: Option<String> = None;

    let search_aql =
        "LET te = @target_emb \
         FOR d IN @@col \
           FILTER d.structural_embedding != null \
           FILTER LENGTH(d.structural_embedding) == @dim \
           FILTER d._id != @target_id \
           LET sim = SUM(FOR i IN 0..@dim_minus_1 RETURN te[i] * d.structural_embedding[i]) \
           SORT sim DESC \
           LIMIT @k \
           RETURN { \
             id: d._id, \
             label: NOT_NULL(d.title, d.name, d._key), \
             collection: @col_name, \
             similarity: ROUND(sim * 10000) / 10000 \
           }";

    for col_info in &collections {
        let bind_vars = json!({
            "@col": col_info.name,
            "col_name": col_info.name,
            "target_emb": target_emb,
            "dim": embed_dim,
            "dim_minus_1": embed_dim - 1,
            "target_id": node_id,
            "k": limit,
        });

        match query::query(
            &pool,
            search_aql,
            Some(&bind_vars),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        {
            Ok(result) => {
                collections_succeeded += 1;
                all_neighbors.extend(result.results);
            }
            Err(e) => {
                // Collections without structural_embedding will return
                // empty results, not errors — but skip any that do error.
                tracing::debug!(
                    collection = col_info.name,
                    error = %e,
                    "skipping collection"
                );
                last_error = Some(e.to_string());
            }
        }
    }

    if collections_succeeded == 0 && !collections.is_empty() {
        anyhow::bail!(
            "all {count} collection queries failed; last error: {err}",
            count = collections.len(),
            err = last_error.as_deref().unwrap_or("unknown"),
        );
    }

    // Step 4: Sort globally and take top-k.
    all_neighbors.sort_by(|a, b| {
        let sa = a.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    all_neighbors.truncate(limit as usize);

    let output = json!({
        "status": "success",
        "data": {
            "query_node": node_id,
            "k": limit,
            "neighbors": all_neighbors,
        },
        "metadata": {
            "count": all_neighbors.len(),
        },
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse and validate a `collection/key` node ID.
///
/// Rejects missing slash, empty collection, and empty key.
fn parse_node_id(node_id: &str) -> Result<(&str, &str)> {
    let (col, key) = node_id
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!(
            "node ID must be in 'collection/key' format, got: {node_id}"
        ))?;

    if col.is_empty() {
        anyhow::bail!("node ID has empty collection name: {node_id}");
    }
    if key.is_empty() {
        anyhow::bail!("node ID has empty key: {node_id}");
    }

    Ok((col, key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_node_id_valid() {
        let (col, key) = parse_node_id("hope_axioms/ax_001").unwrap();
        assert_eq!(col, "hope_axioms");
        assert_eq!(key, "ax_001");
    }

    #[test]
    fn test_parse_node_id_with_slash_in_key() {
        let (col, key) = parse_node_id("collection/key/with/slashes").unwrap();
        assert_eq!(col, "collection");
        assert_eq!(key, "key/with/slashes");
    }

    #[test]
    fn test_parse_node_id_no_slash() {
        let err = parse_node_id("just_a_key").unwrap_err();
        assert!(err.to_string().contains("collection/key"));
    }

    #[test]
    fn test_parse_node_id_empty_collection() {
        let err = parse_node_id("/some_key").unwrap_err();
        assert!(err.to_string().contains("empty collection"));
    }

    #[test]
    fn test_parse_node_id_empty_key() {
        let err = parse_node_id("collection/").unwrap_err();
        assert!(err.to_string().contains("empty key"));
    }
}
