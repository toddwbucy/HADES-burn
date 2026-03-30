//! Vector similarity search — ANN (server-side) with brute-force fallback.
//!
//! Uses the dual-path strategy from Python HADES:
//! - **ANN path**: When a vector index exists and no `doc_filter` is set,
//!   delegates to ArangoDB's APPROX_NEAR_* AQL functions which route
//!   through the server-side FAISS index.
//! - **Brute-force path**: Fetches all embeddings and computes cosine
//!   similarity client-side.  Used when no index exists or when a
//!   `doc_filter` is specified (ANN cannot push filters).

use serde_json::Value;
use tracing::{debug, instrument, trace};

use super::collections::CollectionProfile;
use super::error::ArangoError;
use super::index::{detect_vector_index, VectorMetric};
use super::pool::ArangoPool;
use super::query::{self, ExecutionTarget};

/// Result of a vector similarity search.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The document key from the embeddings collection.
    pub paper_key: String,
    /// Text content from the chunk.
    pub text: Option<String>,
    /// Chunk index within the parent document.
    pub chunk_index: Option<u64>,
    /// Total chunks in the parent document.
    pub total_chunks: Option<u64>,
    /// Title from the metadata document.
    pub title: Option<String>,
    /// ArXiv ID from the metadata document.
    pub arxiv_id: Option<String>,
    /// Similarity score (higher = more similar for cosine/innerProduct).
    pub score: f64,
}

/// Search for similar embeddings using ANN or brute-force fallback.
///
/// Routing decision matches Python HADES:
/// - ANN when a vector index exists AND no `doc_filter` is set.
/// - Brute-force otherwise.
#[instrument(skip(pool, query_embedding), fields(db = %pool.database()))]
pub async fn query_similar(
    pool: &ArangoPool,
    profile: &CollectionProfile,
    query_embedding: &[f64],
    limit: usize,
    doc_filter: Option<&str>,
    n_probe: Option<u32>,
) -> Result<Vec<SimilarityResult>, ArangoError> {
    let detected_metric = detect_vector_index(pool, profile.embeddings).await?;

    match (detected_metric, doc_filter) {
        (Some(metric), None) => {
            debug!(?metric, "routing to ANN search");
            query_ann(pool, profile, query_embedding, limit, n_probe, metric).await
        }
        _ => {
            debug!(
                has_index = detected_metric.is_some(),
                has_filter = doc_filter.is_some(),
                "routing to brute-force search"
            );
            query_brute_force(pool, profile, query_embedding, limit, doc_filter).await
        }
    }
}

// ---------------------------------------------------------------------------
// ANN search (server-side via FAISS vector index)
// ---------------------------------------------------------------------------

/// Map a metric to its APPROX_NEAR_* AQL function name.
fn approx_fn(metric: VectorMetric) -> &'static str {
    match metric {
        VectorMetric::Cosine => "APPROX_NEAR_COSINE",
        VectorMetric::L2 => "APPROX_NEAR_L2",
        VectorMetric::InnerProduct => "APPROX_NEAR_INNER_PRODUCT",
    }
}

/// Whether results should be sorted descending for a given metric.
///
/// Cosine and inner-product return similarity (higher = closer).
/// L2 returns distance (lower = closer).
fn sort_descending(metric: VectorMetric) -> bool {
    match metric {
        VectorMetric::Cosine | VectorMetric::InnerProduct => true,
        VectorMetric::L2 => false,
    }
}

/// ANN search using the server-side FAISS vector index.
async fn query_ann(
    pool: &ArangoPool,
    profile: &CollectionProfile,
    query_embedding: &[f64],
    limit: usize,
    n_probe: Option<u32>,
    metric: VectorMetric,
) -> Result<Vec<SimilarityResult>, ArangoError> {
    let func = approx_fn(metric);
    let sort_order = if sort_descending(metric) { "DESC" } else { "ASC" };

    // Build the options argument for nProbe if specified
    let options_fragment = if n_probe.is_some() {
        ", { nProbe: @n_probe }"
    } else {
        ""
    };

    let aql = format!(
        r#"
        FOR emb IN {embeddings}
            FILTER emb.chunk_key != null
            LET score = {func}(emb.embedding, @query_vec{options_fragment})
            SORT score {sort_order}
            LIMIT @limit
            LET chunk = DOCUMENT(CONCAT("{chunks}/", emb.chunk_key))
            LET meta = DOCUMENT(CONCAT("{metadata}/", emb.paper_key))
            RETURN {{
                paper_key: emb.paper_key,
                text: chunk.text,
                chunk_index: chunk.chunk_index,
                total_chunks: chunk.total_chunks,
                title: meta.title,
                arxiv_id: meta.arxiv_id,
                score: score
            }}
        "#,
        embeddings = profile.embeddings,
        chunks = profile.chunks,
        metadata = profile.metadata,
    );

    let mut bind_vars = serde_json::json!({
        "query_vec": query_embedding,
        "limit": limit,
    });
    if let Some(np) = n_probe {
        bind_vars["n_probe"] = serde_json::json!(np);
    }

    debug!(func, sort_order, limit, "executing ANN search");

    let result = query::query(
        pool,
        &aql,
        Some(&bind_vars),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await?;

    parse_search_results(result.results)
}

// ---------------------------------------------------------------------------
// Brute-force search (client-side cosine similarity)
// ---------------------------------------------------------------------------

/// Brute-force search — fetches all embeddings and computes cosine similarity.
async fn query_brute_force(
    pool: &ArangoPool,
    profile: &CollectionProfile,
    query_embedding: &[f64],
    limit: usize,
    doc_filter: Option<&str>,
) -> Result<Vec<SimilarityResult>, ArangoError> {
    let filter_clause = if doc_filter.is_some() {
        "FILTER emb.paper_key == @paper_key"
    } else {
        ""
    };

    let aql = format!(
        r#"
        FOR emb IN {embeddings}
            FILTER emb.chunk_key != null
            {filter_clause}
            LET chunk = DOCUMENT(CONCAT("{chunks}/", emb.chunk_key))
            LET meta = DOCUMENT(CONCAT("{metadata}/", emb.paper_key))
            RETURN {{
                paper_key: emb.paper_key,
                embedding: emb.embedding,
                text: chunk.text,
                chunk_index: chunk.chunk_index,
                total_chunks: chunk.total_chunks,
                title: meta.title,
                arxiv_id: meta.arxiv_id
            }}
        "#,
        embeddings = profile.embeddings,
        chunks = profile.chunks,
        metadata = profile.metadata,
    );

    let mut bind_vars = serde_json::json!({});
    if let Some(filter) = doc_filter {
        bind_vars["paper_key"] = serde_json::json!(filter);
    }

    debug!(limit, has_filter = doc_filter.is_some(), "executing brute-force search");

    // Large batch_size avoids cursor pagination — the read-only proxy
    // doesn't support cursor continuation.
    let result = query::query(
        pool,
        &aql,
        Some(&bind_vars),
        Some(50_000),
        false,
        ExecutionTarget::Reader,
    )
    .await?;

    // L2-normalize the query vector
    let query_norm = l2_normalize(query_embedding);

    // Score each result by cosine similarity
    let mut scored: Vec<(f64, Value)> = Vec::with_capacity(result.results.len());
    for doc in result.results {
        let embedding = match doc.get("embedding").and_then(|e| e.as_array()) {
            Some(arr) => arr,
            None => {
                trace!("skipping document with missing/null embedding");
                continue;
            }
        };

        let emb_vec: Vec<f64> = embedding
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        if emb_vec.len() != query_embedding.len() {
            trace!(
                expected = query_embedding.len(),
                got = emb_vec.len(),
                "skipping document with mismatched embedding dimension"
            );
            continue;
        }

        let emb_norm = l2_normalize(&emb_vec);
        let score = dot_product(&query_norm, &emb_norm);
        scored.push((score, doc));
    }

    // Sort by score descending (higher cosine = more similar)
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    // Parse into SimilarityResult (strip raw embedding)
    let mut results = Vec::with_capacity(scored.len());
    for (score, doc) in scored {
        let paper_key = doc
            .get("paper_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        results.push(SimilarityResult {
            paper_key,
            text: doc.get("text").and_then(|v| v.as_str()).map(String::from),
            chunk_index: doc.get("chunk_index").and_then(|v| v.as_u64()),
            total_chunks: doc.get("total_chunks").and_then(|v| v.as_u64()),
            title: doc.get("title").and_then(|v| v.as_str()).map(String::from),
            arxiv_id: doc.get("arxiv_id").and_then(|v| v.as_str()).map(String::from),
            score,
        });
    }

    debug!(total_scored = results.len(), "brute-force search complete");
    Ok(results)
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// L2-normalize a vector (with epsilon for numerical stability).
fn l2_normalize(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-8;
    v.iter().map(|x| x / norm).collect()
}

/// Dot product of two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Result parsing
// ---------------------------------------------------------------------------

/// Parse AQL result documents into `SimilarityResult` structs.
fn parse_search_results(docs: Vec<Value>) -> Result<Vec<SimilarityResult>, ArangoError> {
    let mut results = Vec::with_capacity(docs.len());
    for doc in docs {
        let paper_key = doc
            .get("paper_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let score = doc
            .get("score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        results.push(SimilarityResult {
            paper_key,
            text: doc.get("text").and_then(|v| v.as_str()).map(String::from),
            chunk_index: doc.get("chunk_index").and_then(|v| v.as_u64()),
            total_chunks: doc.get("total_chunks").and_then(|v| v.as_u64()),
            title: doc.get("title").and_then(|v| v.as_str()).map(String::from),
            arxiv_id: doc.get("arxiv_id").and_then(|v| v.as_str()).map(String::from),
            score,
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        // norm = 5.0 + 1e-8
        let expected_norm = 5.0 + 1e-8;
        assert!((n[0] - 3.0 / expected_norm).abs() < 1e-6);
        assert!((n[1] - 4.0 / expected_norm).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        // Should not panic — epsilon prevents division by zero
        for x in &n {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let a = l2_normalize(&v);
        let b = l2_normalize(&v);
        let sim = dot_product(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have cosine ~1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = l2_normalize(&[1.0, 0.0]);
        let b = l2_normalize(&[0.0, 1.0]);
        let sim = dot_product(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have cosine ~0.0");
    }

    #[test]
    fn test_approx_fn_mapping() {
        assert_eq!(approx_fn(VectorMetric::Cosine), "APPROX_NEAR_COSINE");
        assert_eq!(approx_fn(VectorMetric::L2), "APPROX_NEAR_L2");
        assert_eq!(approx_fn(VectorMetric::InnerProduct), "APPROX_NEAR_INNER_PRODUCT");
    }

    #[test]
    fn test_sort_order() {
        assert!(sort_descending(VectorMetric::Cosine));
        assert!(sort_descending(VectorMetric::InnerProduct));
        assert!(!sort_descending(VectorMetric::L2));
    }

    #[test]
    fn test_parse_search_results() {
        let docs = vec![serde_json::json!({
            "paper_key": "2501_12345",
            "text": "some text",
            "chunk_index": 0,
            "total_chunks": 3,
            "title": "A Paper",
            "arxiv_id": "2501.12345",
            "score": 0.95
        })];

        let results = parse_search_results(docs).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].paper_key, "2501_12345");
        assert_eq!(results[0].text.as_deref(), Some("some text"));
        assert!((results[0].score - 0.95).abs() < 1e-10);
    }
}
