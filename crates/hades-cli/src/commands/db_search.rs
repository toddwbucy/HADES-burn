//! Native Rust handler for `hades db query` — semantic search.
//!
//! Implements a two-phase vector search pipeline:
//! 1. Embed query text via the embedder gRPC service
//! 2. Fetch stored embeddings from ArangoDB (no vector index required)
//! 3. Brute-force cosine similarity in Rust, select top-K
//! 4. Batch-fetch chunk text + metadata for top-K results
//! 5. (Optional) hybrid reranking: blend cosine score with term-coverage score
//!
//! Efficient for collections up to ~100K embeddings. For larger collections,
//! a vector index (`APPROX_NEAR_COSINE`) would be needed.
//!
//! Outputs results via the shared output formatter with envelope.

use std::collections::HashSet;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tracing::info;

use hades_core::config::HadesConfig;
use hades_core::db::collections::CollectionProfile;
use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::ArangoPool;
use hades_core::persephone::embedding::EmbeddingClient;

use super::output::{self, OutputFormat};

/// `hades db query TEXT [--limit N] [--collection C] [--hybrid] [--structural] [--format F]`
pub async fn run_query(
    config: &HadesConfig,
    search_text: &str,
    limit: u32,
    collection: Option<&str>,
    hybrid: bool,
    structural: bool,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;

    // 1. Resolve collection profile (honors HADES_DEFAULT_COLLECTION env var).
    let default_name;
    let profile_name = match collection {
        Some(name) => name,
        None => {
            default_name = std::env::var("HADES_DEFAULT_COLLECTION")
                .unwrap_or_else(|_| "arxiv".to_string());
            &default_name
        }
    };
    let profile = CollectionProfile::get(profile_name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown collection profile '{profile_name}' — valid: {}",
            CollectionProfile::all()
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    // 2. Embed the query text.
    let client = EmbeddingClient::connect_unix_at(&config.embedding.service.socket)
        .await
        .context("failed to connect to embedding service — is it running?")?;
    let embed_result = client
        .embed(&[search_text.to_string()], "retrieval.query", None)
        .await
        .context("failed to embed query text")?;

    let query_vec = &embed_result.embeddings[0];
    info!(
        dimension = embed_result.dimension,
        model = %embed_result.model,
        duration_ms = embed_result.duration_ms,
        "embedded query"
    );

    // 3. Connect to ArangoDB.
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // 4. Two-phase vector search.
    //
    // Phase 1: Fetch embedding vectors + keys (compact — no text).
    // Phase 2: Compute cosine similarity in Rust, take top-K.
    // Phase 3: Fetch full chunk + metadata documents for top-K only.
    //
    // This avoids needing a vector index (APPROX_NEAR_COSINE) and is
    // efficient for collections up to ~100K embeddings.
    let fk = profile.foreign_key;

    // Phase 1: Fetch all embedding vectors with their keys.
    let fetch_aql = format!(
        "FOR emb IN @@embeddings \
             FILTER emb.chunk_key != null AND emb.{fk} != null AND emb.{fk} != '' \
             RETURN {{ chunk_key: emb.chunk_key, {fk}: emb.{fk}, embedding: emb.embedding }}"
    );
    let fetch_bind = json!({ "@embeddings": profile.embeddings });

    let emb_result = query::query(
        &pool,
        &fetch_aql,
        Some(&fetch_bind),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    .context("failed to fetch embeddings for vector search")?;

    info!(embedding_count = emb_result.results.len(), "fetched embeddings");

    // Phase 2: Compute cosine similarity in Rust, take top-K.
    let mut scored: Vec<(f64, &Value)> = emb_result
        .results
        .iter()
        .filter_map(|doc| {
            let emb = doc["embedding"].as_array()?;
            let stored: Vec<f32> = emb.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if stored.len() != query_vec.len() {
                return None;
            }
            let score = cosine_similarity(query_vec, &stored);
            Some((score as f64, doc))
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit as usize);

    // Phase 3: Batch-fetch chunk + metadata for top-K results.
    let items: Vec<Value> = scored
        .iter()
        .map(|(score, doc)| {
            json!({
                "chunk_key": doc["chunk_key"].as_str().unwrap_or(""),
                "parent_key": doc[fk].as_str().unwrap_or(""),
                "score": score,
            })
        })
        .collect();

    let mut results: Vec<Value> = if items.is_empty() {
        Vec::new()
    } else {
        let detail_aql = format!(
            "FOR item IN @items \
                 LET chunk = DOCUMENT(CONCAT(@chunks_col, '/', item.chunk_key)) \
                 LET meta = DOCUMENT(CONCAT(@metadata_col, '/', item.parent_key)) \
                 RETURN {{ \
                     {fk}: item.parent_key, \
                     text: chunk.text, \
                     chunk_index: chunk.chunk_index, \
                     total_chunks: chunk.total_chunks, \
                     title: meta.title, \
                     arxiv_id: meta.arxiv_id, \
                     score: item.score \
                 }}"
        );
        let detail_bind = json!({
            "chunks_col": profile.chunks,
            "metadata_col": profile.metadata,
            "items": items,
        });

        let detail = query::query(
            &pool,
            &detail_aql,
            Some(&detail_bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .context("failed to fetch result details")?;

        detail.results
    };

    // 5. (Optional) Hybrid keyword reranking.
    if hybrid && !results.is_empty() {
        results = hybrid_rerank(search_text, results);
    }

    // 6. (Optional) Structural graph fusion.
    if structural && !results.is_empty() {
        results = structural_rerank(&pool, profile.metadata, &results).await?;
    }

    // 7. Output.
    let data = json!({
        "query": search_text,
        "collection": profile_name,
        "model": embed_result.model,
        "dimension": embed_result.dimension,
        "result_count": results.len(),
        "results": results,
    });

    output::print_output("db.query", data, &fmt);
    Ok(())
}

// ── Vector math ─────────────────────────────────────────────────────────

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

// ── Structural graph fusion ────────────────────────────────────────────

/// Blend search scores with RGCN structural embedding similarity.
///
/// Algorithm (matches Python `structural_fusion.py:structural_rerank`):
/// 1. Collect document keys from results
/// 2. Batch-fetch `structural_embedding` from the metadata collection
/// 3. Compute centroid from top-3 results' structural embeddings
/// 4. Compute cosine similarity of each result's embedding to centroid
/// 5. Blend: `fused = 0.7 * current_score + 0.3 * structural_cosine`
/// 6. Re-sort by fused score
async fn structural_rerank(
    pool: &ArangoPool,
    metadata_col: &str,
    results: &[Value],
) -> Result<Vec<Value>> {
    // Collect parent keys from results.
    let parent_keys: Vec<&str> = results
        .iter()
        .filter_map(|r| {
            r.get("arxiv_id")
                .and_then(|v| v.as_str())
                .or_else(|| r.get("parent_key").and_then(|v| v.as_str()))
        })
        .collect();

    if parent_keys.is_empty() {
        return Ok(results.to_vec());
    }

    // Batch-fetch structural embeddings from metadata collection.
    let aql = "FOR key IN @keys \
               LET doc = DOCUMENT(CONCAT(@col, '/', key)) \
               RETURN { _key: key, structural_embedding: doc.structural_embedding }";
    let bind = json!({ "keys": parent_keys, "col": metadata_col });

    let emb_result = query::query(
        pool,
        aql,
        Some(&bind),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    .context("failed to fetch structural embeddings")?;

    // Build a map of key → structural embedding.
    let mut emb_map: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for doc in &emb_result.results {
        let key = doc["_key"].as_str().unwrap_or("");
        if let Some(arr) = doc["structural_embedding"].as_array() {
            let vec: Vec<f32> = arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
            if !vec.is_empty() {
                emb_map.insert(key.to_string(), vec);
            }
        }
    }

    if emb_map.is_empty() {
        info!("no structural embeddings found, skipping structural fusion");
        return Ok(results.to_vec());
    }

    // Compute centroid from top-3 results that have structural embeddings.
    let top_vecs: Vec<&Vec<f32>> = results
        .iter()
        .filter_map(|r| {
            let key = r.get("arxiv_id")
                .and_then(|v| v.as_str())
                .or_else(|| r.get("parent_key").and_then(|v| v.as_str()))?;
            emb_map.get(key)
        })
        .take(3)
        .collect();

    if top_vecs.is_empty() {
        return Ok(results.to_vec());
    }

    let centroid = compute_centroid(&top_vecs);

    // Blend scores.
    let mut fused: Vec<Value> = results
        .iter()
        .map(|r| {
            let mut r = r.clone();
            let current_score = r["score"].as_f64().unwrap_or(0.0);
            let key = r.get("arxiv_id")
                .and_then(|v| v.as_str())
                .or_else(|| r.get("parent_key").and_then(|v| v.as_str()))
                .unwrap_or("");

            let structural_sim = emb_map
                .get(key)
                .map(|emb| cosine_similarity_f32(&centroid, emb) as f64)
                .unwrap_or(0.0);

            let fused_score = 0.7 * current_score + 0.3 * structural_sim;
            r["structural_score"] = json!(structural_sim);
            r["score"] = json!(fused_score);
            r
        })
        .collect();

    // Re-sort by fused score descending.
    fused.sort_by(|a, b| {
        let sa = a["score"].as_f64().unwrap_or(0.0);
        let sb = b["score"].as_f64().unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    info!(
        structural_embeddings = emb_map.len(),
        centroid_sources = top_vecs.len(),
        "structural fusion applied"
    );

    Ok(fused)
}

/// Compute the element-wise mean of a set of vectors.
fn compute_centroid(vecs: &[&Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let dim = vecs[0].len();
    let n = vecs.len() as f32;
    let mut centroid = vec![0.0f32; dim];
    for v in vecs {
        for (i, val) in v.iter().enumerate() {
            if i < dim {
                centroid[i] += val;
            }
        }
    }
    for c in &mut centroid {
        *c /= n;
    }
    centroid
}

/// Cosine similarity between two f32 vectors (same as `cosine_similarity` but
/// takes `&[f32]` for ergonomics with the structural embedding code).
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity(a, b)
}

// ── Hybrid keyword reranking ────────────────────────────────────────────

/// Blend vector similarity scores with keyword term-frequency scores.
///
/// Score formula: `0.7 * vector_score + 0.3 * keyword_score`
///
/// The keyword score is the fraction of query terms that appear in the
/// result text (case-insensitive). Simple but effective for filtering
/// out vector-similar but topically irrelevant results.
fn hybrid_rerank(query_text: &str, mut results: Vec<Value>) -> Vec<Value> {
    let query_terms = tokenize(query_text);
    if query_terms.is_empty() {
        return results;
    }

    for result in &mut results {
        let vector_score = result["score"].as_f64().unwrap_or(0.0);
        let text = result["text"].as_str().unwrap_or("");
        let keyword_score = keyword_tf_score(&query_terms, text);
        let blended = 0.7 * vector_score + 0.3 * keyword_score;

        result["vector_score"] = json!(vector_score);
        result["keyword_score"] = json!(keyword_score);
        result["score"] = json!(blended);
    }

    // Re-sort by blended score descending.
    results.sort_by(|a, b| {
        let sa = a["score"].as_f64().unwrap_or(0.0);
        let sb = b["score"].as_f64().unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Tokenize text into lowercase terms, filtering tokens shorter than 2 chars.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Compute term-coverage score: fraction of unique query terms present in text.
fn keyword_tf_score(query_terms: &[String], text: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let text_terms: HashSet<&str> = text_lower
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 2)
        .collect();

    let unique_terms: HashSet<&str> = query_terms.iter().map(|s| s.as_str()).collect();
    if unique_terms.is_empty() {
        return 0.0;
    }

    let matches = unique_terms
        .iter()
        .filter(|qt| text_terms.contains(*qt))
        .count();

    matches as f64 / unique_terms.len() as f64
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("attention mechanism in transformers");
        assert_eq!(tokens, vec!["attention", "mechanism", "in", "transformers"]);
    }

    #[test]
    fn test_tokenize_filters_short() {
        let tokens = tokenize("a b cd ef");
        assert_eq!(tokens, vec!["cd", "ef"]);
    }

    #[test]
    fn test_keyword_tf_score_full_match() {
        let terms = tokenize("nested learning");
        let score = keyword_tf_score(&terms, "This paper introduces nested learning theory");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_keyword_tf_score_partial_match() {
        let terms = tokenize("nested learning optimization");
        let score = keyword_tf_score(&terms, "nested architectures use deep layers");
        // Only "nested" matches → 1/3
        assert!((score - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_keyword_tf_score_no_match() {
        let terms = tokenize("quantum computing");
        let score = keyword_tf_score(&terms, "attention mechanism in transformers");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hybrid_rerank_reorders() {
        let results = vec![
            json!({ "text": "unrelated content about cooking", "score": 0.95 }),
            json!({ "text": "attention mechanism for transformers", "score": 0.90 }),
        ];

        let reranked = hybrid_rerank("attention transformers", results);
        // Second result should move up because it has keyword matches.
        assert!(reranked[0]["text"].as_str().unwrap().contains("attention"));
    }

    #[test]
    fn test_hybrid_preserves_scores() {
        let results = vec![
            json!({ "text": "attention mechanism paper", "score": 0.9 }),
        ];
        let reranked = hybrid_rerank("attention mechanism", results);
        // Should have vector_score and keyword_score fields
        assert!(reranked[0].get("vector_score").is_some());
        assert!(reranked[0].get("keyword_score").is_some());
    }

    // ── Structural fusion unit tests ───────────────────────────────────

    #[test]
    fn test_compute_centroid_single_vec() {
        let v = vec![1.0, 2.0, 3.0];
        let centroid = compute_centroid(&[&v]);
        assert_eq!(centroid, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compute_centroid_multiple_vecs() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0];
        let centroid = compute_centroid(&[&v1, &v2, &v3]);
        let expected = 1.0 / 3.0;
        for c in &centroid {
            assert!((c - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_centroid_empty() {
        let centroid = compute_centroid(&[]);
        assert!(centroid.is_empty());
    }

    #[test]
    fn test_cosine_similarity_f32_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity_f32(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_f32_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity_f32(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
}
