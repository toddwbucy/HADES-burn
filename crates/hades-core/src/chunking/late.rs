//! Late chunking — mean-pool token-level embeddings per chunk.
//!
//! In late chunking, the *entire* document is encoded in one pass by the
//! embedding model, producing a token-level embedding for each token.
//! Chunk boundaries are then applied *after* encoding: each chunk's
//! embedding is the L2-normalized mean of its constituent token embeddings.
//!
//! This preserves cross-chunk context that would be lost if chunks were
//! embedded independently.

/// Configuration for late chunking.
#[derive(Debug, Clone)]
pub struct LateCunkConfig {
    /// Tokens per chunk (default 500).
    pub chunk_size_tokens: usize,
    /// Overlap tokens between consecutive chunks (default 200).
    pub overlap_tokens: usize,
}

impl Default for LateCunkConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 500,
            overlap_tokens: 200,
        }
    }
}

/// Result of late chunking: one embedding vector per chunk, plus the
/// token-index boundaries used.
#[derive(Debug, Clone)]
pub struct LateChunkResult {
    /// One embedding per chunk, L2-normalized.
    pub embeddings: Vec<Vec<f32>>,
    /// Token-index boundaries `(start, end)` for each chunk.
    pub boundaries: Vec<(usize, usize)>,
}

/// Compute chunk boundaries and mean-pool token-level embeddings.
///
/// # Arguments
/// * `token_embeddings` — one `[dimension]` vector per token, from the
///   embedding model's full-document encode pass.
/// * `config` — chunk size and overlap in tokens.
///
/// # Returns
/// A [`LateChunkResult`] with one L2-normalized embedding per chunk.
///
/// # Panics
/// Does not panic. Returns an empty result if `token_embeddings` is empty.
pub fn late_chunk_embeddings(
    token_embeddings: &[Vec<f32>],
    config: &LateCunkConfig,
) -> LateChunkResult {
    if token_embeddings.is_empty() {
        return LateChunkResult {
            embeddings: Vec::new(),
            boundaries: Vec::new(),
        };
    }

    let num_tokens = token_embeddings.len();
    let step = config.chunk_size_tokens.saturating_sub(config.overlap_tokens).max(1);

    let mut boundaries = Vec::new();
    let mut start = 0;
    while start < num_tokens {
        let end = (start + config.chunk_size_tokens).min(num_tokens);
        boundaries.push((start, end));
        if end >= num_tokens {
            break;
        }
        start += step;
    }

    let embeddings = boundaries
        .iter()
        .map(|&(s, e)| mean_pool_and_normalize(&token_embeddings[s..e]))
        .collect();

    LateChunkResult {
        embeddings,
        boundaries,
    }
}

/// Mean-pool a slice of token embeddings and L2-normalize the result.
fn mean_pool_and_normalize(token_embeddings: &[Vec<f32>]) -> Vec<f32> {
    if token_embeddings.is_empty() {
        return Vec::new();
    }

    let dim = token_embeddings[0].len();
    let count = token_embeddings.len() as f32;
    let mut pooled = vec![0.0f32; dim];

    for emb in token_embeddings {
        for (p, &v) in pooled.iter_mut().zip(emb.iter()) {
            *p += v;
        }
    }

    // Mean
    for p in &mut pooled {
        *p /= count;
    }

    // L2-normalize
    l2_normalize_f32(&mut pooled);

    pooled
}

/// In-place L2 normalization for f32 vectors.
fn l2_normalize_f32(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_late_chunk_empty() {
        let result = late_chunk_embeddings(&[], &LateCunkConfig::default());
        assert!(result.embeddings.is_empty());
        assert!(result.boundaries.is_empty());
    }

    #[test]
    fn test_late_chunk_single_chunk() {
        // 3 tokens, chunk_size=10 (larger than input) -> single chunk
        let tokens = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let config = LateCunkConfig {
            chunk_size_tokens: 10,
            overlap_tokens: 2,
        };
        let result = late_chunk_embeddings(&tokens, &config);
        assert_eq!(result.embeddings.len(), 1);
        assert_eq!(result.boundaries, vec![(0, 3)]);

        // Mean of [1,0,0], [0,1,0], [0,0,1] = [0.333, 0.333, 0.333]
        // L2 norm = sqrt(3 * 0.111) ≈ 0.577
        // Normalized ≈ [0.577, 0.577, 0.577]
        let emb = &result.embeddings[0];
        let expected = 1.0 / 3.0_f32.sqrt();
        for &v in emb {
            assert!((v - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_late_chunk_multiple_chunks() {
        // 6 tokens, chunk_size=3, overlap=1 -> step=2
        let tokens: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                v
            })
            .collect();

        let config = LateCunkConfig {
            chunk_size_tokens: 3,
            overlap_tokens: 1,
        };
        let result = late_chunk_embeddings(&tokens, &config);

        // Boundaries: (0,3), (2,5), (4,6)
        assert_eq!(result.boundaries, vec![(0, 3), (2, 5), (4, 6)]);
        assert_eq!(result.embeddings.len(), 3);

        // Each embedding should be L2-normalized (norm ≈ 1.0)
        for emb in &result.embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
        }
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize_f32(&mut v);
        // Should not produce NaN; stays zero
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mean_pool_and_normalize() {
        let tokens = vec![vec![3.0, 0.0], vec![0.0, 4.0]];
        let result = mean_pool_and_normalize(&tokens);
        // Mean = [1.5, 2.0], norm = sqrt(1.5^2 + 2.0^2) = sqrt(6.25) = 2.5
        // Normalized = [0.6, 0.8]
        assert!((result[0] - 0.6).abs() < 1e-5);
        assert!((result[1] - 0.8).abs() < 1e-5);
    }
}
