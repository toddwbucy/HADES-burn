//! Async double-buffered prefetcher for RGCN training.
//!
//! Pre-computes negative samples for upcoming epochs while the GPU
//! trains on the current epoch.  Uses a bounded tokio channel for
//! backpressure — if the GPU is slower than sampling, the producer
//! blocks until the channel has room.
//!
//! ## Usage
//!
//! ```ignore
//! // One-time setup: load graph → split edges → serialize for Python
//! let data = prepare_training_data(&pool, &path, &split_config).await?;
//!
//! // Start prefetcher — background task pre-computes negatives
//! let mut pf = Prefetcher::start(
//!     data.graph.clone(),
//!     data.split.clone(),
//!     PrefetchConfig::default(),
//!     Some(200), // epochs
//! );
//!
//! while let Some(batch) = pf.next_batch().await {
//!     // send batch.train_neg / batch.val_neg to Python TrainingService
//! }
//! ```

use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument};

use hades_core::db::ArangoPool;
use hades_core::graph::loader::GraphLoaderError;
use hades_core::graph::types::{GraphData, IDMap};

use crate::tensor::{
    EdgeSplit, NegativeSamples, SplitConfig, TensorError, negative_sample,
    prepare_and_serialize, split_edges,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from prefetcher setup and the training data pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PrefetchError {
    #[error("graph loading failed: {0}")]
    GraphLoad(#[from] GraphLoaderError),

    #[error("tensor/serialization error: {0}")]
    Tensor(#[from] TensorError),

    #[error("invalid prefetch_depth: must be >= 1")]
    InvalidDepth,

    #[error("invalid neg_sampling_ratio: {0} (must be finite and positive)")]
    InvalidNegRatio(f64),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the async prefetcher.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of epochs to prefetch ahead (channel buffer size).
    ///
    /// Higher values trade memory for latency tolerance.  Each buffered
    /// epoch holds two `NegativeSamples` (train + val), whose size is
    /// proportional to `num_train_edges × neg_sampling_ratio`.
    ///
    /// Default: 2 (double-buffered).
    pub prefetch_depth: usize,

    /// Ratio of negative to positive samples per epoch.
    /// Default: 1.0 (one negative per positive edge).
    pub neg_sampling_ratio: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_depth: 2,
            neg_sampling_ratio: 1.0,
        }
    }
}

impl PrefetchConfig {
    /// Validate configuration values.
    fn validate(&self) -> Result<(), PrefetchError> {
        if self.prefetch_depth == 0 {
            return Err(PrefetchError::InvalidDepth);
        }
        if !self.neg_sampling_ratio.is_finite() || self.neg_sampling_ratio <= 0.0 {
            return Err(PrefetchError::InvalidNegRatio(self.neg_sampling_ratio));
        }
        Ok(())
    }

    /// Estimate data memory per buffered epoch in bytes.
    ///
    /// Each `NegativeSamples` holds two `Vec<u32>` of length
    /// `num_edges × neg_sampling_ratio`.  We buffer train + val negatives.
    /// This counts only element bytes, not per-Vec heap headers (~96 bytes
    /// per batch for four `Vec<u32>`) — negligible for real workloads.
    pub fn estimate_batch_bytes(&self, num_train_edges: usize, num_val_edges: usize) -> usize {
        let train_neg = (num_train_edges as f64 * self.neg_sampling_ratio) as usize;
        let val_neg = (num_val_edges as f64 * self.neg_sampling_ratio) as usize;
        // Each NegativeSamples has src + dst Vec<u32> = 2 × len × 4 bytes
        (train_neg + val_neg) * 2 * std::mem::size_of::<u32>()
    }

    /// Estimate total memory for all buffered epochs.
    pub fn estimate_buffer_bytes(&self, num_train_edges: usize, num_val_edges: usize) -> usize {
        self.estimate_batch_bytes(num_train_edges, num_val_edges) * self.prefetch_depth
    }
}

// ---------------------------------------------------------------------------
// EpochBatch — one epoch's worth of negative samples
// ---------------------------------------------------------------------------

/// Pre-computed negative samples for a single training epoch.
///
/// Each epoch gets fresh random negatives so the model cannot memorize
/// specific negative pairs.
#[derive(Debug, Clone)]
pub struct EpochBatch {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Negative samples for training edges.
    pub train_neg: NegativeSamples,
    /// Negative samples for validation edges.
    pub val_neg: NegativeSamples,
}

// ---------------------------------------------------------------------------
// TrainingData — result of one-time setup
// ---------------------------------------------------------------------------

/// Result of [`prepare_training_data`]: everything needed to start training.
pub struct TrainingData {
    /// The loaded graph (shared ownership for the prefetcher).
    pub graph: Arc<GraphData>,
    /// Bidirectional ArangoDB `_id` ↔ index mapping.
    pub id_map: IDMap,
    /// Train/val/test edge split.
    pub split: Arc<EdgeSplit>,
}

// ---------------------------------------------------------------------------
// Prefetcher
// ---------------------------------------------------------------------------

/// Async prefetcher that pre-computes negative samples on a background task.
///
/// A bounded tokio mpsc channel connects the producer (background task
/// sampling negatives) to the consumer (training loop).  The channel
/// capacity equals `prefetch_depth`, providing natural backpressure.
///
/// The background task uses [`tokio::task::spawn_blocking`] for the
/// CPU-bound negative sampling work to avoid starving the async runtime.
pub struct Prefetcher {
    rx: mpsc::Receiver<EpochBatch>,
    handle: JoinHandle<()>,
}

impl Prefetcher {
    /// Start the prefetcher background task.
    ///
    /// * `graph` — shared graph data (only edges are read for rejection sampling).
    /// * `split` — edge split (used to compute negative sample counts).
    /// * `config` — prefetch depth and sampling ratio.
    /// * `num_epochs` — total epochs to produce.  `None` = unbounded
    ///   (stop by dropping the `Prefetcher` or calling [`stop()`](Self::stop)).
    pub fn start(
        graph: Arc<GraphData>,
        split: Arc<EdgeSplit>,
        config: PrefetchConfig,
        num_epochs: Option<usize>,
    ) -> Result<Self, PrefetchError> {
        config.validate()?;

        let num_train_neg =
            (split.train_idx.len() as f64 * config.neg_sampling_ratio) as usize;
        let num_val_neg =
            (split.val_idx.len() as f64 * config.neg_sampling_ratio) as usize;

        let buffer_bytes =
            config.estimate_buffer_bytes(split.train_idx.len(), split.val_idx.len());

        info!(
            prefetch_depth = config.prefetch_depth,
            num_train_neg,
            num_val_neg,
            buffer_mb = buffer_bytes as f64 / (1024.0 * 1024.0),
            num_epochs = ?num_epochs,
            "prefetcher starting"
        );

        let (tx, rx) = mpsc::channel(config.prefetch_depth);

        let handle = tokio::spawn(Self::producer(
            tx,
            graph,
            num_train_neg,
            num_val_neg,
            num_epochs,
        ));

        Ok(Self { rx, handle })
    }

    /// Background producer task.
    async fn producer(
        tx: mpsc::Sender<EpochBatch>,
        graph: Arc<GraphData>,
        num_train_neg: usize,
        num_val_neg: usize,
        num_epochs: Option<usize>,
    ) {
        let mut epoch = 0;
        loop {
            if let Some(max) = num_epochs
                && epoch >= max
            {
                break;
            }

            // Negative sampling is CPU-bound — run both train and val
            // concurrently on the blocking thread pool.
            let g1 = Arc::clone(&graph);
            let g2 = Arc::clone(&graph);

            let (train_neg, val_neg) = tokio::join!(
                tokio::task::spawn_blocking(move || negative_sample(&g1, num_train_neg)),
                tokio::task::spawn_blocking(move || negative_sample(&g2, num_val_neg)),
            );

            // Unwrap JoinHandle results — panic propagation
            let train_neg = train_neg.expect("train negative sampling panicked");
            let val_neg = val_neg.expect("val negative sampling panicked");

            let batch = EpochBatch {
                epoch,
                train_neg,
                val_neg,
            };

            debug!(epoch, "prefetched batch");

            // Channel send — blocks if buffer is full (backpressure).
            // Returns Err if receiver is dropped → stop producing.
            if tx.send(batch).await.is_err() {
                debug!("prefetcher channel closed, stopping");
                break;
            }

            epoch += 1;
        }

        info!(epochs_produced = epoch, "prefetcher finished");
    }

    /// Receive the next pre-computed batch.
    ///
    /// Returns `None` when all epochs have been produced and consumed,
    /// or when the prefetcher has been stopped.
    pub async fn next_batch(&mut self) -> Option<EpochBatch> {
        self.rx.recv().await
    }

    /// Stop the prefetcher, cancelling the background task.
    ///
    /// Any buffered batches still in the channel are discarded.
    /// Prefer dropping the `Prefetcher` (which also cancels) unless
    /// you need to explicitly await shutdown.
    pub fn stop(self) {
        self.handle.abort();
    }

    /// Check whether the background task is still running.
    pub fn is_running(&self) -> bool {
        !self.handle.is_finished()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

// ---------------------------------------------------------------------------
// One-time training data setup
// ---------------------------------------------------------------------------

/// Load graph from ArangoDB, split edges, and serialize to safetensors.
///
/// This is the one-time setup before training begins:
/// 1. Load the full graph (nodes + edges + embeddings) from ArangoDB.
/// 2. Split edges into train/val/test sets.
/// 3. Generate initial negative samples.
/// 4. Serialize everything to a safetensors file for Python to mmap.
///
/// Returns a [`TrainingData`] containing the graph and split (wrapped in
/// `Arc` for sharing with the prefetcher).
#[instrument(skip_all, fields(path = %output_path.display()))]
pub async fn prepare_training_data(
    pool: &ArangoPool,
    output_path: &Path,
    config: &SplitConfig,
) -> Result<TrainingData, PrefetchError> {
    info!("loading graph from ArangoDB");
    let (graph, id_map) = hades_core::graph::load(pool).await?;

    info!(
        num_nodes = graph.num_nodes,
        num_edges = graph.num_edges,
        embedded = graph.embedded_count(),
        "graph loaded, preparing training data"
    );

    // Wrap in Arc now — avoids a full deep clone into spawn_blocking
    let graph = Arc::new(graph);

    // split + negative sample + serialize — CPU-bound
    let graph_ref = Arc::clone(&graph);
    let config_clone = config.clone();
    let path = output_path.to_path_buf();

    let split = tokio::task::spawn_blocking(move || -> Result<EdgeSplit, TensorError> {
        let split = split_edges(graph_ref.num_edges, &config_clone)?;
        prepare_and_serialize(&path, &graph_ref, &config_clone)?;
        Ok(split)
    })
    .await
    .expect("serialization task panicked")?;

    info!(
        train = split.train_idx.len(),
        val = split.val_idx.len(),
        test = split.test_idx.len(),
        path = %output_path.display(),
        "training data prepared"
    );

    Ok(TrainingData {
        graph,
        id_map,
        split: Arc::new(split),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hades_core::graph::schema::JINA_DIM;

    /// Build a small test graph matching tensor.rs tests.
    fn test_graph() -> GraphData {
        let num_nodes = 10;
        let mut graph = GraphData::with_capacity(num_nodes, 0);

        let edges = [
            (0, 1, 0),
            (1, 2, 0),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 2),
            (5, 6, 3),
            (6, 7, 4),
            (7, 8, 5),
            (8, 9, 6),
            (0, 9, 7),
            (1, 5, 8),
            (2, 8, 9),
            (3, 7, 10),
            (4, 6, 11),
            (5, 0, 0),
            (6, 1, 1),
            (7, 2, 2),
            (8, 3, 3),
            (9, 4, 4),
            (0, 5, 5),
        ];
        for &(s, d, r) in &edges {
            graph.add_edge(s, d, r);
        }

        graph.collection_names = vec!["col_a".into(), "col_b".into()];
        for i in 0..num_nodes {
            graph.node_collections[i] = (i % 2) as u32;
        }

        let emb = vec![1.0f32; JINA_DIM];
        graph.set_node_features(0, &emb);
        graph.set_node_features(3, &emb);
        graph.set_node_features(7, &emb);

        graph
    }

    #[test]
    fn test_prefetch_config_defaults() {
        let config = PrefetchConfig::default();
        assert_eq!(config.prefetch_depth, 2);
        assert!((config.neg_sampling_ratio - 1.0).abs() < f64::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_prefetch_config_validate_depth() {
        let bad = PrefetchConfig {
            prefetch_depth: 0,
            neg_sampling_ratio: 1.0,
        };
        assert!(matches!(bad.validate(), Err(PrefetchError::InvalidDepth)));
    }

    #[test]
    fn test_prefetch_config_validate_neg_ratio() {
        for bad_ratio in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let config = PrefetchConfig {
                prefetch_depth: 2,
                neg_sampling_ratio: bad_ratio,
            };
            assert!(
                matches!(config.validate(), Err(PrefetchError::InvalidNegRatio(_))),
                "expected InvalidNegRatio for {bad_ratio}"
            );
        }
    }

    #[test]
    fn test_estimate_batch_bytes() {
        let config = PrefetchConfig {
            prefetch_depth: 2,
            neg_sampling_ratio: 1.0,
        };
        // 100 train edges × 1.0 ratio = 100 neg, 10 val edges × 1.0 = 10 neg
        // Each neg: 2 vecs × len × 4 bytes = (100 + 10) × 2 × 4 = 880
        let bytes = config.estimate_batch_bytes(100, 10);
        assert_eq!(bytes, 880);
    }

    #[test]
    fn test_estimate_buffer_bytes() {
        let config = PrefetchConfig {
            prefetch_depth: 3,
            neg_sampling_ratio: 1.0,
        };
        let batch = config.estimate_batch_bytes(100, 10);
        let buffer = config.estimate_buffer_bytes(100, 10);
        assert_eq!(buffer, batch * 3);
    }

    #[tokio::test]
    async fn test_prefetcher_produces_batches() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig::default();

        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            config,
            Some(5),
        )
        .unwrap();

        let mut received = Vec::new();
        while let Some(batch) = pf.next_batch().await {
            received.push(batch);
        }

        assert_eq!(received.len(), 5, "should receive exactly 5 batches");

        // Epoch numbers are sequential
        for (i, batch) in received.iter().enumerate() {
            assert_eq!(batch.epoch, i);
        }

        // Each batch has non-empty negatives
        for batch in &received {
            assert!(!batch.train_neg.src.is_empty());
            assert_eq!(batch.train_neg.src.len(), batch.train_neg.dst.len());
            assert!(!batch.val_neg.src.is_empty());
            assert_eq!(batch.val_neg.src.len(), batch.val_neg.dst.len());
        }
    }

    #[tokio::test]
    async fn test_prefetcher_different_negatives_per_epoch() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig::default();

        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            config,
            Some(3),
        )
        .unwrap();

        let b0 = pf.next_batch().await.unwrap();
        let b1 = pf.next_batch().await.unwrap();
        let b2 = pf.next_batch().await.unwrap();

        // It's astronomically unlikely that two random samples are identical
        // on a 10-node graph with 20 edges. Check at least one pair differs.
        let all_same = b0.train_neg.src == b1.train_neg.src
            && b1.train_neg.src == b2.train_neg.src
            && b0.train_neg.dst == b1.train_neg.dst
            && b1.train_neg.dst == b2.train_neg.dst;

        assert!(
            !all_same,
            "negative samples should differ across epochs (re-randomized)"
        );
    }

    #[tokio::test]
    async fn test_prefetcher_stop_early() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig {
            prefetch_depth: 1,
            neg_sampling_ratio: 1.0,
        };

        // Request unbounded but stop after 2
        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            config,
            None, // unbounded
        )
        .unwrap();

        let _b0 = pf.next_batch().await.unwrap();
        let _b1 = pf.next_batch().await.unwrap();

        // Dropping should cancel the background task
        pf.stop();
    }

    #[tokio::test]
    async fn test_prefetcher_drop_cancels() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());

        let pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            PrefetchConfig::default(),
            None,
        )
        .unwrap();

        assert!(pf.is_running());

        // Drop should abort the background task
        drop(pf);

        // Give the runtime a moment to process the abort
        tokio::task::yield_now().await;
    }

    #[tokio::test]
    async fn test_prefetcher_custom_ratio() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig {
            prefetch_depth: 1,
            neg_sampling_ratio: 2.0,
        };

        let expected_train_neg = (split.train_idx.len() as f64 * 2.0) as usize;

        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            config,
            Some(1),
        )
        .unwrap();

        let batch = pf.next_batch().await.unwrap();
        assert_eq!(batch.train_neg.src.len(), expected_train_neg);
    }

    #[tokio::test]
    async fn test_prefetcher_negatives_in_bounds() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());

        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            PrefetchConfig::default(),
            Some(3),
        )
        .unwrap();

        let num_nodes = graph.num_nodes as u32;

        while let Some(batch) = pf.next_batch().await {
            for (&s, &d) in batch.train_neg.src.iter().zip(&batch.train_neg.dst) {
                assert!(s < num_nodes, "train neg src {s} out of bounds");
                assert!(d < num_nodes, "train neg dst {d} out of bounds");
                assert_ne!(s, d, "train neg should not have self-loops");
            }
            for (&s, &d) in batch.val_neg.src.iter().zip(&batch.val_neg.dst) {
                assert!(s < num_nodes, "val neg src {s} out of bounds");
                assert!(d < num_nodes, "val neg dst {d} out of bounds");
                assert_ne!(s, d, "val neg should not have self-loops");
            }
        }
    }

    #[test]
    fn test_epoch_batch_clone() {
        let batch = EpochBatch {
            epoch: 0,
            train_neg: NegativeSamples {
                src: vec![0, 1],
                dst: vec![2, 3],
            },
            val_neg: NegativeSamples {
                src: vec![4],
                dst: vec![5],
            },
        };
        let cloned = batch.clone();
        assert_eq!(cloned.epoch, 0);
        assert_eq!(cloned.train_neg.src, vec![0, 1]);
    }

    #[test]
    fn test_training_data_fields() {
        let graph = test_graph();
        let split = split_edges(graph.num_edges, &SplitConfig::default()).unwrap();

        let data = TrainingData {
            graph: Arc::new(graph),
            id_map: IDMap::new(),
            split: Arc::new(split),
        };

        assert_eq!(data.graph.num_nodes, 10);
        assert_eq!(data.graph.num_edges, 20);
        assert!(data.id_map.is_empty());
        assert_eq!(
            data.split.train_idx.len() + data.split.val_idx.len() + data.split.test_idx.len(),
            20
        );
    }
}
