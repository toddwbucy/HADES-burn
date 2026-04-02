//! Tensor serialization and IPC for the training pipeline.
//!
//! Serializes [`GraphData`] plus edge splits and negative samples into
//! the [safetensors](https://huggingface.co/docs/safetensors) format.
//! The file can be memory-mapped for zero-copy reads by the training loop.
//!
//! ## Safetensors layout
//!
//! **Tensors:**
//! - `node_features`    — `F32  [N, D]`
//! - `has_embedding`    — `BOOL [N]`
//! - `node_collections` — `U32  [N]`
//! - `edge_src`         — `U32  [E]`
//! - `edge_dst`         — `U32  [E]`
//! - `edge_type`        — `U32  [E]`
//! - `train_idx`        — `U32  [E_train]`
//! - `val_idx`          — `U32  [E_val]`
//! - `test_idx`         — `U32  [E_test]`
//! - `neg_src`          — `U32  [E_neg]`
//! - `neg_dst`          — `U32  [E_neg]`
//!
//! **Metadata (JSON header):**
//! - `num_nodes`, `num_edges`, `num_relations`, `feature_dim`
//! - `collection_names` (JSON array)
//! - `val_ratio`, `test_ratio`, `neg_sampling_ratio`

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;
use rand::Rng;
use rand::seq::SliceRandom;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use tracing::{info, warn};

use hades_core::graph::types::GraphData;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("graph has no edges — cannot split or sample")]
    EmptyGraph,

    #[error("tensor '{name}' not found in safetensors file")]
    MissingTensor { name: String },

    #[error("metadata key '{key}' not found")]
    MissingMetadata { key: String },

    #[error("metadata parse error for '{key}': {message}")]
    MetadataParse { key: String, message: String },

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tensor '{name}' has dtype {actual:?}, expected {expected:?}")]
    DtypeMismatch {
        name: String,
        expected: Dtype,
        actual: Dtype,
    },

    #[error("invalid split config: val_ratio ({val}) + test_ratio ({test}) = {sum} > 1.0")]
    InvalidSplitConfig { val: f64, test: f64, sum: f64 },

    #[error("serialization validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("invalid neg_sampling_ratio: {neg} (must be finite and non-negative)")]
    InvalidNegSamplingRatio { neg: f64 },
}

// ---------------------------------------------------------------------------
// Edge split configuration
// ---------------------------------------------------------------------------

/// Configuration for train/val/test edge splitting and negative sampling.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Fraction of edges for validation (default: 0.1).
    pub val_ratio: f64,
    /// Fraction of edges for test (default: 0.1).
    pub test_ratio: f64,
    /// Ratio of negative to positive samples (default: 1.0).
    pub neg_sampling_ratio: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            val_ratio: 0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge split result
// ---------------------------------------------------------------------------

/// Train/val/test edge split — indices into the edge arrays.
#[derive(Debug, Clone)]
pub struct EdgeSplit {
    /// Indices of training edges.
    pub train_idx: Vec<u32>,
    /// Indices of validation edges.
    pub val_idx: Vec<u32>,
    /// Indices of test edges.
    pub test_idx: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Negative samples
// ---------------------------------------------------------------------------

/// Negative edge samples (non-existing node pairs).
#[derive(Debug, Clone)]
pub struct NegativeSamples {
    /// Source node indices.
    pub src: Vec<u32>,
    /// Destination node indices.
    pub dst: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Edge splitting — random permutation
// ---------------------------------------------------------------------------

/// Split edges into train/val/test sets via random permutation.
///
/// Matches Python `RGCNTrainer._split_edges()`.
pub fn split_edges(num_edges: usize, config: &SplitConfig) -> Result<EdgeSplit, TensorError> {
    if num_edges == 0 {
        return Err(TensorError::EmptyGraph);
    }

    let sum = config.val_ratio + config.test_ratio;
    if !config.val_ratio.is_finite()
        || !config.test_ratio.is_finite()
        || sum > 1.0
        || config.val_ratio < 0.0
        || config.test_ratio < 0.0
    {
        return Err(TensorError::InvalidSplitConfig {
            val: config.val_ratio,
            test: config.test_ratio,
            sum,
        });
    }

    let mut perm: Vec<u32> = (0..num_edges as u32).collect();
    perm.shuffle(&mut rand::rng());

    let val_size = (num_edges as f64 * config.val_ratio) as usize;
    let test_size = (num_edges as f64 * config.test_ratio) as usize;
    let train_size = num_edges - val_size - test_size;

    let train_idx = perm[..train_size].to_vec();
    let val_idx = perm[train_size..train_size + val_size].to_vec();
    let test_idx = perm[train_size + val_size..].to_vec();

    info!(train = train_size, val = val_size, test = test_size, "edge split");

    Ok(EdgeSplit {
        train_idx,
        val_idx,
        test_idx,
    })
}

// ---------------------------------------------------------------------------
// Negative sampling — rejection sampling
// ---------------------------------------------------------------------------

/// Generate negative edge samples via rejection sampling.
///
/// Matches Python `RGCNTrainer._negative_sample()`. Samples node pairs
/// that are **not** existing edges and **not** self-loops. The exclusion
/// set is built from the full edge set (not just train edges).
pub fn negative_sample(graph: &GraphData, num_neg: usize) -> NegativeSamples {
    if num_neg == 0 || graph.num_nodes < 2 {
        return NegativeSamples {
            src: Vec::new(),
            dst: Vec::new(),
        };
    }

    // Build existing edge set for O(1) lookup
    let mut existing: HashSet<(u32, u32)> =
        HashSet::with_capacity(graph.num_edges);
    for (&s, &d) in graph.edge_src.iter().zip(&graph.edge_dst) {
        existing.insert((s, d));
    }

    let mut rng = rand::rng();
    let mut neg_src = Vec::with_capacity(num_neg);
    let mut neg_dst = Vec::with_capacity(num_neg);
    let max_attempts = num_neg * 10;
    let mut attempts = 0;
    let num_nodes = graph.num_nodes as u32;

    while neg_src.len() < num_neg && attempts < max_attempts {
        // Generate candidates in batches for efficiency
        let batch_size = num_neg.min(1024);
        for _ in 0..batch_size {
            let s: u32 = rng.random_range(0..num_nodes);
            let d: u32 = rng.random_range(0..num_nodes);
            if s != d && !existing.contains(&(s, d)) {
                neg_src.push(s);
                neg_dst.push(d);
                if neg_src.len() >= num_neg {
                    break;
                }
            }
        }
        attempts += batch_size;
    }

    // Truncate to exactly num_neg (may be short if graph is too dense)
    neg_src.truncate(num_neg);
    neg_dst.truncate(num_neg);

    if neg_src.len() < num_neg {
        warn!(
            requested = num_neg,
            actual = neg_src.len(),
            "negative sampling exhausted max attempts"
        );
    }

    NegativeSamples {
        src: neg_src,
        dst: neg_dst,
    }
}

// ---------------------------------------------------------------------------
// Serialization — GraphData + splits → safetensors
// ---------------------------------------------------------------------------

/// Helper: reinterpret a `&[T]` as `&[u8]`.
///
/// # Safety assumption
/// Safetensors uses little-endian byte order. This transmute is correct
/// on little-endian platforms (x86, ARM LE, RISC-V LE). The compile-time
/// assertion below prevents silent corruption on big-endian targets.
fn as_bytes<T>(slice: &[T]) -> &[u8] {
    const { assert!(cfg!(target_endian = "little"), "safetensors requires little-endian") }
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}

/// Serialize a graph with edge splits and negative samples to safetensors format.
///
/// Returns the serialized bytes. Use [`serialize_to_file`] to write directly
/// to disk.
pub fn serialize_graph(
    graph: &GraphData,
    split: &EdgeSplit,
    neg: &NegativeSamples,
    config: &SplitConfig,
) -> Result<Vec<u8>, TensorError> {
    // Validate split indices are within edge bounds
    let num_edges = graph.num_edges as u32;
    for &idx in split.train_idx.iter().chain(&split.val_idx).chain(&split.test_idx) {
        if idx >= num_edges {
            return Err(TensorError::ValidationFailed {
                message: format!("split index {idx} >= num_edges {}", graph.num_edges),
            });
        }
    }

    // Validate negative sample consistency and node bounds
    if neg.src.len() != neg.dst.len() {
        return Err(TensorError::ValidationFailed {
            message: format!(
                "neg_src length ({}) != neg_dst length ({})",
                neg.src.len(),
                neg.dst.len()
            ),
        });
    }
    let num_nodes = graph.num_nodes as u32;
    for (&s, &d) in neg.src.iter().zip(&neg.dst) {
        if s >= num_nodes || d >= num_nodes {
            return Err(TensorError::ValidationFailed {
                message: format!(
                    "negative sample ({s}, {d}) out of bounds for {num_nodes} nodes"
                ),
            });
        }
    }

    // Convert has_embedding Vec<bool> → Vec<u8> (safetensors BOOL is 1 byte)
    let has_emb_u8: Vec<u8> = graph.has_embedding.iter().map(|&b| b as u8).collect();

    let tensors: Vec<(&str, TensorView<'_>)> = vec![
        (
            "node_features",
            TensorView::new(
                Dtype::F32,
                vec![graph.num_nodes, graph.feature_dim],
                as_bytes(&graph.node_features),
            )?,
        ),
        (
            "has_embedding",
            TensorView::new(Dtype::BOOL, vec![graph.num_nodes], &has_emb_u8)?,
        ),
        (
            "node_collections",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_nodes],
                as_bytes(&graph.node_collections),
            )?,
        ),
        (
            "edge_src",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_edges],
                as_bytes(&graph.edge_src),
            )?,
        ),
        (
            "edge_dst",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_edges],
                as_bytes(&graph.edge_dst),
            )?,
        ),
        (
            "edge_type",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_edges],
                as_bytes(&graph.edge_type),
            )?,
        ),
        (
            "train_idx",
            TensorView::new(
                Dtype::U32,
                vec![split.train_idx.len()],
                as_bytes(&split.train_idx),
            )?,
        ),
        (
            "val_idx",
            TensorView::new(
                Dtype::U32,
                vec![split.val_idx.len()],
                as_bytes(&split.val_idx),
            )?,
        ),
        (
            "test_idx",
            TensorView::new(
                Dtype::U32,
                vec![split.test_idx.len()],
                as_bytes(&split.test_idx),
            )?,
        ),
        (
            "neg_src",
            TensorView::new(
                Dtype::U32,
                vec![neg.src.len()],
                as_bytes(&neg.src),
            )?,
        ),
        (
            "neg_dst",
            TensorView::new(
                Dtype::U32,
                vec![neg.dst.len()],
                as_bytes(&neg.dst),
            )?,
        ),
    ];

    // Metadata: scalars + collection_names as JSON in the header
    let mut metadata = HashMap::new();
    metadata.insert("num_nodes".into(), graph.num_nodes.to_string());
    metadata.insert("num_edges".into(), graph.num_edges.to_string());
    metadata.insert("num_relations".into(), graph.num_relations.to_string());
    metadata.insert("feature_dim".into(), graph.feature_dim.to_string());
    metadata.insert(
        "collection_names".into(),
        serde_json::to_string(&graph.collection_names)?,
    );
    metadata.insert("val_ratio".into(), config.val_ratio.to_string());
    metadata.insert("test_ratio".into(), config.test_ratio.to_string());
    metadata.insert(
        "neg_sampling_ratio".into(),
        config.neg_sampling_ratio.to_string(),
    );

    let bytes = safetensors::tensor::serialize(tensors, Some(metadata))?;
    Ok(bytes)
}

/// Serialize a graph to a safetensors file on disk.
///
/// Uses atomic write semantics: data is written to a temporary file in the
/// same directory, flushed, then renamed to the target path. This prevents
/// concurrent `MappedGraph::open()` from seeing a partial file.
pub fn serialize_to_file(
    path: &Path,
    graph: &GraphData,
    split: &EdgeSplit,
    neg: &NegativeSamples,
    config: &SplitConfig,
) -> Result<(), TensorError> {
    let bytes = serialize_graph(graph, split, neg, config)?;

    // Write to a temp file, then atomically rename
    let tmp_path = path.with_extension("safetensors.tmp");
    let mut file = fs::File::create(&tmp_path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp_path, path)?;

    info!(
        path = %path.display(),
        size_mb = bytes.len() as f64 / (1024.0 * 1024.0),
        "wrote safetensors file"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Memory-mapped reader
// ---------------------------------------------------------------------------

/// Memory-mapped safetensors file for zero-copy tensor access.
///
/// The file is mapped into memory and tensors are accessed as byte slices
/// without copying. The `MappedGraph` holds the mmap and provides typed
/// accessors for the training loop.
///
/// # Safety
///
/// The underlying file must not be modified or truncated while the map is
/// active. External modification can cause SIGBUS or undefined behavior.
/// [`serialize_to_file`] uses atomic write (write-to-tmp + rename) to
/// reduce risk, but concurrent non-atomic writers or file truncation can
/// still corrupt the mapping. For full safety, use exclusive file locking
/// or open a read-only snapshot.
pub struct MappedGraph {
    mmap: Mmap,
}

impl MappedGraph {
    /// Open a safetensors file via memory mapping.
    ///
    /// # Safety
    ///
    /// The caller must ensure the file is not concurrently modified.
    /// See [`MappedGraph`] for details.
    pub fn open(path: &Path) -> Result<Self, TensorError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    /// Get the raw safetensors data.
    fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Parse the safetensors header and access tensors.
    pub fn tensors(&self) -> Result<SafeTensors<'_>, TensorError> {
        Ok(SafeTensors::deserialize(self.data())?)
    }

    /// Read a U32 tensor as a `Vec<u32>`.
    pub fn read_u32_tensor(&self, name: &str) -> Result<Vec<u32>, TensorError> {
        let st = self.tensors()?;
        let view = st.tensor(name).map_err(|_| TensorError::MissingTensor {
            name: name.to_string(),
        })?;
        if view.dtype() != Dtype::U32 {
            return Err(TensorError::DtypeMismatch {
                name: name.to_string(),
                expected: Dtype::U32,
                actual: view.dtype(),
            });
        }
        let bytes = view.data();
        let result: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(result)
    }

    /// Read the F32 node features as a flat `Vec<f32>`.
    pub fn read_node_features(&self) -> Result<Vec<f32>, TensorError> {
        let st = self.tensors()?;
        let view = st
            .tensor("node_features")
            .map_err(|_| TensorError::MissingTensor {
                name: "node_features".into(),
            })?;
        if view.dtype() != Dtype::F32 {
            return Err(TensorError::DtypeMismatch {
                name: "node_features".into(),
                expected: Dtype::F32,
                actual: view.dtype(),
            });
        }
        let bytes = view.data();
        let result: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(result)
    }

    /// Read a metadata value from the safetensors header.
    pub fn metadata_value(&self, key: &str) -> Result<String, TensorError> {
        let (_, meta) = SafeTensors::read_metadata(self.data())?;
        meta.metadata()
            .as_ref()
            .and_then(|m: &HashMap<String, String>| m.get(key))
            .cloned()
            .ok_or_else(|| TensorError::MissingMetadata {
                key: key.to_string(),
            })
    }

    /// Read num_nodes from metadata.
    pub fn num_nodes(&self) -> Result<usize, TensorError> {
        self.metadata_value("num_nodes")?
            .parse()
            .map_err(|e| TensorError::MetadataParse {
                key: "num_nodes".into(),
                message: format!("{e}"),
            })
    }

    /// Read num_edges from metadata.
    pub fn num_edges(&self) -> Result<usize, TensorError> {
        self.metadata_value("num_edges")?
            .parse()
            .map_err(|e| TensorError::MetadataParse {
                key: "num_edges".into(),
                message: format!("{e}"),
            })
    }

    /// Read collection_names from metadata.
    pub fn collection_names(&self) -> Result<Vec<String>, TensorError> {
        let raw = self.metadata_value("collection_names")?;
        serde_json::from_str(&raw).map_err(|e| TensorError::MetadataParse {
            key: "collection_names".into(),
            message: format!("{e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience: full pipeline
// ---------------------------------------------------------------------------

/// Load graph, compute splits + negatives, and serialize to safetensors.
///
/// This is the high-level entry point combining all steps. The `IDMap` is
/// not serialized (it's only needed for embedding export, handled separately).
pub fn prepare_and_serialize(
    path: &Path,
    graph: &GraphData,
    config: &SplitConfig,
) -> Result<(), TensorError> {
    if !config.neg_sampling_ratio.is_finite() || config.neg_sampling_ratio < 0.0 {
        return Err(TensorError::InvalidNegSamplingRatio {
            neg: config.neg_sampling_ratio,
        });
    }

    let split = split_edges(graph.num_edges, config)?;

    // Negative samples: 1 per positive train edge by default
    let num_neg = (split.train_idx.len() as f64 * config.neg_sampling_ratio) as usize;
    let neg = negative_sample(graph, num_neg);

    serialize_to_file(path, graph, &split, &neg, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hades_core::graph::schema::{JINA_DIM, NUM_RELATIONS};

    /// Build a small test graph for unit tests.
    fn test_graph() -> GraphData {
        let num_nodes = 10;
        let mut graph = GraphData::with_capacity(num_nodes, 0);

        // Add some edges across different relation types
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

        // Set collection names and indices
        graph.collection_names = vec!["col_a".into(), "col_b".into()];
        for i in 0..num_nodes {
            graph.node_collections[i] = (i % 2) as u32;
        }

        // Set a few embeddings
        let emb = vec![1.0f32; JINA_DIM];
        graph.set_node_features(0, &emb);
        graph.set_node_features(3, &emb);
        graph.set_node_features(7, &emb);

        graph
    }

    #[test]
    fn test_split_edges_ratios() {
        let split = split_edges(100, &SplitConfig::default()).unwrap();
        assert_eq!(split.train_idx.len() + split.val_idx.len() + split.test_idx.len(), 100);
        assert_eq!(split.val_idx.len(), 10);
        assert_eq!(split.test_idx.len(), 10);
        assert_eq!(split.train_idx.len(), 80);
    }

    #[test]
    fn test_split_edges_no_overlap() {
        let split = split_edges(50, &SplitConfig::default()).unwrap();
        let mut all: Vec<u32> = Vec::new();
        all.extend(&split.train_idx);
        all.extend(&split.val_idx);
        all.extend(&split.test_idx);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 50, "splits must be disjoint and cover all edges");
    }

    #[test]
    fn test_split_edges_empty() {
        let err = split_edges(0, &SplitConfig::default()).unwrap_err();
        assert!(matches!(err, TensorError::EmptyGraph));
    }

    #[test]
    fn test_negative_sample_basic() {
        let graph = test_graph();
        let neg = negative_sample(&graph, 15);
        assert_eq!(neg.src.len(), neg.dst.len());
        assert_eq!(neg.src.len(), 15);

        // No self-loops
        for (&s, &d) in neg.src.iter().zip(&neg.dst) {
            assert_ne!(s, d, "negative sample should not have self-loops");
        }

        // No existing edges
        let existing: HashSet<(u32, u32)> = graph
            .edge_src
            .iter()
            .zip(&graph.edge_dst)
            .map(|(&s, &d)| (s, d))
            .collect();
        for (&s, &d) in neg.src.iter().zip(&neg.dst) {
            assert!(
                !existing.contains(&(s, d)),
                "negative sample ({s}, {d}) is an existing edge"
            );
        }
    }

    #[test]
    fn test_negative_sample_empty() {
        let graph = test_graph();
        let neg = negative_sample(&graph, 0);
        assert!(neg.src.is_empty());
        assert!(neg.dst.is_empty());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let graph = test_graph();
        let split = split_edges(graph.num_edges, &SplitConfig::default()).unwrap();
        let neg = negative_sample(&graph, 10);
        let config = SplitConfig::default();

        let bytes = serialize_graph(&graph, &split, &neg, &config).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize and verify
        let st = SafeTensors::deserialize(&bytes).unwrap();

        // Check tensor shapes
        let nf = st.tensor("node_features").unwrap();
        assert_eq!(nf.shape(), &[graph.num_nodes, graph.feature_dim]);

        let he = st.tensor("has_embedding").unwrap();
        assert_eq!(he.shape(), &[graph.num_nodes]);

        let es = st.tensor("edge_src").unwrap();
        assert_eq!(es.shape(), &[graph.num_edges]);

        let ti = st.tensor("train_idx").unwrap();
        let vi = st.tensor("val_idx").unwrap();
        let tsi = st.tensor("test_idx").unwrap();
        assert_eq!(
            ti.shape()[0] + vi.shape()[0] + tsi.shape()[0],
            graph.num_edges
        );

        // Check metadata
        let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
        let meta = meta.metadata().as_ref().unwrap();
        assert_eq!(meta["num_nodes"], graph.num_nodes.to_string());
        assert_eq!(meta["num_edges"], graph.num_edges.to_string());
        assert_eq!(meta["num_relations"], NUM_RELATIONS.to_string());
    }

    #[test]
    fn test_file_roundtrip() {
        let graph = test_graph();
        let config = SplitConfig::default();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        prepare_and_serialize(&path, &graph, &config).unwrap();
        assert!(path.exists());

        // Read back via MappedGraph
        let mapped = MappedGraph::open(&path).unwrap();
        assert_eq!(mapped.num_nodes().unwrap(), graph.num_nodes);
        assert_eq!(mapped.num_edges().unwrap(), graph.num_edges);

        let names = mapped.collection_names().unwrap();
        assert_eq!(names, graph.collection_names);

        let edge_src = mapped.read_u32_tensor("edge_src").unwrap();
        assert_eq!(edge_src, graph.edge_src);

        let features = mapped.read_node_features().unwrap();
        assert_eq!(features.len(), graph.num_nodes * graph.feature_dim);
        // Node 0 should have all 1.0
        assert_eq!(features[0], 1.0);
        // Node 1 should have all 0.0 (no embedding set)
        assert_eq!(features[JINA_DIM], 0.0);
    }

    #[test]
    fn test_split_config_invalid() {
        let bad = SplitConfig {
            val_ratio: 0.6,
            test_ratio: 0.6,
            neg_sampling_ratio: 1.0,
        };
        let err = split_edges(100, &bad).unwrap_err();
        assert!(matches!(err, TensorError::InvalidSplitConfig { .. }));

        let negative = SplitConfig {
            val_ratio: -0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: 1.0,
        };
        let err = split_edges(100, &negative).unwrap_err();
        assert!(matches!(err, TensorError::InvalidSplitConfig { .. }));
    }

    #[test]
    fn test_split_config_nan_inf() {
        // NaN val_ratio
        let nan_val = SplitConfig {
            val_ratio: f64::NAN,
            test_ratio: 0.1,
            neg_sampling_ratio: 1.0,
        };
        assert!(matches!(
            split_edges(100, &nan_val).unwrap_err(),
            TensorError::InvalidSplitConfig { .. }
        ));

        // Infinity test_ratio
        let inf_test = SplitConfig {
            val_ratio: 0.1,
            test_ratio: f64::INFINITY,
            neg_sampling_ratio: 1.0,
        };
        assert!(matches!(
            split_edges(100, &inf_test).unwrap_err(),
            TensorError::InvalidSplitConfig { .. }
        ));

        // NaN neg_sampling_ratio — caught in prepare_and_serialize
        let nan_neg = SplitConfig {
            val_ratio: 0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: f64::NAN,
        };
        let graph = test_graph();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan.safetensors");
        assert!(matches!(
            prepare_and_serialize(&path, &graph, &nan_neg).unwrap_err(),
            TensorError::InvalidNegSamplingRatio { .. }
        ));

        // Negative neg_sampling_ratio
        let neg_neg = SplitConfig {
            val_ratio: 0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: -1.0,
        };
        assert!(matches!(
            prepare_and_serialize(&path, &graph, &neg_neg).unwrap_err(),
            TensorError::InvalidNegSamplingRatio { .. }
        ));
    }

    #[test]
    fn test_split_config_custom() {
        let config = SplitConfig {
            val_ratio: 0.2,
            test_ratio: 0.2,
            neg_sampling_ratio: 2.0,
        };
        let split = split_edges(100, &config).unwrap();
        assert_eq!(split.val_idx.len(), 20);
        assert_eq!(split.test_idx.len(), 20);
        assert_eq!(split.train_idx.len(), 60);
    }
}
