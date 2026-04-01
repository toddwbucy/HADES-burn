//! Graph data types for the training pipeline.
//!
//! Defines the Rust-side representation of graph tensors loaded from
//! ArangoDB. These are plain `Vec`-based structures; conversion to
//! framework-specific tensors (Burn, PyTorch via IPC) happens in the
//! serialization layer (P4.3).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::schema::{JINA_DIM, NUM_RELATIONS};

// ---------------------------------------------------------------------------
// IDMap — bidirectional ArangoDB _id ↔ integer index
// ---------------------------------------------------------------------------

/// Bidirectional mapping between ArangoDB `_id` and contiguous integer indices.
///
/// Integer indices are used as row indices into feature matrices and as
/// node indices in edge tensors. The mapping is built incrementally as
/// nodes are discovered during edge loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IDMap {
    /// Forward map: ArangoDB `_id` (e.g. `"hope_axioms/container-is"`) → integer index.
    arango_to_idx: HashMap<String, usize>,
    /// Reverse map: integer index → ArangoDB `_id`. Contiguous from 0.
    idx_to_arango: Vec<String>,
}

impl IDMap {
    /// Create a new empty map.
    pub fn new() -> Self {
        Self {
            arango_to_idx: HashMap::new(),
            idx_to_arango: Vec::new(),
        }
    }

    /// Create a map with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arango_to_idx: HashMap::with_capacity(capacity),
            idx_to_arango: Vec::with_capacity(capacity),
        }
    }

    /// Get or create an integer index for an ArangoDB `_id`.
    ///
    /// If the `_id` has been seen before, returns the existing index.
    /// Otherwise assigns the next contiguous index.
    pub fn get_or_create(&mut self, arango_id: &str) -> usize {
        if let Some(&idx) = self.arango_to_idx.get(arango_id) {
            return idx;
        }
        let idx = self.idx_to_arango.len();
        self.arango_to_idx.insert(arango_id.to_string(), idx);
        self.idx_to_arango.push(arango_id.to_string());
        idx
    }

    /// Look up the integer index for an ArangoDB `_id`.
    pub fn get_index(&self, arango_id: &str) -> Option<usize> {
        self.arango_to_idx.get(arango_id).copied()
    }

    /// Look up the ArangoDB `_id` for an integer index.
    pub fn get_arango_id(&self, idx: usize) -> Option<&str> {
        self.idx_to_arango.get(idx).map(|s| s.as_str())
    }

    /// Extract the collection name from an ArangoDB `_id`.
    ///
    /// E.g. `"hope_axioms/container-is"` → `"hope_axioms"`.
    pub fn collection_of(&self, idx: usize) -> Option<&str> {
        self.idx_to_arango
            .get(idx)
            .and_then(|id| id.split('/').next())
    }

    /// Number of mapped nodes.
    pub fn len(&self) -> usize {
        self.idx_to_arango.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.idx_to_arango.is_empty()
    }

    /// Iterate over all `(index, arango_id)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &str)> {
        self.idx_to_arango
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.as_str()))
    }

    /// Group node indices by their collection name.
    ///
    /// Returns a map from collection name → list of `(arango_id, index)`.
    pub fn nodes_by_collection(&self) -> HashMap<&str, Vec<(&str, usize)>> {
        let mut groups: HashMap<&str, Vec<(&str, usize)>> = HashMap::new();
        for (idx, arango_id) in self.idx_to_arango.iter().enumerate() {
            if let Some(col) = arango_id.split('/').next() {
                groups.entry(col).or_default().push((arango_id, idx));
            }
        }
        groups
    }
}

impl Default for IDMap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GraphData — the loaded graph in tensor-like form
// ---------------------------------------------------------------------------

/// Graph data loaded from ArangoDB, ready for training.
///
/// Mirrors the Python `GraphLoader.load()` return value.
/// All "tensors" are flat `Vec` — conversion to framework-specific
/// tensor types is deferred to the serialization layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    /// Node feature matrix, flattened row-major: `[num_nodes × feature_dim]`.
    ///
    /// Nodes without Jina embeddings have zero vectors.
    pub node_features: Vec<f32>,

    /// Boolean mask: `true` if the node has a real Jina embedding.
    /// Length: `num_nodes`.
    pub has_embedding: Vec<bool>,

    /// Collection type index per node. Length: `num_nodes`.
    /// Maps to `collection_names[node_collections[i]]`.
    pub node_collections: Vec<u32>,

    /// Edge source indices. Length: `num_edges`.
    pub edge_src: Vec<u32>,

    /// Edge target indices. Length: `num_edges`.
    pub edge_dst: Vec<u32>,

    /// Relation type index per edge. Length: `num_edges`.
    /// Maps to `EDGE_COLLECTION_NAMES[edge_type[i]]`.
    pub edge_type: Vec<u32>,

    /// Number of nodes.
    pub num_nodes: usize,

    /// Number of edges.
    pub num_edges: usize,

    /// Number of relation types.
    pub num_relations: usize,

    /// Feature dimensionality per node.
    pub feature_dim: usize,

    /// Sorted collection names — `node_collections` values index into this.
    pub collection_names: Vec<String>,
}

impl GraphData {
    /// Create an empty graph with the standard Jina feature dimension.
    pub fn empty() -> Self {
        Self {
            node_features: Vec::new(),
            has_embedding: Vec::new(),
            node_collections: Vec::new(),
            edge_src: Vec::new(),
            edge_dst: Vec::new(),
            edge_type: Vec::new(),
            num_nodes: 0,
            num_edges: 0,
            num_relations: NUM_RELATIONS,
            feature_dim: JINA_DIM,
            collection_names: Vec::new(),
        }
    }

    /// Allocate a graph with known node and edge counts.
    ///
    /// Node features are initialized to zero (no embedding).
    pub fn with_capacity(num_nodes: usize, num_edges: usize) -> Self {
        Self {
            node_features: vec![0.0; num_nodes * JINA_DIM],
            has_embedding: vec![false; num_nodes],
            node_collections: vec![0; num_nodes],
            edge_src: Vec::with_capacity(num_edges),
            edge_dst: Vec::with_capacity(num_edges),
            edge_type: Vec::with_capacity(num_edges),
            num_nodes,
            num_edges: 0,
            num_relations: NUM_RELATIONS,
            feature_dim: JINA_DIM,
            collection_names: Vec::new(),
        }
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, src: u32, dst: u32, rel_type: u32) {
        self.edge_src.push(src);
        self.edge_dst.push(dst);
        self.edge_type.push(rel_type);
        self.num_edges += 1;
    }

    /// Set the node feature vector for a given node index.
    ///
    /// `embedding` must have exactly `feature_dim` elements.
    /// Returns `false` if the index is out of bounds or dimension mismatches.
    pub fn set_node_features(&mut self, node_idx: usize, embedding: &[f32]) -> bool {
        if node_idx >= self.num_nodes || embedding.len() != self.feature_dim {
            return false;
        }
        let start = node_idx * self.feature_dim;
        self.node_features[start..start + self.feature_dim].copy_from_slice(embedding);
        self.has_embedding[node_idx] = true;
        true
    }

    /// Get the node feature vector for a given node index.
    pub fn get_node_features(&self, node_idx: usize) -> Option<&[f32]> {
        if node_idx >= self.num_nodes {
            return None;
        }
        let start = node_idx * self.feature_dim;
        Some(&self.node_features[start..start + self.feature_dim])
    }

    /// Count of nodes that have real embeddings.
    pub fn embedded_count(&self) -> usize {
        self.has_embedding.iter().filter(|&&b| b).count()
    }

    /// Fraction of nodes with embeddings.
    pub fn embedding_coverage(&self) -> f64 {
        if self.num_nodes == 0 {
            return 0.0;
        }
        self.embedded_count() as f64 / self.num_nodes as f64
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.node_features.len() != self.num_nodes * self.feature_dim {
            return Err(format!(
                "node_features length {} != num_nodes {} × feature_dim {}",
                self.node_features.len(),
                self.num_nodes,
                self.feature_dim
            ));
        }
        if self.has_embedding.len() != self.num_nodes {
            return Err(format!(
                "has_embedding length {} != num_nodes {}",
                self.has_embedding.len(),
                self.num_nodes
            ));
        }
        if self.node_collections.len() != self.num_nodes {
            return Err(format!(
                "node_collections length {} != num_nodes {}",
                self.node_collections.len(),
                self.num_nodes
            ));
        }
        if self.edge_src.len() != self.num_edges {
            return Err(format!(
                "edge_src length {} != num_edges {}",
                self.edge_src.len(),
                self.num_edges
            ));
        }
        if self.edge_dst.len() != self.num_edges {
            return Err(format!(
                "edge_dst length {} != num_edges {}",
                self.edge_dst.len(),
                self.num_edges
            ));
        }
        if self.edge_type.len() != self.num_edges {
            return Err(format!(
                "edge_type length {} != num_edges {}",
                self.edge_type.len(),
                self.num_edges
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idmap_get_or_create() {
        let mut map = IDMap::new();
        let idx0 = map.get_or_create("hope_axioms/container-is");
        let idx1 = map.get_or_create("atlas_definitions/def-assoc");
        let idx0_again = map.get_or_create("hope_axioms/container-is");

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx0_again, 0); // idempotent
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_idmap_reverse_lookup() {
        let mut map = IDMap::new();
        map.get_or_create("hope_axioms/container-is");
        map.get_or_create("atlas_definitions/def-assoc");

        assert_eq!(map.get_arango_id(0), Some("hope_axioms/container-is"));
        assert_eq!(map.get_arango_id(1), Some("atlas_definitions/def-assoc"));
        assert_eq!(map.get_arango_id(99), None);
    }

    #[test]
    fn test_idmap_collection_of() {
        let mut map = IDMap::new();
        map.get_or_create("hope_axioms/container-is");
        map.get_or_create("nl_reframings/rf-001");

        assert_eq!(map.collection_of(0), Some("hope_axioms"));
        assert_eq!(map.collection_of(1), Some("nl_reframings"));
        assert_eq!(map.collection_of(99), None);
    }

    #[test]
    fn test_idmap_nodes_by_collection() {
        let mut map = IDMap::new();
        map.get_or_create("hope_axioms/a");
        map.get_or_create("hope_axioms/b");
        map.get_or_create("atlas_definitions/x");

        let groups = map.nodes_by_collection();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["hope_axioms"].len(), 2);
        assert_eq!(groups["atlas_definitions"].len(), 1);
    }

    #[test]
    fn test_idmap_index_lookup() {
        let mut map = IDMap::new();
        map.get_or_create("hope_axioms/a");

        assert_eq!(map.get_index("hope_axioms/a"), Some(0));
        assert_eq!(map.get_index("nonexistent"), None);
    }

    #[test]
    fn test_idmap_iter() {
        let mut map = IDMap::new();
        map.get_or_create("a/1");
        map.get_or_create("b/2");

        let pairs: Vec<_> = map.iter().collect();
        assert_eq!(pairs, vec![(0, "a/1"), (1, "b/2")]);
    }

    #[test]
    fn test_graph_data_empty() {
        let g = GraphData::empty();
        assert_eq!(g.num_nodes, 0);
        assert_eq!(g.num_edges, 0);
        assert_eq!(g.num_relations, NUM_RELATIONS);
        assert_eq!(g.feature_dim, JINA_DIM);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_data_with_capacity() {
        let g = GraphData::with_capacity(100, 500);
        assert_eq!(g.num_nodes, 100);
        assert_eq!(g.num_edges, 0);
        assert_eq!(g.node_features.len(), 100 * JINA_DIM);
        assert_eq!(g.has_embedding.len(), 100);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_data_add_edge() {
        let mut g = GraphData::with_capacity(10, 0);
        g.add_edge(0, 1, 0);
        g.add_edge(1, 2, 3);

        assert_eq!(g.num_edges, 2);
        assert_eq!(g.edge_src, vec![0, 1]);
        assert_eq!(g.edge_dst, vec![1, 2]);
        assert_eq!(g.edge_type, vec![0, 3]);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_data_set_features() {
        let mut g = GraphData::with_capacity(3, 0);

        // All start as zero / no embedding
        assert_eq!(g.embedded_count(), 0);
        assert!(!g.has_embedding[0]);

        // Set features for node 1
        let emb = vec![1.0f32; JINA_DIM];
        assert!(g.set_node_features(1, &emb));
        assert!(g.has_embedding[1]);
        assert_eq!(g.embedded_count(), 1);

        // Verify the feature vector
        let features = g.get_node_features(1).unwrap();
        assert_eq!(features.len(), JINA_DIM);
        assert_eq!(features[0], 1.0);

        // Node 0 still zero
        let f0 = g.get_node_features(0).unwrap();
        assert_eq!(f0[0], 0.0);
    }

    #[test]
    fn test_graph_data_set_features_bounds() {
        let mut g = GraphData::with_capacity(2, 0);

        // Out of bounds
        assert!(!g.set_node_features(99, &[1.0; JINA_DIM]));

        // Wrong dimension
        assert!(!g.set_node_features(0, &[1.0; 128]));
    }

    #[test]
    fn test_graph_data_embedding_coverage() {
        let mut g = GraphData::with_capacity(4, 0);
        assert_eq!(g.embedding_coverage(), 0.0);

        let emb = vec![1.0f32; JINA_DIM];
        g.set_node_features(0, &emb);
        g.set_node_features(1, &emb);
        assert!((g.embedding_coverage() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_graph_data_validate_catches_mismatch() {
        let mut g = GraphData::with_capacity(2, 0);
        g.has_embedding.push(false); // extra element
        assert!(g.validate().is_err());
    }

    #[test]
    fn test_graph_data_empty_coverage() {
        let g = GraphData::empty();
        assert_eq!(g.embedding_coverage(), 0.0);
    }

    #[test]
    fn test_idmap_serialization_roundtrip() {
        let mut map = IDMap::new();
        map.get_or_create("hope_axioms/a");
        map.get_or_create("atlas_definitions/b");

        let json = serde_json::to_string(&map).unwrap();
        let restored: IDMap = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get_index("hope_axioms/a"), Some(0));
        assert_eq!(restored.get_arango_id(1), Some("atlas_definitions/b"));
    }
}
