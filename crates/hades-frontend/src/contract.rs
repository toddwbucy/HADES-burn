//! The graph-JSON contract.
//!
//! This is the single data shape the frontend produces and the WebGL viewer
//! consumes. Both the snapshot path (shelling out to `hades db export`) and a
//! future live path (querying the warm daemon) emit *this* structure, so the
//! viewer never needs to know which backend fed it.
//!
//! Slice-awareness is baked in from v1: `meta.partial` plus the `*_total`
//! counts let the viewer honestly say "showing 2,000 of 45,000" instead of
//! pretending it holds the whole graph. That is the one concession the "graphs
//! get large fast" requirement forces into the schema up front.

use serde::Serialize;
use std::collections::BTreeMap;

/// A complete graph payload: metadata + nodes + edges.
#[derive(Debug, Serialize)]
pub struct GraphSnapshot {
    pub meta: SnapshotMeta,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Serialize)]
pub struct SnapshotMeta {
    /// Source database the snapshot was assembled from.
    pub database: String,
    /// Named graph the topology was drawn from.
    pub named_graph: String,
    /// How this payload was produced: "snapshot" (CLI export) or "live" (daemon).
    pub source: &'static str,
    /// True when the payload is a bounded slice (a per-collection limit was hit),
    /// so the viewer must not treat node/edge counts as the whole graph.
    pub partial: bool,
    /// Vertex collections present, in discovery order.
    pub node_collections: Vec<String>,
    /// Edge collections present, in discovery order.
    pub edge_collections: Vec<String>,
    /// Union of document attribute keys per collection, discovered by sampling.
    /// This is what powers the viewer's "color by / size by / dash-if" dropdowns
    /// with zero domain assumptions.
    pub attribute_index: BTreeMap<String, Vec<String>>,
    /// Nodes/edges actually included in this payload.
    pub node_count: usize,
    pub edge_count: usize,
    /// Total nodes/edges in the source collections (before any slice limit).
    pub node_total: usize,
    pub edge_total: usize,
}

/// A vertex. `attrs` carries every non-internal document field verbatim, so the
/// inspection panel and attribute-driven styling have the full record.
#[derive(Debug, Serialize)]
pub struct Node {
    /// ArangoDB `_id` ("collection/key") — the stable graph key.
    pub id: String,
    pub collection: String,
    /// Human-friendly label derived from a conventional field (see `label_for`).
    pub label: String,
    /// Undirected degree within this payload — a cheap size-by-importance default.
    pub degree: u32,
    pub attrs: serde_json::Map<String, serde_json::Value>,
}

/// An edge. `source`/`target` are ArangoDB `_from`/`_to` (`_id` refs), named to
/// match what graphology/sigma expect on import.
#[derive(Debug, Serialize)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    /// The edge collection name — in HADES the collection name *is* the relation
    /// type (ontology decision D6), so this is the primary edge-styling key.
    pub collection: String,
    pub attrs: serde_json::Map<String, serde_json::Value>,
}

/// One chunk of a file's source text, ordered by `chunk_index`.
#[derive(Debug, Serialize)]
pub struct ChunkText {
    pub chunk_index: u32,
    pub text: String,
}

/// A partial payload returned by neighborhood expansion, for the viewer to
/// merge into the graph it already holds. Same `Node`/`Edge` shapes as a full
/// snapshot, minus the meta — the client dedupes and recomputes degree on merge.
#[derive(Debug, Serialize)]
pub struct GraphDelta {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// Internal ArangoDB fields we strip from `attrs` (kept only where they carry
/// graph meaning, e.g. `_id`/`_from`/`_to` are promoted to typed fields above).
pub const INTERNAL_FIELDS: &[&str] = &["_id", "_key", "_rev", "_from", "_to"];

/// Derive a display label from a document, trying conventional name-bearing
/// fields before falling back to the key. Domain-agnostic: it prefers whatever
/// human-readable field the document happens to carry.
pub fn label_for(doc: &serde_json::Map<String, serde_json::Value>, key: &str) -> String {
    for field in ["name", "qualified_name", "title", "path", "label"] {
        if let Some(serde_json::Value::String(s)) = doc.get(field)
            && !s.is_empty()
        {
            return s.clone();
        }
    }
    key.to_string()
}
