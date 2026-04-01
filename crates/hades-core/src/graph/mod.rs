//! Graph schema and data types for RGCN training.
//!
//! Ports the NL knowledge graph schema from Python HADES
//! (`core/database/nl_graph_schema.py`) and defines the tensor-like
//! data structures consumed by the graph loader and training pipeline.

pub mod schema;
pub mod types;

pub use schema::{
    EdgeCollectionDef, NamedGraphDef, NlGraphSchema, ALL_EDGE_COLLECTIONS,
    ALL_NAMED_GRAPHS, EDGE_COLLECTION_NAMES, NL_GRAPH_SCHEMA, NUM_RELATIONS,
};
pub use types::{GraphData, IDMap};
