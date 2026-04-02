//! Graph schema and data types for RGCN training.
//!
//! Ports the NL knowledge graph schema from Python HADES
//! (`core/database/nl_graph_schema.py`) and defines the tensor-like
//! data structures consumed by the graph loader and training pipeline.

pub mod export;
pub mod loader;
pub mod schema;
pub mod types;

pub use export::{ExportConfig, ExportError, ExportResult, decode_f32_embeddings, export_embeddings};
pub use loader::{GraphLoaderError, load};
pub use schema::{
    EdgeCollectionDef, NamedGraphDef, NlGraphSchema, ALL_EDGE_COLLECTIONS,
    ALL_NAMED_GRAPHS, EDGE_COLLECTION_NAMES, JINA_DIM, NL_GRAPH_SCHEMA, NUM_RELATIONS,
    relation_index,
};
pub use types::{GraphData, GraphDataError, IDMap};
