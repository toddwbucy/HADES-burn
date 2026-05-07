//! Graph data types and runtime schema for RGCN training.
//!
//! The schema is loaded from each database's `hades_schema` collection at
//! runtime — there is no compile-time ontology baked into the binary.

pub mod export;
pub mod loader;
pub mod runtime_schema;
pub mod types;

pub use export::{ExportConfig, ExportError, ExportResult, decode_f32_embeddings, export_embeddings};
pub use loader::{GraphLoaderError, load};
pub use runtime_schema::{RuntimeEdgeDef, RuntimeNamedGraph, RuntimeSchema, SchemaMeta, SchemaError};
pub use types::{GraphData, GraphDataError, IDMap};
