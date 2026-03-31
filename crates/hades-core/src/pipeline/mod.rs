//! Document processing pipeline — extract → chunk → embed → store.
//!
//! Orchestrates the Persephone extraction and embedding clients with the
//! chunking engine and ArangoDB storage.  Supports single-document and
//! batch processing with two-phase GPU memory optimization.

mod orchestrator;

pub use orchestrator::{
    DocumentResult, Pipeline, PipelineConfig, PipelineError, PipelineSummary,
};
