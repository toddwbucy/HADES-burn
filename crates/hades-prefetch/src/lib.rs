//! Async graph-aware batch prefetcher for RGCN training.
//!
//! Provides tensor serialization (safetensors), edge splitting,
//! negative sampling, and memory-mapped graph access for the
//! training pipeline.

pub mod tensor;

pub use tensor::{
    EdgeSplit, MappedGraph, NegativeSamples, SplitConfig, TensorError, negative_sample,
    prepare_and_serialize, serialize_graph, serialize_to_file, split_edges,
};
