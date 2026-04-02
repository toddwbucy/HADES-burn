//! Async graph-aware batch prefetcher for RGCN training.
//!
//! Provides tensor serialization (safetensors), edge splitting,
//! negative sampling, memory-mapped graph access, async double-buffered
//! prefetching, and a training orchestrator for the training pipeline.

pub mod orchestrator;
pub mod prefetcher;
pub mod tensor;

pub use orchestrator::{
    EpochMetrics, Orchestrator, OrchestratorError, TrainConfig, TrainResult,
};
pub use prefetcher::{
    EpochBatch, PrefetchConfig, PrefetchError, Prefetcher, TrainingData,
    prepare_training_data,
};
pub use tensor::{
    EdgeSplit, MappedGraph, NegativeSamples, SplitConfig, TensorError, negative_sample,
    prepare_and_serialize, serialize_graph, serialize_to_file, split_edges,
};
