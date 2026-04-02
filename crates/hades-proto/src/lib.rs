//! Persephone gRPC/protobuf definitions.
//!
//! Generated from `.proto` files in the `proto/persephone/` directory.
//! Provides both client stubs and server traits for the embedding and
//! extraction services.

/// Common types shared across Persephone services.
pub mod common {
    tonic::include_proto!("persephone.common");
}

/// Embedding service — vector embedding generation.
pub mod embedding {
    tonic::include_proto!("persephone.embedding");
}

/// Extraction service — document content extraction.
pub mod extraction {
    tonic::include_proto!("persephone.extraction");
}

/// Training service — RGCN link prediction on knowledge graphs.
pub mod training {
    tonic::include_proto!("persephone.training");
}
