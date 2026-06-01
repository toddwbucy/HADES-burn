//! HADES gRPC/protobuf definitions.
//!
//! Generated from `.proto` files under the `proto/` directory. Provides both
//! client stubs and server traits. The `persephone.*` packages (embedding,
//! extraction) are HADES-owned compute services that carry the legacy
//! provider-protocol brand; `hades.training` was decoupled from it (issue
//! #106) — none of these are the Persephone PM system.

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

/// Training service — RGCN link prediction on knowledge graphs (HADES-owned;
/// decoupled from the `persephone.*` provider brand — issue #106).
pub mod training {
    tonic::include_proto!("hades.training");
}
