//! Persephone service clients.
//!
//! Typed gRPC clients for the Persephone embedding, extraction, and
//! training services.  Connects over Unix domain sockets or TCP.

pub mod embedder_http;
pub mod embedding;
pub mod extraction;
pub mod training;
