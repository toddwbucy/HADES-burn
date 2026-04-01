//! rust-analyzer LSP integration for cross-file Rust analysis.
//!
//! Spawns a rust-analyzer subprocess, communicates over JSON-RPC, and
//! extracts rich symbol data and cross-file edges (calls, implements,
//! defines) that single-file `syn` parsing cannot provide.
//!
//! # Architecture
//!
//! - [`client::LspClient`] — JSON-RPC transport over stdin/stdout
//! - [`session::RustAnalyzerSession`] — LSP lifecycle, initialization, readiness
//! - [`symbols::RustSymbolExtractor`] — symbol extraction from documentSymbol + hover
//! - [`edges::RustEdgeResolver`] — edge materialization (defines, calls, implements, pyo3, ffi)

pub mod client;
pub mod edges;
pub mod session;
pub mod symbols;

pub use edges::{CrateEdge, EdgeKind, RustEdgeResolver};
pub use session::{RustAnalyzerSession, find_crate_root, group_files_by_crate};
pub use symbols::RustSymbolExtractor;

use thiserror::Error;

/// Errors from rust-analyzer integration.
#[derive(Debug, Error)]
pub enum RustAnalyzerError {
    /// rust-analyzer binary not found.
    #[error("rust-analyzer not found: {0}")]
    NotFound(String),

    /// The LSP server process died or failed to start.
    #[error("LSP process error: {0}")]
    Process(String),

    /// JSON-RPC protocol error.
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc {
        code: i64,
        message: String,
    },

    /// Request timed out.
    #[error("LSP request timed out: {0}")]
    Timeout(String),

    /// I/O error communicating with the LSP server.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid Cargo.toml or crate root.
    #[error("invalid crate root: {0}")]
    InvalidCrate(String),

    /// Serde error parsing LSP responses.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}
