//! Shared LSP infrastructure and dedicated semantic language servers.

pub mod client;
pub mod edges;
pub mod go_symbols;
pub mod gopls;
pub mod rust_analyzer;
pub mod rust_symbols;
pub mod session;
pub mod symbols;

pub use edges::{CrateEdge, EdgeKind, LspEdgeResolver};
pub use gopls::{Gopls, GoplsSession, find_go_module_root, group_files_by_go_module};
pub use rust_analyzer::{RustAnalyzer, RustAnalyzerSession, find_crate_root, group_files_by_crate};
pub use rust_symbols::RustSymbolExtractor;

use thiserror::Error;

/// Errors shared by all language-server integrations.
#[derive(Debug, Error)]
pub enum LspError {
    #[error("language server not found: {0}")]
    NotFound(String),
    #[error("LSP process error: {0}")]
    Process(String),
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc { code: i64, message: String },
    #[error("LSP request timed out: {0}")]
    Timeout(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid language workspace: {0}")]
    InvalidWorkspace(String),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

// Compatibility name while downstream callers migrate to the generic module.
pub type RustAnalyzerError = LspError;
