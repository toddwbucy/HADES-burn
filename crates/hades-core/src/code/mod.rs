//! Multi-language code analysis engine.
//!
//! Provides AST-level parsing, symbol extraction, code metrics, and
//! structure-aware chunking for Python and Rust source files.
//!
//! - **Python**: [`rustpython-parser`] for full AST analysis
//! - **Rust**: [`syn`] for single-file AST analysis
//!
//! The architecture is extensible — tree-sitter or rust-analyzer
//! integration can be added alongside these parsers without changing
//! the public API.

mod language;
mod symbols;
mod python;
mod rust_ast;
mod chunking;

pub use language::Language;
pub use symbols::{
    CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef,
};
pub use chunking::AstChunking;

/// Analyze a source file, extracting symbols, metrics, and structure.
///
/// Detects the language from the file extension, then dispatches to
/// the appropriate parser.  Returns `None` if the language is not
/// supported.
pub fn analyze(source: &str, file_path: &str) -> Option<FileAnalysis> {
    let lang = Language::from_path(file_path)?;
    Some(analyze_with_language(source, lang))
}

/// Analyze source code with an explicitly specified language.
pub fn analyze_with_language(source: &str, lang: Language) -> FileAnalysis {
    match lang {
        Language::Python => python::analyze(source),
        Language::Rust => rust_ast::analyze(source),
    }
}

/// Typed error for code analysis operations.
#[derive(Debug, thiserror::Error)]
pub enum CodeAnalysisError {
    /// The file's language is not supported for analysis.
    #[error("unsupported language for file: {0}")]
    UnsupportedLanguage(String),

    /// The parser failed to produce a valid AST.
    #[error("parse error: {0}")]
    ParseError(String),

    /// I/O error reading source files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
