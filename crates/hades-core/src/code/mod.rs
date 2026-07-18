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

use std::path::PathBuf;

mod chunking;
mod cpp;
pub mod cpp_edges;
mod language;
mod python;
pub mod python_calls;
pub mod rust_analyzer;
mod rust_ast;
pub mod rust_imports;
mod symbols;

pub use chunking::AstChunking;
pub use language::Language;
pub use symbols::{CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef};

/// Optional project context used by analyzers that need build-system data.
#[derive(Debug, Clone, Default)]
pub struct AnalysisOptions {
    /// Path to `compile_commands.json` or the directory containing it.
    /// C/C++/CUDA analysis uses the matching command's include paths,
    /// defines, language standard, and CUDA target flags.
    pub compilation_database: Option<PathBuf>,
}

/// Analyze a source file, extracting symbols, metrics, and structure.
///
/// Detects the language from the file extension, then dispatches to
/// the appropriate parser.  Returns an error if the language is not
/// supported or parsing fails.
pub fn analyze(source: &str, file_path: &str) -> Result<FileAnalysis, CodeAnalysisError> {
    let lang = Language::from_path(file_path)
        .ok_or_else(|| CodeAnalysisError::UnsupportedLanguage(file_path.to_string()))?;
    analyze_with_language(source, lang, file_path)
}

/// Analyze source code with an explicitly specified language.
///
/// `file_path` is used by analyzers that need it (the C++ path passes it to
/// libclang for CUDA detection and include resolution); the Python and Rust
/// paths ignore it.
pub fn analyze_with_language(
    source: &str,
    lang: Language,
    file_path: &str,
) -> Result<FileAnalysis, CodeAnalysisError> {
    analyze_with_options(source, lang, file_path, &AnalysisOptions::default())
}

/// Analyze source with per-project build context.
pub fn analyze_with_options(
    source: &str,
    lang: Language,
    file_path: &str,
    options: &AnalysisOptions,
) -> Result<FileAnalysis, CodeAnalysisError> {
    match lang {
        Language::Python => python::analyze(source),
        Language::Rust => rust_ast::analyze(source),
        Language::Cpp => cpp::analyze(source, file_path, options.compilation_database.as_deref()),
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
