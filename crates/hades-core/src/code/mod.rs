//! Multi-language code analysis engine.
//!
//! Provides per-file analyzer dispatch, symbol extraction, code metrics, and
//! structure-aware chunking for multilingual source trees.
//!
//! Dedicated analyzers are preferred. Registered Tree-sitter grammars provide
//! an explicitly structural fallback, with raw text handled by the ingest
//! layer only after both higher tiers fail.

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
pub mod tree_sitter;
pub mod tree_sitter_edges;

pub use chunking::AstChunking;
pub use language::Language;
pub use symbols::{
    AnalysisTier, CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef, compute_content_hash,
};

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
    match analyze_with_fallback(source, lang, file_path, options) {
        AnalyzerOutcome::Success(analysis) => Ok(analysis),
        AnalyzerOutcome::Failed { analyzer, reason } => Err(CodeAnalysisError::ParseError(
            format!("{analyzer}: {reason}"),
        )),
    }
}

/// Typed result of per-file analyzer selection.
#[derive(Debug)]
pub enum AnalyzerOutcome {
    Success(FileAnalysis),
    Failed {
        analyzer: &'static str,
        reason: String,
    },
}

/// Select the highest-fidelity available analyzer for one file.
pub fn analyze_with_fallback(
    source: &str,
    lang: Language,
    file_path: &str,
    options: &AnalysisOptions,
) -> AnalyzerOutcome {
    let semantic = analyze_semantic(source, lang, file_path, options);
    match semantic {
        Ok(mut analysis) => {
            annotate(
                &mut analysis,
                AnalysisTier::Semantic,
                semantic_analyzer(lang),
                None,
            );
            AnalyzerOutcome::Success(analysis)
        }
        Err(semantic_failure) => {
            let fallback_reason = semantic_failure.to_string();
            match tree_sitter::analyze(source, lang) {
                Ok(mut analysis) => {
                    annotate(
                        &mut analysis,
                        AnalysisTier::Structural,
                        "tree-sitter",
                        Some(fallback_reason),
                    );
                    AnalyzerOutcome::Success(analysis)
                }
                Err(reason) => AnalyzerOutcome::Failed {
                    analyzer: "tree-sitter",
                    reason: format!("semantic: {fallback_reason}; structural: {reason}"),
                },
            }
        }
    }
}

fn analyze_semantic(
    source: &str,
    lang: Language,
    file_path: &str,
    options: &AnalysisOptions,
) -> Result<FileAnalysis, CodeAnalysisError> {
    match lang {
        Language::Python => python::analyze(source),
        Language::Rust => rust_ast::analyze(source),
        Language::Cpp => cpp::analyze(source, file_path, options.compilation_database.as_deref()),
        Language::Go => Err(CodeAnalysisError::AnalyzerUnavailable(
            "gopls integration is tracked by #99".to_string(),
        )),
    }
}

fn semantic_analyzer(lang: Language) -> &'static str {
    match lang {
        Language::Python => "rustpython-parser",
        Language::Rust => "syn",
        Language::Cpp => "libclang",
        Language::Go => "gopls",
    }
}

fn annotate(
    analysis: &mut FileAnalysis,
    tier: AnalysisTier,
    analyzer: &str,
    fallback_reason: Option<String>,
) {
    analysis.analysis_tier = tier;
    analysis.analyzer = analyzer.to_string();
    analysis.fallback_reason = fallback_reason;
    for symbol in &mut analysis.symbols {
        if !symbol.metadata.is_object() {
            symbol.metadata = serde_json::json!({});
        }
        symbol.metadata["analysis_tier"] = serde_json::json!(tier.as_str());
        symbol.metadata["analyzer"] = serde_json::json!(analyzer);
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

    /// A dedicated analyzer is not installed or not yet registered.
    #[error("analyzer unavailable: {0}")]
    AnalyzerUnavailable(String),

    /// I/O error reading source files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_language_dispatch_is_per_file() {
        let rust = analyze_with_fallback(
            "fn main() {}",
            Language::Rust,
            "main.rs",
            &AnalysisOptions::default(),
        );
        let go = analyze_with_fallback(
            "package main\nfunc main() {}",
            Language::Go,
            "main.go",
            &AnalysisOptions::default(),
        );
        assert!(matches!(
            rust,
            AnalyzerOutcome::Success(FileAnalysis {
                analysis_tier: AnalysisTier::Semantic,
                ..
            })
        ));
        assert!(matches!(
            go,
            AnalyzerOutcome::Success(FileAnalysis {
                analysis_tier: AnalysisTier::Structural,
                ..
            })
        ));
    }

    #[test]
    fn empty_file_is_success_but_malformed_file_is_failed() {
        let empty =
            analyze_with_fallback("", Language::Go, "empty.go", &AnalysisOptions::default());
        assert!(matches!(empty, AnalyzerOutcome::Success(_)));

        let malformed = analyze_with_fallback(
            "func (",
            Language::Go,
            "broken.go",
            &AnalysisOptions::default(),
        );
        assert!(matches!(malformed, AnalyzerOutcome::Failed { .. }));
    }
}
