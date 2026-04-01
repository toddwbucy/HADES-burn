//! Core types for code analysis results.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::Language;

/// A symbol extracted from source code.
#[derive(Debug, Clone, Serialize)]
pub struct Symbol {
    /// Symbol name (e.g. `"process_batch"`, `"Config"`).
    pub name: String,
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// Start line (1-based).
    pub start_line: usize,
    /// End line (1-based, inclusive).
    pub end_line: usize,
    /// Language-specific metadata.
    ///
    /// Python: decorators, docstring, bases, is_async, parameters.
    /// Rust: visibility, is_async, is_unsafe, generics, derives, impl_trait.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

/// Classification of extracted symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Class,
    Struct,
    Enum,
    Trait,
    Import,
    Constant,
    Variable,
    TypeAlias,
    Macro,
    Impl,
    Module,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Function => "function",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Import => "import",
            Self::Constant => "constant",
            Self::Variable => "variable",
            Self::TypeAlias => "type_alias",
            Self::Macro => "macro",
            Self::Impl => "impl",
            Self::Module => "module",
        };
        f.write_str(s)
    }
}

/// Code complexity and size metrics.
#[derive(Debug, Clone, Serialize)]
pub struct CodeMetrics {
    /// Total lines in the file.
    pub total_lines: usize,
    /// Non-blank, non-comment lines.
    pub lines_of_code: usize,
    /// Blank lines.
    pub blank_lines: usize,
    /// Comment lines.
    pub comment_lines: usize,
    /// McCabe cyclomatic complexity.
    pub cyclomatic_complexity: usize,
    /// Maximum nesting depth of control flow.
    pub max_nesting_depth: usize,
}

/// A top-level definition boundary used for AST-aligned chunking.
#[derive(Debug, Clone, Serialize)]
pub struct TopLevelDef {
    /// Definition name.
    pub name: String,
    /// What kind (function, class, struct, etc.).
    pub kind: SymbolKind,
    /// Start line (1-based).
    pub start_line: usize,
    /// End line (1-based, inclusive).
    pub end_line: usize,
    /// Start byte offset in source.
    pub start_byte: usize,
    /// End byte offset in source (exclusive).
    pub end_byte: usize,
}

/// Complete analysis result for a single source file.
#[derive(Debug, Clone, Serialize)]
pub struct FileAnalysis {
    /// Detected language.
    pub language: Language,
    /// All extracted symbols.
    pub symbols: Vec<Symbol>,
    /// Code metrics.
    pub metrics: CodeMetrics,
    /// SHA-256 hash of sorted symbol names for change detection.
    pub symbol_hash: String,
    /// Top-level definition boundaries for chunking.
    pub top_level_defs: Vec<TopLevelDef>,
}

/// Compute a deterministic hash of symbol names for change detection.
///
/// Sorts symbol names alphabetically, joins with newlines, and returns
/// the hex-encoded SHA-256 digest.  Two files with the same symbol
/// names (regardless of body changes) produce the same hash — this
/// is intentional, matching the Python HADES behavior where only
/// structural changes (added/removed/renamed symbols) trigger
/// re-processing.
pub fn compute_symbol_hash(symbols: &[Symbol]) -> String {
    let mut names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    let joined = names.join("\n");
    let digest = Sha256::digest(joined.as_bytes());
    hex_encode(&digest)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_hash_deterministic() {
        let symbols = vec![
            Symbol {
                name: "foo".into(),
                kind: SymbolKind::Function,
                start_line: 1,
                end_line: 5,
                metadata: serde_json::Value::Null,
            },
            Symbol {
                name: "bar".into(),
                kind: SymbolKind::Class,
                start_line: 7,
                end_line: 20,
                metadata: serde_json::Value::Null,
            },
        ];
        let h1 = compute_symbol_hash(&symbols);
        let h2 = compute_symbol_hash(&symbols);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_symbol_hash_order_independent() {
        let s1 = vec![
            Symbol { name: "a".into(), kind: SymbolKind::Function, start_line: 1, end_line: 1, metadata: serde_json::Value::Null },
            Symbol { name: "b".into(), kind: SymbolKind::Function, start_line: 2, end_line: 2, metadata: serde_json::Value::Null },
        ];
        let s2 = vec![
            Symbol { name: "b".into(), kind: SymbolKind::Function, start_line: 2, end_line: 2, metadata: serde_json::Value::Null },
            Symbol { name: "a".into(), kind: SymbolKind::Function, start_line: 1, end_line: 1, metadata: serde_json::Value::Null },
        ];
        assert_eq!(compute_symbol_hash(&s1), compute_symbol_hash(&s2));
    }

    #[test]
    fn test_symbol_hash_differs_on_name_change() {
        let s1 = vec![
            Symbol { name: "foo".into(), kind: SymbolKind::Function, start_line: 1, end_line: 1, metadata: serde_json::Value::Null },
        ];
        let s2 = vec![
            Symbol { name: "bar".into(), kind: SymbolKind::Function, start_line: 1, end_line: 1, metadata: serde_json::Value::Null },
        ];
        assert_ne!(compute_symbol_hash(&s1), compute_symbol_hash(&s2));
    }

    #[test]
    fn test_symbol_kind_display() {
        assert_eq!(SymbolKind::Function.to_string(), "function");
        assert_eq!(SymbolKind::Class.to_string(), "class");
        assert_eq!(SymbolKind::Impl.to_string(), "impl");
    }
}
