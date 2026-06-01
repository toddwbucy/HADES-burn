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

impl Symbol {
    /// The collision-free name used to derive this symbol's graph `_key`.
    ///
    /// The bare `name` is NOT unique within a file — Rust has one `new`,
    /// `default`, `fmt`, … per `impl` block, and Python one `__init__` per
    /// class. Keying on the bare name collapses them onto a single `_key`,
    /// silently discarding all but one (see issue #113). This method consults
    /// the parent context the analyzers already record in `metadata`:
    ///
    /// - Rust methods carry `impl_context` (`"Config"`, `"Display for Config"`,
    ///   set in `rust_ast.rs`) → `"<ctx>::<name>"`.
    /// - Python methods carry `parent_symbol` (the enclosing class) →
    ///   `"<parent>.<name>"`, matching the dotted convention the Python call
    ///   resolver indexes on (`python_calls.rs`).
    /// - Everything else (free functions, top-level types) is already unique at
    ///   file scope and returns the bare `name`.
    ///
    /// Mirrors the qualified names produced by the rust-analyzer enrichment
    /// path, which has always keyed correctly.
    pub fn qualified_name(&self) -> String {
        if let Some(ctx) = self.metadata.get("impl_context").and_then(|v| v.as_str())
            && !ctx.is_empty()
        {
            return format!("{ctx}::{}", self.name);
        }
        if let Some(parent) = self.metadata.get("parent_symbol").and_then(|v| v.as_str())
            && !parent.is_empty()
        {
            return format!("{parent}.{}", self.name);
        }
        self.name.clone()
    }
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

impl SymbolKind {
    /// Map to the universal graph semantic primitive.
    ///
    /// Five primitives: `file`, `module`, `type`, `callable`, `value`.
    /// Returns `None` for non-primitive kinds (`Import`, `Impl`) that
    /// should not be stored as symbol vertices in the graph.
    ///
    /// - `Import` → produces edges, not vertices
    /// - `Impl` → scaffolding; methods extracted individually as `callable`
    ///
    /// **Keep in sync with `universal_kind_from_lsp()`** in
    /// `code::rust_analyzer::edges`, which maps LSP kind strings to the
    /// same four primitives for the rust-analyzer enrichment path.
    pub fn universal_kind(&self) -> Option<&'static str> {
        match self {
            Self::Function => Some("callable"),
            Self::Macro => Some("callable"),
            Self::Struct => Some("type"),
            Self::Enum => Some("type"),
            Self::Trait => Some("type"),
            Self::Class => Some("type"),
            Self::TypeAlias => Some("type"),
            Self::Constant => Some("value"),
            Self::Variable => Some("value"),
            Self::Module => Some("module"),
            Self::Import => None,
            Self::Impl => None,
        }
    }

    /// Whether this kind represents a graph primitive that should be stored
    /// as a vertex in `codebase_symbols`.
    pub fn is_primitive(&self) -> bool {
        self.universal_kind().is_some()
    }

    /// The language-specific kind string (stored as `lang_kind` in ArangoDB).
    pub fn lang_kind(&self) -> &'static str {
        match self {
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
        }
    }
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.lang_kind())
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
    use serde_json::json;

    fn sym(name: &str, metadata: serde_json::Value) -> Symbol {
        Symbol {
            name: name.into(),
            kind: SymbolKind::Function,
            start_line: 1,
            end_line: 1,
            metadata,
        }
    }

    #[test]
    fn test_qualified_name_rust_impl_method() {
        // Two `new`s in different impl blocks must qualify differently (#113).
        let a = sym("new", json!({ "impl_context": "Config" }));
        let b = sym("new", json!({ "impl_context": "Embedder" }));
        assert_eq!(a.qualified_name(), "Config::new");
        assert_eq!(b.qualified_name(), "Embedder::new");
        assert_ne!(a.qualified_name(), b.qualified_name());
    }

    #[test]
    fn test_qualified_name_rust_trait_impl() {
        let fmt = sym("fmt", json!({ "impl_context": "Display for Config" }));
        assert_eq!(fmt.qualified_name(), "Display for Config::fmt");
    }

    #[test]
    fn test_qualified_name_python_method() {
        let save = sym("save", json!({ "parent_symbol": "Model" }));
        assert_eq!(save.qualified_name(), "Model.save");
    }

    #[test]
    fn test_qualified_name_free_function_is_bare() {
        assert_eq!(sym("helper", json!({})).qualified_name(), "helper");
        assert_eq!(
            sym("helper", serde_json::Value::Null).qualified_name(),
            "helper"
        );
        // Empty context must not produce a dangling separator.
        assert_eq!(
            sym("helper", json!({ "impl_context": "" })).qualified_name(),
            "helper"
        );
        assert_eq!(
            sym("helper", json!({ "parent_symbol": "" })).qualified_name(),
            "helper"
        );
    }

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
            Symbol {
                name: "a".into(),
                kind: SymbolKind::Function,
                start_line: 1,
                end_line: 1,
                metadata: serde_json::Value::Null,
            },
            Symbol {
                name: "b".into(),
                kind: SymbolKind::Function,
                start_line: 2,
                end_line: 2,
                metadata: serde_json::Value::Null,
            },
        ];
        let s2 = vec![
            Symbol {
                name: "b".into(),
                kind: SymbolKind::Function,
                start_line: 2,
                end_line: 2,
                metadata: serde_json::Value::Null,
            },
            Symbol {
                name: "a".into(),
                kind: SymbolKind::Function,
                start_line: 1,
                end_line: 1,
                metadata: serde_json::Value::Null,
            },
        ];
        assert_eq!(compute_symbol_hash(&s1), compute_symbol_hash(&s2));
    }

    #[test]
    fn test_symbol_hash_differs_on_name_change() {
        let s1 = vec![Symbol {
            name: "foo".into(),
            kind: SymbolKind::Function,
            start_line: 1,
            end_line: 1,
            metadata: serde_json::Value::Null,
        }];
        let s2 = vec![Symbol {
            name: "bar".into(),
            kind: SymbolKind::Function,
            start_line: 1,
            end_line: 1,
            metadata: serde_json::Value::Null,
        }];
        assert_ne!(compute_symbol_hash(&s1), compute_symbol_hash(&s2));
    }

    #[test]
    fn test_symbol_kind_display() {
        assert_eq!(SymbolKind::Function.to_string(), "function");
        assert_eq!(SymbolKind::Class.to_string(), "class");
        assert_eq!(SymbolKind::Impl.to_string(), "impl");
    }

    #[test]
    fn test_universal_kind_mapping() {
        assert_eq!(SymbolKind::Function.universal_kind(), Some("callable"));
        assert_eq!(SymbolKind::Macro.universal_kind(), Some("callable"));
        assert_eq!(SymbolKind::Struct.universal_kind(), Some("type"));
        assert_eq!(SymbolKind::Enum.universal_kind(), Some("type"));
        assert_eq!(SymbolKind::Trait.universal_kind(), Some("type"));
        assert_eq!(SymbolKind::Class.universal_kind(), Some("type"));
        assert_eq!(SymbolKind::TypeAlias.universal_kind(), Some("type"));
        assert_eq!(SymbolKind::Constant.universal_kind(), Some("value"));
        assert_eq!(SymbolKind::Variable.universal_kind(), Some("value"));
        assert_eq!(SymbolKind::Module.universal_kind(), Some("module"));
        assert_eq!(SymbolKind::Import.universal_kind(), None);
        assert_eq!(SymbolKind::Impl.universal_kind(), None);
    }

    #[test]
    fn test_is_primitive() {
        assert!(SymbolKind::Function.is_primitive());
        assert!(SymbolKind::Struct.is_primitive());
        assert!(!SymbolKind::Import.is_primitive());
        assert!(!SymbolKind::Impl.is_primitive());
    }

    #[test]
    fn test_lang_kind() {
        assert_eq!(SymbolKind::Function.lang_kind(), "function");
        assert_eq!(SymbolKind::Import.lang_kind(), "import");
        assert_eq!(SymbolKind::Impl.lang_kind(), "impl");
    }
}
