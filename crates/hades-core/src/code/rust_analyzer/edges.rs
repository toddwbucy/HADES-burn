//! Edge resolution from rust-analyzer extraction data.
//!
//! Materializes symbol nodes and edges from file-level extraction data
//! for storage in ArangoDB graph collections:
//! - `codebase_symbols` — per-symbol documents
//! - `codebase_edges` — defines, calls, implements, imports, pyo3_exposes, ffi_exposes

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::db::keys;
use super::symbols::FileExtraction;

/// Edge types produced by the resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// File defines a symbol.
    Defines,
    /// Symbol calls another symbol.
    Calls,
    /// Symbol implements a trait method.
    Implements,
    /// Symbol is exposed to Python via PyO3.
    Pyo3Exposes,
    /// Symbol is exposed via FFI boundary.
    FfiExposes,
    /// File imports a symbol from another file.
    Imports,
}

impl EdgeKind {
    /// String representation for storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Defines => "defines",
            Self::Calls => "calls",
            Self::Implements => "implements",
            Self::Pyo3Exposes => "pyo3_exposes",
            Self::FfiExposes => "ffi_exposes",
            Self::Imports => "imports",
        }
    }
}

/// A resolved edge ready for ArangoDB insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateEdge {
    /// Source vertex ID (e.g., `codebase_files/src_lib_rs`).
    pub from: String,
    /// Target vertex ID (e.g., `codebase_symbols/src_lib_rs__Config__new`).
    pub to: String,
    /// Edge type.
    pub kind: EdgeKind,
    /// Additional metadata.
    #[serde(flatten)]
    pub metadata: serde_json::Value,
}

/// A symbol document ready for ArangoDB insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDocument {
    /// ArangoDB document key.
    #[serde(rename = "_key")]
    pub key: String,
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub visibility: String,
    pub signature: String,
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_symbol: Option<String>,
    pub impl_trait: Option<String>,
    pub is_pyo3: bool,
    pub is_ffi: bool,
    pub is_unsafe: bool,
    pub derives: Vec<String>,
    pub python_name: Option<String>,
    pub analyzed_at: String,
}

/// Resolves file-level extraction data into symbol documents and edges.
///
/// Takes a map of `rel_path → FileExtraction` (produced by
/// `RustSymbolExtractor::extract_crate`) and produces:
/// - Symbol documents for the `codebase_symbols` collection
/// - Edge documents for the `codebase_edges` collection
pub struct RustEdgeResolver {
    /// Input: rel_path → extraction data.
    file_data: HashMap<String, FileExtraction>,
    /// Index: qualified_name → vec of (rel_path, symbol_key).
    symbol_index: HashMap<String, Vec<(String, String)>>,
}

impl RustEdgeResolver {
    /// Create a new resolver from extraction data.
    pub fn new(file_data: HashMap<String, FileExtraction>) -> Self {
        let mut resolver = Self {
            file_data,
            symbol_index: HashMap::new(),
        };
        resolver.build_index();
        resolver
    }

    /// Build symbol documents for the `codebase_symbols` collection.
    pub fn build_symbol_documents(&self) -> Vec<SymbolDocument> {
        let mut documents = Vec::new();

        for (rel_path, extraction) in &self.file_data {
            for sym in &extraction.symbols {
                let sk = keys::symbol_key(
                    &keys::file_key(rel_path),
                    &sym.qualified_name,
                );

                documents.push(SymbolDocument {
                    key: sk,
                    name: sym.name.clone(),
                    qualified_name: sym.qualified_name.clone(),
                    kind: sym.kind.clone(),
                    visibility: sym.visibility.clone(),
                    signature: sym.signature.clone(),
                    file_path: rel_path.clone(),
                    start_line: sym.start_line,
                    end_line: sym.end_line,
                    parent_symbol: sym.parent_symbol.clone(),
                    impl_trait: sym.impl_trait.clone(),
                    is_pyo3: sym.is_pyo3,
                    is_ffi: sym.is_ffi,
                    is_unsafe: sym.is_unsafe,
                    derives: sym.derives.clone(),
                    python_name: sym.python_name.clone(),
                    analyzed_at: extraction.analyzed_at.clone(),
                });
            }
        }

        info!("built {} symbol documents from {} files",
            documents.len(), self.file_data.len());
        documents
    }

    /// Build edges for the `codebase_edges` collection.
    pub fn build_edges(&self) -> Vec<CrateEdge> {
        let mut edges = Vec::new();
        let mut seen: HashSet<(String, String, &str)> = HashSet::new();

        for (rel_path, extraction) in &self.file_data {
            let fk = keys::file_key(rel_path);

            for sym in &extraction.symbols {
                let sk = keys::symbol_key(&fk, &sym.qualified_name);

                // 1. defines: file → symbol
                let from = format!("codebase_files/{fk}");
                let to = format!("codebase_symbols/{sk}");
                if seen.insert((from.clone(), to.clone(), "defines")) {
                    edges.push(CrateEdge {
                        from,
                        to,
                        kind: EdgeKind::Defines,
                        metadata: serde_json::json!({
                            "file_path": rel_path,
                            "symbol_name": sym.qualified_name,
                        }),
                    });
                }

                // 2. calls: symbol → called symbol
                for call in &sym.calls {
                    if let Some(target_sk) = self.resolve_call_target(call, rel_path) {
                        let from = format!("codebase_symbols/{sk}");
                        let to = format!("codebase_symbols/{target_sk}");
                        if seen.insert((from.clone(), to.clone(), "calls")) {
                            edges.push(CrateEdge {
                                from,
                                to,
                                kind: EdgeKind::Calls,
                                metadata: serde_json::json!({
                                    "caller": sym.qualified_name,
                                    "callee": call.qualified_name,
                                }),
                            });
                        }
                    }
                }

                // 3. implements: method with impl_trait → trait symbol
                if let Some(ref trait_name) = sym.impl_trait
                    && matches!(sym.kind.as_str(), "method" | "function")
                    && let Some(trait_sk) = self.resolve_trait(trait_name, rel_path)
                {
                    let from = format!("codebase_symbols/{sk}");
                    let to = format!("codebase_symbols/{trait_sk}");
                    if seen.insert((from.clone(), to.clone(), "implements")) {
                        edges.push(CrateEdge {
                            from,
                            to,
                            kind: EdgeKind::Implements,
                            metadata: serde_json::json!({
                                "implementor": sym.qualified_name,
                                "trait": trait_name,
                            }),
                        });
                    }
                }

                // 4. pyo3_exposes: self-edge marker
                if sym.is_pyo3 {
                    let vertex = format!("codebase_symbols/{sk}");
                    let key = (vertex.clone(), vertex.clone(), "pyo3_exposes");
                    if seen.insert(key) {
                        edges.push(CrateEdge {
                            from: vertex.clone(),
                            to: vertex,
                            kind: EdgeKind::Pyo3Exposes,
                            metadata: serde_json::json!({
                                "symbol_name": sym.qualified_name,
                                "python_name": sym.python_name
                                    .as_deref()
                                    .unwrap_or(&sym.name),
                            }),
                        });
                    }
                }

                // 5. ffi_exposes: self-edge marker
                if sym.is_ffi {
                    let vertex = format!("codebase_symbols/{sk}");
                    let key = (vertex.clone(), vertex.clone(), "ffi_exposes");
                    if seen.insert(key) {
                        edges.push(CrateEdge {
                            from: vertex.clone(),
                            to: vertex,
                            kind: EdgeKind::FfiExposes,
                            metadata: serde_json::json!({
                                "symbol_name": sym.qualified_name,
                            }),
                        });
                    }
                }
            }
        }

        info!("built {} edges from {} files", edges.len(), self.file_data.len());
        edges
    }

    // ── Internal ─────────────────────────────────────────────────

    /// Build the symbol index for call resolution.
    fn build_index(&mut self) {
        for (rel_path, extraction) in &self.file_data {
            let fk = keys::file_key(rel_path);
            for sym in &extraction.symbols {
                if sym.qualified_name.is_empty() {
                    continue;
                }
                let sk = keys::symbol_key(&fk, &sym.qualified_name);
                let entry = (rel_path.clone(), sk);

                self.symbol_index
                    .entry(sym.qualified_name.clone())
                    .or_default()
                    .push(entry.clone());

                // Also index by bare name for cross-file resolution.
                if sym.name != sym.qualified_name {
                    let file_scoped = format!("{}::{}", rel_path, sym.name);
                    self.symbol_index
                        .entry(file_scoped)
                        .or_default()
                        .push(entry);
                }
            }
        }
    }

    /// Resolve a call target to a symbol key.
    fn resolve_call_target(
        &self,
        call: &super::symbols::CallTarget,
        caller_file: &str,
    ) -> Option<String> {
        let prefer_file = if call.file.is_empty() { caller_file } else { &call.file };

        // Strategy 1: exact qualified name.
        if let Some(entries) = self.symbol_index.get(&call.qualified_name)
            && let Some(sk) = pick_best_match(entries, prefer_file)
        {
            return Some(sk);
        }

        // Strategy 2: file-scoped name.
        if !call.file.is_empty() {
            let file_scoped = format!("{}::{}", call.file, call.name);
            if let Some(entries) = self.symbol_index.get(&file_scoped)
                && let Some(sk) = pick_best_match(entries, &call.file)
            {
                return Some(sk);
            }
        }

        // Strategy 3: same-file name.
        let same_file = format!("{}::{}", caller_file, call.name);
        if let Some(entries) = self.symbol_index.get(&same_file)
            && let Some(sk) = pick_best_match(entries, caller_file)
        {
            return Some(sk);
        }

        // Strategy 4: bare name.
        if let Some(entries) = self.symbol_index.get(&call.name)
            && let Some(sk) = pick_best_match(entries, prefer_file)
        {
            return Some(sk);
        }

        None
    }

    /// Resolve a trait name to a symbol key.
    fn resolve_trait(&self, trait_name: &str, prefer_file: &str) -> Option<String> {
        if let Some(entries) = self.symbol_index.get(trait_name) {
            return pick_best_match(entries, prefer_file);
        }
        None
    }
}

/// Pick the best symbol key from candidates, preferring same-file matches.
fn pick_best_match(entries: &[(String, String)], prefer_file: &str) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    // Prefer same-file match.
    for (rel_path, sk) in entries {
        if rel_path == prefer_file {
            return Some(sk.clone());
        }
    }
    // Fallback: first entry.
    Some(entries[0].1.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::symbols::{CallTarget, ExtractedSymbol, FileExtraction};

    fn make_symbol(name: &str, kind: &str) -> ExtractedSymbol {
        ExtractedSymbol {
            name: name.to_string(),
            qualified_name: name.to_string(),
            kind: kind.to_string(),
            visibility: "pub".to_string(),
            signature: String::new(),
            start_line: 0,
            end_line: 10,
            parent_symbol: None,
            impl_trait: None,
            is_pyo3: false,
            is_ffi: false,
            is_unsafe: false,
            derives: Vec::new(),
            python_name: None,
            calls: Vec::new(),
        }
    }

    fn make_extraction(symbols: Vec<ExtractedSymbol>) -> FileExtraction {
        FileExtraction {
            symbols,
            impl_blocks: Vec::new(),
            pyo3_exports: Vec::new(),
            ffi_boundaries: Vec::new(),
            analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_defines_edges() {
        let mut file_data = HashMap::new();
        file_data.insert(
            "src/lib.rs".to_string(),
            make_extraction(vec![make_symbol("Config", "struct")]),
        );

        let resolver = RustEdgeResolver::new(file_data);
        let edges = resolver.build_edges();

        let defines: Vec<_> = edges.iter().filter(|e| e.kind == EdgeKind::Defines).collect();
        assert_eq!(defines.len(), 1);
        assert!(defines[0].from.starts_with("codebase_files/"));
        assert!(defines[0].to.starts_with("codebase_symbols/"));
    }

    #[test]
    fn test_calls_edges() {
        let mut file_data = HashMap::new();

        let mut caller = make_symbol("main", "function");
        caller.calls = vec![CallTarget {
            qualified_name: "Config".to_string(),
            name: "Config".to_string(),
            file: "src/config.rs".to_string(),
            line: 5,
        }];

        file_data.insert(
            "src/main.rs".to_string(),
            make_extraction(vec![caller]),
        );
        file_data.insert(
            "src/config.rs".to_string(),
            make_extraction(vec![make_symbol("Config", "struct")]),
        );

        let resolver = RustEdgeResolver::new(file_data);
        let edges = resolver.build_edges();

        let calls: Vec<_> = edges.iter().filter(|e| e.kind == EdgeKind::Calls).collect();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_pyo3_marker_edge() {
        let mut file_data = HashMap::new();
        let mut sym = make_symbol("my_func", "function");
        sym.is_pyo3 = true;
        sym.python_name = Some("my_func".to_string());
        file_data.insert("src/lib.rs".to_string(), make_extraction(vec![sym]));

        let resolver = RustEdgeResolver::new(file_data);
        let edges = resolver.build_edges();

        let pyo3: Vec<_> = edges.iter().filter(|e| e.kind == EdgeKind::Pyo3Exposes).collect();
        assert_eq!(pyo3.len(), 1);
        assert_eq!(pyo3[0].from, pyo3[0].to); // self-edge
    }

    #[test]
    fn test_implements_edge() {
        let mut file_data = HashMap::new();

        let trait_sym = make_symbol("Display", "interface");
        let mut method = make_symbol("fmt", "method");
        method.impl_trait = Some("Display".to_string());
        method.parent_symbol = Some("Config".to_string());

        file_data.insert(
            "src/lib.rs".to_string(),
            make_extraction(vec![trait_sym, method]),
        );

        let resolver = RustEdgeResolver::new(file_data);
        let edges = resolver.build_edges();

        let implements: Vec<_> = edges.iter()
            .filter(|e| e.kind == EdgeKind::Implements)
            .collect();
        assert_eq!(implements.len(), 1);
    }

    #[test]
    fn test_symbol_documents() {
        let mut file_data = HashMap::new();
        file_data.insert(
            "src/lib.rs".to_string(),
            make_extraction(vec![
                make_symbol("Config", "struct"),
                make_symbol("new", "function"),
            ]),
        );

        let resolver = RustEdgeResolver::new(file_data);
        let docs = resolver.build_symbol_documents();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().all(|d| !d.key.is_empty()));
    }

    #[test]
    fn test_edge_deduplication() {
        let mut file_data = HashMap::new();
        file_data.insert(
            "src/lib.rs".to_string(),
            make_extraction(vec![make_symbol("Config", "struct")]),
        );

        let resolver = RustEdgeResolver::new(file_data);
        let edges = resolver.build_edges();

        // Only one defines edge for Config.
        let defines: Vec<_> = edges.iter().filter(|e| e.kind == EdgeKind::Defines).collect();
        assert_eq!(defines.len(), 1);
    }
}
