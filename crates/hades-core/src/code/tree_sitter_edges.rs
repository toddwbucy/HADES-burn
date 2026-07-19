//! Materialize explicitly syntactic relationships from Tree-sitter metadata.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::{Value, json};

use crate::db::{collections::CODEBASE, keys};

use super::{Symbol, SymbolKind};

#[derive(Debug, Default)]
pub struct StructuralEdges {
    pub calls: Vec<Value>,
    pub imports: Vec<Value>,
}

/// Resolve only unambiguous name relationships. These edges remain marked as
/// syntactic and must never be interpreted as compiler-resolved facts.
pub fn resolve(files: &HashMap<String, Vec<Symbol>>) -> StructuralEdges {
    let mut targets: HashMap<&str, Vec<(&str, &Symbol)>> = HashMap::new();
    for (path, symbols) in files {
        for symbol in symbols.iter().filter(|s| s.kind.is_primitive()) {
            targets
                .entry(symbol.name.as_str())
                .or_default()
                .push((path, symbol));
        }
    }

    let mut result = StructuralEdges::default();
    let mut seen_calls = HashSet::new();
    for (source_path, symbols) in files {
        let source_file_key = keys::file_key(source_path);
        for caller in symbols.iter().filter(|s| s.kind == SymbolKind::Function) {
            let from_key = keys::symbol_key(
                &source_file_key,
                &caller.qualified_name(),
                caller.start_line,
            );
            let Some(calls) = caller.metadata.get("calls").and_then(Value::as_array) else {
                continue;
            };
            for call in calls {
                let Some(name) = call.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let Some(candidates) = targets.get(name).filter(|items| items.len() == 1) else {
                    continue;
                };
                let (target_path, target) = candidates[0];
                let target_file_key = keys::file_key(target_path);
                let to_key = keys::symbol_key(
                    &target_file_key,
                    &target.qualified_name(),
                    target.start_line,
                );
                if from_key == to_key || !seen_calls.insert((from_key.clone(), to_key.clone())) {
                    continue;
                }
                result.calls.push(json!({
                    "_key": keys::edge_key(&from_key, "calls", &to_key),
                    "_from": format!("{}/{}", CODEBASE.symbols, from_key),
                    "_to": format!("{}/{}", CODEBASE.symbols, to_key),
                    "source_file": source_path,
                    "target_file": target_path,
                    "call_line": call.get("call_line"),
                    "is_kernel_launch": call.get("is_kernel_launch").and_then(Value::as_bool).unwrap_or(false),
                    "analysis_tier": "structural",
                    "analyzer": "tree-sitter",
                    "resolution": "syntactic",
                }));
            }
        }
    }

    // Go imports name packages (directories), not individual files. Resolve a
    // package only when the ingest set contains one unambiguous representative
    // file for that directory; otherwise leave it unresolved rather than guess.
    let mut package_files: HashMap<String, Vec<&str>> = HashMap::new();
    for path in files.keys() {
        if let Some(package) = Path::new(path)
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
        {
            package_files
                .entry(package.to_string())
                .or_default()
                .push(path);
        }
    }
    let mut seen_imports = HashSet::new();
    for (source_path, symbols) in files {
        for import in symbols.iter().filter(|s| s.kind == SymbolKind::Import) {
            let Some(path) = import.metadata.get("path").and_then(Value::as_str) else {
                continue;
            };
            let package = path.rsplit('/').next().unwrap_or(path);
            let Some(targets) = package_files.get(package).filter(|items| items.len() == 1) else {
                continue;
            };
            let target_path = targets[0];
            if source_path == target_path
                || !seen_imports.insert((source_path.clone(), target_path.to_string()))
            {
                continue;
            }
            let from_key = keys::file_key(source_path);
            let to_key = keys::file_key(target_path);
            result.imports.push(json!({
                "_key": keys::edge_key(&from_key, "imports", &to_key),
                "_from": format!("{}/{}", CODEBASE.files, from_key),
                "_to": format!("{}/{}", CODEBASE.files, to_key),
                "import_path": path,
                "analysis_tier": "structural",
                "analyzer": "tree-sitter",
                "resolution": "syntactic",
            }));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::{Language, tree_sitter};

    #[test]
    fn emits_only_unambiguous_syntactic_edges() {
        let mut files = HashMap::new();
        files.insert(
            "cmd/main.go".to_string(),
            tree_sitter::analyze(
                "package main\nimport \"project/helper\"\nfunc run(){ Help() }",
                Language::Go,
            )
            .unwrap()
            .symbols,
        );
        files.insert(
            "helper/helper.go".to_string(),
            tree_sitter::analyze("package helper\nfunc Help() {}", Language::Go)
                .unwrap()
                .symbols,
        );
        let edges = resolve(&files);
        assert_eq!(edges.calls.len(), 1);
        assert_eq!(edges.imports.len(), 1);
        assert_eq!(edges.calls[0]["resolution"], "syntactic");
    }
}
