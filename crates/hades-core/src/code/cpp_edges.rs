//! Materialize libclang-resolved C/C++/CUDA calls as graph edges.
//!
//! [`super::cpp`] stores each call's target USR, definition location, and
//! qualified name on the caller symbol. This module resolves those records
//! against all ingested C-family symbols after the per-file pass, so calls can
//! cross translation units while still targeting HADES's span-based keys.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::{Value, json};

use super::{Symbol, SymbolKind};
use crate::db::{collections::CODEBASE, keys};

#[derive(Clone)]
struct IndexedSymbol {
    path: String,
    key: String,
    line: usize,
}

/// Resolve semantic call records into `codebase_calls_edges` documents.
pub fn resolve_cpp_calls(base: &Path, file_symbols: &HashMap<String, Vec<Symbol>>) -> Vec<Value> {
    let mut by_usr: HashMap<String, Vec<IndexedSymbol>> = HashMap::new();
    let mut by_location: HashMap<(String, usize), Vec<IndexedSymbol>> = HashMap::new();
    let mut by_qname: HashMap<String, Vec<IndexedSymbol>> = HashMap::new();

    for (rel_path, symbols) in file_symbols {
        let fkey = keys::file_key(rel_path);
        for symbol in symbols.iter().filter(|s| s.kind != SymbolKind::Import) {
            let qname = symbol.qualified_name();
            let indexed = IndexedSymbol {
                path: rel_path.clone(),
                key: keys::symbol_key(&fkey, &qname, symbol.start_line),
                line: symbol.start_line,
            };
            by_qname.entry(qname).or_default().push(indexed.clone());
            by_location
                .entry((rel_path.clone(), symbol.start_line))
                .or_default()
                .push(indexed.clone());
            if let Some(usr) = symbol.metadata.get("usr").and_then(Value::as_str) {
                by_usr.entry(usr.to_string()).or_default().push(indexed);
            }
        }
    }

    let mut edges = Vec::new();
    let mut seen = HashSet::new();
    for (source_path, symbols) in file_symbols {
        let source_fkey = keys::file_key(source_path);
        for caller in symbols.iter().filter(|s| s.kind != SymbolKind::Import) {
            let Some(calls) = caller.metadata.get("calls").and_then(Value::as_array) else {
                continue;
            };
            let caller_qname = caller.qualified_name();
            let caller_key = keys::symbol_key(&source_fkey, &caller_qname, caller.start_line);

            for call in calls {
                let target_rel = call
                    .get("target_file")
                    .and_then(Value::as_str)
                    .and_then(|path| relative_path(base, Path::new(path)));
                let target_line = call
                    .get("target_line")
                    .and_then(Value::as_u64)
                    .map(|line| line as usize);
                let target_usr = call.get("target_usr").and_then(Value::as_str);
                let target_qname = call
                    .get("qualified_name")
                    .and_then(Value::as_str)
                    .unwrap_or("");

                let target = target_usr
                    .and_then(|usr| by_usr.get(usr))
                    .and_then(|entries| pick(entries, target_rel.as_deref(), target_line))
                    .or_else(|| {
                        let path = target_rel.as_ref()?;
                        let line = target_line?;
                        by_location
                            .get(&(path.clone(), line))
                            .and_then(|entries| pick(entries, Some(path), Some(line)))
                    })
                    .or_else(|| {
                        by_qname
                            .get(target_qname)
                            .and_then(|entries| pick(entries, target_rel.as_deref(), target_line))
                    });

                let Some(target) = target else {
                    // External/library calls have no vertex in this database.
                    continue;
                };
                if target.key == caller_key {
                    continue;
                }

                let edge_key = keys::edge_key(&caller_key, "calls", &target.key);
                if !seen.insert(edge_key.clone()) {
                    continue;
                }
                edges.push(json!({
                    "_key": edge_key,
                    "_from": format!("{}/{}", CODEBASE.symbols, caller_key),
                    "_to": format!("{}/{}", CODEBASE.symbols, target.key),
                    "caller": caller_qname,
                    "callee": target_qname,
                    "source_path": source_path,
                    "target_path": target.path,
                    "call_line": call.get("call_line").cloned().unwrap_or(Value::Null),
                    "kernel_launch": call
                        .get("is_kernel_launch")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    "resolution": "semantic",
                    "analyzer": "libclang",
                }));
            }
        }
    }
    edges
}

fn relative_path(base: &Path, path: &Path) -> Option<String> {
    let base = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let absolute = absolute.canonicalize().unwrap_or(absolute);
    absolute
        .strip_prefix(&base)
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

fn pick<'a>(
    entries: &'a [IndexedSymbol],
    preferred_path: Option<&str>,
    preferred_line: Option<usize>,
) -> Option<&'a IndexedSymbol> {
    entries
        .iter()
        .find(|entry| {
            preferred_path.is_none_or(|path| entry.path == path)
                && preferred_line.is_none_or(|line| entry.line == line)
        })
        .or_else(|| {
            entries
                .iter()
                .find(|entry| preferred_path.is_some_and(|path| entry.path == path))
        })
        .or_else(|| entries.first())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn function(name: &str, line: usize, metadata: Value) -> Symbol {
        Symbol {
            name: name.into(),
            kind: SymbolKind::Function,
            start_line: line,
            end_line: line + 1,
            metadata,
        }
    }

    #[test]
    fn resolves_usr_to_span_key_and_marks_kernel_launch() {
        let temp = tempfile::tempdir().unwrap();
        let caller_path = temp.path().join("launch.cu");
        let target_path = temp.path().join("kernel.cuh");
        std::fs::write(&caller_path, "").unwrap();
        std::fs::write(&target_path, "").unwrap();

        let mut files = HashMap::new();
        files.insert(
            "launch.cu".into(),
            vec![function(
                "launch",
                3,
                json!({
                    "qualified_name": "launch",
                    "usr": "caller",
                    "calls": [{
                        "name": "kernel",
                        "qualified_name": "gpu::kernel",
                        "target_usr": "kernel-usr",
                        "target_file": target_path,
                        "target_line": 7,
                        "call_line": 4,
                        "is_kernel_launch": true
                    }]
                }),
            )],
        );
        files.insert(
            "kernel.cuh".into(),
            vec![function(
                "kernel",
                7,
                json!({"qualified_name": "gpu::kernel", "usr": "kernel-usr"}),
            )],
        );

        let edges = resolve_cpp_calls(temp.path(), &files);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["kernel_launch"], true);
        assert_eq!(edges[0]["resolution"], "semantic");
        assert_eq!(edges[0]["target_path"], "kernel.cuh");
    }

    #[test]
    fn drops_external_and_self_calls() {
        let mut files = HashMap::new();
        files.insert(
            "a.cpp".into(),
            vec![function(
                "a",
                1,
                json!({
                    "qualified_name": "a",
                    "usr": "a-usr",
                    "calls": [
                        {"name": "a", "qualified_name": "a", "target_usr": "a-usr"},
                        {"name": "printf", "qualified_name": "printf", "target_usr": "external"}
                    ]
                }),
            )],
        );
        assert!(resolve_cpp_calls(Path::new("."), &files).is_empty());
    }
}
