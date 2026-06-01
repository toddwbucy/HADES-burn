//! Python call-graph edge resolution.
//!
//! Reads `calls` and `parent_symbol` metadata produced by [`super::python::analyze`]
//! and resolves each call site against the cross-file symbol index to emit
//! `codebase_calls_edges` documents (symbol → symbol).
//!
//! Resolution strategy mirrors the legacy `python_edge_resolver.py`:
//!
//! 1. **Exact qualified-name match.** The call's `qualified_name` (e.g.
//!    `"json.load"`, `"Config.save"`) is looked up directly in the qualified
//!    index. Same-file matches are preferred when multiple candidates exist.
//! 2. **`self.method` rewrite.** When the callee starts with `"self."` and
//!    the caller is a method (has `parent_symbol`), rewrite to
//!    `"ParentClass.method"` and look up again.
//! 3. **Bare-name fallback.** Look up the call's `name` in the bare-name
//!    index. Lower confidence — picks the first match preferring same-file.
//!
//! Unresolved calls are silently dropped: external library calls
//! (`os.path.join`, `json.load`) typically don't have symbols in the local
//! codebase, so emitting unresolved edges would create noise without
//! actionable graph data.

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use super::symbols::{Symbol, SymbolKind};
use crate::db::collections::CODEBASE;
use crate::db::keys;

/// Build a qualified-name → vec<(rel_path, symbol_key)> index for Python symbols.
///
/// For methods (symbols with `parent_symbol` metadata), the qualified name is
/// `"ParentClass.method_name"`. For top-level functions, classes, and other
/// primitives it is just the symbol name. Imports are skipped — they produce
/// import edges, not call edges.
pub fn build_qualified_index(
    file_symbols: &HashMap<String, Vec<Symbol>>,
) -> HashMap<String, Vec<(String, String)>> {
    let mut index: HashMap<String, Vec<(String, String)>> = HashMap::new();

    for (rel_path, symbols) in file_symbols {
        let fkey = keys::file_key(rel_path);
        for sym in symbols {
            if sym.kind == SymbolKind::Import {
                continue;
            }
            let qname = qualified_name(sym);
            // Value keys on the qualified name so it matches the stored vertex
            // `_key`; the index is *keyed* by qname for qualified-exact lookup.
            let skey = keys::symbol_key(&fkey, &qname);
            let entry = (rel_path.clone(), skey);
            index.entry(qname).or_default().push(entry);
        }
    }

    index
}

/// Resolve Python call sites into `codebase_calls_edges` documents.
///
/// Each symbol's `calls` metadata is a list of `{name, qualified_name}` records
/// produced by AST extraction. For each call we attempt resolution in three
/// stages (qualified-exact → self-rewrite → bare-name) and emit an edge
/// document when a target is found. Self-edges (caller == callee) are skipped.
///
/// Returns the deduplicated edge list, ready for batch insert into
/// [`CODEBASE.calls_edges`].
pub fn resolve_python_calls(
    file_symbols: &HashMap<String, Vec<Symbol>>,
    qualified_index: &HashMap<String, Vec<(String, String)>>,
    bare_name_index: &HashMap<String, Vec<(String, String)>>,
) -> Vec<Value> {
    let mut edges = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for (rel_path, symbols) in file_symbols {
        let fkey = keys::file_key(rel_path);

        for sym in symbols {
            // Only definition symbols can call other symbols.
            if sym.kind == SymbolKind::Import {
                continue;
            }

            // Skip if the symbol has no calls recorded.
            let Some(call_array) = sym.metadata.get("calls").and_then(|v| v.as_array()) else {
                continue;
            };

            let caller_qname = qualified_name(sym);
            let caller_skey = keys::symbol_key(&fkey, &caller_qname);
            let caller_parent = sym
                .metadata
                .get("parent_symbol")
                .and_then(|v| v.as_str())
                .map(str::to_string);

            for call in call_array {
                let call_name = call.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let call_qname = call
                    .get("qualified_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(call_name);

                if call_name.is_empty() {
                    continue;
                }

                let resolved = resolve_call_target(
                    call_name,
                    call_qname,
                    caller_parent.as_deref(),
                    rel_path,
                    qualified_index,
                    bare_name_index,
                );

                if let Some((target_path, target_skey)) = resolved {
                    // No self-edges (a symbol calling itself).
                    if target_skey == caller_skey {
                        continue;
                    }

                    let edge_key = keys::edge_key(&caller_skey, "calls", target_skey);
                    if !seen.insert(edge_key.clone()) {
                        continue;
                    }

                    edges.push(json!({
                        "_from": format!("{}/{}", CODEBASE.symbols, caller_skey),
                        "_to": format!("{}/{}", CODEBASE.symbols, target_skey),
                        "_key": edge_key,
                        "caller": caller_qname,
                        "callee": call_qname,
                        "source_path": rel_path,
                        "target_path": target_path,
                    }));
                }
            }
        }
    }

    edges
}

/// Compute the qualified name for a Python symbol.
///
/// For symbols with `parent_symbol` metadata, returns `"Parent.name"`.
/// Otherwise returns the bare symbol name.
fn qualified_name(sym: &Symbol) -> String {
    // Single source of truth: `Symbol::qualified_name()` produces the same
    // `"Parent.method"` form for Python (and `::` form for Rust). Kept as a thin
    // wrapper so call sites in this module read naturally.
    sym.qualified_name()
}

/// Attempt to resolve a call site to a target symbol.
///
/// Returns `(target_rel_path, target_symbol_key)` on success.
fn resolve_call_target<'a>(
    call_name: &str,
    call_qname: &str,
    caller_parent_class: Option<&str>,
    caller_file: &str,
    qualified_index: &'a HashMap<String, Vec<(String, String)>>,
    bare_name_index: &'a HashMap<String, Vec<(String, String)>>,
) -> Option<(&'a str, &'a str)> {
    // Strategy 1: exact qualified-name match.
    if let Some(entries) = qualified_index.get(call_qname)
        && let Some(pick) = pick_best(entries, caller_file)
    {
        return Some(pick);
    }

    // Strategy 2: self.method → ParentClass.method
    if let Some(method) = call_qname.strip_prefix("self.")
        && let Some(parent) = caller_parent_class
    {
        let rewritten = format!("{parent}.{method}");
        if let Some(entries) = qualified_index.get(&rewritten)
            && let Some(pick) = pick_best(entries, caller_file)
        {
            return Some(pick);
        }
    }

    // Strategy 3: bare-name fallback.
    if !call_name.is_empty()
        && let Some(entries) = bare_name_index.get(call_name)
        && let Some(pick) = pick_best(entries, caller_file)
    {
        return Some(pick);
    }

    None
}

/// Pick the best candidate from an index entry list.
///
/// Prefers same-file matches over cross-file. Falls back to the first entry
/// when no same-file match exists.
fn pick_best<'a>(entries: &'a [(String, String)], prefer_file: &str) -> Option<(&'a str, &'a str)> {
    if entries.is_empty() {
        return None;
    }
    for (path, skey) in entries {
        if path == prefer_file {
            return Some((path.as_str(), skey.as_str()));
        }
    }
    entries.first().map(|(p, k)| (p.as_str(), k.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn func(name: &str, parent: Option<&str>, calls: Value) -> Symbol {
        let mut meta = json!({});
        if let Some(p) = parent {
            meta["parent_symbol"] = json!(p);
        }
        if !calls.as_array().map(|a| a.is_empty()).unwrap_or(true) {
            meta["calls"] = calls;
        }
        Symbol {
            name: name.into(),
            kind: SymbolKind::Function,
            start_line: 1,
            end_line: 10,
            metadata: meta,
        }
    }

    fn klass(name: &str) -> Symbol {
        Symbol {
            name: name.into(),
            kind: SymbolKind::Class,
            start_line: 1,
            end_line: 100,
            metadata: Value::Null,
        }
    }

    fn import(name: &str) -> Symbol {
        Symbol {
            name: name.into(),
            kind: SymbolKind::Import,
            start_line: 1,
            end_line: 1,
            metadata: json!({ "type": "import", "module": name }),
        }
    }

    fn build_bare_index(
        file_symbols: &HashMap<String, Vec<Symbol>>,
    ) -> HashMap<String, Vec<(String, String)>> {
        let mut index: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for (rel_path, symbols) in file_symbols {
            let fkey = keys::file_key(rel_path);
            for sym in symbols {
                if sym.kind == SymbolKind::Import {
                    continue;
                }
                let skey = keys::symbol_key(&fkey, &sym.qualified_name());
                index
                    .entry(sym.name.clone())
                    .or_default()
                    .push((rel_path.clone(), skey));
            }
        }
        index
    }

    #[test]
    fn test_qualified_index_top_level() {
        let mut files = HashMap::new();
        files.insert("a.py".to_string(), vec![func("hello", None, json!([]))]);
        let idx = build_qualified_index(&files);
        assert!(idx.contains_key("hello"));
        assert_eq!(idx["hello"].len(), 1);
    }

    #[test]
    fn test_qualified_index_method() {
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![klass("Config"), func("load", Some("Config"), json!([]))],
        );
        let idx = build_qualified_index(&files);
        assert!(idx.contains_key("Config"), "class indexed by bare name");
        assert!(
            idx.contains_key("Config.load"),
            "method indexed as Class.method"
        );
        assert!(
            !idx.contains_key("load"),
            "method should not appear under bare name in qualified index"
        );
    }

    #[test]
    fn test_qualified_index_skips_imports() {
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![import("json"), func("hello", None, json!([]))],
        );
        let idx = build_qualified_index(&files);
        assert!(!idx.contains_key("json"));
        assert!(idx.contains_key("hello"));
    }

    #[test]
    fn test_resolve_exact_qualified() {
        // a.py calls b.py's "helper" via "module.helper" — qualified match.
        let mut files = HashMap::new();
        files.insert(
            "caller.py".to_string(),
            vec![func(
                "do",
                None,
                json!([{"name": "helper", "qualified_name": "mod.helper"}]),
            )],
        );
        files.insert("mod.py".to_string(), vec![func("helper", None, json!([]))]);

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        // Inject a synthetic "mod.helper" entry so Strategy 1 can fire.
        let mut qual_with_mod = qual_idx.clone();
        let helper_entry = qual_idx["helper"].clone();
        qual_with_mod.insert("mod.helper".to_string(), helper_entry);

        let edges = resolve_python_calls(&files, &qual_with_mod, &bare_idx);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["callee"], "mod.helper");
    }

    #[test]
    fn test_resolve_self_method_rewrite() {
        // class Config: def load(self): self.save()  →  Config.save
        let mut files = HashMap::new();
        files.insert(
            "config.py".to_string(),
            vec![
                klass("Config"),
                func(
                    "load",
                    Some("Config"),
                    json!([{"name": "save", "qualified_name": "self.save"}]),
                ),
                func("save", Some("Config"), json!([])),
            ],
        );

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        let edges = resolve_python_calls(&files, &qual_idx, &bare_idx);
        assert_eq!(edges.len(), 1, "expected 1 edge, got {:?}", edges);
        assert_eq!(edges[0]["caller"], "Config.load");
        assert_eq!(edges[0]["callee"], "self.save");
    }

    #[test]
    fn test_resolve_bare_name_fallback() {
        // top-level "main" calls "process" by bare name → bare-name match.
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![func(
                "main",
                None,
                json!([{"name": "process", "qualified_name": "process"}]),
            )],
        );
        files.insert("b.py".to_string(), vec![func("process", None, json!([]))]);

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        let edges = resolve_python_calls(&files, &qual_idx, &bare_idx);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["target_path"], "b.py");
    }

    #[test]
    fn test_resolve_no_self_edges() {
        // A function calling itself (recursion) should not produce an edge.
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![func(
                "fib",
                None,
                json!([{"name": "fib", "qualified_name": "fib"}]),
            )],
        );

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        let edges = resolve_python_calls(&files, &qual_idx, &bare_idx);
        assert!(edges.is_empty(), "self-recursion must not emit an edge");
    }

    #[test]
    fn test_resolve_dedup() {
        // A function calling the same target twice produces one edge.
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![func(
                "main",
                None,
                json!([
                    {"name": "helper", "qualified_name": "helper"},
                    {"name": "helper", "qualified_name": "helper"},
                ]),
            )],
        );
        files.insert("b.py".to_string(), vec![func("helper", None, json!([]))]);

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        let edges = resolve_python_calls(&files, &qual_idx, &bare_idx);
        assert_eq!(
            edges.len(),
            1,
            "duplicate call sites should produce one edge"
        );
    }

    #[test]
    fn test_resolve_prefers_same_file() {
        // Two definitions of `helper` exist; caller in a.py should resolve
        // to a.py's `helper` over b.py's.
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![
                func("helper", None, json!([])),
                func(
                    "caller",
                    None,
                    json!([{"name": "helper", "qualified_name": "helper"}]),
                ),
            ],
        );
        files.insert("b.py".to_string(), vec![func("helper", None, json!([]))]);

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        let edges = resolve_python_calls(&files, &qual_idx, &bare_idx);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["target_path"], "a.py");
    }

    #[test]
    fn test_resolve_unresolved_dropped() {
        // Call to a symbol that doesn't exist anywhere — drop silently.
        let mut files = HashMap::new();
        files.insert(
            "a.py".to_string(),
            vec![func(
                "main",
                None,
                json!([{"name": "os_path_join", "qualified_name": "os.path.join"}]),
            )],
        );

        let qual_idx = build_qualified_index(&files);
        let bare_idx = build_bare_index(&files);

        let edges = resolve_python_calls(&files, &qual_idx, &bare_idx);
        assert!(
            edges.is_empty(),
            "unresolved external calls must not emit edges"
        );
    }
}
