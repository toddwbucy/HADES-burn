//! Rust `use`-statement expansion and import edge resolution.
//!
//! The basic syn parser (`rust_ast.rs`) already extracts `use` items as
//! `SymbolKind::Import` symbols with the full path string as the name.
//! This module provides utilities to:
//!
//! 1. **Expand** grouped imports (`use serde::{Deserialize, Serialize}`)
//!    into individual fully-qualified paths.
//! 2. **Strip** crate-local qualifiers (`crate::`, `self::`, `super::`)
//!    that aren't meaningful for cross-file resolution.
//! 3. **Resolve** expanded paths against an in-memory symbol index to
//!    produce file→symbol import edges.

use std::collections::HashMap;

use serde_json::json;

use super::symbols::{Symbol, SymbolKind};
use crate::db::collections::CODEBASE;
use crate::db::keys;

/// Collect raw use-path strings from a file's symbol list.
///
/// Filters for `SymbolKind::Import` symbols and expands any grouped
/// imports into individual paths.
pub fn collect_use_paths(symbols: &[Symbol]) -> Vec<String> {
    let mut paths = Vec::new();
    for sym in symbols {
        if sym.kind == SymbolKind::Import {
            let expanded = expand_use_group(&sym.name);
            paths.extend(expanded);
        }
    }
    paths
}

/// Expand a use-path that may contain `{...}` groups into individual paths.
///
/// # Examples
/// ```
/// # use hades_core::code::rust_imports::expand_use_group;
/// assert_eq!(
///     expand_use_group("serde::{Deserialize, Serialize}"),
///     vec!["serde::Deserialize", "serde::Serialize"],
/// );
/// assert_eq!(
///     expand_use_group("std::collections::HashMap"),
///     vec!["std::collections::HashMap"],
/// );
/// ```
pub fn expand_use_group(use_path: &str) -> Vec<String> {
    // Fast path: no braces means single import.
    if !use_path.contains('{') {
        return vec![use_path.to_string()];
    }

    // Find the first '{' and its matching '}'.
    let brace_start = match use_path.find('{') {
        Some(i) => i,
        None => return vec![use_path.to_string()],
    };

    // The prefix is everything before the brace (e.g., "serde::").
    let prefix = &use_path[..brace_start];

    // Find matching closing brace with depth tracking.
    let brace_end = match find_matching_brace(use_path, brace_start) {
        Some(i) => i,
        None => return vec![use_path.to_string()], // malformed, return as-is
    };

    // The suffix is everything after the closing brace (usually empty).
    let suffix = &use_path[brace_end + 1..];

    // Split the contents at top-level commas (respecting nested braces).
    let inner = &use_path[brace_start + 1..brace_end];
    let parts = split_at_top_level_commas(inner);

    let mut result = Vec::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // `use foo::{self}` means "import foo itself" → emit "foo", not "foo::self".
        if part == "self" {
            let module = prefix.trim_end_matches("::");
            if !module.is_empty() {
                result.push(format!("{module}{suffix}"));
            }
            continue;
        }
        // Recursively expand nested groups.
        let full = format!("{prefix}{part}{suffix}");
        result.extend(expand_use_group(&full));
    }

    result
}

/// Strip crate-local qualifiers from a use path.
///
/// Removes leading `crate::`, `self::`, and `super::` prefixes that
/// are meaningless for cross-file symbol resolution.
pub fn strip_crate_qualifiers(path: &str) -> &str {
    let mut p = path;
    loop {
        if let Some(rest) = p.strip_prefix("crate::") {
            p = rest;
        } else if let Some(rest) = p.strip_prefix("self::") {
            p = rest;
        } else if let Some(rest) = p.strip_prefix("super::") {
            p = rest;
        } else {
            break;
        }
    }
    p
}

/// Extract the leaf (final segment) from a use path.
///
/// `"std::collections::HashMap"` → `"HashMap"`
/// `"serde::Deserialize"` → `"Deserialize"`
/// `"HashMap"` → `"HashMap"`
/// `"foo::Bar as Baz"` → `"Bar"` (definition name, not alias)
pub fn leaf_name(path: &str) -> &str {
    // Handle "as" renames: "foo::Bar as Baz" → use "Bar" (the definition name).
    // The symbol index is keyed by definition names, not aliases.
    let path = if let Some((left, _alias)) = path.rsplit_once(" as ") {
        left.trim()
    } else {
        path
    };
    path.rsplit("::").next().unwrap_or(path)
}

/// Build a symbol index from all ingested files' symbols.
///
/// Returns a map from bare symbol name → vec of (rel_path, symbol_key).
/// Only definition symbols are indexed (imports are skipped).
pub fn build_symbol_index(
    file_symbols: &HashMap<String, Vec<Symbol>>,
) -> HashMap<String, Vec<(String, String)>> {
    let mut index: HashMap<String, Vec<(String, String)>> = HashMap::new();

    for (rel_path, symbols) in file_symbols {
        let fkey = keys::file_key(rel_path);
        for sym in symbols {
            // Skip imports themselves — we only want definitions.
            if sym.kind == SymbolKind::Import {
                continue;
            }

            let skey = keys::symbol_key(&fkey, &sym.name);
            let entry = (rel_path.clone(), skey);

            // Index by symbol name.
            index
                .entry(sym.name.clone())
                .or_default()
                .push(entry);
        }
    }

    index
}

/// Resolve Rust use-imports to file→symbol edges.
///
/// For each importing file, expands its use-paths, strips qualifiers,
/// extracts the leaf name, and looks it up in the symbol index.
/// Creates edges from the importing file to the target symbol.
pub fn resolve_rust_imports(
    rust_imports: &HashMap<String, Vec<String>>,
    symbol_index: &HashMap<String, Vec<(String, String)>>,
) -> Vec<serde_json::Value> {
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (source_path, use_paths) in rust_imports {
        let source_fkey = keys::file_key(source_path);

        for use_path in use_paths {
            let stripped = strip_crate_qualifiers(use_path);

            // Skip glob imports — can't resolve `*` to a specific symbol.
            if stripped.ends_with("::*") || stripped == "*" {
                continue;
            }

            let leaf = leaf_name(stripped);

            // Look up the leaf name in the symbol index.
            if let Some(targets) = symbol_index.get(leaf) {
                // Pick the best match: prefer the one whose file path
                // overlaps with the use-path module structure.
                let target = pick_best_import_target(targets, stripped, source_path);

                if let Some((target_path, target_skey)) = target {
                    // Don't create self-edges (file importing its own symbols).
                    if target_path == source_path {
                        continue;
                    }

                    let edge_key = keys::edge_key(&source_fkey, "imports", target_skey);
                    if !seen.insert(edge_key.clone()) {
                        continue; // already emitted
                    }

                    edges.push(json!({
                        "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                        "_to": format!("{}/{}", CODEBASE.symbols, target_skey),
                        "_key": edge_key,
                        "type": "imports",
                        "source_path": source_path,
                        "target_path": target_path,
                        "symbol_name": leaf,
                        "use_path": use_path,
                    }));
                }
            }
        }
    }

    edges
}

// ── Internal helpers ───────────────────────────────────────────────

/// Find the matching `}` for a `{` at position `start`, handling nesting.
fn find_matching_brace(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a string at commas that are at brace depth 0.
fn split_at_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    // Don't forget the last segment.
    if start < s.len() {
        parts.push(&s[start..]);
    }

    parts
}

/// Pick the best target from candidates for an import.
///
/// Prefers targets whose file path structurally matches the use-path.
/// For example, `use crate::db::keys` should prefer a symbol in
/// `src/db/keys.rs` over one in `src/config.rs`.
///
/// Scoring compares use-path module segments against discrete path
/// components (not substrings) to avoid false positives like "db"
/// matching "debug".
fn pick_best_import_target<'a>(
    targets: &'a [(String, String)],
    use_path: &str,
    source_path: &str,
) -> Option<(&'a str, &'a str)> {
    if targets.is_empty() {
        return None;
    }

    // Convert use-path segments to path-like form for matching.
    // "db::keys::file_key" → ["db", "keys"]  (drop leaf, it's the symbol name)
    let segments: Vec<&str> = use_path.split("::").collect();
    let module_segments = if segments.len() > 1 {
        &segments[..segments.len() - 1]
    } else {
        &segments[..]
    };

    let mut best: Option<(&str, &str, usize)> = None;

    for (rel_path, skey) in targets {
        // Skip same-file matches.
        if rel_path == source_path {
            continue;
        }

        // Split rel_path into discrete components for segment matching.
        // "src/db/keys.rs" → ["src", "db", "keys"]
        let path_parts: Vec<&str> = rel_path
            .trim_end_matches(".rs")
            .split(['/', '\\'])
            .collect();

        // Score: how many module segments match a discrete path component?
        let score = module_segments
            .iter()
            .filter(|seg| path_parts.contains(seg))
            .count();

        if best.is_none() || score > best.unwrap().2 {
            best = Some((rel_path.as_str(), skey.as_str(), score));
        }
    }

    best.map(|(path, skey, _)| (path, skey))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_simple() {
        assert_eq!(
            expand_use_group("std::collections::HashMap"),
            vec!["std::collections::HashMap"],
        );
    }

    #[test]
    fn test_expand_group() {
        let mut result = expand_use_group("serde::{Deserialize, Serialize}");
        result.sort();
        assert_eq!(result, vec!["serde::Deserialize", "serde::Serialize"]);
    }

    #[test]
    fn test_expand_nested_group() {
        let mut result = expand_use_group("foo::{bar::{A, B}, baz}");
        result.sort();
        assert_eq!(result, vec!["foo::bar::A", "foo::bar::B", "foo::baz"]);
    }

    #[test]
    fn test_expand_self_import() {
        let result = expand_use_group("crate::code::{self, Language}");
        // `self` in a use group means "import the module itself" → "crate::code"
        assert!(result.contains(&"crate::code".to_string()));
        assert!(result.contains(&"crate::code::Language".to_string()));
    }

    #[test]
    fn test_expand_glob() {
        assert_eq!(expand_use_group("std::io::*"), vec!["std::io::*"]);
    }

    #[test]
    fn test_strip_crate_qualifiers() {
        assert_eq!(strip_crate_qualifiers("crate::db::keys"), "db::keys");
        assert_eq!(strip_crate_qualifiers("self::config"), "config");
        assert_eq!(strip_crate_qualifiers("super::parent"), "parent");
        assert_eq!(
            strip_crate_qualifiers("super::super::grandparent"),
            "grandparent"
        );
        assert_eq!(
            strip_crate_qualifiers("std::collections::HashMap"),
            "std::collections::HashMap",
        );
    }

    #[test]
    fn test_leaf_name() {
        assert_eq!(leaf_name("std::collections::HashMap"), "HashMap");
        assert_eq!(leaf_name("serde::Deserialize"), "Deserialize");
        assert_eq!(leaf_name("HashMap"), "HashMap");
    }

    #[test]
    fn test_leaf_name_with_rename() {
        // Returns the definition name, not the alias — symbol index is keyed by definitions.
        assert_eq!(leaf_name("foo::Bar as Baz"), "Bar");
    }

    #[test]
    fn test_resolve_basic() {
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/main.rs".to_string(),
            vec!["crate::config::Config".to_string()],
        );

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "Config".to_string(),
            vec![("src/config.rs".to_string(), "src_config_rs__Config__abcd1234".to_string())],
        );

        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["type"], "imports");
        assert_eq!(edges[0]["source_path"], "src/main.rs");
        assert_eq!(edges[0]["target_path"], "src/config.rs");
        assert_eq!(edges[0]["symbol_name"], "Config");
    }

    #[test]
    fn test_resolve_no_self_edge() {
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/lib.rs".to_string(),
            vec!["crate::MyStruct".to_string()],
        );

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "MyStruct".to_string(),
            vec![("src/lib.rs".to_string(), "src_lib_rs__MyStruct__abcd1234".to_string())],
        );

        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert!(edges.is_empty(), "should not create self-edges");
    }

    #[test]
    fn test_resolve_skips_glob() {
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/main.rs".to_string(),
            vec!["std::io::*".to_string()],
        );

        let symbol_index = HashMap::new();
        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_resolve_dedup() {
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/main.rs".to_string(),
            vec![
                "crate::config::Config".to_string(),
                "crate::config::Config".to_string(), // duplicate
            ],
        );

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "Config".to_string(),
            vec![("src/config.rs".to_string(), "src_config_rs__Config__abcd1234".to_string())],
        );

        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert_eq!(edges.len(), 1, "duplicate imports should be deduplicated");
    }

    #[test]
    fn test_resolve_prefers_path_match() {
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/main.rs".to_string(),
            vec!["crate::db::keys::file_key".to_string()],
        );

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "file_key".to_string(),
            vec![
                ("src/utils.rs".to_string(), "src_utils_rs__file_key__aaaa0000".to_string()),
                ("src/db/keys.rs".to_string(), "src_db_keys_rs__file_key__bbbb1111".to_string()),
            ],
        );

        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["target_path"], "src/db/keys.rs");
    }

    #[test]
    fn test_resolve_aliased_import() {
        // `use crate::config::Config as Cfg` should look up "Config", not "Cfg".
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/main.rs".to_string(),
            vec!["crate::config::Config as Cfg".to_string()],
        );

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "Config".to_string(),
            vec![("src/config.rs".to_string(), "src_config_rs__Config__abcd1234".to_string())],
        );

        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["symbol_name"], "Config");
        assert_eq!(edges[0]["target_path"], "src/config.rs");
    }

    #[test]
    fn test_resolve_no_substring_false_positive() {
        // "db" segment should NOT match "debug.rs" — only discrete path components.
        let mut rust_imports = HashMap::new();
        rust_imports.insert(
            "src/main.rs".to_string(),
            vec!["crate::db::Pool".to_string()],
        );

        let mut symbol_index = HashMap::new();
        symbol_index.insert(
            "Pool".to_string(),
            vec![
                ("src/debug.rs".to_string(), "src_debug_rs__Pool__aaaa0000".to_string()),
                ("src/db/pool.rs".to_string(), "src_db_pool_rs__Pool__bbbb1111".to_string()),
            ],
        );

        let edges = resolve_rust_imports(&rust_imports, &symbol_index);
        assert_eq!(edges.len(), 1);
        // Must pick src/db/pool.rs (segment match), not src/debug.rs (substring match).
        assert_eq!(edges[0]["target_path"], "src/db/pool.rs");
    }

    #[test]
    fn test_build_symbol_index_skips_imports() {
        let mut file_symbols = HashMap::new();
        file_symbols.insert(
            "src/lib.rs".to_string(),
            vec![
                Symbol {
                    name: "Config".to_string(),
                    kind: SymbolKind::Struct,
                    start_line: 10,
                    end_line: 20,
                    metadata: serde_json::Value::Null,
                },
                Symbol {
                    name: "std::collections::HashMap".to_string(),
                    kind: SymbolKind::Import,
                    start_line: 1,
                    end_line: 1,
                    metadata: serde_json::Value::Null,
                },
            ],
        );

        let index = build_symbol_index(&file_symbols);
        assert!(index.contains_key("Config"), "should index struct");
        assert!(
            !index.contains_key("std::collections::HashMap"),
            "should skip import symbols"
        );
    }
}
