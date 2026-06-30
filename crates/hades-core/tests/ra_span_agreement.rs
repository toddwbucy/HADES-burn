//! Probe for issue #148: does `syn`'s identifier-span line agree with
//! rust-analyzer's `selectionRange` line (after 0-based -> 1-based) for the same
//! symbol?
//!
//! The span-based symbol-identity contract depends on this agreement. If it
//! holds, anchoring the symbol key on the name-token line both fixes the
//! sibling-module collision (#148) and preserves syn/RA overwrite-not-duplicate
//! (the two analyzers still land on the same key for the same symbol).
//!
//! Requires rust-analyzer on PATH. SKIPS (does not fail) if it is unavailable.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use hades_core::code::rust_analyzer::RustAnalyzerSession;
use serde_json::Value;

// Line numbers matter. The identifier of `Top` is on line 2 even though the
// item carries an attribute on line 1 -- that is the case that breaks
// item-start anchoring but not name-token anchoring. `Cfg` and `build` each
// appear twice in sibling modules (the #148 collision): same qualified name,
// distinct name-token lines.
const FIXTURE: &str = r#"#[derive(Debug)]
pub struct Top {
    pub n: usize,
}

impl Top {
    pub fn new() -> Self {
        Self { n: 0 }
    }
}

pub fn free() -> u32 {
    1
}

mod a {
    pub struct Cfg;
    impl Cfg {
        pub fn build() {}
    }
}

mod b {
    pub struct Cfg;
    impl Cfg {
        pub fn build() {}
    }
}
"#;

/// syn side: name -> set of identifier-span lines (1-based).
fn syn_ident_lines(src: &str) -> BTreeMap<String, BTreeSet<usize>> {
    let file = syn::parse_file(src).expect("fixture parses");
    let mut map: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for item in &file.items {
        walk_item(item, &mut map);
    }
    map
}

fn record(map: &mut BTreeMap<String, BTreeSet<usize>>, name: String, span: proc_macro2::Span) {
    map.entry(name).or_default().insert(span.start().line);
}

fn walk_item(item: &syn::Item, map: &mut BTreeMap<String, BTreeSet<usize>>) {
    match item {
        syn::Item::Fn(f) => record(map, f.sig.ident.to_string(), f.sig.ident.span()),
        syn::Item::Struct(s) => record(map, s.ident.to_string(), s.ident.span()),
        syn::Item::Enum(e) => record(map, e.ident.to_string(), e.ident.span()),
        syn::Item::Trait(t) => record(map, t.ident.to_string(), t.ident.span()),
        syn::Item::Mod(m) => {
            record(map, m.ident.to_string(), m.ident.span());
            if let Some((_, items)) = &m.content {
                for it in items {
                    walk_item(it, map);
                }
            }
        }
        syn::Item::Impl(imp) => {
            for it in &imp.items {
                if let syn::ImplItem::Fn(method) = it {
                    record(map, method.sig.ident.to_string(), method.sig.ident.span());
                }
            }
        }
        _ => {}
    }
}

/// RA side: name -> set of selectionRange start lines (0-based, as reported).
fn ra_sel_lines(symbols: &[Value]) -> BTreeMap<String, BTreeSet<u64>> {
    fn walk(node: &Value, map: &mut BTreeMap<String, BTreeSet<u64>>) {
        if let (Some(name), Some(line)) = (
            node.get("name").and_then(Value::as_str),
            node.pointer("/selectionRange/start/line").and_then(Value::as_u64),
        ) {
            map.entry(name.to_string()).or_default().insert(line);
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for c in children {
                walk(c, map);
            }
        }
    }
    let mut map = BTreeMap::new();
    for s in symbols {
        walk(s, &mut map);
    }
    map
}

#[tokio::test]
async fn syn_ident_line_agrees_with_ra_selection_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"probe148\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
    )
    .unwrap();
    std::fs::write(root.join("src/lib.rs"), FIXTURE).unwrap();

    // Honor an explicit binary override (HADES_RA_BIN) -- the default
    // `rust-analyzer` on PATH may be a rustup proxy resolving to a toolchain
    // that lacks the component.
    let ra_bin = std::env::var("HADES_RA_BIN").ok();
    let start = match ra_bin.as_deref() {
        Some(bin) => RustAnalyzerSession::start_with_options(root, Some(bin), 60).await,
        None => RustAnalyzerSession::start(root).await,
    };
    let session = match start {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: rust-analyzer unavailable: {e}");
            return;
        }
    };

    let symbols = session
        .document_symbols(Path::new("src/lib.rs"))
        .await
        .expect("documentSymbol request");
    let _ = session.shutdown().await;

    let syn_map = syn_ident_lines(FIXTURE);
    let ra_map = ra_sel_lines(&symbols);

    assert!(
        !ra_map.is_empty(),
        "rust-analyzer returned no symbols (indexing not ready?)"
    );

    // The load-bearing assumption: for every name present in both analyzers,
    // syn ident lines (1-based) == RA selectionRange lines (0-based) + 1.
    let mut compared = 0usize;
    for (name, syn_lines) in &syn_map {
        if let Some(ra_lines) = ra_map.get(name) {
            let ra_norm: BTreeSet<usize> = ra_lines.iter().map(|l| *l as usize + 1).collect();
            assert_eq!(
                *syn_lines, ra_norm,
                "name-token line disagreement for `{name}`: syn={syn_lines:?} ra+1={ra_norm:?}"
            );
            compared += 1;
        }
    }
    assert!(
        compared >= 5,
        "expected to compare several symbols, only compared {compared}"
    );

    // #148 core: `build` collides on qualified_name (Cfg::build in both modules)
    // but has two DISTINCT name-token lines, so a span-anchored key disambiguates.
    let build = syn_map.get("build").expect("`build` present");
    assert_eq!(
        build.len(),
        2,
        "expected two sibling-module `build` methods at distinct lines: {build:?}"
    );
    println!("OK: compared {compared} symbols; syn ident line == RA selectionRange line + 1");
    println!("    sibling `build` name-token lines (distinct): {build:?}");

    // THE invariant the symbol key relies on: `keys::symbol_key` is fed
    // `Symbol::start_line` (syn item-span, 1-based) on the syn side and
    // `start_line + 1` (RA `range`, 0-based) on the rust-analyzer side. Those
    // must match for the same symbol, or RA enrichment duplicates the vertex
    // instead of overwriting it. Assert it directly.
    let syn_item = syn_item_lines(FIXTURE);
    let ra_range = ra_range_lines(&symbols);
    let mut item_compared = 0usize;
    for (name, syn_lines) in &syn_item {
        if let Some(r) = ra_range.get(name) {
            let ra_norm: BTreeSet<usize> = r.iter().map(|l| *l as usize + 1).collect();
            assert_eq!(
                *syn_lines, ra_norm,
                "item-span disagreement for `{name}`: syn={syn_lines:?} ra_range+1={ra_norm:?}"
            );
            item_compared += 1;
        }
    }
    assert!(item_compared >= 5, "compared only {item_compared} item spans");
    println!("OK: syn item-span line == RA range line + 1 for {item_compared} symbols");
}

fn syn_item_lines(src: &str) -> BTreeMap<String, BTreeSet<usize>> {
    use syn::spanned::Spanned;
    fn walk(item: &syn::Item, map: &mut BTreeMap<String, BTreeSet<usize>>) {
        let (name, span) = match item {
            syn::Item::Fn(f) => (f.sig.ident.to_string(), f.span()),
            syn::Item::Struct(s) => (s.ident.to_string(), s.span()),
            syn::Item::Mod(m) => {
                if let Some((_, items)) = &m.content {
                    for it in items {
                        walk(it, map);
                    }
                }
                (m.ident.to_string(), m.span())
            }
            syn::Item::Impl(imp) => {
                for it in &imp.items {
                    if let syn::ImplItem::Fn(method) = it {
                        map.entry(method.sig.ident.to_string())
                            .or_default()
                            .insert(method.span().start().line);
                    }
                }
                return;
            }
            _ => return,
        };
        map.entry(name).or_default().insert(span.start().line);
    }
    let file = syn::parse_file(src).expect("parses");
    let mut map = BTreeMap::new();
    for item in &file.items {
        walk(item, &mut map);
    }
    map
}

fn ra_range_lines(symbols: &[Value]) -> BTreeMap<String, BTreeSet<u64>> {
    fn walk(node: &Value, map: &mut BTreeMap<String, BTreeSet<u64>>) {
        if let (Some(name), Some(line)) = (
            node.get("name").and_then(Value::as_str),
            node.pointer("/range/start/line").and_then(Value::as_u64),
        ) {
            map.entry(name.to_string()).or_default().insert(line);
        }
        if let Some(children) = node.get("children").and_then(Value::as_array) {
            for c in children {
                walk(c, map);
            }
        }
    }
    let mut map = BTreeMap::new();
    for s in symbols {
        walk(s, &mut map);
    }
    map
}
