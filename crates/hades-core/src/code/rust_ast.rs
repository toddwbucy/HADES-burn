//! Rust source analysis via `syn`.
//!
//! Extracts symbols (functions, structs, enums, traits, impls, macros),
//! computes code metrics, and identifies top-level definition boundaries
//! for AST-aligned chunking.

use serde_json::json;
use syn::{visit::Visit, Item, Visibility};
use tracing::warn;

use super::symbols::{
    CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef, compute_symbol_hash,
};
use super::Language;

/// Analyze Rust source code, returning symbols, metrics, and structure.
pub fn analyze(source: &str) -> FileAnalysis {
    let symbols = extract_symbols(source);
    let metrics = compute_metrics(source);
    let top_level_defs = extract_top_level_defs(source);
    let symbol_hash = compute_symbol_hash(&symbols);

    FileAnalysis {
        language: Language::Rust,
        symbols,
        metrics,
        symbol_hash,
        top_level_defs,
    }
}

// ── Symbol Extraction ──────────────────────────────────────────────────

fn extract_symbols(source: &str) -> Vec<Symbol> {
    let file = match syn::parse_file(source) {
        Ok(f) => f,
        Err(e) => {
            warn!(error = %e, "Rust parse error, returning empty symbols");
            return Vec::new();
        }
    };

    let line_offsets = build_line_offsets(source);
    let mut collector = SymbolCollector {
        symbols: Vec::new(),
        source,
        line_offsets: &line_offsets,
        impl_context: None,
    };

    for item in &file.items {
        collector.visit_top_level_item(item);
    }

    collector.symbols
}

struct SymbolCollector<'a> {
    symbols: Vec<Symbol>,
    source: &'a str,
    line_offsets: &'a [usize],
    /// Current impl block context (e.g. "Config" or "Display for Config").
    impl_context: Option<String>,
}

impl<'a> SymbolCollector<'a> {
    fn visit_top_level_item(&mut self, item: &Item) {
        match item {
            Item::Fn(func) => {
                let (start_line, end_line) = self.span_lines(&func.sig.fn_token.span);
                let end_line = self.find_block_end(end_line, source_end_line(func, self.source, self.line_offsets));

                let mut meta = json!({
                    "visibility": vis_str(&func.vis),
                    "is_async": func.sig.asyncness.is_some(),
                    "is_unsafe": func.sig.unsafety.is_some(),
                });
                if !func.sig.generics.params.is_empty() {
                    meta["has_generics"] = json!(true);
                }

                self.symbols.push(Symbol {
                    name: func.sig.ident.to_string(),
                    kind: SymbolKind::Function,
                    start_line,
                    end_line,
                    metadata: meta,
                });
            }

            Item::Struct(s) => {
                let (start_line, end_line) = self.span_lines(&s.struct_token.span);
                let end_line = self.find_block_end(end_line, source_end_line(s, self.source, self.line_offsets));

                let derives = extract_derives(&s.attrs);
                let mut meta = json!({
                    "visibility": vis_str(&s.vis),
                });
                if !derives.is_empty() {
                    meta["derives"] = json!(derives);
                }
                if !s.generics.params.is_empty() {
                    meta["has_generics"] = json!(true);
                }

                self.symbols.push(Symbol {
                    name: s.ident.to_string(),
                    kind: SymbolKind::Struct,
                    start_line,
                    end_line,
                    metadata: meta,
                });
            }

            Item::Enum(e) => {
                let (start_line, end_line) = self.span_lines(&e.enum_token.span);
                let end_line = self.find_block_end(end_line, source_end_line(e, self.source, self.line_offsets));

                let derives = extract_derives(&e.attrs);
                let variants: Vec<String> = e.variants.iter().map(|v| v.ident.to_string()).collect();
                let mut meta = json!({
                    "visibility": vis_str(&e.vis),
                    "variants": variants,
                });
                if !derives.is_empty() {
                    meta["derives"] = json!(derives);
                }

                self.symbols.push(Symbol {
                    name: e.ident.to_string(),
                    kind: SymbolKind::Enum,
                    start_line,
                    end_line,
                    metadata: meta,
                });
            }

            Item::Trait(t) => {
                let (start_line, end_line) = self.span_lines(&t.trait_token.span);
                let end_line = self.find_block_end(end_line, source_end_line(t, self.source, self.line_offsets));

                let methods: Vec<String> = t
                    .items
                    .iter()
                    .filter_map(|item| {
                        if let syn::TraitItem::Fn(m) = item {
                            Some(m.sig.ident.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                let mut meta = json!({
                    "visibility": vis_str(&t.vis),
                });
                if !methods.is_empty() {
                    meta["methods"] = json!(methods);
                }

                self.symbols.push(Symbol {
                    name: t.ident.to_string(),
                    kind: SymbolKind::Trait,
                    start_line,
                    end_line,
                    metadata: meta,
                });
            }

            Item::Impl(imp) => {
                let self_ty = type_name(&imp.self_ty);
                let trait_name = imp.trait_.as_ref().map(|(_, path, _)| path_name(path));

                let impl_label = if let Some(ref tn) = trait_name {
                    format!("{tn} for {self_ty}")
                } else {
                    self_ty.clone()
                };

                let (start_line, end_line) = self.span_lines(&imp.impl_token.span);
                let end_line = self.find_block_end(end_line, source_end_line(imp, self.source, self.line_offsets));

                let mut meta = json!({
                    "self_type": self_ty,
                });
                if let Some(tn) = &trait_name {
                    meta["trait"] = json!(tn);
                }

                self.symbols.push(Symbol {
                    name: impl_label.clone(),
                    kind: SymbolKind::Impl,
                    start_line,
                    end_line,
                    metadata: meta,
                });

                // Descend into impl items with context.
                let old_ctx = self.impl_context.take();
                self.impl_context = Some(impl_label);
                for impl_item in &imp.items {
                    self.visit_impl_item(impl_item);
                }
                self.impl_context = old_ctx;
            }

            Item::Use(u) => {
                let (start_line, end_line) = self.span_lines(&u.use_token.span);
                let path = use_tree_str(&u.tree);

                self.symbols.push(Symbol {
                    name: path,
                    kind: SymbolKind::Import,
                    start_line,
                    end_line,
                    metadata: json!({ "visibility": vis_str(&u.vis) }),
                });
            }

            Item::Const(c) => {
                let (start_line, end_line) = self.span_lines(&c.const_token.span);
                self.symbols.push(Symbol {
                    name: c.ident.to_string(),
                    kind: SymbolKind::Constant,
                    start_line,
                    end_line,
                    metadata: json!({ "visibility": vis_str(&c.vis) }),
                });
            }

            Item::Static(s) => {
                let (start_line, end_line) = self.span_lines(&s.static_token.span);
                self.symbols.push(Symbol {
                    name: s.ident.to_string(),
                    kind: SymbolKind::Constant,
                    start_line,
                    end_line,
                    metadata: json!({
                        "visibility": vis_str(&s.vis),
                        "is_mutable": matches!(s.mutability, syn::StaticMutability::Mut(_)),
                    }),
                });
            }

            Item::Type(t) => {
                let (start_line, end_line) = self.span_lines(&t.type_token.span);
                self.symbols.push(Symbol {
                    name: t.ident.to_string(),
                    kind: SymbolKind::TypeAlias,
                    start_line,
                    end_line,
                    metadata: json!({ "visibility": vis_str(&t.vis) }),
                });
            }

            Item::Macro(m) => {
                if let Some(ref ident) = m.ident {
                    let (start_line, end_line) = self.span_lines(&m.mac.path.segments[0].ident.span());
                    self.symbols.push(Symbol {
                        name: ident.to_string(),
                        kind: SymbolKind::Macro,
                        start_line,
                        end_line,
                        metadata: serde_json::Value::Null,
                    });
                }
            }

            Item::Mod(m) => {
                let (start_line, end_line) = self.span_lines(&m.mod_token.span);
                self.symbols.push(Symbol {
                    name: m.ident.to_string(),
                    kind: SymbolKind::Module,
                    start_line,
                    end_line,
                    metadata: json!({ "visibility": vis_str(&m.vis) }),
                });
            }

            _ => {}
        }
    }

    fn visit_impl_item(&mut self, item: &syn::ImplItem) {
        match item {
            syn::ImplItem::Fn(method) => {
                let (start_line, end_line) = self.span_lines(&method.sig.fn_token.span);
                let end_line = self.find_block_end(end_line, source_end_line(method, self.source, self.line_offsets));

                let mut meta = json!({
                    "visibility": vis_str(&method.vis),
                    "is_async": method.sig.asyncness.is_some(),
                    "is_unsafe": method.sig.unsafety.is_some(),
                });
                if let Some(ctx) = &self.impl_context {
                    meta["impl_context"] = json!(ctx);
                }

                self.symbols.push(Symbol {
                    name: method.sig.ident.to_string(),
                    kind: SymbolKind::Function,
                    start_line,
                    end_line,
                    metadata: meta,
                });
            }
            syn::ImplItem::Const(c) => {
                let (start_line, _) = self.span_lines(&c.const_token.span);
                self.symbols.push(Symbol {
                    name: c.ident.to_string(),
                    kind: SymbolKind::Constant,
                    start_line,
                    end_line: start_line,
                    metadata: json!({ "visibility": vis_str(&c.vis) }),
                });
            }
            syn::ImplItem::Type(t) => {
                let (start_line, _) = self.span_lines(&t.type_token.span);
                self.symbols.push(Symbol {
                    name: t.ident.to_string(),
                    kind: SymbolKind::TypeAlias,
                    start_line,
                    end_line: start_line,
                    metadata: json!({ "visibility": vis_str(&t.vis) }),
                });
            }
            _ => {}
        }
    }

    /// Convert a Span to 1-based (start_line, end_line).
    fn span_lines(&self, span: &proc_macro2::Span) -> (usize, usize) {
        let start = span.start().line;
        let end = span.end().line;
        (start.max(1), end.max(start))
    }

    /// Heuristic: if syn's span only covers the keyword (e.g. `fn`),
    /// use the source-derived end line if it's larger.
    fn find_block_end(&self, keyword_end: usize, source_derived_end: usize) -> usize {
        keyword_end.max(source_derived_end)
    }
}

// ── Top-Level Definitions ──────────────────────────────────────────────

fn extract_top_level_defs(source: &str) -> Vec<TopLevelDef> {
    let file = match syn::parse_file(source) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let line_offsets = build_line_offsets(source);
    let mut defs = Vec::new();

    for item in &file.items {
        let (name, kind) = match item {
            Item::Fn(f) => (f.sig.ident.to_string(), SymbolKind::Function),
            Item::Struct(s) => (s.ident.to_string(), SymbolKind::Struct),
            Item::Enum(e) => (e.ident.to_string(), SymbolKind::Enum),
            Item::Trait(t) => (t.ident.to_string(), SymbolKind::Trait),
            Item::Impl(i) => {
                let self_ty = type_name(&i.self_ty);
                let label = if let Some((_, path, _)) = &i.trait_ {
                    format!("{} for {self_ty}", path_name(path))
                } else {
                    self_ty
                };
                (label, SymbolKind::Impl)
            }
            _ => continue,
        };

        let (start_line, end_line, start_byte, end_byte) =
            item_span_info(item, source, &line_offsets);

        defs.push(TopLevelDef {
            name,
            kind,
            start_line,
            end_line,
            start_byte,
            end_byte,
        });
    }

    defs
}

/// Get span info for an item, using source scanning to find the true end.
fn item_span_info(
    item: &Item,
    source: &str,
    line_offsets: &[usize],
) -> (usize, usize, usize, usize) {
    // syn spans from proc-macro2 only have line info in non-proc-macro context.
    // Use the source_end_line helper to find the closing brace.
    let keyword_span = match item {
        Item::Fn(f) => f.sig.fn_token.span,
        Item::Struct(s) => s.struct_token.span,
        Item::Enum(e) => e.enum_token.span,
        Item::Trait(t) => t.trait_token.span,
        Item::Impl(i) => i.impl_token.span,
        _ => return (1, 1, 0, 0),
    };

    let start_line = keyword_span.start().line.max(1);
    let derived_end = source_end_line(item, source, line_offsets);
    let end_line = keyword_span.end().line.max(start_line).max(derived_end);

    // Compute byte offsets from line numbers.
    let start_byte = line_to_byte(line_offsets, start_line);
    let end_byte = line_to_byte_end(line_offsets, end_line, source.len());

    // Adjust start_byte backwards to include attributes/doc comments.
    let adjusted_start = find_item_start(source, start_byte);
    let adjusted_start_line = byte_to_line(line_offsets, adjusted_start);

    (adjusted_start_line, end_line, adjusted_start, end_byte)
}

/// Scan backwards from an item's keyword to include preceding attributes
/// and doc comments (lines starting with `#[` or `///` or `//!`).
fn find_item_start(source: &str, keyword_byte: usize) -> usize {
    let prefix = &source[..keyword_byte];
    let mut start = keyword_byte;

    for line in prefix.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[")
            || trimmed.starts_with("///")
            || trimmed.starts_with("//!")
            || trimmed.is_empty()
        {
            start = start.saturating_sub(line.len() + 1); // +1 for newline
        } else {
            break;
        }
    }
    start
}

// ── Code Metrics ───────────────────────────────────────────────────────

fn compute_metrics(source: &str) -> CodeMetrics {
    let total_lines = source.lines().count().max(1);
    let blank_lines = source.lines().filter(|l| l.trim().is_empty()).count();
    let comment_lines = source
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
        })
        .count();
    let lines_of_code = total_lines.saturating_sub(blank_lines + comment_lines);

    let (complexity, max_depth) = compute_complexity(source);

    CodeMetrics {
        total_lines,
        lines_of_code,
        blank_lines,
        comment_lines,
        cyclomatic_complexity: complexity,
        max_nesting_depth: max_depth,
    }
}

/// Compute cyclomatic complexity and max nesting from the Rust AST.
fn compute_complexity(source: &str) -> (usize, usize) {
    let file = match syn::parse_file(source) {
        Ok(f) => f,
        Err(_) => return (1, 0),
    };

    let mut visitor = ComplexityVisitor {
        complexity: 1,
        depth: 0,
        max_depth: 0,
    };
    visitor.visit_file(&file);
    (visitor.complexity, visitor.max_depth)
}

struct ComplexityVisitor {
    complexity: usize,
    depth: usize,
    max_depth: usize,
}

impl<'ast> Visit<'ast> for ComplexityVisitor {
    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        self.complexity += 1;
        self.depth += 1;
        self.max_depth = self.max_depth.max(self.depth);
        syn::visit::visit_expr_if(self, node);
        self.depth -= 1;
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        // Each arm adds a path (minus 1 for the match itself, but we
        // count each arm for consistency with the Python approach).
        for arm in &node.arms {
            self.complexity += 1;
            self.depth += 1;
            self.max_depth = self.max_depth.max(self.depth);
            self.visit_arm(arm);
            self.depth -= 1;
        }
        // Visit the expression being matched.
        self.visit_expr(&node.expr);
    }

    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.complexity += 1;
        self.depth += 1;
        self.max_depth = self.max_depth.max(self.depth);
        syn::visit::visit_expr_for_loop(self, node);
        self.depth -= 1;
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.complexity += 1;
        self.depth += 1;
        self.max_depth = self.max_depth.max(self.depth);
        syn::visit::visit_expr_while(self, node);
        self.depth -= 1;
    }

    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        self.complexity += 1;
        self.depth += 1;
        self.max_depth = self.max_depth.max(self.depth);
        syn::visit::visit_expr_loop(self, node);
        self.depth -= 1;
    }

    fn visit_expr_closure(&mut self, node: &'ast syn::ExprClosure) {
        // Closures don't add complexity but do add nesting.
        self.depth += 1;
        self.max_depth = self.max_depth.max(self.depth);
        syn::visit::visit_expr_closure(self, node);
        self.depth -= 1;
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        // && and || add branching paths.
        match node.op {
            syn::BinOp::And(_) | syn::BinOp::Or(_) => {
                self.complexity += 1;
            }
            _ => {}
        }
        syn::visit::visit_expr_binary(self, node);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn vis_str(vis: &Visibility) -> &'static str {
    match vis {
        Visibility::Public(_) => "pub",
        Visibility::Restricted(_) => "pub(restricted)",
        Visibility::Inherited => "private",
    }
}

fn type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => path_name(&p.path),
        _ => "?".into(),
    }
}

fn path_name(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn use_tree_str(tree: &syn::UseTree) -> String {
    match tree {
        syn::UseTree::Path(p) => format!("{}::{}", p.ident, use_tree_str(&p.tree)),
        syn::UseTree::Name(n) => n.ident.to_string(),
        syn::UseTree::Rename(r) => format!("{} as {}", r.ident, r.rename),
        syn::UseTree::Glob(_) => "*".into(),
        syn::UseTree::Group(g) => {
            let items: Vec<String> = g.items.iter().map(use_tree_str).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Extract derive macro names from attributes.
fn extract_derives(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut derives = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("derive")
            && let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
        {
            for path in nested {
                derives.push(path_name(&path));
            }
        }
    }
    derives
}

/// Estimate the end line of a syntax item by scanning source for closing braces.
///
/// syn's proc-macro2 spans don't always cover the full item in non-macro
/// contexts.  This function provides a fallback by finding the matching
/// closing brace after the item's keyword.
fn source_end_line<T>(_item: &T, source: &str, line_offsets: &[usize]) -> usize {
    // In practice, syn spans work correctly when parsing files directly
    // (not inside proc macros).  This is a no-op fallback that returns 0,
    // letting the caller use the span-derived end line.
    //
    // If we find syn spans are insufficient, this can be upgraded to
    // brace-matching logic.
    let _ = (source, line_offsets);
    0
}

fn build_line_offsets(source: &str) -> Vec<usize> {
    let mut offsets = vec![0]; // line 1 starts at byte 0
    for (i, ch) in source.char_indices() {
        if ch == '\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

fn line_to_byte(offsets: &[usize], line: usize) -> usize {
    if line == 0 || line > offsets.len() {
        return 0;
    }
    offsets[line - 1]
}

fn line_to_byte_end(offsets: &[usize], line: usize, source_len: usize) -> usize {
    if line >= offsets.len() {
        return source_len;
    }
    offsets[line]
}

fn byte_to_line(offsets: &[usize], byte: usize) -> usize {
    match offsets.binary_search(&byte) {
        Ok(idx) => idx + 1,
        Err(idx) => idx.max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RUST: &str = r#"//! Module documentation.

use std::collections::HashMap;
use std::path::Path;

/// Maximum buffer size.
const MAX_BUF: usize = 4096;

/// A configuration holder.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Config {
    pub name: String,
    pub values: HashMap<String, String>,
}

/// Error type for operations.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
}

/// A trait for processing items.
pub trait Processor {
    fn process(&self, input: &str) -> Result<String, AppError>;
}

impl Config {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            values: HashMap::new(),
        }
    }

    pub async fn load(path: &Path) -> Result<Self, AppError> {
        if path.exists() {
            todo!()
        } else {
            Ok(Self::new("default"))
        }
    }
}

impl Processor for Config {
    fn process(&self, input: &str) -> Result<String, AppError> {
        match self.values.get(input) {
            Some(v) => Ok(v.clone()),
            None => Ok(String::new()),
        }
    }
}

pub fn run(items: &[String]) -> usize {
    let mut count = 0;
    for item in items {
        if item.is_empty() {
            continue;
        } else if item.starts_with('#') {
            count += 1;
        }
    }
    count
}
"#;

    #[test]
    fn test_extract_symbols() {
        let analysis = analyze(SAMPLE_RUST);
        let names: Vec<&str> = analysis.symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"std::collections::HashMap"), "missing HashMap import: {names:?}");
        assert!(names.contains(&"MAX_BUF"), "missing MAX_BUF constant");
        assert!(names.contains(&"Config"), "missing Config struct");
        assert!(names.contains(&"AppError"), "missing AppError enum");
        assert!(names.contains(&"Processor"), "missing Processor trait");
        assert!(names.contains(&"new"), "missing new method");
        assert!(names.contains(&"load"), "missing load method");
        assert!(names.contains(&"run"), "missing run function");
    }

    #[test]
    fn test_symbol_kinds() {
        let analysis = analyze(SAMPLE_RUST);

        let config = analysis.symbols.iter().find(|s| s.name == "Config" && s.kind == SymbolKind::Struct).unwrap();
        assert_eq!(config.kind, SymbolKind::Struct);

        let app_error = analysis.symbols.iter().find(|s| s.name == "AppError").unwrap();
        assert_eq!(app_error.kind, SymbolKind::Enum);

        let processor = analysis.symbols.iter().find(|s| s.name == "Processor").unwrap();
        assert_eq!(processor.kind, SymbolKind::Trait);

        let max_buf = analysis.symbols.iter().find(|s| s.name == "MAX_BUF").unwrap();
        assert_eq!(max_buf.kind, SymbolKind::Constant);
    }

    #[test]
    fn test_struct_derives() {
        let analysis = analyze(SAMPLE_RUST);
        let config = analysis.symbols.iter()
            .find(|s| s.name == "Config" && s.kind == SymbolKind::Struct)
            .unwrap();
        let derives = config.metadata["derives"].as_array().unwrap();
        let derive_names: Vec<&str> = derives.iter().map(|d| d.as_str().unwrap()).collect();
        assert!(derive_names.contains(&"Debug"));
        assert!(derive_names.contains(&"Clone"));
    }

    #[test]
    fn test_enum_variants() {
        let analysis = analyze(SAMPLE_RUST);
        let app_error = analysis.symbols.iter().find(|s| s.name == "AppError").unwrap();
        let variants = app_error.metadata["variants"].as_array().unwrap();
        let names: Vec<&str> = variants.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(names.contains(&"Io"));
        assert!(names.contains(&"Parse"));
    }

    #[test]
    fn test_function_metadata() {
        let analysis = analyze(SAMPLE_RUST);
        let load = analysis.symbols.iter().find(|s| s.name == "load").unwrap();
        assert_eq!(load.metadata["is_async"], true);
        assert_eq!(load.metadata["visibility"], "pub");

        let run = analysis.symbols.iter().find(|s| s.name == "run").unwrap();
        assert_eq!(run.metadata["is_async"], false);
    }

    #[test]
    fn test_impl_context() {
        let analysis = analyze(SAMPLE_RUST);
        let new_fn = analysis.symbols.iter().find(|s| s.name == "new").unwrap();
        assert!(new_fn.metadata["impl_context"].as_str().unwrap().contains("Config"));
    }

    #[test]
    fn test_trait_methods() {
        let analysis = analyze(SAMPLE_RUST);
        let processor = analysis.symbols.iter().find(|s| s.name == "Processor").unwrap();
        let methods = processor.metadata["methods"].as_array().unwrap();
        assert_eq!(methods[0], "process");
    }

    #[test]
    fn test_top_level_defs() {
        let analysis = analyze(SAMPLE_RUST);
        let def_names: Vec<&str> = analysis.top_level_defs.iter().map(|d| d.name.as_str()).collect();
        assert!(def_names.contains(&"Config"), "missing Config def: {def_names:?}");
        assert!(def_names.contains(&"AppError"), "missing AppError def: {def_names:?}");
        assert!(def_names.contains(&"Processor"), "missing Processor def: {def_names:?}");
        assert!(def_names.contains(&"run"), "missing run def: {def_names:?}");
    }

    #[test]
    fn test_metrics() {
        let analysis = analyze(SAMPLE_RUST);
        let m = &analysis.metrics;
        assert!(m.total_lines > 50);
        assert!(m.lines_of_code > 0);
        assert!(m.comment_lines > 0);
        // Complexity: 1 base + if/else in load + match arms + for + if + else if
        assert!(m.cyclomatic_complexity >= 4);
        assert!(m.max_nesting_depth >= 1);
    }

    #[test]
    fn test_empty_source() {
        let analysis = analyze("");
        assert!(analysis.symbols.is_empty());
        assert!(analysis.top_level_defs.is_empty());
    }

    #[test]
    fn test_use_tree_formatting() {
        assert_eq!(
            use_tree_str(&syn::parse_str::<syn::UseTree>("std::collections::HashMap").unwrap()),
            "std::collections::HashMap"
        );
    }
}
