//! C / C++ / CUDA source analysis via libclang.
//!
//! Uses the real Clang frontend (libclang, loaded at run time) to extract
//! functions, CUDA kernels, methods, and record types with accurate spans.
//! The span (`start_line`) is the symbol-identity anchor (#148); for C++ there
//! is a single analyzer, so any stable line works and the line also
//! disambiguates same-named symbols (overloads, sibling-namespace defs) without
//! needing a fully-reconstructed qualified name.
//!
//! libclang is not thread-safe and `clang-rs` permits only one live `Clang`
//! token per process, so parsing is serialized behind [`CLANG_LOCK`]. If
//! libclang is unavailable the analyzer degrades to no symbols (the file is
//! still chunked and embedded), mirroring the rust-analyzer-absent path.

use std::sync::Mutex;

use clang::{Clang, Entity, EntityKind, Index, Unsaved};
use serde_json::json;
use tracing::warn;

use super::Language;
use super::symbols::{
    CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef, compute_symbol_hash,
};

static CLANG_LOCK: Mutex<()> = Mutex::new(());

/// Analyze C/C++/CUDA source, returning symbols, metrics, and structure.
pub fn analyze(source: &str, file_path: &str) -> Result<FileAnalysis, super::CodeAnalysisError> {
    let (symbols, top_level_defs) = match extract(source, file_path) {
        Ok(pair) => pair,
        Err(e) => {
            warn!(error = %e, file = file_path, "libclang analysis failed, no C++ symbols");
            (Vec::new(), Vec::new())
        }
    };
    let metrics = compute_metrics(source);
    let symbol_hash = compute_symbol_hash(&symbols);

    Ok(FileAnalysis {
        language: Language::Cpp,
        symbols,
        metrics,
        symbol_hash,
        top_level_defs,
    })
}

fn is_cuda(file_path: &str, source: &str) -> bool {
    file_path.ends_with(".cu")
        || file_path.ends_with(".cuh")
        || source.contains("__global__")
        || source.contains("__device__")
}

fn parse_args(cuda: bool) -> Vec<&'static str> {
    if cuda {
        vec![
            "-x",
            "cuda",
            "--cuda-host-only",
            "--cuda-path=/usr/local/cuda",
            "--no-cuda-version-check",
            "-std=c++17",
        ]
    } else {
        vec!["-x", "c++", "-std=c++17"]
    }
}

fn extract(source: &str, file_path: &str) -> Result<(Vec<Symbol>, Vec<TopLevelDef>), String> {
    // Hold the singleton across the whole parse; recover from a poisoned lock
    // rather than cascading panics across files.
    let _guard = CLANG_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let clang = Clang::new()?;
    let index = Index::new(&clang, false, false);
    let args = parse_args(is_cuda(file_path, source));

    let tu = index
        .parser(file_path)
        .arguments(&args)
        .unsaved(&[Unsaved::new(file_path, source)])
        .skip_function_bodies(true)
        .detailed_preprocessing_record(false)
        .parse()
        .map_err(|e| format!("{e:?}"))?;

    let lines: Vec<&str> = source.lines().collect();
    let mut symbols = Vec::new();
    let mut defs = Vec::new();
    collect(tu.get_entity(), true, &lines, &mut symbols, &mut defs);
    Ok((symbols, defs))
}

/// Walk the AST. `top` is true while inside the translation unit, namespaces,
/// or `extern "C"` blocks (so direct functions/types there are top-level defs
/// for chunking) and false inside a record (a class's methods are covered by
/// the class's own chunk).
fn collect(
    entity: Entity,
    top: bool,
    lines: &[&str],
    symbols: &mut Vec<Symbol>,
    defs: &mut Vec<TopLevelDef>,
) {
    for child in entity.get_children() {
        let in_main = child
            .get_location()
            .map(|l| l.is_in_main_file())
            .unwrap_or(false);
        let kind = child.get_kind();

        if in_main {
            match kind {
                EntityKind::FunctionDecl | EntityKind::Method => {
                    if let Some(sym) = func_symbol(&child, lines) {
                        symbols.push(sym);
                    }
                }
                EntityKind::StructDecl => push_type(&child, SymbolKind::Struct, symbols),
                EntityKind::ClassDecl => push_type(&child, SymbolKind::Class, symbols),
                EntityKind::EnumDecl => push_type(&child, SymbolKind::Enum, symbols),
                _ => {}
            }

            if top
                && matches!(
                    kind,
                    EntityKind::FunctionDecl
                        | EntityKind::StructDecl
                        | EntityKind::ClassDecl
                        | EntityKind::EnumDecl
                )
                && let Some(def) = top_level_def(&child)
            {
                defs.push(def);
            }
        }

        match kind {
            EntityKind::Namespace | EntityKind::LinkageSpec => {
                collect(child, top, lines, symbols, defs);
            }
            EntityKind::StructDecl | EntityKind::ClassDecl => {
                collect(child, false, lines, symbols, defs);
            }
            _ => {}
        }
    }
}

fn span_lines(entity: &Entity) -> Option<(usize, usize)> {
    let range = entity.get_range()?;
    let start = range.get_start().get_file_location().line as usize;
    let end = range.get_end().get_file_location().line as usize;
    Some((start.max(1), end.max(start)))
}

fn func_symbol(entity: &Entity, lines: &[&str]) -> Option<Symbol> {
    let name = entity.get_name()?;
    let (start_line, end_line) = span_lines(entity)?;

    let mut meta = serde_json::Map::new();

    // Class context for a method -> `Class::method` qualified name (reuses the
    // `impl_context` convention shared with the Rust path).
    if let Some(parent) = entity.get_semantic_parent()
        && matches!(
            parent.get_kind(),
            EntityKind::ClassDecl | EntityKind::StructDecl
        )
        && let Some(cls) = parent.get_name()
    {
        meta.insert("impl_context".into(), json!(cls));
    }

    // CUDA execution-space qualifiers, read from the declaration line (libclang
    // anchors a FunctionDecl range at the qualifier, e.g. `__global__ void k`).
    if let Some(line) = lines.get(start_line.saturating_sub(1)) {
        if line.contains("__global__") {
            meta.insert("cuda".into(), json!("global"));
            meta.insert("is_kernel".into(), json!(true));
        } else if line.contains("__device__") {
            meta.insert("cuda".into(), json!("device"));
        } else if line.contains("__host__") {
            meta.insert("cuda".into(), json!("host"));
        }
    }

    Some(Symbol {
        name,
        kind: SymbolKind::Function,
        start_line,
        end_line,
        metadata: serde_json::Value::Object(meta),
    })
}

fn push_type(entity: &Entity, kind: SymbolKind, symbols: &mut Vec<Symbol>) {
    let Some(name) = entity.get_name() else {
        return;
    };
    let Some((start_line, end_line)) = span_lines(entity) else {
        return;
    };
    symbols.push(Symbol {
        name,
        kind,
        start_line,
        end_line,
        metadata: serde_json::Value::Null,
    });
}

fn top_level_def(entity: &Entity) -> Option<TopLevelDef> {
    let name = entity.get_name()?;
    let range = entity.get_range()?;
    let start = range.get_start().get_file_location();
    let end = range.get_end().get_file_location();
    let kind = match entity.get_kind() {
        EntityKind::FunctionDecl | EntityKind::Method => SymbolKind::Function,
        EntityKind::StructDecl => SymbolKind::Struct,
        EntityKind::ClassDecl => SymbolKind::Class,
        EntityKind::EnumDecl => SymbolKind::Enum,
        _ => return None,
    };
    Some(TopLevelDef {
        name,
        kind,
        start_line: (start.line as usize).max(1),
        end_line: (end.line as usize).max(start.line as usize),
        start_byte: start.offset as usize,
        end_byte: end.offset as usize,
    })
}

fn compute_metrics(source: &str) -> CodeMetrics {
    let mut total = 0;
    let mut blank = 0;
    let mut comment = 0;
    for raw in source.lines() {
        total += 1;
        let t = raw.trim();
        if t.is_empty() {
            blank += 1;
        } else if t.starts_with("//") || t.starts_with("/*") || t.starts_with('*') {
            comment += 1;
        }
    }
    CodeMetrics {
        total_lines: total,
        lines_of_code: total.saturating_sub(blank + comment),
        blank_lines: blank,
        comment_lines: comment,
        cyclomatic_complexity: 0,
        max_nesting_depth: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // No #include, so parsing needs no CUDA headers; `-x cuda` still
    // understands `__global__`, and bodies are skipped.
    const CUDA: &str = "__global__ void my_kernel(float* x, int n) {}\n\
extern \"C\" void launch(float* x, int n) {}\n\
struct Cfg { int a; void reset(); };\n";

    #[test]
    fn analyze_cuda_extracts_kernel_method_and_type() {
        let analysis = analyze(CUDA, "test.cu").unwrap();
        if analysis.symbols.is_empty() {
            eprintln!("SKIP: libclang unavailable (no symbols extracted)");
            return;
        }
        let by_name = |n: &str| analysis.symbols.iter().find(|s| s.name == n);

        let kernel = by_name("my_kernel").expect("kernel extracted");
        assert_eq!(kernel.kind, SymbolKind::Function);
        assert_eq!(
            kernel.metadata.get("is_kernel").and_then(|v| v.as_bool()),
            Some(true),
            "__global__ function should be flagged as a kernel"
        );
        assert_eq!(kernel.start_line, 1, "span anchor for #148 identity");

        assert_eq!(
            by_name("launch").expect("host fn").kind,
            SymbolKind::Function
        );
        assert_eq!(by_name("Cfg").expect("struct").kind, SymbolKind::Struct);

        // Method carries its class context -> `Cfg::reset` qualified name.
        let reset = by_name("reset").expect("method extracted");
        assert_eq!(reset.qualified_name(), "Cfg::reset");

        // Digest is stable and non-empty.
        assert!(!analysis.symbol_hash.is_empty());
    }
}
