//! Python source analysis via `rustpython-parser`.
//!
//! Extracts symbols (functions, classes, imports, variables),
//! computes code metrics, and identifies top-level definition
//! boundaries for AST-aligned chunking.
//!
//! **API note**: rustpython-parser 0.4 uses `TextRange` with byte offsets
//! (not line/column). Async functions are a separate `Stmt::AsyncFunctionDef`
//! variant. Class bases are in `.bases`, not `.arguments`.

use rustpython_parser::{self as parser, ast};
use serde_json::json;
use tracing::warn;

use super::symbols::{
    CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef, compute_symbol_hash,
};
use super::Language;

/// Analyze Python source code, returning symbols, metrics, and structure.
pub fn analyze(source: &str) -> Result<FileAnalysis, super::CodeAnalysisError> {
    let line_offsets = build_line_offsets(source);

    // Parse once — share the AST across all extraction phases.
    let module = parse_module(source);

    let symbols = match &module {
        Some(m) => extract_symbols_from_module(m, &line_offsets),
        None => Vec::new(),
    };
    let metrics = match &module {
        Some(m) => compute_metrics_from_module(source, m),
        None => compute_metrics_fallback(source),
    };
    let top_level_defs = match &module {
        Some(m) => extract_top_level_defs_from_module(m, &line_offsets),
        None => Vec::new(),
    };
    let symbol_hash = compute_symbol_hash(&symbols);

    Ok(FileAnalysis {
        language: Language::Python,
        symbols,
        metrics,
        symbol_hash,
        top_level_defs,
    })
}

/// Parse Python source into a Module. Returns None on parse error.
fn parse_module(source: &str) -> Option<ast::ModModule> {
    let parsed = match parser::parse(source, parser::Mode::Module, "<input>") {
        Ok(parsed) => parsed,
        Err(e) => {
            warn!(error = %e, "Python parse error");
            return None;
        }
    };
    match parsed {
        ast::Mod::Module(m) => Some(m),
        _ => None,
    }
}

// ── Symbol Extraction ──────────────────────────────────────────────────

fn extract_symbols_from_module(module: &ast::ModModule, offsets: &[usize]) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    for stmt in &module.body {
        extract_stmt_symbols(stmt, &mut symbols, &module.body, offsets);
    }
    symbols
}

fn extract_stmt_symbols(
    stmt: &ast::Stmt,
    symbols: &mut Vec<Symbol>,
    _parent_body: &[ast::Stmt],
    offsets: &[usize],
) {
    match stmt {
        ast::Stmt::FunctionDef(func) => {
            extract_funcdef(func, false, symbols, offsets);
        }
        ast::Stmt::AsyncFunctionDef(func) => {
            extract_async_funcdef(func, symbols, offsets);
        }

        ast::Stmt::ClassDef(class) => {
            let decorators: Vec<String> = class
                .decorator_list
                .iter()
                .filter_map(expr_name)
                .collect();

            let bases: Vec<String> = class
                .bases
                .iter()
                .filter_map(expr_name)
                .collect();

            let docstring = extract_docstring(&class.body);

            let (start_line, end_line) = range_lines(&class.range, offsets);

            let mut meta = json!({});
            if !decorators.is_empty() {
                meta["decorators"] = json!(decorators);
            }
            if !bases.is_empty() {
                meta["bases"] = json!(bases);
            }
            if let Some(doc) = &docstring {
                meta["docstring"] = json!(doc);
            }

            symbols.push(Symbol {
                name: class.name.to_string(),
                kind: SymbolKind::Class,
                start_line,
                end_line,
                metadata: meta,
            });

            // Extract methods defined inside the class.
            for body_stmt in &class.body {
                match body_stmt {
                    ast::Stmt::FunctionDef(_) | ast::Stmt::AsyncFunctionDef(_) => {
                        extract_stmt_symbols(body_stmt, symbols, &class.body, offsets);
                    }
                    _ => {}
                }
            }
        }

        ast::Stmt::Import(imp) => {
            let (start_line, end_line) = range_lines(&imp.range, offsets);
            for alias in &imp.names {
                // Use asname (alias) if present, otherwise first dotted
                // component of the module name (what Python actually binds).
                let bound_name = if let Some(ref asname) = alias.asname {
                    asname.to_string()
                } else {
                    alias.name.split('.')
                        .next()
                        .unwrap_or("")
                        .to_string()
                };
                if bound_name.is_empty() {
                    continue;
                }
                symbols.push(Symbol {
                    name: bound_name,
                    kind: SymbolKind::Import,
                    start_line,
                    end_line,
                    metadata: json!({
                        "type": "import",
                        "module": alias.name.to_string(),
                    }),
                });
            }
        }

        ast::Stmt::ImportFrom(imp) => {
            let module_name = imp
                .module
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_default();
            let level = imp.level.map(|l| l.to_u32()).unwrap_or(0);

            let (start_line, end_line) = range_lines(&imp.range, offsets);
            for alias in &imp.names {
                // Skip wildcard imports (from ... import *)
                if alias.name.as_str() == "*" {
                    continue;
                }
                // Use asname if present, otherwise the imported name.
                let bound_name = if let Some(ref asname) = alias.asname {
                    asname.to_string()
                } else {
                    alias.name.to_string()
                };
                symbols.push(Symbol {
                    name: bound_name,
                    kind: SymbolKind::Import,
                    start_line,
                    end_line,
                    metadata: json!({
                        "type": "from_import",
                        "module": module_name,
                        "level": level,
                        "original_name": alias.name.to_string(),
                    }),
                });
            }
        }

        ast::Stmt::Assign(assign) => {
            let (start_line, end_line) = range_lines(&assign.range, offsets);
            for target in &assign.targets {
                if let Some(name) = expr_name(target) {
                    let kind = if name.chars().all(|c| c.is_uppercase() || c == '_') {
                        SymbolKind::Constant
                    } else {
                        SymbolKind::Variable
                    };
                    symbols.push(Symbol {
                        name,
                        kind,
                        start_line,
                        end_line,
                        metadata: serde_json::Value::Null,
                    });
                }
            }
        }

        ast::Stmt::AnnAssign(assign) => {
            if let Some(name) = expr_name(&assign.target) {
                let (start_line, end_line) = range_lines(&assign.range, offsets);
                let kind = if name.chars().all(|c| c.is_uppercase() || c == '_') {
                    SymbolKind::Constant
                } else {
                    SymbolKind::Variable
                };
                symbols.push(Symbol {
                    name,
                    kind,
                    start_line,
                    end_line,
                    metadata: serde_json::Value::Null,
                });
            }
        }

        _ => {}
    }
}

/// Extract a sync function definition into symbols.
fn extract_funcdef(
    func: &ast::StmtFunctionDef,
    is_async: bool,
    symbols: &mut Vec<Symbol>,
    offsets: &[usize],
) {
    let decorators: Vec<String> = func
        .decorator_list
        .iter()
        .filter_map(expr_name)
        .collect();

    let params: Vec<String> = func
        .args
        .args
        .iter()
        .map(|p| p.def.arg.to_string())
        .collect();

    let docstring = extract_docstring(&func.body);
    let (start_line, end_line) = range_lines(&func.range, offsets);

    let mut meta = json!({ "is_async": is_async });
    if !decorators.is_empty() {
        meta["decorators"] = json!(decorators);
    }
    if !params.is_empty() {
        meta["parameters"] = json!(params);
    }
    if let Some(doc) = &docstring {
        meta["docstring"] = json!(doc);
    }

    symbols.push(Symbol {
        name: func.name.to_string(),
        kind: SymbolKind::Function,
        start_line,
        end_line,
        metadata: meta,
    });
}

/// Extract an async function definition into symbols.
fn extract_async_funcdef(
    func: &ast::StmtAsyncFunctionDef,
    symbols: &mut Vec<Symbol>,
    offsets: &[usize],
) {
    let decorators: Vec<String> = func
        .decorator_list
        .iter()
        .filter_map(expr_name)
        .collect();

    let params: Vec<String> = func
        .args
        .args
        .iter()
        .map(|p| p.def.arg.to_string())
        .collect();

    let docstring = extract_docstring(&func.body);
    let (start_line, end_line) = range_lines(&func.range, offsets);

    let mut meta = json!({ "is_async": true });
    if !decorators.is_empty() {
        meta["decorators"] = json!(decorators);
    }
    if !params.is_empty() {
        meta["parameters"] = json!(params);
    }
    if let Some(doc) = &docstring {
        meta["docstring"] = json!(doc);
    }

    symbols.push(Symbol {
        name: func.name.to_string(),
        kind: SymbolKind::Function,
        start_line,
        end_line,
        metadata: meta,
    });
}

/// Extract a simple name from an expression.
fn expr_name(expr: &ast::Expr) -> Option<String> {
    match expr {
        ast::Expr::Name(n) => Some(n.id.to_string()),
        ast::Expr::Attribute(attr) => {
            let prefix = expr_name(&attr.value)?;
            Some(format!("{prefix}.{}", attr.attr))
        }
        ast::Expr::Call(call) => expr_name(&call.func),
        _ => None,
    }
}

/// Extract docstring from the first statement of a body.
fn extract_docstring(body: &[ast::Stmt]) -> Option<String> {
    if let Some(ast::Stmt::Expr(expr_stmt)) = body.first()
        && let ast::Expr::Constant(c) = expr_stmt.value.as_ref()
        && let ast::Constant::Str(s) = &c.value
    {
        let trimmed = s.trim();
        if trimmed.chars().count() > 500 {
            let truncated: String = trimmed.chars().take(497).collect();
            return Some(format!("{truncated}..."));
        }
        return Some(trimmed.to_string());
    }
    None
}

// ── Top-Level Definitions ──────────────────────────────────────────────

fn extract_top_level_defs_from_module(module: &ast::ModModule, offsets: &[usize]) -> Vec<TopLevelDef> {
    let mut defs = Vec::new();
    for stmt in &module.body {
        let (name, kind, range) = match stmt {
            ast::Stmt::FunctionDef(f) => {
                (f.name.to_string(), SymbolKind::Function, &f.range)
            }
            ast::Stmt::AsyncFunctionDef(f) => {
                (f.name.to_string(), SymbolKind::Function, &f.range)
            }
            ast::Stmt::ClassDef(c) => {
                (c.name.to_string(), SymbolKind::Class, &c.range)
            }
            _ => continue,
        };

        let (start_line, end_line) = range_lines(range, offsets);
        let start_byte = range.start().to_usize();
        let end_byte = range.end().to_usize();

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

// ── Code Metrics ───────────────────────────────────────────────────────

fn compute_metrics_from_module(source: &str, module: &ast::ModModule) -> CodeMetrics {
    let total_lines = if source.is_empty() { 0 } else { source.lines().count() };
    let blank_lines = source.lines().filter(|l| l.trim().is_empty()).count();
    let comment_lines = source
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .count();
    let lines_of_code = total_lines.saturating_sub(blank_lines + comment_lines);

    let mut complexity = 1usize;
    let mut max_depth = 0usize;
    count_complexity_stmts(&module.body, 0, &mut complexity, &mut max_depth);

    CodeMetrics {
        total_lines,
        lines_of_code,
        blank_lines,
        comment_lines,
        cyclomatic_complexity: complexity,
        max_nesting_depth: max_depth,
    }
}

/// Fallback metrics when parsing fails (line counting only).
fn compute_metrics_fallback(source: &str) -> CodeMetrics {
    let total_lines = if source.is_empty() { 0 } else { source.lines().count() };
    let blank_lines = source.lines().filter(|l| l.trim().is_empty()).count();
    let comment_lines = source
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .count();
    let lines_of_code = total_lines.saturating_sub(blank_lines + comment_lines);

    CodeMetrics {
        total_lines,
        lines_of_code,
        blank_lines,
        comment_lines,
        cyclomatic_complexity: 1,
        max_nesting_depth: 0,
    }
}

fn count_complexity_stmts(
    stmts: &[ast::Stmt],
    depth: usize,
    complexity: &mut usize,
    max_depth: &mut usize,
) {
    for stmt in stmts {
        count_complexity_stmt(stmt, depth, complexity, max_depth);
    }
}

fn count_complexity_stmt(
    stmt: &ast::Stmt,
    depth: usize,
    complexity: &mut usize,
    max_depth: &mut usize,
) {
    match stmt {
        ast::Stmt::If(if_stmt) => {
            *complexity += 1;
            let new_depth = depth + 1;
            *max_depth = (*max_depth).max(new_depth);
            count_complexity_stmts(&if_stmt.body, new_depth, complexity, max_depth);
            // orelse may contain a nested If (elif) or plain else body.
            if orelse_is_elif(&if_stmt.orelse) {
                // elif: count as branch at same nesting level.
                count_complexity_stmts(&if_stmt.orelse, depth, complexity, max_depth);
            } else {
                count_complexity_stmts(&if_stmt.orelse, new_depth, complexity, max_depth);
            }
        }
        ast::Stmt::For(for_stmt) => {
            *complexity += 1;
            let new_depth = depth + 1;
            *max_depth = (*max_depth).max(new_depth);
            count_complexity_stmts(&for_stmt.body, new_depth, complexity, max_depth);
            count_complexity_stmts(&for_stmt.orelse, depth, complexity, max_depth);
        }
        ast::Stmt::While(while_stmt) => {
            *complexity += 1;
            let new_depth = depth + 1;
            *max_depth = (*max_depth).max(new_depth);
            count_complexity_stmts(&while_stmt.body, new_depth, complexity, max_depth);
            count_complexity_stmts(&while_stmt.orelse, depth, complexity, max_depth);
        }
        ast::Stmt::Try(try_stmt) => {
            let new_depth = depth + 1;
            *max_depth = (*max_depth).max(new_depth);
            count_complexity_stmts(&try_stmt.body, new_depth, complexity, max_depth);
            for handler in &try_stmt.handlers {
                *complexity += 1;
                let ast::ExceptHandler::ExceptHandler(h) = handler;
                count_complexity_stmts(&h.body, new_depth, complexity, max_depth);
            }
            count_complexity_stmts(&try_stmt.orelse, depth, complexity, max_depth);
            count_complexity_stmts(&try_stmt.finalbody, depth, complexity, max_depth);
        }
        ast::Stmt::With(with_stmt) => {
            *complexity += 1;
            let new_depth = depth + 1;
            *max_depth = (*max_depth).max(new_depth);
            count_complexity_stmts(&with_stmt.body, new_depth, complexity, max_depth);
        }
        ast::Stmt::FunctionDef(func) => {
            count_complexity_stmts(&func.body, depth, complexity, max_depth);
        }
        ast::Stmt::AsyncFunctionDef(func) => {
            count_complexity_stmts(&func.body, depth, complexity, max_depth);
        }
        ast::Stmt::ClassDef(class) => {
            count_complexity_stmts(&class.body, depth, complexity, max_depth);
        }
        ast::Stmt::Match(match_stmt) => {
            let new_depth = depth + 1;
            *max_depth = (*max_depth).max(new_depth);
            for case in &match_stmt.cases {
                *complexity += 1;
                count_complexity_stmts(&case.body, new_depth, complexity, max_depth);
            }
        }
        _ => {}
    }
}

/// Check if an orelse block is an elif (single nested If statement).
fn orelse_is_elif(orelse: &[ast::Stmt]) -> bool {
    orelse.len() == 1 && matches!(&orelse[0], ast::Stmt::If(_))
}

// ── Utilities ──────────────────────────────────────────────────────────

/// Build a table mapping byte offsets to 1-based line numbers.
fn build_line_offsets(source: &str) -> Vec<usize> {
    let mut offsets = vec![0]; // line 1 starts at byte 0
    for (i, ch) in source.char_indices() {
        if ch == '\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

/// Convert a TextRange to (start_line, end_line), both 1-based.
fn range_lines(range: &rustpython_parser::text_size::TextRange, offsets: &[usize]) -> (usize, usize) {
    let start_byte = range.start().to_usize();
    let end_byte = range.end().to_usize();
    (byte_to_line(offsets, start_byte), byte_to_line(offsets, end_byte))
}

/// Convert a byte offset to a 1-based line number.
fn byte_to_line(offsets: &[usize], byte: usize) -> usize {
    match offsets.binary_search(&byte) {
        Ok(idx) => idx + 1,
        Err(idx) => idx.max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PYTHON: &str = r#"#!/usr/bin/env python3
"""Module docstring."""

import os
from pathlib import Path

MAX_SIZE = 1024

class Config:
    """Configuration manager."""

    def __init__(self, path: str):
        self.path = path
        self.data = {}

    def load(self):
        """Load configuration from file."""
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)

    async def save(self):
        pass

def process(items: list) -> int:
    """Process items and return count."""
    count = 0
    for item in items:
        if item.is_valid():
            count += 1
        elif item.is_pending():
            count += 2
    return count
"#;

    #[test]
    fn test_extract_symbols() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();
        let names: Vec<&str> = analysis.symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"os"), "missing os: {names:?}");
        assert!(names.contains(&"Path"), "missing Path: {names:?}");
        assert!(names.contains(&"MAX_SIZE"), "missing MAX_SIZE: {names:?}");
        assert!(names.contains(&"Config"), "missing Config: {names:?}");
        assert!(names.contains(&"__init__"), "missing __init__: {names:?}");
        assert!(names.contains(&"load"), "missing load: {names:?}");
        assert!(names.contains(&"save"), "missing save: {names:?}");
        assert!(names.contains(&"process"), "missing process: {names:?}");
    }

    #[test]
    fn test_symbol_kinds() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();

        let config = analysis.symbols.iter().find(|s| s.name == "Config").unwrap();
        assert_eq!(config.kind, SymbolKind::Class);

        let process = analysis.symbols.iter().find(|s| s.name == "process").unwrap();
        assert_eq!(process.kind, SymbolKind::Function);

        let max_size = analysis.symbols.iter().find(|s| s.name == "MAX_SIZE").unwrap();
        assert_eq!(max_size.kind, SymbolKind::Constant);

        let os_import = analysis.symbols.iter().find(|s| s.name == "os").unwrap();
        assert_eq!(os_import.kind, SymbolKind::Import);
    }

    #[test]
    fn test_function_metadata() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();
        let save = analysis.symbols.iter().find(|s| s.name == "save").unwrap();
        assert_eq!(save.metadata["is_async"], true);

        let process = analysis.symbols.iter().find(|s| s.name == "process").unwrap();
        assert_eq!(process.metadata["is_async"], false);
        let params = process.metadata["parameters"].as_array().unwrap();
        assert_eq!(params[0], "items");
    }

    #[test]
    fn test_class_metadata() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();
        let config = analysis.symbols.iter().find(|s| s.name == "Config").unwrap();
        assert!(config.metadata.get("docstring").is_some());
    }

    #[test]
    fn test_import_metadata() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();
        let path_import = analysis.symbols.iter().find(|s| s.name == "Path").unwrap();
        assert_eq!(path_import.metadata["type"], "from_import");
        assert_eq!(path_import.metadata["module"], "pathlib");
    }

    #[test]
    fn test_top_level_defs() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();
        assert_eq!(analysis.top_level_defs.len(), 2, "defs: {:?}", analysis.top_level_defs);
        assert_eq!(analysis.top_level_defs[0].name, "Config");
        assert_eq!(analysis.top_level_defs[0].kind, SymbolKind::Class);
        assert_eq!(analysis.top_level_defs[1].name, "process");
        assert_eq!(analysis.top_level_defs[1].kind, SymbolKind::Function);
    }

    #[test]
    fn test_metrics() {
        let analysis = analyze(SAMPLE_PYTHON).unwrap();
        let m = &analysis.metrics;
        assert!(m.total_lines > 20);
        assert!(m.lines_of_code > 0);
        assert!(m.blank_lines > 0);
        assert!(m.comment_lines > 0);
        assert!(m.cyclomatic_complexity >= 5, "complexity: {}", m.cyclomatic_complexity);
        assert!(m.max_nesting_depth >= 2, "depth: {}", m.max_nesting_depth);
    }

    #[test]
    fn test_empty_source() {
        let analysis = analyze("").unwrap();
        assert!(analysis.symbols.is_empty());
        assert!(analysis.top_level_defs.is_empty());
    }

    #[test]
    fn test_syntax_error_graceful() {
        let analysis = analyze("def foo(:\n    pass");
        let _ = analysis; // Should not panic — returns Ok with empty symbols.
    }
}
