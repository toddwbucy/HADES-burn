//! Symbol extraction from rust-analyzer LSP responses.
//!
//! Walks the hierarchical documentSymbol tree, enriches symbols with
//! hover-derived type signatures, detects impl blocks, PyO3 exports,
//! and FFI boundaries.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;
use tracing::{debug, warn};

use super::session::RustAnalyzerSession;
use super::RustAnalyzerError;

/// LSP SymbolKind → human-readable name.
fn symbol_kind_name(kind: u64) -> &'static str {
    match kind {
        1 => "file",
        2 => "module",
        3 => "namespace",
        4 => "package",
        5 => "class",
        6 => "method",
        7 => "property",
        8 => "field",
        9 => "constructor",
        10 => "enum",
        11 => "interface",
        12 => "function",
        13 => "variable",
        14 => "constant",
        15 => "string",
        16 => "number",
        17 => "boolean",
        18 => "array",
        19 => "object",
        20 => "key",
        21 => "null",
        22 => "enum_member",
        23 => "struct",
        24 => "event",
        25 => "operator",
        26 => "type_parameter",
        _ => "unknown",
    }
}

/// PyO3 attribute pattern.
static PYO3_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"#\[(?:pyclass|pymethods|pyfunction|pymodule|pyproto)(?:\([^]]*\))?\]|#\[pyo3\([^]]*\)\]"
    ).expect("invalid pyo3 regex")
});

/// FFI attribute/keyword pattern.
static FFI_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(extern\s+"C"|#\[(?:unsafe\()?no_mangle(?:\))?\]|#\[(?:unsafe\()?export_name\s*=\s*"[^"]+"(?:\))?\])"#
    ).expect("invalid ffi regex")
});

/// Regex to extract `#[pyo3(name = "...")]` python name.
static PYO3_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"pyo3\(.*name\s*=\s*"([^"]+)""#).expect("invalid pyo3 name regex")
});

/// Regex for Rust code blocks in markdown hover.
static RUST_CODE_BLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```rust\n(.*?)```").expect("invalid code block regex")
});

/// Regex for `#[derive(...)]` attributes.
static DERIVE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"#\[derive\(([^)]+)\)\]").expect("invalid derive regex")
});

/// A symbol extracted from rust-analyzer, enriched and ready for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedSymbol {
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub visibility: String,
    pub signature: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_symbol: Option<String>,
    pub impl_trait: Option<String>,
    pub is_pyo3: bool,
    pub is_ffi: bool,
    pub is_unsafe: bool,
    pub derives: Vec<String>,
    pub python_name: Option<String>,
    /// Outgoing calls from this symbol (populated if call hierarchy is enabled).
    pub calls: Vec<CallTarget>,
}

/// A call target resolved from callHierarchy/outgoingCalls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallTarget {
    pub qualified_name: String,
    pub name: String,
    pub file: String,
    pub line: u32,
}

/// Result of extracting a single file via rust-analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileExtraction {
    pub symbols: Vec<ExtractedSymbol>,
    pub impl_blocks: Vec<ImplBlock>,
    pub pyo3_exports: Vec<String>,
    pub ffi_boundaries: Vec<String>,
    pub analyzed_at: String,
}

/// An impl block grouping methods by self type and optional trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplBlock {
    pub self_type: String,
    pub trait_name: Option<String>,
    pub methods: Vec<String>,
}

/// Extracts structured symbol data from Rust files via rust-analyzer.
pub struct RustSymbolExtractor<'a> {
    session: &'a RustAnalyzerSession,
    include_calls: bool,
}

impl<'a> RustSymbolExtractor<'a> {
    /// Create a new extractor attached to a session.
    pub fn new(session: &'a RustAnalyzerSession, include_calls: bool) -> Self {
        Self { session, include_calls }
    }

    /// Extract all symbol data for a single Rust file.
    pub async fn extract_file(
        &self,
        file_path: &std::path::Path,
    ) -> Result<FileExtraction, RustAnalyzerError> {
        let content = tokio::fs::read_to_string(
            if file_path.is_absolute() {
                file_path.to_path_buf()
            } else {
                self.session.crate_root().join(file_path)
            }
        ).await.unwrap_or_default();

        let lines: Vec<&str> = content.lines().collect();

        // Get document symbols.
        let raw_symbols = self.session.document_symbols(file_path).await?;

        // Get the file URI for hover/call requests.
        let uri = self.session.open_file(file_path).await?;

        // Walk and enrich symbols.
        let mut symbols = Vec::new();
        for sym in &raw_symbols {
            self.walk_symbol(sym, &uri, &lines, None, None, false, false, &mut symbols).await;
        }

        // Group impl blocks.
        let impl_blocks = extract_impl_blocks(&symbols);

        let pyo3_exports: Vec<String> = symbols.iter()
            .filter(|s| s.is_pyo3)
            .map(|s| s.name.clone())
            .collect();

        let ffi_boundaries: Vec<String> = symbols.iter()
            .filter(|s| s.is_ffi)
            .map(|s| s.name.clone())
            .collect();

        Ok(FileExtraction {
            symbols,
            impl_blocks,
            pyo3_exports,
            ffi_boundaries,
            analyzed_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Extract all .rs files in a crate.
    pub async fn extract_crate(
        &self,
        rs_files: &[&std::path::Path],
    ) -> std::collections::HashMap<String, FileExtraction> {
        let mut results = std::collections::HashMap::new();

        for &file_path in rs_files {
            let rel = file_path.to_string_lossy().to_string();
            match self.extract_file(file_path).await {
                Ok(data) => {
                    debug!("extracted {} symbols from {}", data.symbols.len(), rel);
                    results.insert(rel, data);
                }
                Err(e) => {
                    warn!("failed to extract symbols from {}: {e}", rel);
                    results.insert(rel, FileExtraction {
                        symbols: Vec::new(),
                        impl_blocks: Vec::new(),
                        pyo3_exports: Vec::new(),
                        ffi_boundaries: Vec::new(),
                        analyzed_at: chrono::Utc::now().to_rfc3339(),
                    });
                }
            }
        }

        results
    }

    /// Recursively walk a documentSymbol node and its children.
    #[allow(clippy::too_many_arguments)]
    async fn walk_symbol(
        &self,
        sym: &Value,
        uri: &str,
        lines: &[&str],
        parent_name: Option<&str>,
        impl_trait: Option<&str>,
        parent_is_pyo3: bool,
        parent_is_ffi: bool,
        out: &mut Vec<ExtractedSymbol>,
    ) {
        let name = sym["name"].as_str().unwrap_or("");
        let kind_id = sym["kind"].as_u64().unwrap_or(0);
        let kind = symbol_kind_name(kind_id);
        let detail = sym["detail"].as_str().unwrap_or("");

        // Range info.
        let start_line = sym.pointer("/range/start/line")
            .and_then(Value::as_u64).unwrap_or(0) as u32;
        let end_line = sym.pointer("/range/end/line")
            .and_then(Value::as_u64).unwrap_or(0) as u32;
        let sel_line = sym.pointer("/selectionRange/start/line")
            .and_then(Value::as_u64).unwrap_or(start_line as u64) as u32;
        let sel_char = sym.pointer("/selectionRange/start/character")
            .and_then(Value::as_u64).unwrap_or(0) as u32;

        // Detect impl blocks — rust-analyzer names them "impl Foo" or "impl Bar for Foo".
        let mut current_impl_trait = impl_trait.map(String::from);
        let mut impl_self_type = parent_name.map(String::from);

        if let Some(parent) = parent_name
            && let Some(impl_body) = parent.strip_prefix("impl ")
        {
            if let Some((trait_name, self_type)) = impl_body.split_once(" for ") {
                current_impl_trait = Some(trait_name.trim().to_string());
                impl_self_type = Some(self_type.trim().to_string());
            } else {
                impl_self_type = Some(impl_body.trim().to_string());
            }
        }

        // Scan source lines for attributes.
        let attrs = scan_attributes(lines, start_line, sel_line);

        let is_pyo3 = parent_is_pyo3 || attrs.iter().any(|a| PYO3_PATTERN.is_match(a));
        let is_ffi = parent_is_ffi || attrs.iter().any(|a| FFI_PATTERN.is_match(a));
        let is_unsafe = detail.contains("unsafe") || attrs.iter().any(|a| a.contains("unsafe"));

        // impl blocks are containers — skip emitting, but propagate to children.
        if name.starts_with("impl ") {
            if let Some(children) = sym.get("children").and_then(Value::as_array) {
                for child in children {
                    Box::pin(self.walk_symbol(
                        child, uri, lines,
                        Some(name),
                        current_impl_trait.as_deref(),
                        is_pyo3, is_ffi,
                        out,
                    )).await;
                }
            }
            return;
        }

        // Build qualified name.
        let qualified_name = if let Some(ref self_type) = impl_self_type {
            if parent_name.is_some_and(|p| !p.starts_with("impl ")) {
                // Not from an impl block context directly.
                if let Some(parent) = parent_name {
                    format!("{parent}::{name}")
                } else {
                    name.to_string()
                }
            } else {
                format!("{self_type}::{name}")
            }
        } else if let Some(parent) = parent_name {
            if parent.starts_with("impl ") {
                name.to_string()
            } else {
                format!("{parent}::{name}")
            }
        } else {
            name.to_string()
        };

        // Parse visibility from source.
        let visibility = parse_visibility(lines, sel_line as usize);

        // Get hover for signature (pub items only to limit LSP calls).
        let mut signature = detail.to_string();
        if matches!(visibility.as_str(), "pub" | "pub(crate)")
            && matches!(kind, "function" | "method" | "struct" | "enum" | "constant")
            && let Ok(Some(hover)) = self.session.hover(uri, sel_line, sel_char).await
            && let Some(sig) = extract_signature_from_hover(&hover)
        {
            signature = sig;
        }

        let derives = extract_derives(&attrs);

        // Extract python_name from #[pyo3(name = "...")].
        let python_name = if is_pyo3 {
            attrs.iter()
                .find_map(|a| PYO3_NAME.captures(a))
                .map(|c| c[1].to_string())
                .or_else(|| Some(name.to_string()))
        } else {
            None
        };

        // Call hierarchy.
        let calls = if self.include_calls && matches!(kind, "function" | "method") {
            self.get_outgoing_calls(uri, sel_line, sel_char).await
        } else {
            Vec::new()
        };

        out.push(ExtractedSymbol {
            name: name.to_string(),
            qualified_name,
            kind: kind.to_string(),
            visibility,
            signature,
            start_line,
            end_line,
            parent_symbol: impl_self_type.clone().or_else(|| parent_name.map(String::from)),
            impl_trait: current_impl_trait.clone(),
            is_pyo3,
            is_ffi,
            is_unsafe,
            derives,
            python_name,
            calls,
        });

        // Recurse into children (non-impl containers like struct, enum, module).
        if let Some(children) = sym.get("children").and_then(Value::as_array) {
            let child_parent = if matches!(kind, "struct" | "enum" | "module" | "interface") {
                Some(out.last().map(|s| s.qualified_name.clone()).unwrap_or_default())
            } else {
                parent_name.map(String::from)
            };

            for child in children {
                Box::pin(self.walk_symbol(
                    child, uri, lines,
                    child_parent.as_deref(),
                    current_impl_trait.as_deref(),
                    is_pyo3, is_ffi,
                    out,
                )).await;
            }
        }
    }

    /// Get outgoing calls for a symbol.
    async fn get_outgoing_calls(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Vec<CallTarget> {
        let raw = match self.session.call_hierarchy_outgoing(uri, line, character).await {
            Ok(calls) => calls,
            Err(e) => {
                debug!("call hierarchy failed for {uri}:{line}:{character}: {e}");
                return Vec::new();
            }
        };

        raw.iter().filter_map(|item| {
            let to = item.get("to")?;
            let target_name = to["name"].as_str()?;
            let target_detail = to["detail"].as_str().unwrap_or(target_name);
            let target_uri = to["uri"].as_str().unwrap_or("");
            let target_line = to.pointer("/range/start/line")
                .and_then(Value::as_u64).unwrap_or(0) as u32;

            let target_file = uri_to_rel_path(target_uri, self.session.crate_root());

            Some(CallTarget {
                qualified_name: target_detail.to_string(),
                name: target_name.to_string(),
                file: target_file,
                line: target_line,
            })
        }).collect()
    }
}

/// Scan lines above a symbol for Rust attributes (#[...]).
fn scan_attributes(lines: &[&str], start_line: u32, sel_line: u32) -> Vec<String> {
    let mut attrs = Vec::new();
    let start = start_line as usize;
    let sel = sel_line as usize;

    // Attributes between range.start and selectionRange.start.
    for line in &lines[start..sel.min(lines.len())] {
        let line = line.trim();
        if line.starts_with("#[") || line.starts_with("#![") {
            attrs.push(line.to_string());
        }
    }

    // Look above range.start for more attributes.
    if start > 0 {
        let mut i = start - 1;
        loop {
            if i >= lines.len() {
                break;
            }
            let line = lines[i].trim();
            if line.starts_with("#[") || line.starts_with("#![") {
                attrs.push(line.to_string());
            } else if line.starts_with("//") || line.is_empty() {
                // Skip comments and blanks.
            } else {
                break;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    attrs.reverse();
    attrs
}

/// Parse visibility from source text at a symbol's start line.
fn parse_visibility(lines: &[&str], start_line: usize) -> String {
    if start_line >= lines.len() {
        return "private".to_string();
    }

    let line = lines[start_line].trim();

    // Check for pub(crate), pub(super), pub(in path).
    if let Some(rest) = line.strip_prefix("pub") {
        if let Some(rest) = rest.trim_start().strip_prefix('(')
            && let Some(end) = rest.find(')')
        {
            return format!("pub({})", rest[..end].trim());
        }
        if rest.starts_with(' ') || rest.starts_with('(') {
            return "pub".to_string();
        }
    }

    "private".to_string()
}

/// Extract signature from a hover response.
fn extract_signature_from_hover(hover: &Value) -> Option<String> {
    let contents = hover.get("contents")?;
    let value = if let Some(v) = contents.get("value").and_then(Value::as_str) {
        v.to_string()
    } else if let Some(s) = contents.as_str() {
        s.to_string()
    } else if let Some(arr) = contents.as_array() {
        arr.iter()
            .filter_map(|c| {
                c.get("value").and_then(Value::as_str)
                    .or_else(|| c.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        return None;
    };

    // Extract Rust code blocks from markdown.
    let blocks: Vec<String> = RUST_CODE_BLOCK.captures_iter(&value)
        .filter_map(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
        .collect();

    if !blocks.is_empty() {
        // Prefer blocks containing fn/struct/enum/trait/const/type/impl.
        let keywords = ["fn ", "struct ", "enum ", "trait ", "const ", "type ", "impl "];
        for block in blocks.iter().rev() {
            if keywords.iter().any(|kw| block.contains(kw)) {
                return Some(block.clone());
            }
        }
        return Some(blocks.last().unwrap().clone());
    }

    // Fallback: return the whole value if it looks like a signature.
    if value.contains("fn ") || value.contains("struct ") || value.contains("enum ") {
        return Some(value.trim().to_string());
    }

    None
}

/// Extract derive macro names from attributes.
fn extract_derives(attrs: &[String]) -> Vec<String> {
    let mut derives = Vec::new();
    for attr in attrs {
        if let Some(caps) = DERIVE_PATTERN.captures(attr) {
            for name in caps[1].split(',') {
                let name = name.trim();
                if !name.is_empty() {
                    derives.push(name.to_string());
                }
            }
        }
    }
    derives
}

/// Group symbols into impl blocks by parent and trait.
fn extract_impl_blocks(symbols: &[ExtractedSymbol]) -> Vec<ImplBlock> {
    let mut blocks: std::collections::HashMap<(String, Option<String>), Vec<String>> =
        std::collections::HashMap::new();

    for sym in symbols {
        if matches!(sym.kind.as_str(), "method" | "function")
            && let Some(ref parent) = sym.parent_symbol
        {
            let key = (parent.clone(), sym.impl_trait.clone());
            blocks.entry(key).or_default().push(sym.name.clone());
        }
    }

    blocks.into_iter()
        .map(|((self_type, trait_name), methods)| ImplBlock {
            self_type,
            trait_name,
            methods,
        })
        .collect()
}

/// Convert a file:// URI to a path relative to the crate root.
fn uri_to_rel_path(uri: &str, crate_root: &std::path::Path) -> String {
    if let Some(abs_path) = uri.strip_prefix("file://") {
        let abs = std::path::Path::new(abs_path);
        abs.strip_prefix(crate_root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| abs_path.to_string())
    } else {
        uri.to_string()
    }
}
