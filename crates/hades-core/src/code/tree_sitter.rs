//! Lower-fidelity structural analysis via registered Tree-sitter grammars.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tree_sitter::{Language as TsLanguage, Node, Parser};

use super::Language;
use super::symbols::{AnalysisTier, CodeMetrics, FileAnalysis, Symbol, SymbolKind, TopLevelDef};

/// Whether a structural grammar is registered for a language.
pub fn has_grammar(language: Language) -> bool {
    matches!(
        language,
        Language::Cpp | Language::Go | Language::Python | Language::Rust
    )
}

fn grammar(language: Language) -> Option<TsLanguage> {
    match language {
        Language::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
        Language::Go => Some(tree_sitter_go::LANGUAGE.into()),
        Language::Python => Some(tree_sitter_python::LANGUAGE.into()),
        Language::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
    }
}

/// Parse a file structurally. The result never claims compiler resolution.
pub fn analyze(source: &str, language: Language) -> Result<FileAnalysis, String> {
    let grammar = grammar(language).ok_or_else(|| format!("no grammar for {language}"))?;
    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .map_err(|e| format!("failed to load {language} grammar: {e}"))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| format!("Tree-sitter returned no {language} syntax tree"))?;

    let mut collector = Collector {
        source,
        language,
        symbols: Vec::new(),
        defs: Vec::new(),
    };
    let root = tree.root_node();
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        collector.walk(child, &[], None, true);
    }

    // Tree-sitter can recover around dialect-specific constructs (notably
    // CUDA). Accept a recovered tree when it yielded real structure, but do
    // not mistake a wholly malformed file for a successful empty analysis.
    if root.has_error() && collector.symbols.is_empty() && !source.trim().is_empty() {
        return Err(format!(
            "{language} grammar could not recover useful structure"
        ));
    }

    let encoded = serde_json::to_vec(&collector.symbols).unwrap_or_default();
    let symbol_hash = Sha256::digest(encoded)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(FileAnalysis {
        language,
        symbols: collector.symbols,
        metrics: compute_metrics(source),
        symbol_hash,
        top_level_defs: collector.defs,
        analysis_tier: AnalysisTier::Structural,
        analyzer: "tree-sitter".to_string(),
        fallback_reason: None,
    })
}

struct Collector<'a> {
    source: &'a str,
    language: Language,
    symbols: Vec<Symbol>,
    defs: Vec<TopLevelDef>,
}

impl Collector<'_> {
    fn walk(
        &mut self,
        node: Node<'_>,
        scopes: &[String],
        current_callable: Option<usize>,
        top_level: bool,
    ) {
        if is_call(node, self.language)
            && let Some(index) = current_callable
            && let Some(call) = self.call_metadata(node)
        {
            push_metadata_array(&mut self.symbols[index].metadata, "calls", call);
        }

        let mut child_scopes = scopes.to_vec();
        let mut child_callable = current_callable;
        if let Some((name, kind)) = self.definition(node) {
            let qualified_name = if scopes.is_empty() {
                name.clone()
            } else {
                format!("{}::{name}", scopes.join("::"))
            };
            let mut metadata = json!({
                "qualified_name": qualified_name,
                "analysis_tier": "structural",
                "analyzer": "tree-sitter",
                "resolution": "syntactic",
            });
            if node.has_error() {
                metadata["recovered_parse"] = json!(true);
            }
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;
            let index = self.symbols.len();
            self.symbols.push(Symbol {
                name: name.clone(),
                kind,
                start_line,
                end_line,
                metadata,
            });
            if self.language == Language::Cpp && kind == SymbolKind::Function {
                for call in self.cuda_launches(node) {
                    push_metadata_array(&mut self.symbols[index].metadata, "calls", call);
                }
            }
            if top_level && kind.is_primitive() {
                self.defs.push(TopLevelDef {
                    name: name.clone(),
                    kind,
                    start_line,
                    end_line,
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                });
            }
            if kind == SymbolKind::Function {
                child_callable = Some(index);
            }
            if matches!(
                kind,
                SymbolKind::Class | SymbolKind::Struct | SymbolKind::Module
            ) {
                child_scopes.push(name);
            }
        } else if let Some(import) = self.import_symbol(node) {
            self.symbols.push(import);
        }

        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            self.walk(child, &child_scopes, child_callable, false);
        }
    }

    fn definition(&self, node: Node<'_>) -> Option<(String, SymbolKind)> {
        let kind = match (self.language, node.kind()) {
            (Language::Go, "function_declaration" | "method_declaration") => SymbolKind::Function,
            (Language::Go, "type_spec") => match node.child_by_field_name("type").map(|n| n.kind())
            {
                Some("struct_type") => SymbolKind::Struct,
                Some("interface_type") => SymbolKind::Trait,
                _ => SymbolKind::TypeAlias,
            },
            (Language::Cpp, "function_definition") => SymbolKind::Function,
            (Language::Cpp, "class_specifier") => SymbolKind::Class,
            (Language::Cpp, "struct_specifier" | "union_specifier") => SymbolKind::Struct,
            (Language::Cpp, "enum_specifier") => SymbolKind::Enum,
            (Language::Cpp, "namespace_definition") => SymbolKind::Module,
            (Language::Cpp, "alias_declaration" | "type_definition") => SymbolKind::TypeAlias,
            (Language::Python, "function_definition") => SymbolKind::Function,
            (Language::Python, "class_definition") => SymbolKind::Class,
            (Language::Rust, "function_item") => SymbolKind::Function,
            (Language::Rust, "struct_item") => SymbolKind::Struct,
            (Language::Rust, "enum_item") => SymbolKind::Enum,
            (Language::Rust, "trait_item") => SymbolKind::Trait,
            (Language::Rust, "type_item") => SymbolKind::TypeAlias,
            (Language::Rust, "mod_item") => SymbolKind::Module,
            _ => return None,
        };

        let name_node = match self.language {
            Language::Cpp if node.kind() == "function_definition" => node
                .child_by_field_name("declarator")
                .and_then(cpp_declarator_name_node),
            _ => node.child_by_field_name("name"),
        }?;
        let name = self.text(name_node).trim().to_string();
        (!name.is_empty()).then_some((name, kind))
    }

    fn import_symbol(&self, node: Node<'_>) -> Option<Symbol> {
        let is_import = matches!(
            (self.language, node.kind()),
            (Language::Go, "import_spec")
                | (Language::Cpp, "preproc_include")
                | (
                    Language::Python,
                    "import_statement" | "import_from_statement"
                )
                | (Language::Rust, "use_declaration")
        );
        if !is_import {
            return None;
        }
        let raw = self.text(node).trim();
        let name = match self.language {
            Language::Go => node
                .child_by_field_name("path")
                .map(|n| self.text(n))
                .unwrap_or(raw),
            Language::Cpp => raw.strip_prefix("#include").map(str::trim).unwrap_or(raw),
            Language::Python => raw
                .strip_prefix("from ")
                .or_else(|| raw.strip_prefix("import "))
                .unwrap_or(raw),
            Language::Rust => raw
                .strip_prefix("use ")
                .unwrap_or(raw)
                .trim_end_matches(';'),
        }
        .trim_matches(['"', '<', '>'])
        .to_string();
        Some(Symbol {
            name: name.clone(),
            kind: SymbolKind::Import,
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            metadata: json!({
                "path": name,
                "analysis_tier": "structural",
                "analyzer": "tree-sitter",
                "resolution": "syntactic",
            }),
        })
    }

    fn call_metadata(&self, node: Node<'_>) -> Option<Value> {
        let callee = node
            .child_by_field_name("function")
            .or_else(|| node.named_child(0))?;
        let expression = self.text(callee).trim();
        let name = expression
            .rsplit([':', '.', '>'])
            .find(|part| !part.is_empty() && *part != "-")?
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
            .to_string();
        if name.is_empty() {
            return None;
        }
        Some(json!({
            "name": name,
            "expression": expression,
            "call_line": node.start_position().row + 1,
            "is_kernel_launch": self.text(node).contains("<<<"),
            "analysis_tier": "structural",
            "analyzer": "tree-sitter",
            "resolution": "syntactic",
        }))
    }

    fn cuda_launches(&self, node: Node<'_>) -> Vec<Value> {
        let snippet = self.text(node);
        let mut calls = Vec::new();
        for (launch_offset, _) in snippet.match_indices("<<<") {
            let prefix = &snippet[..launch_offset];
            let name_start = prefix
                .char_indices()
                .rev()
                .find(|(_, ch)| !ch.is_alphanumeric() && *ch != '_')
                .map_or(0, |(index, ch)| index + ch.len_utf8());
            let name = prefix[name_start..].trim();
            if name.is_empty() {
                continue;
            }
            let absolute = node.start_byte() + launch_offset;
            let call_line = self.source[..absolute]
                .bytes()
                .filter(|b| *b == b'\n')
                .count()
                + 1;
            calls.push(json!({
                "name": name,
                "expression": name,
                "call_line": call_line,
                "is_kernel_launch": true,
                "analysis_tier": "structural",
                "analyzer": "tree-sitter",
                "resolution": "syntactic",
                "recovered_cuda_syntax": true,
            }));
        }
        calls
    }

    fn text(&self, node: Node<'_>) -> &str {
        self.source.get(node.byte_range()).unwrap_or("")
    }
}

fn cpp_declarator_name_node(node: Node<'_>) -> Option<Node<'_>> {
    if matches!(
        node.kind(),
        "identifier" | "field_identifier" | "operator_name" | "destructor_name"
    ) {
        return Some(node);
    }
    if let Some(declarator) = node.child_by_field_name("declarator") {
        return cpp_declarator_name_node(declarator);
    }
    if let Some(name) = node.child_by_field_name("name") {
        return cpp_declarator_name_node(name).or(Some(name));
    }
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter_map(cpp_declarator_name_node)
        .last()
}

fn is_call(node: Node<'_>, language: Language) -> bool {
    match language {
        Language::Rust => matches!(node.kind(), "call_expression" | "macro_invocation"),
        _ => node.kind() == "call_expression",
    }
}

fn push_metadata_array(metadata: &mut Value, key: &str, value: Value) {
    if !metadata.get(key).is_some_and(Value::is_array) {
        metadata[key] = json!([]);
    }
    if let Some(values) = metadata[key].as_array_mut()
        && !values.contains(&value)
    {
        values.push(value);
    }
}

fn compute_metrics(source: &str) -> CodeMetrics {
    let mut blank_lines = 0;
    let mut comment_lines = 0;
    let mut lines_of_code = 0;
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() {
            blank_lines += 1;
        } else if line.starts_with("//") || line.starts_with('#') || line.starts_with("/*") {
            comment_lines += 1;
        } else {
            lines_of_code += 1;
        }
    }
    CodeMetrics {
        total_lines: source.lines().count(),
        lines_of_code,
        blank_lines,
        comment_lines,
        cyclomatic_complexity: 1,
        max_nesting_depth: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_symbols_calls_imports_and_chunks_are_structural() {
        let source = r#"package demo
import "fmt"
type Widget struct { Name string }
func (w Widget) Run() { fmt.Println(w.Name) }
"#;
        let analysis = analyze(source, Language::Go).unwrap();
        assert_eq!(analysis.analysis_tier, AnalysisTier::Structural);
        assert!(analysis.symbols.iter().any(|s| s.name == "Widget"));
        let run = analysis.symbols.iter().find(|s| s.name == "Run").unwrap();
        assert_eq!(run.metadata["calls"][0]["name"], "Println");
        assert!(!analysis.top_level_defs.is_empty());
    }

    #[test]
    fn cpp_and_cuda_structure_survives_without_semantic_resolution() {
        let source = r#"
namespace gpu {
template <typename T> __global__ void kernel(T* out) { out[0] = T{}; }
void launch(float* out) { kernel<<<1, 1>>>(out); }
}
"#;
        let analysis = analyze(source, Language::Cpp).unwrap();
        assert!(analysis.symbols.iter().any(|s| s.name == "gpu"));
        assert!(analysis.symbols.iter().any(|s| s.name == "kernel"));
        let launch = analysis
            .symbols
            .iter()
            .find(|s| s.name == "launch")
            .unwrap();
        assert!(launch.metadata["calls"].as_array().is_some_and(|calls| {
            calls
                .iter()
                .any(|call| call["name"] == "kernel" && call["is_kernel_launch"] == true)
        }));
    }

    #[test]
    fn malformed_source_is_a_typed_failure_not_empty_success() {
        assert!(analyze("func (", Language::Go).is_err());
        let empty = analyze("", Language::Go).unwrap();
        assert!(empty.symbols.is_empty());
    }
}
