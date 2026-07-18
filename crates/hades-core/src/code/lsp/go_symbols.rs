//! Semantic Go symbol, call, and interface extraction through gopls.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;
use tracing::{debug, warn};

use super::LspError;
use super::gopls::GoplsSession;
use super::session::uri_to_path;
use super::symbols::{
    CallTarget, ExtractedSymbol, FileExtraction, ImplementationTarget, symbol_kind_name,
};

pub struct GoSymbolExtractor<'a> {
    session: &'a GoplsSession,
    include_calls: bool,
}

impl<'a> GoSymbolExtractor<'a> {
    pub fn new(session: &'a GoplsSession, include_calls: bool) -> Self {
        Self {
            session,
            include_calls,
        }
    }

    pub async fn extract_file(&self, file_path: &Path) -> Result<FileExtraction, LspError> {
        let raw_symbols = self.session.document_symbols(file_path).await?;
        let uri = self.session.open_file(file_path).await?;
        let mut extraction = FileExtraction::empty();
        for symbol in &raw_symbols {
            self.walk_symbol(symbol, &uri, None, None, &mut extraction)
                .await;
        }
        Ok(extraction)
    }

    pub async fn extract_module(&self, go_files: &[&Path]) -> HashMap<String, FileExtraction> {
        let mut results = HashMap::new();
        // gopls discovers cross-package interface implementations from its
        // loaded workspace. Open every module file before querying the first
        // one so results do not depend on input order.
        for file in go_files {
            if let Err(error) = self.session.open_file(file).await {
                warn!(path = %file.display(), %error, "gopls failed to preload file");
            }
        }
        if let Err(error) = self
            .session
            .workspace_symbols("__hades_gopls_preload__")
            .await
        {
            warn!(%error, "gopls workspace preload did not complete cleanly");
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        for file in go_files {
            let path = file.to_string_lossy().to_string();
            match self.extract_file(file).await {
                Ok(extraction) => {
                    debug!(
                        path,
                        symbols = extraction.symbols.len(),
                        "gopls extracted file"
                    );
                    results.insert(path, extraction);
                }
                Err(error) => {
                    warn!(path, %error, "gopls extraction failed; Tree-sitter data retained");
                }
            }
        }
        results
    }

    fn walk_symbol<'b>(
        &'b self,
        symbol: &'b Value,
        uri: &'b str,
        parent: Option<&'b str>,
        interface_owner: Option<&'b str>,
        extraction: &'b mut FileExtraction,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'b>> {
        Box::pin(async move {
            let raw_name = symbol["name"].as_str().unwrap_or("");
            if raw_name.is_empty() {
                return;
            }
            // gopls renders receiver methods as `(Type).Method`. Keep the
            // receiver as parent metadata, but use the bare method name so the
            // semantic document replaces the Tree-sitter fallback document.
            let (receiver, name) = go_symbol_name(raw_name);
            let kind = symbol_kind_name(symbol["kind"].as_u64().unwrap_or(0));
            let start_line = symbol
                .pointer("/range/start/line")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let end_line = symbol
                .pointer("/range/end/line")
                .and_then(Value::as_u64)
                .unwrap_or(start_line as u64) as u32;
            let selection_line = symbol
                .pointer("/selectionRange/start/line")
                .and_then(Value::as_u64)
                .unwrap_or(start_line as u64) as u32;
            let selection_character = symbol
                .pointer("/selectionRange/start/character")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let qualified_name = name.to_string();
            let symbol_parent = receiver.or(parent);
            let visibility = name
                .chars()
                .next()
                .filter(|character| character.is_uppercase())
                .map_or("private", |_| "pub")
                .to_string();
            let mut signature = symbol["detail"].as_str().unwrap_or("").to_string();
            if let Ok(Some(hover)) = self
                .session
                .hover(uri, selection_line, selection_character)
                .await
                && let Some(value) = hover_text(&hover)
            {
                signature = value;
            }
            let calls = if self.include_calls && matches!(kind, "function" | "method") {
                self.outgoing_calls(uri, selection_line, selection_character)
                    .await
            } else {
                Vec::new()
            };

            extraction.symbols.push(ExtractedSymbol {
                name: name.to_string(),
                qualified_name: qualified_name.clone(),
                kind: kind.to_string(),
                visibility,
                signature,
                start_line,
                end_line,
                parent_symbol: symbol_parent.map(str::to_string),
                impl_trait: None,
                is_pyo3: false,
                is_ffi: false,
                is_unsafe: false,
                derives: Vec::new(),
                python_name: None,
                calls,
            });

            // gopls resolves interface satisfaction from an interface method,
            // not from the interface type declaration. Its result points at
            // the corresponding concrete method, which becomes the source of
            // an `implements` edge to the owning interface.
            if let Some(interface_name) = interface_owner
                && matches!(kind, "function" | "method")
            {
                for target in self
                    .session
                    .implementations(uri, selection_line, selection_character)
                    .await
                    .unwrap_or_default()
                {
                    let target_uri = target["uri"]
                        .as_str()
                        .or_else(|| target["targetUri"].as_str())
                        .unwrap_or("");
                    let target_line = target
                        .pointer("/range/start/line")
                        .or_else(|| target.pointer("/targetRange/start/line"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32;
                    let file = uri_relative_path(target_uri, self.session.workspace_root());
                    if !file.is_empty() {
                        extraction.implementations.push(ImplementationTarget {
                            interface_name: interface_name.to_string(),
                            interface_qualified_name: interface_name.to_string(),
                            implementor_file: file,
                            implementor_line: target_line,
                        });
                    }
                }
            }

            let child_parent = matches!(kind, "struct" | "interface" | "class" | "module")
                .then_some(qualified_name.as_str())
                .or(parent);
            let child_interface_owner = if kind == "interface" {
                Some(qualified_name.as_str())
            } else {
                interface_owner
            };
            if let Some(children) = symbol.get("children").and_then(Value::as_array) {
                for child in children {
                    self.walk_symbol(child, uri, child_parent, child_interface_owner, extraction)
                        .await;
                }
            }
        })
    }

    async fn outgoing_calls(&self, uri: &str, line: u32, character: u32) -> Vec<CallTarget> {
        self.session
            .call_hierarchy_outgoing(uri, line, character)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|call| {
                let target = call.get("to")?;
                let raw_name = target["name"].as_str()?;
                let (_, name) = go_symbol_name(raw_name);
                let target_uri = target["uri"].as_str().unwrap_or("");
                Some(CallTarget {
                    qualified_name: name.to_string(),
                    name: name.to_string(),
                    file: uri_relative_path(target_uri, self.session.workspace_root()),
                    line: target
                        .pointer("/range/start/line")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32,
                })
            })
            .collect()
    }
}

fn go_symbol_name(raw_name: &str) -> (Option<&str>, &str) {
    if let Some((receiver, method)) = raw_name
        .strip_prefix('(')
        .and_then(|name| name.split_once(")."))
    {
        (Some(receiver), method)
    } else {
        (None, raw_name.rsplit('.').next().unwrap_or(raw_name))
    }
}

fn uri_relative_path(uri: &str, root: &Path) -> String {
    uri_to_path(uri)
        .and_then(|path| path.strip_prefix(root).ok().map(Path::to_path_buf))
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn hover_text(hover: &Value) -> Option<String> {
    let contents = hover.get("contents")?;
    if let Some(value) = contents.get("value").and_then(Value::as_str) {
        return Some(value.to_string());
    }
    if let Some(value) = contents.as_str() {
        return Some(value.to_string());
    }
    contents.as_array().and_then(|items| {
        items.iter().find_map(|item| {
            item.get("value")
                .and_then(Value::as_str)
                .or_else(|| item.as_str())
                .map(str::to_string)
        })
    })
}
