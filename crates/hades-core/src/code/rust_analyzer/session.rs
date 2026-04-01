//! rust-analyzer session lifecycle manager.
//!
//! Handles the full LSP initialization handshake, waits for indexing
//! to complete, and provides convenience methods for common LSP
//! requests (documentSymbol, hover, callHierarchy).

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use tracing::{debug, info, warn};

use super::client::LspClient;
use super::RustAnalyzerError;

/// Default timeout for rust-analyzer indexing (seconds).
const DEFAULT_INDEX_TIMEOUT_SECS: u64 = 120;

/// Default per-request timeout (seconds).
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

/// Manages a rust-analyzer LSP session for one Rust crate.
///
/// Handles initialization handshake, waits for indexing, provides
/// convenience methods for documentSymbol, hover, and callHierarchy.
pub struct RustAnalyzerSession {
    client: LspClient,
    crate_root: PathBuf,
    ready: bool,
    request_timeout: Duration,
}

impl RustAnalyzerSession {
    /// Start a new rust-analyzer session for the given crate root.
    ///
    /// Spawns rust-analyzer, performs the initialize/initialized handshake,
    /// and waits for indexing to complete.
    pub async fn start(crate_root: &Path) -> Result<Self, RustAnalyzerError> {
        Self::start_with_options(crate_root, None, DEFAULT_INDEX_TIMEOUT_SECS).await
    }

    /// Start with custom binary path and timeout.
    pub async fn start_with_options(
        crate_root: &Path,
        rust_analyzer_cmd: Option<&str>,
        index_timeout_secs: u64,
    ) -> Result<Self, RustAnalyzerError> {
        let crate_root = crate_root.canonicalize().map_err(|e| {
            RustAnalyzerError::InvalidCrate(format!("{}: {e}", crate_root.display()))
        })?;

        // Validate crate root.
        let cargo_toml = crate_root.join("Cargo.toml");
        if !cargo_toml.exists() {
            return Err(RustAnalyzerError::InvalidCrate(format!(
                "no Cargo.toml at {}",
                crate_root.display()
            )));
        }

        let cmd = rust_analyzer_cmd
            .map(String::from)
            .unwrap_or_else(|| find_rust_analyzer().unwrap_or_else(|_| "rust-analyzer".into()));

        let client = LspClient::start(&cmd, &[], &crate_root).await?;

        let mut session = Self {
            client,
            crate_root,
            ready: false,
            request_timeout: Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
        };

        session.initialize().await?;
        session.wait_for_ready(Duration::from_secs(index_timeout_secs)).await;

        Ok(session)
    }

    /// The crate root directory.
    pub fn crate_root(&self) -> &Path {
        &self.crate_root
    }

    /// Whether rust-analyzer has finished indexing.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Open a Rust file in the LSP session via textDocument/didOpen.
    pub async fn open_file(&self, file_path: &Path) -> Result<String, RustAnalyzerError> {
        let abs_path = if file_path.is_absolute() {
            file_path.to_path_buf()
        } else {
            self.crate_root.join(file_path)
        };

        let abs_path = abs_path.canonicalize().map_err(RustAnalyzerError::Io)?;
        let uri = format!("file://{}", abs_path.display());

        let content = tokio::fs::read_to_string(&abs_path).await?;

        self.client.notify(
            "textDocument/didOpen",
            serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": "rust",
                    "version": 1,
                    "text": content,
                }
            }),
        ).await?;

        debug!("opened file: {}", abs_path.display());
        Ok(uri)
    }

    /// Close a previously opened file.
    pub async fn close_file(&self, uri: &str) -> Result<(), RustAnalyzerError> {
        self.client.notify(
            "textDocument/didClose",
            serde_json::json!({
                "textDocument": { "uri": uri }
            }),
        ).await
    }

    /// Get hierarchical document symbols for a Rust file.
    pub async fn document_symbols(
        &self,
        file_path: &Path,
    ) -> Result<Vec<Value>, RustAnalyzerError> {
        let uri = self.open_file(file_path).await?;

        let result = self.client.request(
            "textDocument/documentSymbol",
            serde_json::json!({
                "textDocument": { "uri": uri }
            }),
            self.request_timeout,
        ).await?;

        match result {
            Value::Array(arr) => Ok(arr),
            Value::Null => Ok(Vec::new()),
            _ => Ok(Vec::new()),
        }
    }

    /// Get hover information at a position.
    ///
    /// Retries on null results since rust-analyzer may still be
    /// completing semantic analysis after indexing reports done.
    pub async fn hover(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Option<Value>, RustAnalyzerError> {
        for attempt in 0..3 {
            let result = self.client.request(
                "textDocument/hover",
                serde_json::json!({
                    "textDocument": { "uri": uri },
                    "position": { "line": line, "character": character },
                }),
                Duration::from_secs(10),
            ).await?;

            if !result.is_null() {
                return Ok(Some(result));
            }

            if attempt < 2 {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
        Ok(None)
    }

    /// Get outgoing calls from a symbol at the given position.
    ///
    /// Two-step: prepareCallHierarchy → callHierarchy/outgoingCalls.
    pub async fn call_hierarchy_outgoing(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<Value>, RustAnalyzerError> {
        // Step 1: Prepare call hierarchy item.
        let items = self.client.request(
            "textDocument/prepareCallHierarchy",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
            }),
            self.request_timeout,
        ).await?;

        let items = match items {
            Value::Array(arr) if !arr.is_empty() => arr,
            _ => return Ok(Vec::new()),
        };

        // Step 2: Get outgoing calls for the first item.
        let outgoing = self.client.request(
            "callHierarchy/outgoingCalls",
            serde_json::json!({ "item": items[0] }),
            self.request_timeout,
        ).await?;

        match outgoing {
            Value::Array(arr) => Ok(arr),
            _ => Ok(Vec::new()),
        }
    }

    /// Get incoming calls to a symbol at the given position.
    pub async fn call_hierarchy_incoming(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<Value>, RustAnalyzerError> {
        let items = self.client.request(
            "textDocument/prepareCallHierarchy",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
            }),
            self.request_timeout,
        ).await?;

        let items = match items {
            Value::Array(arr) if !arr.is_empty() => arr,
            _ => return Ok(Vec::new()),
        };

        let incoming = self.client.request(
            "callHierarchy/incomingCalls",
            serde_json::json!({ "item": items[0] }),
            self.request_timeout,
        ).await?;

        match incoming {
            Value::Array(arr) => Ok(arr),
            _ => Ok(Vec::new()),
        }
    }

    /// Gracefully shut down the session.
    pub async fn shutdown(self) -> Result<(), RustAnalyzerError> {
        self.client.shutdown().await
    }

    // ── Internal ─────────────────────────────────────────────────

    /// Perform the LSP initialize/initialized handshake.
    async fn initialize(&mut self) -> Result<(), RustAnalyzerError> {
        let root_uri = format!("file://{}", self.crate_root.display());

        let result = self.client.request(
            "initialize",
            serde_json::json!({
                "processId": null,
                "rootUri": root_uri,
                "rootPath": self.crate_root.to_str().unwrap_or(""),
                "workspaceFolders": [{
                    "uri": root_uri,
                    "name": self.crate_root.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("workspace"),
                }],
                "capabilities": {
                    "textDocument": {
                        "documentSymbol": {
                            "hierarchicalDocumentSymbolSupport": true,
                            "symbolKind": {
                                "valueSet": (1..=26).collect::<Vec<i32>>(),
                            },
                        },
                        "hover": {
                            "contentFormat": ["markdown", "plaintext"],
                        },
                        "callHierarchy": {
                            "dynamicRegistration": false,
                        },
                        "references": {},
                        "definition": {},
                        "publishDiagnostics": {
                            "relatedInformation": true,
                        },
                    },
                    "window": {
                        "workDoneProgress": true,
                    },
                },
                "initializationOptions": {
                    "workDoneProgress": true,
                },
            }),
            Duration::from_secs(60),
        ).await?;

        debug!(
            "rust-analyzer initialized for {}",
            self.crate_root.display(),
        );

        // Log server capabilities for debugging.
        if let Some(caps) = result.get("capabilities") {
            debug!("server capabilities: {}", caps);
        }

        // Send initialized notification.
        self.client.notify("initialized", serde_json::json!({})).await?;

        info!("rust-analyzer handshake complete for {}", self.crate_root.display());
        Ok(())
    }

    /// Wait for rust-analyzer to finish indexing.
    ///
    /// Monitors $/progress notifications, then sends a workspace/symbol
    /// probe to confirm readiness.
    async fn wait_for_ready(&mut self, timeout: Duration) {
        let start = tokio::time::Instant::now();
        let poll_interval = Duration::from_millis(500);

        while start.elapsed() < timeout {
            // Drain progress notifications.
            let notifications = self.client
                .drain_notifications(Some("$/progress"))
                .await;

            for notif in &notifications {
                if let Some(value) = notif.pointer("/params/value") {
                    let kind = value.get("kind").and_then(Value::as_str).unwrap_or("");
                    let title = value.get("title").and_then(Value::as_str).unwrap_or("");
                    let message = value.get("message").and_then(Value::as_str).unwrap_or("");

                    if !title.is_empty() || !message.is_empty() {
                        debug!("rust-analyzer progress: {title}: {message}");
                    }

                    if kind == "end" {
                        debug!("progress phase ended: {title}");
                    }
                }
            }

            // Readiness probe: can the server respond to workspace/symbol?
            match self.client.request(
                "workspace/symbol",
                serde_json::json!({ "query": "__hades_readiness_probe__" }),
                Duration::from_secs(3),
            ).await {
                Ok(_) => {
                    self.ready = true;
                    let elapsed = start.elapsed();
                    info!("rust-analyzer ready in {elapsed:.1?}");
                    return;
                }
                Err(_) => {
                    // Not ready yet — keep waiting.
                }
            }

            tokio::time::sleep(poll_interval).await;
        }

        warn!(
            "rust-analyzer did not confirm readiness within {timeout:?}"
        );
    }
}

/// Locate the rust-analyzer binary on the system.
fn find_rust_analyzer() -> Result<String, RustAnalyzerError> {
    // Check PATH via which.
    if let Ok(output) = std::process::Command::new("which")
        .arg("rust-analyzer")
        .output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    // Check ~/.cargo/bin/
    if let Some(home) = dirs_path() {
        let cargo_bin = home.join(".cargo/bin/rust-analyzer");
        if cargo_bin.exists() {
            return Ok(cargo_bin.to_string_lossy().into_owned());
        }
    }

    Err(RustAnalyzerError::NotFound(
        "rust-analyzer not found. Install via: rustup component add rust-analyzer".into(),
    ))
}

/// Get the user's home directory.
fn dirs_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Find the nearest Cargo.toml above a file path.
///
/// Walks up the directory tree from `file_path` looking for Cargo.toml.
/// Returns the directory containing the Cargo.toml (the crate root).
pub fn find_crate_root(file_path: &Path) -> Option<PathBuf> {
    let mut dir = if file_path.is_file() {
        file_path.parent()?.to_path_buf()
    } else {
        file_path.to_path_buf()
    };

    loop {
        if dir.join("Cargo.toml").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Group .rs files by their nearest Cargo.toml (crate root).
///
/// Returns a map of `crate_root → Vec<file_path>`. Files without a
/// Cargo.toml ancestor are silently dropped.
pub fn group_files_by_crate(rs_files: &[PathBuf]) -> std::collections::HashMap<PathBuf, Vec<PathBuf>> {
    let mut groups: std::collections::HashMap<PathBuf, Vec<PathBuf>> = std::collections::HashMap::new();

    for file_path in rs_files {
        if let Some(crate_root) = find_crate_root(file_path) {
            groups.entry(crate_root).or_default().push(file_path.clone());
        } else {
            tracing::warn!("no Cargo.toml found for {}, skipping", file_path.display());
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_find_crate_root_from_source_file() {
        // This test uses the HADES-Burn repo itself as a test fixture.
        let our_lib = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
        let root = find_crate_root(&our_lib);
        assert!(root.is_some());
        assert!(root.unwrap().join("Cargo.toml").exists());
    }

    #[test]
    fn test_find_crate_root_nonexistent() {
        let result = find_crate_root(Path::new("/tmp/definitely/not/a/rust/crate/foo.rs"));
        // May find a Cargo.toml somewhere in /tmp, or may not.
        // The important thing is it doesn't panic.
        let _ = result;
    }

    #[test]
    fn test_group_files_by_crate() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let files = vec![
            manifest_dir.join("src/lib.rs"),
            manifest_dir.join("src/code/mod.rs"),
        ];
        let groups = group_files_by_crate(&files);
        // Both files belong to the same crate.
        assert_eq!(groups.len(), 1);
        let (root, grouped_files) = groups.into_iter().next().unwrap();
        assert!(root.join("Cargo.toml").exists());
        assert_eq!(grouped_files.len(), 2);
    }
}
