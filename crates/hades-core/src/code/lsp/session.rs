//! Generic language-server lifecycle and common semantic requests.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use url::Url;

use super::LspError;
use super::client::LspClient;

pub const DEFAULT_INDEX_TIMEOUT_SECS: u64 = 120;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

/// Server-specific configuration layered over the shared LSP transport.
pub trait LanguageServer: Send + Sync + 'static {
    const NAME: &'static str;
    const LANGUAGE_ID: &'static str;

    fn find_binary() -> Result<String, LspError>;
    fn validate_root(root: &Path) -> Result<PathBuf, LspError>;

    fn args() -> &'static [&'static str] {
        &[]
    }

    fn initialization_options() -> Value {
        serde_json::json!({ "workDoneProgress": true })
    }

    /// Recognize server-specific indexing completion messages. A readiness
    /// request is still sent afterward so a missing progress notification is
    /// not mistaken for failure.
    fn progress_is_ready(_value: &Value) -> bool {
        false
    }
}

/// One initialized language-server process for a workspace root.
pub struct LspSession<S: LanguageServer> {
    client: LspClient,
    root: PathBuf,
    ready: bool,
    request_timeout: Duration,
    open_documents: Mutex<HashSet<String>>,
    _server: PhantomData<S>,
}

impl<S: LanguageServer> LspSession<S> {
    pub async fn start(root: &Path) -> Result<Self, LspError> {
        Self::start_with_options(root, None, DEFAULT_INDEX_TIMEOUT_SECS).await
    }

    pub async fn start_with_options(
        root: &Path,
        command: Option<&str>,
        index_timeout_secs: u64,
    ) -> Result<Self, LspError> {
        let root = S::validate_root(root)?;
        let command = command
            .map(str::to_string)
            .map_or_else(S::find_binary, Ok)?;
        let client = LspClient::start(&command, S::args(), &root).await?;
        let mut session = Self {
            client,
            root,
            ready: false,
            request_timeout: Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
            open_documents: Mutex::new(HashSet::new()),
            _server: PhantomData,
        };
        session.initialize().await?;
        session
            .wait_for_ready(Duration::from_secs(index_timeout_secs))
            .await;
        Ok(session)
    }

    pub fn workspace_root(&self) -> &Path {
        &self.root
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub async fn open_file(&self, file_path: &Path) -> Result<String, LspError> {
        let abs_path = if file_path.is_absolute() {
            file_path.to_path_buf()
        } else {
            self.root.join(file_path)
        };
        let abs_path = abs_path.canonicalize()?;
        let uri = path_to_uri(&abs_path)?;
        if !self.open_documents.lock().await.insert(uri.clone()) {
            return Ok(uri);
        }
        let content = tokio::fs::read_to_string(&abs_path).await?;
        if let Err(error) = self
            .client
            .notify(
                "textDocument/didOpen",
                serde_json::json!({
                    "textDocument": {
                        "uri": uri,
                        "languageId": S::LANGUAGE_ID,
                        "version": 1,
                        "text": content,
                    }
                }),
            )
            .await
        {
            self.open_documents.lock().await.remove(&uri);
            return Err(error);
        }
        Ok(uri)
    }

    pub async fn close_file(&self, uri: &str) -> Result<(), LspError> {
        self.client
            .notify(
                "textDocument/didClose",
                serde_json::json!({ "textDocument": { "uri": uri } }),
            )
            .await?;
        self.open_documents.lock().await.remove(uri);
        Ok(())
    }

    pub async fn document_symbols(&self, file_path: &Path) -> Result<Vec<Value>, LspError> {
        let uri = self.open_file(file_path).await?;
        let result = self
            .client
            .request(
                "textDocument/documentSymbol",
                serde_json::json!({ "textDocument": { "uri": uri } }),
                self.request_timeout,
            )
            .await?;
        Ok(result.as_array().cloned().unwrap_or_default())
    }

    /// Synchronization request useful after opening a batch of workspace files.
    pub async fn workspace_symbols(&self, query: &str) -> Result<Vec<Value>, LspError> {
        let result = self
            .client
            .request(
                "workspace/symbol",
                serde_json::json!({ "query": query }),
                self.request_timeout,
            )
            .await?;
        Ok(result.as_array().cloned().unwrap_or_default())
    }

    /// Wait until the server answers two consecutive workspace probes.
    ///
    /// Opening a batch of files can trigger another package load after the
    /// initial handshake. Requiring consecutive successful probes avoids a
    /// fixed delay while still allowing that work to settle.
    pub async fn wait_for_workspace_ready(&self, timeout: Duration) -> bool {
        let start = tokio::time::Instant::now();
        let mut successful_probes = 0;
        while start.elapsed() < timeout {
            let remaining = timeout.saturating_sub(start.elapsed());
            let probe_timeout = remaining.min(Duration::from_secs(3));
            let ready = self
                .client
                .request(
                    "workspace/symbol",
                    serde_json::json!({ "query": "__hades_readiness_probe__" }),
                    probe_timeout,
                )
                .await
                .is_ok();
            successful_probes = if ready { successful_probes + 1 } else { 0 };
            if successful_probes >= 2 {
                debug!(server = S::NAME, elapsed = ?start.elapsed(), "workspace is ready");
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        false
    }

    pub async fn hover(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Option<Value>, LspError> {
        for attempt in 0..3 {
            let result = self
                .position_request("textDocument/hover", uri, line, character)
                .await?;
            if !result.is_null() {
                return Ok(Some(result));
            }
            if attempt < 2 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        Ok(None)
    }

    pub async fn call_hierarchy_outgoing(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<Value>, LspError> {
        let items = self
            .position_request("textDocument/prepareCallHierarchy", uri, line, character)
            .await?;
        let Some(item) = items.as_array().and_then(|items| items.first()) else {
            return Ok(Vec::new());
        };
        let outgoing = self
            .client
            .request(
                "callHierarchy/outgoingCalls",
                serde_json::json!({ "item": item }),
                self.request_timeout,
            )
            .await?;
        Ok(outgoing.as_array().cloned().unwrap_or_default())
    }

    pub async fn call_hierarchy_incoming(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<Value>, LspError> {
        let items = self
            .position_request("textDocument/prepareCallHierarchy", uri, line, character)
            .await?;
        let Some(item) = items.as_array().and_then(|items| items.first()) else {
            return Ok(Vec::new());
        };
        let incoming = self
            .client
            .request(
                "callHierarchy/incomingCalls",
                serde_json::json!({ "item": item }),
                self.request_timeout,
            )
            .await?;
        Ok(incoming.as_array().cloned().unwrap_or_default())
    }

    /// Resolve implementations for an interface/type at a source position.
    pub async fn implementations(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<Value>, LspError> {
        let result = self
            .position_request("textDocument/implementation", uri, line, character)
            .await?;
        Ok(result.as_array().cloned().unwrap_or_default())
    }

    pub async fn shutdown(self) -> Result<(), LspError> {
        self.client.shutdown().await
    }

    async fn position_request(
        &self,
        method: &str,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Value, LspError> {
        self.client
            .request(
                method,
                serde_json::json!({
                    "textDocument": { "uri": uri },
                    "position": { "line": line, "character": character },
                }),
                self.request_timeout,
            )
            .await
    }

    async fn initialize(&mut self) -> Result<(), LspError> {
        let root_uri = path_to_uri(&self.root)?;
        let result = self
            .client
            .request(
                "initialize",
                serde_json::json!({
                    "processId": null,
                    "rootUri": root_uri,
                    "rootPath": self.root.to_str().unwrap_or(""),
                    "workspaceFolders": [{
                        "uri": root_uri,
                        "name": self.root.file_name().and_then(|n| n.to_str()).unwrap_or("workspace"),
                    }],
                    "capabilities": {
                        "workspace": { "configuration": true, "workspaceFolders": true },
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": true,
                                "symbolKind": { "valueSet": (1..=26).collect::<Vec<i32>>() },
                            },
                            "hover": { "contentFormat": ["markdown", "plaintext"] },
                            "callHierarchy": { "dynamicRegistration": false },
                            "implementation": { "dynamicRegistration": false },
                            "references": {},
                            "definition": {},
                            "publishDiagnostics": { "relatedInformation": true },
                        },
                        "window": { "workDoneProgress": true },
                    },
                    "initializationOptions": S::initialization_options(),
                }),
                Duration::from_secs(60),
            )
            .await?;
        debug!(server = S::NAME, capabilities = %result["capabilities"], "LSP initialized");
        self.client
            .notify("initialized", serde_json::json!({}))
            .await?;
        info!(server = S::NAME, root = %self.root.display(), "LSP handshake complete");
        Ok(())
    }

    async fn wait_for_ready(&mut self, timeout: Duration) {
        let start = tokio::time::Instant::now();
        let mut saw_ready_progress = false;
        while start.elapsed() < timeout {
            for notification in self.client.drain_notifications(Some("$/progress")).await {
                if let Some(value) = notification.pointer("/params/value") {
                    saw_ready_progress |= S::progress_is_ready(value);
                    debug!(server = S::NAME, progress = %value, "LSP progress");
                }
            }
            let probe_timeout = if saw_ready_progress {
                Duration::from_secs(10)
            } else {
                Duration::from_secs(3)
            };
            if self
                .client
                .request(
                    "workspace/symbol",
                    serde_json::json!({ "query": "__hades_readiness_probe__" }),
                    probe_timeout,
                )
                .await
                .is_ok()
            {
                self.ready = true;
                info!(server = S::NAME, elapsed = ?start.elapsed(), "LSP ready");
                return;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        warn!(server = S::NAME, ?timeout, "LSP did not confirm readiness");
    }
}

pub fn path_to_uri(path: &Path) -> Result<String, LspError> {
    Url::from_file_path(path)
        .map(Into::into)
        .map_err(|_| LspError::InvalidFileUri(path.display().to_string()))
}

pub fn uri_to_path(uri: &str) -> Option<PathBuf> {
    Url::parse(uri).ok()?.to_file_path().ok()
}

#[cfg(test)]
mod tests {
    use super::{path_to_uri, uri_to_path};
    use std::path::Path;

    #[test]
    fn file_uri_round_trips_spaces_and_unicode() {
        let path = Path::new("/tmp/hades source/λ.go");
        let uri = path_to_uri(path).unwrap();
        assert_eq!(uri, "file:///tmp/hades%20source/%CE%BB.go");
        assert_eq!(uri_to_path(&uri).as_deref(), Some(path));
    }

    #[test]
    fn relative_path_returns_an_error() {
        let error = path_to_uri(Path::new("src/lib.rs")).unwrap_err();
        assert!(error.to_string().contains("src/lib.rs"));
    }

    #[test]
    fn rejects_non_file_uris() {
        assert_eq!(uri_to_path("https://example.test/main.go"), None);
    }
}

/// Probe an analyzer binary FROM the workspace directory and return its
/// version line.
///
/// The probe must run with `current_dir = workspace` because the rustup shim
/// resolves per-directory: a `rust-analyzer` that works in an arbitrary shell
/// dies inside a repo whose rust-toolchain.toml pins a toolchain missing the
/// component ("Unknown binary ... in official toolchain"). Probing from
/// anywhere else validates the wrong toolchain (#164, observed three ways in
/// two days). Failures carry the binary's own stderr so the operator sees the
/// shim's actual message.
pub fn preflight_binary(
    command: &str,
    version_args: &[&str],
    workspace: &std::path::Path,
) -> Result<String, LspError> {
    let out = std::process::Command::new(command)
        .args(version_args)
        .current_dir(workspace)
        .output()
        .map_err(|e| LspError::Process(format!("failed to spawn {command}: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(LspError::Process(format!(
            "{command} {} failed (run from {}): {}",
            version_args.join(" "),
            workspace.display(),
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The version invocation each analyzer understands. gopls has no
/// `--version` flag — its form is the `version` subcommand ("flag provided
/// but not defined: -version", verified against a live gopls v0.21.0 and
/// matching the probe in tests/gopls_semantic.rs). Getting this wrong fails
/// the preflight on a WORKING analyzer, which blocks ingest for the wrong
/// reason.
pub fn version_args_for(analyzer: &str) -> &'static [&'static str] {
    match analyzer {
        "gopls" => &["version"],
        _ => &["--version"],
    }
}

/// One analyzer's resolution and live probe result.
pub struct AnalyzerStatus {
    /// The command that will actually be spawned.
    pub command: String,
    /// Where it came from: "config/env" (pinned) or "PATH".
    pub source: &'static str,
    /// Whether it was explicitly pinned by the operator.
    pub configured: bool,
    /// Version line on success, the probe's error on failure.
    pub outcome: Result<String, String>,
}

/// Resolve an analyzer (pin wins over PATH) and probe it from `workspace`.
///
/// The single shared implementation for both the ingest preflight and
/// `hades tools status`, so the two cannot drift on resolution order or
/// version-invocation form.
pub fn resolve_and_probe(
    analyzer: &str,
    configured: Option<&str>,
    workspace: &std::path::Path,
) -> AnalyzerStatus {
    let (command, source, is_pinned) = match configured {
        Some(p) => (p.to_string(), "config/env", true),
        None => (analyzer.to_string(), "PATH", false),
    };
    let outcome = preflight_binary(&command, version_args_for(analyzer), workspace)
        .map_err(|e| e.to_string());
    AnalyzerStatus {
        command,
        source,
        configured: is_pinned,
        outcome,
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::{preflight_binary, resolve_and_probe, version_args_for};

    #[test]
    fn preflight_reports_missing_binary() {
        let err = preflight_binary(
            "definitely-not-a-real-analyzer",
            &["--version"],
            std::path::Path::new("/tmp"),
        );
        assert!(err.is_err(), "a nonexistent binary must fail the preflight");
    }

    #[test]
    fn preflight_accepts_a_working_binary() {
        let v = preflight_binary("rustc", &["--version"], std::path::Path::new("/tmp"))
            .expect("rustc must preflight");
        assert!(v.contains("rustc"), "version line captured: {v}");
    }

    #[test]
    fn gopls_uses_the_version_subcommand_form() {
        assert_eq!(version_args_for("gopls"), &["version"]);
        assert_eq!(version_args_for("rust-analyzer"), &["--version"]);
    }

    /// A binary that rejects `--version` but accepts `version` — the gopls
    /// shape — must preflight with the right form and fail with the wrong
    /// one. Uses a script so the test needs no gopls installed.
    #[test]
    #[cfg(unix)]
    fn subcommand_only_binary_preflights_with_the_right_form() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let script = dir.path().join("fake-gopls");
        std::fs::write(
            &script,
            "#!/bin/sh\nif [ \"$1\" = version ]; then echo fake v1; exit 0; fi\necho 'flag provided but not defined' >&2; exit 2\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cmd = script.to_string_lossy();

        let ok = preflight_binary(&cmd, &["version"], dir.path());
        assert!(ok.is_ok(), "subcommand form must pass: {ok:?}");
        let bad = preflight_binary(&cmd, &["--version"], dir.path());
        assert!(bad.is_err(), "flag form must fail on a gopls-shaped binary");
    }

    #[test]
    fn resolution_reports_pin_vs_path() {
        let pinned = resolve_and_probe(
            "rust-analyzer",
            Some("/nonexistent/ra"),
            std::path::Path::new("/tmp"),
        );
        assert!(pinned.configured && pinned.source == "config/env" && pinned.outcome.is_err());
        let path = resolve_and_probe("rustc", None, std::path::Path::new("/tmp"));
        assert!(!path.configured && path.source == "PATH" && path.outcome.is_ok());
    }
}
