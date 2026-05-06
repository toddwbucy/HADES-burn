//! Persephone Embedding Client — OpenAI-compatible HTTP client for vector
//! embedding generation.
//!
//! HADES is **engine-agnostic** at the protocol layer: any embedding engine
//! that exposes the OpenAI `/v1/embeddings` surface (vLLM, HuggingFace TEI,
//! `hades-weaver-bridge`, llama.cpp-server, ollama where capable, the
//! upstream OpenAI/Anthropic APIs, etc.) is a valid backend. Engines speak
//! the same wire shape; HADES doesn't care which one is running.
//!
//! HADES is **model-bound** at the data layer to Jina V4 (or a future model
//! with the same capability profile: 2048d, 32k context, multimodal,
//! late-chunking-capable). Wrong model → invalidated stored vectors. The
//! engine is fungible, the model is not.
//!
//! Wire protocol: plain HTTP/1.1 JSON, OpenAI-compatible shape.
//!   - `GET  {base}/models`     → list of available models (used for [`info`])
//!   - `POST {base}/embeddings` → embedding generation (used for [`embed`])
//!
//! `task` and `batch_size` are sent as non-standard top-level fields. Engines
//! that don't recognize them ignore them (per JSON convention). Engines that
//! do (vLLM-serving-Jina, etc.) use them for retrieval-quality hints.
//!
//! Endpoint can be either an HTTP base URL (`http://localhost:8000/v1`) or a
//! Unix socket path (`/run/.../embedder.sock`). The Unix socket path is
//! intended for the `hades-weaver-bridge` adapter that exposes Weaver's gRPC
//! embedder via a local OpenAI-compatible HTTP surface.

use std::path::PathBuf;
use std::time::Duration;

use http::header::CONTENT_TYPE;
use http::{Method, Request, Uri};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use hyperlocal::{UnixClientExt, UnixConnector};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

/// Default timeout for embedding requests (5 min for large batches).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the embedding client.
#[derive(Debug, Clone)]
pub struct EmbeddingClientConfig {
    /// Endpoint for the OpenAI-compatible embedding service.
    pub endpoint: EmbeddingEndpoint,
    /// Model identifier sent in every request (`model` field of the OpenAI
    /// embeddings request body). HADES is bound to Jina V4 capabilities;
    /// configure this to match whatever model your engine has loaded.
    pub model: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

/// Endpoint for the embedding service.
///
/// Both variants speak HTTP/1.1 JSON in the OpenAI-compatible shape; the
/// only difference is the transport. Unix is intended for the
/// `hades-weaver-bridge` adapter (Weaver coexistence mode); HTTP is the
/// default for everything else.
#[derive(Debug, Clone)]
pub enum EmbeddingEndpoint {
    /// Unix domain socket path. The server listening on this socket must
    /// expose `/v1/embeddings` and `/v1/models`.
    Unix(PathBuf),
    /// HTTP base URL including the `/v1` prefix
    /// (e.g. `http://localhost:8000/v1`). The client appends `/embeddings`
    /// and `/models` to form the full request URI.
    Tcp(String),
}

/// Default model identifier. Jina V4 is HADES's reference model; future
/// capability-equivalent models can be substituted by setting this.
const DEFAULT_MODEL: &str = "jinaai/jina-embeddings-v4";
/// Default endpoint: vLLM-style local URL. Override via config or env.
const DEFAULT_ENDPOINT_URL: &str = "http://localhost:8000/v1";

impl Default for EmbeddingClientConfig {
    fn default() -> Self {
        Self {
            endpoint: EmbeddingEndpoint::Tcp(DEFAULT_ENDPOINT_URL.to_string()),
            model: DEFAULT_MODEL.to_string(),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Error type for embedding client operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Transport/connection error.
    #[error("connection error: {0}")]
    Connection(String),

    /// HTTP error from the service.
    #[error("service error (HTTP {status}): {message}")]
    Http { status: u16, message: String },

    /// Invalid or unparseable response.
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// Request timed out.
    #[error("request timed out after {0}s")]
    Timeout(u64),

    /// I/O error (socket not found, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<hyper_util::client::legacy::Error> for EmbeddingError {
    fn from(e: hyper_util::client::legacy::Error) -> Self {
        EmbeddingError::Connection(e.to_string())
    }
}

impl From<http::Error> for EmbeddingError {
    fn from(e: http::Error) -> Self {
        EmbeddingError::Connection(e.to_string())
    }
}

impl From<serde_json::Error> for EmbeddingError {
    fn from(e: serde_json::Error) -> Self {
        EmbeddingError::InvalidResponse(e.to_string())
    }
}

/// Provider info derived from the OpenAI `/v1/models` endpoint.
///
/// `device` and `dimension` are NOT part of the OpenAI standard; they're
/// engine-specific and may be `None` depending on the backend. `model_loaded`
/// is true iff the configured model appears in the engine's `/v1/models`
/// listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Configured model name (echoed from `EmbeddingClientConfig.model`).
    pub model_name: String,
    /// Device the model is loaded on (e.g. "cuda:0"). `None` for engines
    /// that don't expose this via a standard endpoint.
    #[serde(default)]
    pub device: Option<String>,
    /// Whether the configured model appears in the engine's model list.
    pub model_loaded: bool,
    /// Embedding dimension. `None` unless the engine advertises it (most
    /// don't via `/v1/models`); HADES expects 2048 for Jina V4 regardless.
    #[serde(default)]
    pub dimension: Option<u32>,
}

/// Client for the embedding service.
///
/// Speaks HTTP/1.1 JSON over Unix domain socket or TCP.
#[derive(Clone)]
pub struct EmbeddingClient {
    config: EmbeddingClientConfig,
    unix_client: Option<Client<UnixConnector, Full<Bytes>>>,
    tcp_client: Option<Client<
        hyper_util::client::legacy::connect::HttpConnector,
        Full<Bytes>,
    >>,
}

impl EmbeddingClient {
    /// Connect to the embedding service.
    ///
    /// For Unix sockets, validates the socket file exists. For TCP,
    /// validates the URL parses. The actual HTTP connection is made
    /// on the first request.
    #[instrument(skip_all)]
    pub async fn connect(config: EmbeddingClientConfig) -> Result<Self, EmbeddingError> {
        match &config.endpoint {
            EmbeddingEndpoint::Unix(path) => {
                if !path.exists() {
                    return Err(EmbeddingError::Connection(format!(
                        "socket not found: {}",
                        path.display()
                    )));
                }
                debug!(socket = %path.display(), "embedding client targeting UDS");
                let client = Client::unix();
                info!("embedding client ready (Unix socket)");
                Ok(Self {
                    config,
                    unix_client: Some(client),
                    tcp_client: None,
                })
            }
            EmbeddingEndpoint::Tcp(addr) => {
                debug!(addr, "embedding client targeting TCP");
                let connector = hyper_util::client::legacy::connect::HttpConnector::new();
                let client = Client::builder(TokioExecutor::new())
                    .pool_idle_timeout(Duration::from_secs(90))
                    .build(connector);
                info!("embedding client ready (TCP)");
                Ok(Self {
                    config,
                    unix_client: None,
                    tcp_client: Some(client),
                })
            }
        }
    }

    /// Connect to the embedding service with default configuration.
    pub async fn connect_default() -> Result<Self, EmbeddingError> {
        Self::connect(EmbeddingClientConfig::default()).await
    }

    /// Connect to an embedding service at the given Unix socket path.
    pub async fn connect_unix_at(path: impl Into<PathBuf>) -> Result<Self, EmbeddingError> {
        let config = EmbeddingClientConfig {
            endpoint: EmbeddingEndpoint::Unix(path.into()),
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Connect to an embedding service at the given endpoint string,
    /// auto-detecting the transport from the prefix.
    ///
    /// - `http://...` or `https://...` → HTTP/TCP endpoint (the base URL,
    ///   typically including `/v1`)
    /// - `unix:///path/to/socket`      → Unix domain socket
    /// - `/path/to/socket`             → Unix domain socket (bare absolute path)
    ///
    /// Anything else is rejected with [`EmbeddingError::Connection`].
    pub async fn connect_at(endpoint_str: &str) -> Result<Self, EmbeddingError> {
        let endpoint = parse_endpoint(endpoint_str)?;
        let config = EmbeddingClientConfig {
            endpoint,
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Embed a batch of texts into vectors.
    ///
    /// Sends `POST {base}/embeddings` in the OpenAI-compatible shape:
    /// `{"model": ..., "input": [...]}`. The HADES-specific `task` hint
    /// (e.g. `"retrieval.query"`, `"retrieval.passage"` for Jina V4) and
    /// `batch_size` are sent as non-standard top-level fields — engines
    /// that don't understand them ignore them.
    ///
    /// Returns one embedding vector per input text, in input order.
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    pub async fn embed(
        &self,
        texts: &[String],
        task: &str,
        batch_size: Option<u32>,
    ) -> Result<EmbedResult, EmbeddingError> {
        let mut body = serde_json::json!({
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        });
        if !task.is_empty() {
            body["task"] = serde_json::json!(task);
        }
        if let Some(bs) = batch_size
            && bs > 0
        {
            body["batch_size"] = serde_json::json!(bs);
        }

        let started = std::time::Instant::now();
        let resp = self.request(Method::POST, "/embeddings", Some(&body)).await?;
        let duration_ms = started.elapsed().as_millis() as u64;

        // OpenAI shape: { "object": "list", "data": [{"object":"embedding","embedding":[...],"index":N}], "model": "...", "usage": {...} }
        let data = resp["data"].as_array().ok_or_else(|| {
            EmbeddingError::InvalidResponse(
                "missing 'data' array in /v1/embeddings response".into(),
            )
        })?;

        // Sort by `index` to guarantee input-order alignment regardless of
        // server-side reordering (some engines parallelize and may reorder).
        let mut indexed: Vec<(usize, Vec<f32>)> = data
            .iter()
            .map(|item| {
                let idx = item["index"].as_u64().unwrap_or(0) as usize;
                let embedding: Vec<f32> = item["embedding"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                (idx, embedding)
            })
            .collect();
        indexed.sort_by_key(|(i, _)| *i);
        let embeddings: Vec<Vec<f32>> = indexed.into_iter().map(|(_, v)| v).collect();

        if embeddings.len() != texts.len() {
            return Err(EmbeddingError::InvalidResponse(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }

        // Derive dimension from first non-empty embedding; verify all match.
        let dimension = embeddings
            .first()
            .map(|v| v.len() as u32)
            .unwrap_or(0);
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() as u32 != dimension {
                return Err(EmbeddingError::InvalidResponse(format!(
                    "embedding[{}] has dimension {}, expected {}",
                    i,
                    emb.len(),
                    dimension
                )));
            }
        }

        // Engine echoes the model name back in the response; fall back to
        // the configured one if absent.
        let model = resp["model"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| self.config.model.clone());

        debug!(
            count = embeddings.len(),
            dimension,
            duration_ms,
            model = %model,
            "embedding complete"
        );

        Ok(EmbedResult {
            embeddings,
            model,
            dimension,
            duration_ms,
        })
    }

    /// Embed a single text string.
    pub async fn embed_one(
        &self,
        text: &str,
        task: &str,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let result = self
            .embed(&[text.to_string()], task, None)
            .await?;
        Ok(result.embeddings.into_iter().next().unwrap())
    }

    /// Query the embedding provider's model availability via OpenAI's
    /// `/v1/models` endpoint.
    ///
    /// Returns `model_loaded = true` iff the configured model
    /// (`EmbeddingClientConfig.model`) appears in the engine's listing.
    /// `device` and `dimension` are not part of the OpenAI standard and
    /// will be `None` for engines that don't extend the response.
    #[instrument(skip(self))]
    pub async fn info(&self) -> Result<ProviderInfo, EmbeddingError> {
        let resp = self.request(Method::GET, "/models", None).await?;

        let configured_model = &self.config.model;

        // OpenAI shape: { "object": "list", "data": [{"id": "...", ...}, ...] }
        let data = resp["data"].as_array();
        let model_loaded = data
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item["id"].as_str())
                    .any(|id| id == configured_model)
            })
            .unwrap_or(false);

        // Some engines (vLLM, llama.cpp-server) attach extra fields we can
        // opportunistically read. Standard says no, but if they're there we
        // surface them.
        let device = data
            .and_then(|arr| {
                arr.iter()
                    .find(|item| item["id"].as_str() == Some(configured_model.as_str()))
            })
            .and_then(|item| item["device"].as_str().map(String::from));
        let dimension = data
            .and_then(|arr| {
                arr.iter()
                    .find(|item| item["id"].as_str() == Some(configured_model.as_str()))
            })
            .and_then(|item| item["dimension"].as_u64().map(|n| n as u32));

        debug!(
            model = %configured_model,
            loaded = model_loaded,
            "provider info retrieved"
        );

        Ok(ProviderInfo {
            model_name: configured_model.clone(),
            device,
            model_loaded,
            dimension,
        })
    }

    /// Check if the service is reachable.
    ///
    /// Uses a short timeout so a stalled service returns `false` quickly
    /// rather than blocking for the full request timeout.
    pub async fn health_check(&self) -> bool {
        match tokio::time::timeout(Duration::from_secs(5), self.info()).await {
            Ok(Ok(_)) => true,
            Ok(Err(e)) => {
                warn!(error = %e, "embedding service health check failed");
                false
            }
            Err(_) => {
                warn!("embedding service health check timed out");
                false
            }
        }
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &EmbeddingEndpoint {
        &self.config.endpoint
    }

    // -----------------------------------------------------------------------
    // HTTP transport
    // -----------------------------------------------------------------------

    /// Send an HTTP request to the embedding service.
    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, EmbeddingError> {
        let uri = self.build_uri(path)?;

        let body_bytes = match body {
            Some(v) => serde_json::to_vec(v)?,
            None => Vec::new(),
        };

        let mut builder = Request::builder().method(method).uri(uri);
        if body.is_some() {
            builder = builder.header(CONTENT_TYPE, "application/json");
        }

        let req = builder
            .body(Full::new(Bytes::copy_from_slice(&body_bytes)))?;

        let timeout = self.config.timeout;
        let response_future = if let Some(ref client) = self.unix_client {
            client.request(req)
        } else if let Some(ref client) = self.tcp_client {
            client.request(req)
        } else {
            return Err(EmbeddingError::Connection(
                "no transport configured".to_string(),
            ));
        };

        let response = tokio::time::timeout(timeout, response_future)
            .await
            .map_err(|_| EmbeddingError::Timeout(timeout.as_secs()))??;

        let status = response.status();
        let resp_bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|e| EmbeddingError::Connection(e.to_string()))?
            .to_bytes();

        if !status.is_success() {
            let message = String::from_utf8_lossy(&resp_bytes).into_owned();
            return Err(EmbeddingError::Http {
                status: status.as_u16(),
                message,
            });
        }

        serde_json::from_slice(&resp_bytes).map_err(|e| {
            EmbeddingError::InvalidResponse(format!(
                "failed to parse response JSON: {e}"
            ))
        })
    }

    /// Build a URI for the given path, using Unix socket or TCP.
    ///
    /// For Unix endpoints, the path is appended directly (e.g.
    /// `/v1/embeddings`). For HTTP endpoints, the path is appended to the
    /// configured base URL (e.g. `http://localhost:8000/v1` + `/embeddings`).
    fn build_uri(&self, path: &str) -> Result<Uri, EmbeddingError> {
        match &self.config.endpoint {
            EmbeddingEndpoint::Unix(socket) => {
                // Bridge servers on the Unix socket expose the full /v1/...
                // path; we add the /v1 prefix here so `path` argument stays
                // bare (`/embeddings`, `/models`).
                let full = format!("/v1{}", path);
                Ok(hyperlocal::Uri::new(socket, &full).into())
            }
            EmbeddingEndpoint::Tcp(base) => {
                // HTTP endpoints already include /v1 in the configured base.
                let url = format!("{}{}", base.trim_end_matches('/'), path);
                url.parse()
                    .map_err(|e| EmbeddingError::Connection(format!("invalid URI: {e}")))
            }
        }
    }
}

/// Parse an endpoint string into an [`EmbeddingEndpoint`].
///
/// Recognized prefixes:
/// - `http://`, `https://`           → HTTP/TCP base URL (must include `/v1`)
/// - `unix:///path/to/socket`        → Unix domain socket
/// - `/path/to/socket` (bare path)   → Unix domain socket
fn parse_endpoint(endpoint_str: &str) -> Result<EmbeddingEndpoint, EmbeddingError> {
    if endpoint_str.starts_with("http://") || endpoint_str.starts_with("https://") {
        Ok(EmbeddingEndpoint::Tcp(endpoint_str.to_string()))
    } else if let Some(path) = endpoint_str.strip_prefix("unix://") {
        Ok(EmbeddingEndpoint::Unix(PathBuf::from(path)))
    } else if endpoint_str.starts_with('/') {
        Ok(EmbeddingEndpoint::Unix(PathBuf::from(endpoint_str)))
    } else {
        Err(EmbeddingError::Connection(format!(
            "endpoint must start with http://, https://, unix://, or be an absolute path; got '{endpoint_str}'"
        )))
    }
}

/// Result of an embedding operation.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// Embedding vectors, one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Model identifier used.
    pub model: String,
    /// Embedding dimension.
    pub dimension: u32,
    /// Wall-clock time in milliseconds.
    pub duration_ms: u64,
}

impl std::fmt::Debug for EmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingClient")
            .field("endpoint", &self.config.endpoint)
            .field("model", &self.config.model)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_endpoint_http() {
        let ep = parse_endpoint("http://localhost:8000/v1").unwrap();
        assert!(matches!(ep, EmbeddingEndpoint::Tcp(ref u) if u == "http://localhost:8000/v1"));
    }

    #[test]
    fn parse_endpoint_https() {
        let ep = parse_endpoint("https://api.openai.com/v1").unwrap();
        assert!(matches!(ep, EmbeddingEndpoint::Tcp(ref u) if u == "https://api.openai.com/v1"));
    }

    #[test]
    fn parse_endpoint_unix_scheme() {
        let ep = parse_endpoint("unix:///run/foo/bar.sock").unwrap();
        match ep {
            EmbeddingEndpoint::Unix(p) => assert_eq!(p, PathBuf::from("/run/foo/bar.sock")),
            _ => panic!("expected Unix variant"),
        }
    }

    #[test]
    fn parse_endpoint_bare_path() {
        let ep = parse_endpoint("/run/foo/bar.sock").unwrap();
        match ep {
            EmbeddingEndpoint::Unix(p) => assert_eq!(p, PathBuf::from("/run/foo/bar.sock")),
            _ => panic!("expected Unix variant"),
        }
    }

    #[test]
    fn parse_endpoint_rejects_unknown() {
        assert!(parse_endpoint("relative/path").is_err());
        assert!(parse_endpoint("ftp://example.com").is_err());
        assert!(parse_endpoint("").is_err());
    }
}
