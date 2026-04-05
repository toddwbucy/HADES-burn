//! Lightweight HTTP client for the FastAPI embedder service.
//!
//! The running embedder at `/run/hades/embedder.sock` is a FastAPI HTTP service
//! (not gRPC).  This module uses `hyperlocal` to talk to it over a Unix socket,
//! matching the same stack that `db/transport.rs` uses for ArangoDB.
//!
//! The gRPC `EmbeddingClient` in `embedding.rs` is for the future Persephone
//! provider architecture — it cannot talk to the current HTTP service.

use std::path::PathBuf;
use std::time::Duration;

use http::header::CONTENT_TYPE;
use http::{Method, Request, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyperlocal::{UnixClientExt, UnixConnector};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace};

/// Default request timeout (5 minutes for large texts).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Error type for embedder HTTP operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    /// Connection to the embedder socket failed.
    #[error("connection error: {0}")]
    Connection(String),

    /// Request timed out.
    #[error("request timed out after {0}s")]
    Timeout(u64),

    /// Non-success HTTP status.
    #[error("HTTP {status}: {message}")]
    Http { status: u16, message: String },

    /// Response body could not be parsed.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

/// HTTP client for the FastAPI embedder service.
pub struct EmbedderHttpClient {
    client: Client<UnixConnector, Full<Bytes>>,
    socket_path: PathBuf,
    timeout: Duration,
}

impl EmbedderHttpClient {
    /// Create a new client targeting the given Unix socket path.
    pub fn new(socket_path: &str) -> Self {
        Self {
            client: Client::unix(),
            socket_path: PathBuf::from(socket_path),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Embed a single text string.
    ///
    /// Sends `POST /embed` with `{"texts": [text], "task": task}`.
    pub async fn embed_text(
        &self,
        text: &str,
        task: &str,
    ) -> Result<EmbedTextResponse, EmbedderError> {
        let body = serde_json::json!({
            "texts": [text],
            "task": task,
        });
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| EmbedderError::InvalidResponse(e.to_string()))?;

        let raw: RawEmbedResponse = self.post_json("/embed", &body_bytes).await?;

        let embedding = raw
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::InvalidResponse("no embeddings returned".into()))?;

        Ok(EmbedTextResponse {
            embedding,
            model: raw.model,
            dimension: raw.dimension,
            duration_ms: raw.duration_ms,
        })
    }

    /// Probe the embedder service health.
    ///
    /// Sends `GET /health` and returns service metadata.
    pub async fn health(&self) -> Result<HealthResponse, EmbedderError> {
        self.get_json("/health").await
    }

    // -----------------------------------------------------------------------
    // HTTP helpers
    // -----------------------------------------------------------------------

    async fn get_json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
    ) -> Result<T, EmbedderError> {
        let uri: http::Uri = hyperlocal::Uri::new(&self.socket_path, path).into();
        trace!(%uri, "embedder GET");

        let req = Request::builder()
            .method(Method::GET)
            .uri(uri)
            .body(Full::new(Bytes::new()))
            .map_err(|e| EmbedderError::Connection(e.to_string()))?;

        self.send_and_parse(req).await
    }

    async fn post_json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &[u8],
    ) -> Result<T, EmbedderError> {
        let uri: http::Uri = hyperlocal::Uri::new(&self.socket_path, path).into();
        trace!(%uri, body_len = body.len(), "embedder POST");

        let req = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::copy_from_slice(body)))
            .map_err(|e| EmbedderError::Connection(e.to_string()))?;

        self.send_and_parse(req).await
    }

    async fn send_and_parse<T: for<'de> Deserialize<'de>>(
        &self,
        req: Request<Full<Bytes>>,
    ) -> Result<T, EmbedderError> {
        let timeout = self.timeout;
        let response = tokio::time::timeout(timeout, self.client.request(req))
            .await
            .map_err(|_| EmbedderError::Timeout(timeout.as_secs()))?
            .map_err(|e| EmbedderError::Connection(e.to_string()))?;

        let status = response.status();
        let body_bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|e| EmbedderError::Connection(e.to_string()))?
            .to_bytes();

        debug!(status = status.as_u16(), body_len = body_bytes.len(), "embedder response");

        if !status.is_success() {
            let message = if body_bytes.is_empty() {
                status.canonical_reason().unwrap_or("unknown").to_string()
            } else {
                String::from_utf8_lossy(&body_bytes).into_owned()
            };
            return Err(EmbedderError::Http {
                status: status.as_u16(),
                message,
            });
        }

        if status == StatusCode::NO_CONTENT || body_bytes.is_empty() {
            return Err(EmbedderError::InvalidResponse("empty response body".into()));
        }

        serde_json::from_slice(&body_bytes)
            .map_err(|e| EmbedderError::InvalidResponse(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Raw JSON response from `POST /embed`.
#[derive(Debug, Deserialize)]
struct RawEmbedResponse {
    embeddings: Vec<Vec<f32>>,
    model: String,
    dimension: u32,
    duration_ms: f64,
}

/// Parsed single-text embedding result.
#[derive(Debug, Clone, Serialize)]
pub struct EmbedTextResponse {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Model identifier.
    pub model: String,
    /// Vector dimension.
    pub dimension: u32,
    /// Wall-clock time in milliseconds.
    pub duration_ms: f64,
}

/// Response from `GET /health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub device: String,
    pub model_name: String,
    pub uptime_seconds: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_embed_response() {
        let json = r#"{
            "embeddings": [[0.1, 0.2, 0.3]],
            "model": "jinaai/jina-embeddings-v4",
            "dimension": 3,
            "duration_ms": 42.5
        }"#;
        let raw: RawEmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(raw.embeddings.len(), 1);
        assert_eq!(raw.embeddings[0].len(), 3);
        assert_eq!(raw.model, "jinaai/jina-embeddings-v4");
        assert_eq!(raw.dimension, 3);
    }

    #[test]
    fn test_parse_health_response() {
        let json = r#"{
            "status": "ok",
            "model_loaded": true,
            "device": "cuda:2",
            "model_name": "jinaai/jina-embeddings-v4",
            "uptime_seconds": 12345.6
        }"#;
        let health: HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(health.status, "ok");
        assert!(health.model_loaded);
        assert_eq!(health.device, "cuda:2");
    }

    #[test]
    fn test_parse_embed_response_empty_embeddings() {
        let json = r#"{
            "embeddings": [],
            "model": "test",
            "dimension": 0,
            "duration_ms": 0.0
        }"#;
        let raw: RawEmbedResponse = serde_json::from_str(json).unwrap();
        assert!(raw.embeddings.is_empty());
    }
}
