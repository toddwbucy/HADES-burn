//! Persephone Embedding Client — gRPC client for vector embedding generation.
//!
//! Connects to the Persephone embedding service over a Unix domain socket
//! or TCP endpoint.  Wraps the generated tonic client with connection
//! management, health checking, and ergonomic Rust types.

use std::path::{Path, PathBuf};
use std::time::Duration;

use hades_proto::embedding::embedding_service_client::EmbeddingServiceClient;
use hades_proto::embedding::{EmbedRequest, EmbedResponse, InfoRequest, ProviderInfo};
use hyper_util::rt::TokioIo;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;
use tracing::{debug, info, instrument, warn};

/// Default timeout for embedding requests.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300); // 5 min for large batches
/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the embedding client.
#[derive(Debug, Clone)]
pub struct EmbeddingClientConfig {
    /// Unix socket path or TCP address for the embedding service.
    pub endpoint: EmbeddingEndpoint,
    /// Request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

/// Endpoint for the embedding service.
#[derive(Debug, Clone)]
pub enum EmbeddingEndpoint {
    /// Unix domain socket path.
    Unix(PathBuf),
    /// TCP address (e.g. "http://localhost:50051").
    Tcp(String),
}

impl Default for EmbeddingClientConfig {
    fn default() -> Self {
        Self {
            endpoint: EmbeddingEndpoint::Unix(
                PathBuf::from("/run/hades/embedder.sock"),
            ),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Error type for embedding client operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// gRPC transport/connection error.
    #[error("connection error: {0}")]
    Connection(#[from] tonic::transport::Error),

    /// gRPC status error from the service.
    #[error("service error: {0}")]
    Status(#[from] tonic::Status),

    /// Invalid response from the service.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

/// Client for the Persephone embedding service.
///
/// Provides ergonomic methods for embedding text and querying
/// provider capabilities.
#[derive(Clone)]
pub struct EmbeddingClient {
    inner: EmbeddingServiceClient<Channel>,
    config: EmbeddingClientConfig,
}

impl EmbeddingClient {
    /// Connect to the embedding service.
    #[instrument(skip_all)]
    pub async fn connect(config: EmbeddingClientConfig) -> Result<Self, EmbeddingError> {
        let channel = match &config.endpoint {
            EmbeddingEndpoint::Unix(path) => {
                debug!(socket = %path.display(), "connecting to embedding service via UDS");
                Self::connect_unix(path, &config).await?
            }
            EmbeddingEndpoint::Tcp(addr) => {
                debug!(addr, "connecting to embedding service via TCP");
                Self::connect_tcp(addr, &config).await?
            }
        };

        let inner = EmbeddingServiceClient::new(channel);
        info!("connected to embedding service");

        Ok(Self { inner, config })
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

    /// Embed a batch of texts into vectors.
    ///
    /// Returns a vector of embedding vectors, one per input text.
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    pub async fn embed(
        &self,
        texts: &[String],
        task: &str,
        batch_size: Option<u32>,
    ) -> Result<EmbedResult, EmbeddingError> {
        let request = EmbedRequest {
            texts: texts.to_vec(),
            task: task.to_string(),
            batch_size: batch_size.unwrap_or(0),
        };

        let response: EmbedResponse = self
            .inner
            .clone()
            .embed(request)
            .await?
            .into_inner();

        let embeddings: Vec<Vec<f32>> = response
            .embeddings
            .into_iter()
            .map(|e| e.values)
            .collect();

        if embeddings.len() != texts.len() {
            return Err(EmbeddingError::InvalidResponse(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }

        let expected_dim = response.dimension as usize;
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != expected_dim {
                return Err(EmbeddingError::InvalidResponse(format!(
                    "embedding[{}] has dimension {}, expected {}",
                    i,
                    emb.len(),
                    expected_dim
                )));
            }
        }

        debug!(
            count = embeddings.len(),
            dimension = response.dimension,
            duration_ms = response.duration_ms,
            model = %response.model,
            "embedding complete"
        );

        Ok(EmbedResult {
            embeddings,
            model: response.model,
            dimension: response.dimension,
            duration_ms: response.duration_ms,
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

    /// Query the embedding provider's capabilities and model info.
    #[instrument(skip(self))]
    pub async fn info(&self) -> Result<ProviderInfo, EmbeddingError> {
        let response = self
            .inner
            .clone()
            .info(InfoRequest {})
            .await?
            .into_inner();

        debug!(
            model = %response.model_name,
            dimension = response.dimension,
            device = %response.device,
            loaded = response.model_loaded,
            "provider info retrieved"
        );

        Ok(response)
    }

    /// Check if the service is reachable by calling Info.
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
    // Connection helpers
    // -----------------------------------------------------------------------

    async fn connect_unix(
        path: &Path,
        config: &EmbeddingClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let path = path.to_path_buf();

        // tonic requires a URI even for UDS; the authority is ignored.
        let channel = Endpoint::from_static("http://[::]:50051")
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .connect_with_connector(service_fn(move |_: Uri| {
                let path = path.clone();
                async move {
                    let stream = tokio::net::UnixStream::connect(path).await?;
                    Ok::<_, std::io::Error>(TokioIo::new(stream))
                }
            }))
            .await?;

        Ok(channel)
    }

    async fn connect_tcp(
        addr: &str,
        config: &EmbeddingClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let channel = Endpoint::from_shared(addr.to_string())?
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .connect()
            .await?;

        Ok(channel)
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
            .finish()
    }
}
