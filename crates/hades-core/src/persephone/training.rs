//! Persephone Training Client — gRPC client for RGCN training on GPU.
//!
//! Connects to a Python training provider that owns the GPU, model
//! parameters, and optimizer state. The Rust orchestrator drives the
//! training loop and issues per-step RPCs.

use std::path::{Path, PathBuf};
use std::time::Duration;

use hades_proto::training::training_service_client::TrainingServiceClient;
use hades_proto::training::{
    CheckpointRequest, EvaluateRequest, GetEmbeddingsRequest, InitModelRequest,
    LoadCheckpointRequest, LoadGraphRequest, ModelConfig, OptimizerConfig, TrainStepRequest,
};
use hyper_util::rt::TokioIo;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;
use tracing::{debug, info, instrument};

/// Default timeout for training requests (individual steps are fast).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
/// Timeout for potentially slow operations (model init, graph load, embeddings).
const SLOW_OP_TIMEOUT: Duration = Duration::from_secs(600);
/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the training client.
#[derive(Debug, Clone)]
pub struct TrainingClientConfig {
    /// Unix socket path or TCP address for the training service.
    pub endpoint: TrainingEndpoint,
    /// Timeout for fast operations (train steps, evaluate).
    pub timeout: Duration,
    /// Timeout for slow operations (init, load graph, get embeddings).
    pub slow_timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

/// Endpoint for the training service.
#[derive(Debug, Clone)]
pub enum TrainingEndpoint {
    /// Unix domain socket path.
    Unix(PathBuf),
    /// TCP address (e.g. "http://localhost:50052").
    Tcp(String),
}

impl Default for TrainingClientConfig {
    fn default() -> Self {
        Self {
            endpoint: TrainingEndpoint::Unix(PathBuf::from("/run/hades/training.sock")),
            timeout: DEFAULT_TIMEOUT,
            slow_timeout: SLOW_OP_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for training client operations.
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    /// gRPC transport/connection error.
    #[error("connection error: {0}")]
    Connection(#[from] tonic::transport::Error),

    /// gRPC status error from the service.
    #[error("service error: {0}")]
    Status(#[from] tonic::Status),

    /// Invalid response from the service.
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// Path contains invalid UTF-8.
    #[error("path contains invalid UTF-8: {0}")]
    InvalidPath(PathBuf),
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of model initialization.
#[derive(Debug, Clone)]
pub struct InitResult {
    /// Total number of trainable parameters.
    pub num_parameters: u64,
    /// Device the model was placed on.
    pub device: String,
}

/// Result of loading graph data.
#[derive(Debug, Clone)]
pub struct LoadGraphResult {
    /// Number of nodes loaded.
    pub num_nodes: u64,
    /// Number of edges loaded.
    pub num_edges: u64,
    /// Feature dimension.
    pub feature_dim: u32,
    /// Approximate GPU memory usage in bytes.
    pub gpu_memory_bytes: u64,
}

/// Result of a single training step.
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    /// Binary cross-entropy loss.
    pub loss: f32,
    /// Training accuracy.
    pub accuracy: f32,
}

/// Result of evaluation on a val/test split.
#[derive(Debug, Clone, Copy)]
pub struct EvalResult {
    /// Binary cross-entropy loss.
    pub loss: f32,
    /// Classification accuracy.
    pub accuracy: f32,
    /// Area under the ROC curve.
    pub auc: f32,
}

/// Result of the get-embeddings operation.
#[derive(Debug, Clone)]
pub struct EmbeddingsResult {
    /// Number of nodes embedded.
    pub num_nodes: u64,
    /// Embedding dimension.
    pub embed_dim: u32,
    /// Inline embeddings (empty if written to file).
    pub embeddings: Vec<u8>,
    /// Output file path (empty if returned inline).
    pub output_path: String,
}

/// Result of saving a checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointResult {
    /// Path where the checkpoint was saved.
    pub path: String,
    /// Checkpoint file size in bytes.
    pub size_bytes: u64,
}

/// Convert a path to a UTF-8 string, returning an error for non-UTF-8 paths.
#[allow(clippy::result_large_err)] // TrainingError is large due to tonic::Status
fn path_to_string(path: &Path) -> Result<String, TrainingError> {
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| TrainingError::InvalidPath(path.to_path_buf()))
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Client for the Persephone training service.
///
/// Provides typed methods for each training lifecycle step: init, load,
/// train, evaluate, and checkpoint.
#[derive(Clone)]
pub struct TrainingClient {
    inner: TrainingServiceClient<Channel>,
    config: TrainingClientConfig,
}

impl TrainingClient {
    /// Connect to the training service.
    #[instrument(skip_all)]
    pub async fn connect(config: TrainingClientConfig) -> Result<Self, TrainingError> {
        let channel = match &config.endpoint {
            TrainingEndpoint::Unix(path) => {
                debug!(socket = %path.display(), "connecting to training service via UDS");
                Self::connect_unix(path, &config).await?
            }
            TrainingEndpoint::Tcp(addr) => {
                debug!(addr, "connecting to training service via TCP");
                Self::connect_tcp(addr, &config).await?
            }
        };

        let inner = TrainingServiceClient::new(channel);
        info!("connected to training service");

        Ok(Self { inner, config })
    }

    /// Connect with default configuration.
    pub async fn connect_default() -> Result<Self, TrainingError> {
        Self::connect(TrainingClientConfig::default()).await
    }

    /// Connect to a training service at the given Unix socket path.
    pub async fn connect_unix_at(path: impl Into<PathBuf>) -> Result<Self, TrainingError> {
        let config = TrainingClientConfig {
            endpoint: TrainingEndpoint::Unix(path.into()),
            ..Default::default()
        };
        Self::connect(config).await
    }

    // -----------------------------------------------------------------------
    // Lifecycle RPCs
    // -----------------------------------------------------------------------

    /// Initialize the model on the training provider.
    ///
    /// Must be called before `load_graph` or `train_step`.
    #[instrument(skip(self))]
    pub async fn init_model(
        &self,
        model: ModelConfig,
        optimizer: OptimizerConfig,
        device: &str,
    ) -> Result<InitResult, TrainingError> {
        let request = InitModelRequest {
            model: Some(model),
            optimizer: Some(optimizer),
            device: device.to_string(),
        };

        let mut req = tonic::Request::new(request);
        req.set_timeout(self.config.slow_timeout);

        let response = self
            .inner
            .clone()
            .init_model(req)
            .await?
            .into_inner();

        info!(
            num_parameters = response.num_parameters,
            device = %response.device,
            "model initialized"
        );

        Ok(InitResult {
            num_parameters: response.num_parameters,
            device: response.device,
        })
    }

    /// Load a safetensors graph file into the provider's GPU memory.
    #[instrument(skip_all)]
    pub async fn load_graph(
        &self,
        safetensors_path: impl AsRef<Path>,
    ) -> Result<LoadGraphResult, TrainingError> {
        let request = LoadGraphRequest {
            safetensors_path: path_to_string(safetensors_path.as_ref())?,
        };

        let mut req = tonic::Request::new(request);
        req.set_timeout(self.config.slow_timeout);

        let response = self
            .inner
            .clone()
            .load_graph(req)
            .await?
            .into_inner();

        info!(
            num_nodes = response.num_nodes,
            num_edges = response.num_edges,
            feature_dim = response.feature_dim,
            gpu_mb = response.gpu_memory_bytes as f64 / (1024.0 * 1024.0),
            "graph loaded onto GPU"
        );

        Ok(LoadGraphResult {
            num_nodes: response.num_nodes,
            num_edges: response.num_edges,
            feature_dim: response.feature_dim,
            gpu_memory_bytes: response.gpu_memory_bytes,
        })
    }

    /// Execute one training step.
    #[instrument(skip(self, train_edge_indices, neg_src, neg_dst),
                 fields(edges = train_edge_indices.len(), neg = neg_src.len()))]
    pub async fn train_step(
        &self,
        train_edge_indices: Vec<u32>,
        neg_src: Vec<u32>,
        neg_dst: Vec<u32>,
    ) -> Result<StepResult, TrainingError> {
        let request = TrainStepRequest {
            train_edge_indices,
            neg_src,
            neg_dst,
        };

        let response = self
            .inner
            .clone()
            .train_step(request)
            .await?
            .into_inner();

        Ok(StepResult {
            loss: response.loss,
            accuracy: response.accuracy,
        })
    }

    /// Evaluate on a set of edges (val or test split).
    #[instrument(skip(self, edge_indices, neg_src, neg_dst),
                 fields(edges = edge_indices.len()))]
    pub async fn evaluate(
        &self,
        edge_indices: Vec<u32>,
        neg_src: Vec<u32>,
        neg_dst: Vec<u32>,
    ) -> Result<EvalResult, TrainingError> {
        let request = EvaluateRequest {
            edge_indices,
            neg_src,
            neg_dst,
        };

        let response = self
            .inner
            .clone()
            .evaluate(request)
            .await?
            .into_inner();

        debug!(
            loss = response.loss,
            accuracy = response.accuracy,
            auc = response.auc,
            "evaluation complete"
        );

        Ok(EvalResult {
            loss: response.loss,
            accuracy: response.accuracy,
            auc: response.auc,
        })
    }

    /// Run a full forward pass and return structural embeddings.
    ///
    /// For large graphs, set `output_path` to write embeddings to a file
    /// instead of returning them inline over gRPC.
    #[instrument(skip(self))]
    pub async fn get_embeddings(
        &self,
        output_path: Option<&Path>,
    ) -> Result<EmbeddingsResult, TrainingError> {
        let request = GetEmbeddingsRequest {
            output_path: output_path
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };

        let mut req = tonic::Request::new(request);
        req.set_timeout(self.config.slow_timeout);

        let response = self
            .inner
            .clone()
            .get_embeddings(req)
            .await?
            .into_inner();

        info!(
            num_nodes = response.num_nodes,
            embed_dim = response.embed_dim,
            inline_bytes = response.embeddings.len(),
            output_path = %response.output_path,
            "embeddings retrieved"
        );

        Ok(EmbeddingsResult {
            num_nodes: response.num_nodes,
            embed_dim: response.embed_dim,
            embeddings: response.embeddings,
            output_path: response.output_path,
        })
    }

    /// Save the current model state to a checkpoint file.
    #[instrument(skip_all)]
    pub async fn checkpoint(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<CheckpointResult, TrainingError> {
        let request = CheckpointRequest {
            path: path_to_string(path.as_ref())?,
        };

        let mut req = tonic::Request::new(request);
        req.set_timeout(self.config.slow_timeout);

        let response = self
            .inner
            .clone()
            .checkpoint(req)
            .await?
            .into_inner();

        info!(
            path = %response.path,
            size_mb = response.size_bytes as f64 / (1024.0 * 1024.0),
            "checkpoint saved"
        );

        Ok(CheckpointResult {
            path: response.path,
            size_bytes: response.size_bytes,
        })
    }

    /// Load a previously saved checkpoint, restoring model weights.
    #[instrument(skip_all)]
    pub async fn load_checkpoint(
        &self,
        path: impl AsRef<Path>,
        device: Option<&str>,
    ) -> Result<InitResult, TrainingError> {
        let request = LoadCheckpointRequest {
            path: path_to_string(path.as_ref())?,
            device: device.unwrap_or_default().to_string(),
        };

        let mut req = tonic::Request::new(request);
        req.set_timeout(self.config.slow_timeout);

        let response = self
            .inner
            .clone()
            .load_checkpoint(req)
            .await?
            .into_inner();

        info!(
            num_parameters = response.num_parameters,
            "checkpoint loaded"
        );

        let resolved_device = if response.device.is_empty() {
            device.unwrap_or("unknown").to_string()
        } else {
            response.device
        };

        Ok(InitResult {
            num_parameters: response.num_parameters,
            device: resolved_device,
        })
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &TrainingEndpoint {
        &self.config.endpoint
    }

    // -----------------------------------------------------------------------
    // Connection helpers
    // -----------------------------------------------------------------------

    async fn connect_unix(
        path: &Path,
        config: &TrainingClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let path = path.to_path_buf();

        let channel = Endpoint::from_static("http://[::]:50052")
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
        config: &TrainingClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let channel = Endpoint::from_shared(addr.to_string())?
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .connect()
            .await?;

        Ok(channel)
    }
}

impl std::fmt::Debug for TrainingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingClient")
            .field("endpoint", &self.config.endpoint)
            .finish()
    }
}
