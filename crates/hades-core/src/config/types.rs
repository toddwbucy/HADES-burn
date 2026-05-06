//! Configuration types mirroring hades.yaml schema.
//!
//! Every field has a serde default matching the Python HADES config,
//! so missing YAML keys produce identical behavior to the Python system.

use std::env;
use std::path::PathBuf;

use anyhow::bail;
use serde::Deserialize;
use tracing::warn;

// ---------------------------------------------------------------------------
// Root config
// ---------------------------------------------------------------------------

/// Top-level HADES configuration.
///
/// Maps 1:1 to the structure of `hades.yaml`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct HadesConfig {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub gpu: GpuConfig,
    pub vector_index: VectorIndexConfig,
    pub search: SearchConfig,
    pub rocchio: RocchioConfig,
    pub sync: SyncConfig,
    pub arxiv: ArxivConfig,
    pub logging: LoggingConfig,
    pub batch_processing: BatchProcessingConfig,
}

impl HadesConfig {
    /// Apply environment variable overrides.
    ///
    /// Called after YAML loading, before CLI overrides.
    /// Returns `Err` if an env var contains an invalid value.
    pub fn apply_env_overrides(&mut self) -> anyhow::Result<()> {
        // Database
        if let Ok(v) = env::var("ARANGO_PASSWORD") {
            self.database.password = Some(v);
        }
        if let Ok(v) = env::var("ARANGO_HOST") {
            self.database.host = v;
        }
        if let Ok(v) = env::var("ARANGO_PORT") {
            match v.parse::<u16>() {
                Ok(port) => self.database.port = port,
                Err(_) => warn!("ARANGO_PORT={v} is not a valid port number, ignoring"),
            }
        }
        if let Ok(v) = env::var("HADES_DATABASE") {
            self.database.name = v;
        }
        if let Ok(v) = env::var("ARANGO_RO_SOCKET") {
            self.database.sockets.readonly = Some(v);
        }
        if let Ok(v) = env::var("ARANGO_RW_SOCKET") {
            self.database.sockets.readwrite = Some(v);
        }

        // GPU
        if let Ok(v) = env::var("HADES_USE_GPU") {
            match v.to_lowercase().trim() {
                "true" | "1" | "yes" => self.gpu.enabled = true,
                "false" | "0" | "no" => self.gpu.enabled = false,
                other => bail!(
                    "HADES_USE_GPU={other:?} is not valid. \
                     Expected: true, false, 1, 0, yes, or no."
                ),
            }
        }
        if let Ok(v) = env::var("CUDA_VISIBLE_DEVICES") {
            self.gpu.cuda_visible_devices = Some(v);
        }

        // Embedding service
        if let Ok(v) = env::var("HADES_EMBEDDER_SOCKET") {
            self.embedding.service.socket = v;
        }

        // ArXiv paths
        if let Ok(v) = env::var("HADES_PDF_PATH") {
            self.arxiv.pdf_base_path = PathBuf::from(v);
        }
        if let Ok(v) = env::var("HADES_LATEX_PATH") {
            self.arxiv.latex_base_path = PathBuf::from(v);
        }

        Ok(())
    }

    /// Apply CLI argument overrides (highest priority).
    ///
    /// Only non-`None` values override the config.
    pub fn apply_cli_overrides(&mut self, database: Option<&str>, gpu_device: Option<u32>) {
        if let Some(db) = database {
            self.database.name = db.to_string();
        }
        if let Some(device_idx) = gpu_device {
            self.gpu.device = format!("cuda:{device_idx}");
            self.gpu.enabled = true;
        }
    }

    /// Get the effective database name after all overrides.
    pub fn effective_database(&self) -> &str {
        &self.database.name
    }

    /// Databases that are allowed to receive writes.
    ///
    /// Production databases (e.g. `NestedLearning`) are **read-only** to
    /// prevent accidental data corruption.  Only databases in this list
    /// may be targeted by write operations (sync, ingest, etc.).
    const WRITABLE_DATABASES: &[&str] = &["bident_burn"];

    /// Check that the effective database is writable.
    ///
    /// Returns `Ok(())` if the database is in the allow-list, or `Err`
    /// with a descriptive message otherwise.  Call this before any
    /// command that writes to ArangoDB.
    pub fn require_writable_database(&self) -> anyhow::Result<()> {
        let db = self.effective_database();
        if Self::WRITABLE_DATABASES.contains(&db) {
            Ok(())
        } else {
            anyhow::bail!(
                "refusing to write to database '{db}': only {:?} are writable. \
                 Use --database to target a writable database.",
                Self::WRITABLE_DATABASES,
            )
        }
    }

    /// Get the effective socket path for ArangoDB connections.
    ///
    /// Returns read-only or read-write socket depending on the `read_only` flag.
    /// Returns `None` if no socket is configured (fall back to host:port).
    pub fn effective_socket(&self, read_only: bool) -> Option<&str> {
        let socket = if read_only {
            self.database.sockets.readonly.as_deref()
        } else {
            self.database.sockets.readwrite.as_deref()
        };
        // Fall back to the other socket if preferred one isn't set
        socket.or(if read_only {
            self.database.sockets.readwrite.as_deref()
        } else {
            self.database.sockets.readonly.as_deref()
        })
    }

    /// Get the effective GPU device string.
    ///
    /// Returns "cpu" if GPU is disabled.
    pub fn effective_device(&self) -> &str {
        if self.gpu.enabled {
            &self.gpu.device
        } else {
            "cpu"
        }
    }

    /// Get the ArangoDB base URL for HTTP API requests.
    ///
    /// Used when connecting over TCP instead of Unix socket.
    pub fn database_url(&self) -> String {
        format!(
            "http://{}:{}/_db/{}",
            self.database.host, self.database.port, self.database.name
        )
    }

    /// Get the password, returning an error message if unset.
    pub fn require_password(&self) -> anyhow::Result<&str> {
        self.database
            .password
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("ARANGO_PASSWORD environment variable is required"))
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    /// Database name. Called "database" in the YAML to match ArangoDB convention.
    #[serde(alias = "database")]
    pub name: String,
    pub username: String,
    /// Password is never stored in YAML — always from ARANGO_PASSWORD env var.
    #[serde(skip)]
    pub password: Option<String>,
    pub sockets: SocketConfig,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "localhost".into(),
            port: 8529,
            name: "NestedLearning".into(),
            username: "root".into(),
            password: None,
            sockets: SocketConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SocketConfig {
    pub readonly: Option<String>,
    pub readwrite: Option<String>,
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub service: EmbeddingServiceConfig,
    pub model: EmbeddingModelConfig,
    pub batch: BatchConfig,
    pub chunking: ChunkingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingServiceConfig {
    pub socket: String,
    pub fallback_to_local: bool,
    pub timeout_ms: u64,
    pub idle_timeout: u64,
}

impl Default for EmbeddingServiceConfig {
    fn default() -> Self {
        Self {
            // OpenAI-compatible HTTP endpoint. vLLM default port is 8000;
            // override per machine in /etc/hades/hades.yaml. For Weaver
            // coexistence, point at a hades-weaver-bridge Unix socket
            // (`unix:///run/...`) instead.
            socket: "http://localhost:8000/v1".into(),
            fallback_to_local: true,
            timeout_ms: 30000,
            idle_timeout: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingModelConfig {
    pub name: String,
    pub dimension: u32,
    pub max_tokens: u32,
    pub use_fp16: bool,
    pub normalize: bool,
}

impl Default for EmbeddingModelConfig {
    fn default() -> Self {
        Self {
            name: "jinaai/jina-embeddings-v4".into(),
            dimension: 2048,
            max_tokens: 32768,
            use_fp16: true,
            normalize: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BatchConfig {
    pub size: u32,
    pub size_small: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            size: 48,
            size_small: 8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ChunkingConfig {
    pub size_tokens: u32,
    pub overlap_tokens: u32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            size_tokens: 500,
            overlap_tokens: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    pub device: String,
    pub enabled: bool,
    /// Set by CUDA_VISIBLE_DEVICES env var — not in YAML.
    #[serde(skip)]
    pub cuda_visible_devices: Option<String>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device: "cuda:2".into(),
            enabled: true,
            cuda_visible_devices: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Vector index
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VectorIndexConfig {
    pub default_n_probe: u32,
    pub metric: String,
    pub auto_n_lists: bool,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            default_n_probe: 10,
            metric: "cosine".into(),
            auto_n_lists: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub limit: u32,
    pub max_limit: u32,
    pub hybrid: HybridConfig,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: 10,
            max_limit: 100,
            hybrid: HybridConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HybridConfig {
    pub vector_weight: f64,
    pub keyword_weight: f64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            keyword_weight: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Rocchio
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RocchioConfig {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Default for RocchioConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.75,
            gamma: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SyncConfig {
    pub default_lookback_days: u32,
    pub batch_size: u32,
    pub max_results: u32,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            default_lookback_days: 7,
            batch_size: 8,
            max_results: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch processing
// ---------------------------------------------------------------------------

/// Batch processing configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BatchProcessingConfig {
    /// Maximum concurrent items in flight.
    pub concurrency: usize,
    /// Minimum seconds between progress updates.
    pub progress_interval_secs: f64,
    /// Requests per second for rate limiting (0 = unlimited).
    pub rate_limit_rps: f64,
    /// Maximum retry attempts for rate-limited requests.
    pub rate_limit_retries: u32,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            progress_interval_secs: 1.0,
            rate_limit_rps: 0.0,
            rate_limit_retries: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// ArXiv paths
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ArxivConfig {
    pub pdf_base_path: PathBuf,
    pub latex_base_path: PathBuf,
}

impl Default for ArxivConfig {
    fn default() -> Self {
        Self {
            pdf_base_path: PathBuf::from("/bulk-store/arxiv-data/pdf"),
            latex_base_path: PathBuf::from("/bulk-store/arxiv-data/src"),
        }
    }
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "INFO".into(),
            format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s".into(),
        }
    }
}
