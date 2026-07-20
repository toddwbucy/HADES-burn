//! Configuration types mirroring hades.yaml schema.
//!
//! Every field has a serde default matching the Python HADES config,
//! so missing YAML keys produce identical behavior to the Python system.

use std::env;

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
    pub logging: LoggingConfig,
    pub batch_processing: BatchProcessingConfig,
    /// External semantic-analyzer binaries (rust-analyzer, gopls).
    #[serde(default)]
    pub analyzers: AnalyzersConfig,
}

impl HadesConfig {
    /// Apply environment variable overrides.
    ///
    /// Called after YAML loading, before CLI overrides.
    /// Returns `Err` if an env var contains an invalid value.
    pub fn apply_env_overrides(&mut self) -> anyhow::Result<()> {
        if let Ok(v) = std::env::var("HADES_RUST_ANALYZER_PATH") {
            self.analyzers.rust_analyzer = Some(v);
        }
        if let Ok(v) = std::env::var("HADES_GOPLS_PATH") {
            self.analyzers.gopls = Some(v);
        }
        // Database
        if let Ok(v) = env::var("ARANGO_USERNAME") {
            self.database.username = v;
        }
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
            self.database.name = Some(v);
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

        Ok(())
    }

    /// Apply CLI argument overrides (highest priority).
    ///
    /// Only non-`None` values override the config.
    pub fn apply_cli_overrides(&mut self, database: Option<&str>, gpu_device: Option<u32>) {
        if let Some(db) = database {
            self.database.name = Some(db.to_string());
        }
        if let Some(device_idx) = gpu_device {
            self.gpu.device = format!("cuda:{device_idx}");
            self.gpu.enabled = true;
        }
    }

    /// Test helper: build a config with a specific database name.
    ///
    /// Production code should never call this — it sidesteps the YAML/env
    /// loading path. Used by unit tests that need a config with a known
    /// database name.
    pub fn with_database(name: &str) -> Self {
        let mut cfg = Self::default();
        cfg.database.name = Some(name.to_string());
        cfg
    }

    /// Get the effective database name after all overrides.
    ///
    /// Returns an error if no database is configured. There is **no implicit
    /// default** — every command that touches a database must require
    /// `--db <name>` (or set `HADES_DATABASE`) explicitly.
    pub fn effective_database(&self) -> anyhow::Result<&str> {
        self.database.name.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "no database specified. Pass --db <name> (or --database <name>), \
                 or set HADES_DATABASE in the environment."
            )
        })
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
    /// Used when connecting over TCP instead of Unix socket. Returns an
    /// error if no database is configured (call `effective_database` first
    /// or use `--db`).
    pub fn database_url(&self) -> anyhow::Result<String> {
        let db = self.effective_database()?;
        Ok(format!(
            "http://{}:{}/_db/{}",
            self.database.host, self.database.port, db
        ))
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
    ///
    /// `None` means no database is configured. Commands that need a database
    /// must require `--db <name>` (or the `HADES_DATABASE` env var) — there
    /// is **no implicit default**, by design. Defaulting to a real database
    /// is a footgun: an agent without context could write to the wrong place
    /// or read stale data thinking it was current.
    #[serde(alias = "database")]
    pub name: Option<String>,
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
            name: None,
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
            // HADES-owned embedder, OpenAI-compatible HTTP endpoint.
            // Port 8087 chosen to avoid collisions with vLLM/uvicorn (8000)
            // and weaver-serve's LLM chat API (8080). Override per machine
            // in /etc/hades/hades.yaml or via HADES_EMBEDDER_SOCKET env.
            // For Weaver coexistence (forensic queries), point at a
            // hades-weaver-bridge Unix socket (`unix:///run/...`) instead.
            socket: "http://localhost:8087/v1".into(),
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

/// Paths to external analyzer binaries used by `codebase ingest` enrichment.
///
/// Resolution order per analyzer: this config -> the environment override ->
/// the PATH lookup. A configured path is used verbatim (no PATH search), which
/// is what makes enrichment deterministic across machines — the rustup shim on
/// PATH resolves per-directory via rust-toolchain.toml, so "is rust-analyzer
/// installed" has no directory-independent answer without a pin (#167).
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyzersConfig {
    /// Absolute path to a rust-analyzer binary. Env: HADES_RUST_ANALYZER_PATH.
    #[serde(default)]
    pub rust_analyzer: Option<String>,
    /// Absolute path to a gopls binary. Env: HADES_GOPLS_PATH.
    #[serde(default)]
    pub gopls: Option<String>,
}
