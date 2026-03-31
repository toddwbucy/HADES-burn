//! Persephone Extraction Client — gRPC client for document content extraction.
//!
//! Connects to the Persephone extraction service over a Unix domain socket
//! or TCP endpoint.  Wraps the generated tonic client with connection
//! management, health checking, and ergonomic Rust types.

use std::path::{Path, PathBuf};
use std::time::Duration;

use hades_proto::extraction::extraction_service_client::ExtractionServiceClient;
use hades_proto::extraction::{
    CapabilitiesRequest, ExtractRequest, ExtractResponse, ExtractorInfo, SourceType,
};
use hyper_util::rt::TokioIo;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;
use tracing::{debug, info, instrument, warn};

/// Default timeout for extraction requests (large PDFs can be slow).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600); // 10 min
/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the extraction client.
#[derive(Debug, Clone)]
pub struct ExtractionClientConfig {
    /// Unix socket path or TCP address for the extraction service.
    pub endpoint: ExtractionEndpoint,
    /// Request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

/// Endpoint for the extraction service.
#[derive(Debug, Clone)]
pub enum ExtractionEndpoint {
    /// Unix domain socket path.
    Unix(PathBuf),
    /// TCP address (e.g. "http://localhost:50052").
    Tcp(String),
}

impl Default for ExtractionClientConfig {
    fn default() -> Self {
        Self {
            endpoint: ExtractionEndpoint::Unix(
                PathBuf::from("/run/hades/extractor.sock"),
            ),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Error type for extraction client operations.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
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

/// Client for the Persephone extraction service.
///
/// Provides ergonomic methods for extracting structured content
/// from documents and querying extractor capabilities.
#[derive(Clone)]
pub struct ExtractionClient {
    inner: ExtractionServiceClient<Channel>,
    config: ExtractionClientConfig,
}

impl ExtractionClient {
    /// Connect to the extraction service.
    #[instrument(skip_all)]
    pub async fn connect(config: ExtractionClientConfig) -> Result<Self, ExtractionError> {
        let channel = match &config.endpoint {
            ExtractionEndpoint::Unix(path) => {
                debug!(socket = %path.display(), "connecting to extraction service via UDS");
                Self::connect_unix(path, &config).await?
            }
            ExtractionEndpoint::Tcp(addr) => {
                debug!(addr, "connecting to extraction service via TCP");
                Self::connect_tcp(addr, &config).await?
            }
        };

        let inner = ExtractionServiceClient::new(channel);
        info!("connected to extraction service");

        Ok(Self { inner, config })
    }

    /// Connect to the extraction service with default configuration.
    pub async fn connect_default() -> Result<Self, ExtractionError> {
        Self::connect(ExtractionClientConfig::default()).await
    }

    /// Connect to an extraction service at the given Unix socket path.
    pub async fn connect_unix_at(path: impl Into<PathBuf>) -> Result<Self, ExtractionError> {
        let config = ExtractionClientConfig {
            endpoint: ExtractionEndpoint::Unix(path.into()),
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Extract structured content from a file on the service's filesystem.
    ///
    /// The extraction service must have read access to `file_path`.
    #[instrument(skip(self, file_path), fields(path = %file_path.as_ref().display()))]
    pub async fn extract_file(
        &self,
        file_path: impl AsRef<Path>,
        options: ExtractOptions,
    ) -> Result<ExtractResult, ExtractionError> {
        let request = ExtractRequest {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            content: Vec::new(),
            source_type: options.source_type.unwrap_or(SourceType::Unknown).into(),
            extract_tables: options.extract_tables,
            extract_equations: options.extract_equations,
            extract_images: options.extract_images,
            use_ocr: options.use_ocr,
        };

        self.do_extract(request).await
    }

    /// Extract structured content from in-memory bytes.
    ///
    /// `file_name` is used for source-type detection when `source_type` is not
    /// explicitly set.
    #[instrument(skip(self, content), fields(file_name, content_len = content.len()))]
    pub async fn extract_bytes(
        &self,
        file_name: &str,
        content: Vec<u8>,
        options: ExtractOptions,
    ) -> Result<ExtractResult, ExtractionError> {
        let request = ExtractRequest {
            file_path: file_name.to_string(),
            content,
            source_type: options.source_type.unwrap_or(SourceType::Unknown).into(),
            extract_tables: options.extract_tables,
            extract_equations: options.extract_equations,
            extract_images: options.extract_images,
            use_ocr: options.use_ocr,
        };

        self.do_extract(request).await
    }

    /// Query the extractor's capabilities and supported formats.
    #[instrument(skip(self))]
    pub async fn capabilities(&self) -> Result<ExtractorInfo, ExtractionError> {
        let response = self
            .inner
            .clone()
            .capabilities(CapabilitiesRequest {})
            .await?
            .into_inner();

        debug!(
            extensions = ?response.supported_extensions,
            features = ?response.features,
            gpu = response.gpu_available,
            "extractor capabilities retrieved"
        );

        Ok(response)
    }

    /// Check if the service is reachable by calling Capabilities.
    ///
    /// Uses a short timeout so a stalled service returns `false` quickly
    /// rather than blocking for the full request timeout.
    pub async fn health_check(&self) -> bool {
        match tokio::time::timeout(Duration::from_secs(5), self.capabilities()).await {
            Ok(Ok(_)) => true,
            Ok(Err(e)) => {
                warn!(error = %e, "extraction service health check failed");
                false
            }
            Err(_) => {
                warn!("extraction service health check timed out");
                false
            }
        }
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &ExtractionEndpoint {
        &self.config.endpoint
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    async fn do_extract(
        &self,
        request: ExtractRequest,
    ) -> Result<ExtractResult, ExtractionError> {
        let response: ExtractResponse = self
            .inner
            .clone()
            .extract(request)
            .await?
            .into_inner();

        if response.full_text.is_empty()
            && response.tables.is_empty()
            && response.equations.is_empty()
            && response.images.is_empty()
        {
            return Err(ExtractionError::InvalidResponse(
                "extraction returned no content (empty text, no tables, equations, or images)"
                    .to_string(),
            ));
        }

        let source_type = response.source_type.try_into().unwrap_or(SourceType::Unknown);

        debug!(
            text_len = response.full_text.len(),
            tables = response.tables.len(),
            equations = response.equations.len(),
            images = response.images.len(),
            source_type = ?source_type,
            "extraction complete"
        );

        Ok(ExtractResult {
            full_text: response.full_text,
            tables: response.tables,
            equations: response.equations,
            images: response.images,
            metadata: response.metadata,
            source_type,
        })
    }

    async fn connect_unix(
        path: &Path,
        config: &ExtractionClientConfig,
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
        config: &ExtractionClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let channel = Endpoint::from_shared(addr.to_string())?
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .connect()
            .await?;

        Ok(channel)
    }
}

/// Options for extraction requests.
#[derive(Debug, Clone, Default)]
pub struct ExtractOptions {
    /// Source type hint. `None` means auto-detect.
    pub source_type: Option<SourceType>,
    /// Whether to extract tables.
    pub extract_tables: bool,
    /// Whether to extract equations.
    pub extract_equations: bool,
    /// Whether to extract images/figures.
    pub extract_images: bool,
    /// Whether to use OCR for scanned content.
    pub use_ocr: bool,
}

impl ExtractOptions {
    /// Create options that extract all supported content types.
    ///
    /// OCR is left disabled because it is expensive and significantly
    /// increases extraction time.  Enable it explicitly when needed:
    /// `ExtractOptions { use_ocr: true, ..ExtractOptions::all() }`.
    pub fn all() -> Self {
        Self {
            source_type: None,
            extract_tables: true,
            extract_equations: true,
            extract_images: true,
            use_ocr: false,
        }
    }
}

/// Result of an extraction operation.
#[derive(Debug, Clone)]
pub struct ExtractResult {
    /// Full extracted text content.
    pub full_text: String,
    /// Extracted tables.
    pub tables: Vec<hades_proto::extraction::Table>,
    /// Extracted equations.
    pub equations: Vec<hades_proto::extraction::Equation>,
    /// Extracted image references.
    pub images: Vec<hades_proto::extraction::ImageRef>,
    /// Additional metadata from the extraction process.
    pub metadata: std::collections::HashMap<String, String>,
    /// Source type used for extraction.
    pub source_type: SourceType,
}

impl std::fmt::Debug for ExtractionClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractionClient")
            .field("endpoint", &self.config.endpoint)
            .finish()
    }
}
