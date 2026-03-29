//! HTTP transport for ArangoDB over Unix socket or TCP.
//!
//! Uses `hyperlocal` for Unix domain socket connections (the primary path)
//! and falls back to TCP via `hyper-util` when no socket is available.

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{Method, Request, StatusCode, Uri};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use hyperlocal::{UnixClientExt, UnixConnector};
use serde_json::Value;
use tracing::{debug, trace};

use super::error::{ArangoError, ArangoErrorBody};
use crate::config::HadesConfig;

/// Async ArangoDB client.
///
/// Sends HTTP requests to ArangoDB over a Unix domain socket (preferred)
/// or TCP.  Each instance targets a single database.
/// Default request timeout (30 seconds, matches Python read_timeout).
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Default pool idle timeout (90 seconds).
const DEFAULT_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone)]
pub struct ArangoClient {
    /// Unix socket path, if connected via UDS.
    socket_path: Option<PathBuf>,
    /// Base URL for TCP connections (e.g. `http://localhost:8529`).
    base_url: String,
    /// Target database name.
    database: String,
    /// Pre-encoded Basic auth header value.
    auth_header: Option<String>,
    /// Timeout for individual HTTP requests.
    request_timeout: Duration,
    /// The hyper client for Unix socket connections.
    unix_client: Option<Client<UnixConnector, Full<Bytes>>>,
    /// The hyper client for TCP connections.
    tcp_client: Option<Client<
        hyper_util::client::legacy::connect::HttpConnector,
        Full<Bytes>,
    >>,
}

impl std::fmt::Debug for ArangoClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArangoClient")
            .field("socket_path", &self.socket_path)
            .field("base_url", &self.base_url)
            .field("database", &self.database)
            .field("has_auth", &self.auth_header.is_some())
            .finish()
    }
}

impl ArangoClient {
    /// Create a client from the HADES configuration.
    ///
    /// Socket discovery priority:
    /// 1. Explicit socket from config (readonly/readwrite)
    /// 2. Direct socket: `/run/arangodb3/arangodb.sock`
    /// 3. TCP fallback: `http://{host}:{port}`
    ///
    /// `read_only` selects the read-only or read-write socket path.
    pub fn from_config(config: &HadesConfig, read_only: bool) -> Result<Self, ArangoError> {
        let socket_path = resolve_socket(config, read_only);
        let base_url = config.database_url();
        let database = config.effective_database().to_string();

        let auth_header = config.database.password.as_deref().map(|pw| {
            let credentials = format!("{}:{}", config.database.username, pw);
            let encoded = base64::engine::general_purpose::STANDARD.encode(credentials);
            format!("Basic {encoded}")
        });

        let mut client = Self {
            socket_path: socket_path.clone(),
            base_url,
            database,
            auth_header,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            unix_client: None,
            tcp_client: None,
        };

        if let Some(ref sock) = socket_path {
            client.unix_client = Some(Client::unix());
            debug!(
                socket = %sock.display(),
                db = %client.database,
                "ArangoDB client using Unix socket"
            );
        } else {
            let connector = hyper_util::client::legacy::connect::HttpConnector::new();
            client.tcp_client = Some(
                Client::builder(TokioExecutor::new())
                    .pool_idle_timeout(DEFAULT_POOL_IDLE_TIMEOUT)
                    .build(connector),
            );
            debug!(
                url = %client.base_url,
                db = %client.database,
                "ArangoDB client using TCP"
            );
        }

        Ok(client)
    }

    /// Create a client with an explicit socket path (for testing).
    pub fn with_socket(
        socket_path: PathBuf,
        database: &str,
        username: &str,
        password: &str,
    ) -> Self {
        let credentials = format!("{username}:{password}");
        let encoded = base64::engine::general_purpose::STANDARD.encode(credentials);

        Self {
            socket_path: Some(socket_path),
            base_url: String::new(),
            database: database.to_string(),
            auth_header: Some(format!("Basic {encoded}")),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            unix_client: Some(Client::unix()),
            tcp_client: None,
        }
    }

    /// Create a client with an explicit TCP URL (for testing).
    pub fn with_url(
        base_url: &str,
        database: &str,
        username: &str,
        password: &str,
    ) -> Self {
        let credentials = format!("{username}:{password}");
        let encoded = base64::engine::general_purpose::STANDARD.encode(credentials);
        let connector = hyper_util::client::legacy::connect::HttpConnector::new();

        Self {
            socket_path: None,
            base_url: base_url.to_string(),
            database: database.to_string(),
            auth_header: Some(format!("Basic {encoded}")),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            unix_client: None,
            tcp_client: Some(
                Client::builder(TokioExecutor::new())
                    .pool_idle_timeout(DEFAULT_POOL_IDLE_TIMEOUT)
                    .build(connector),
            ),
        }
    }

    /// The database this client targets.
    pub fn database(&self) -> &str {
        &self.database
    }

    /// The socket path, if using Unix domain socket.
    pub fn socket_path(&self) -> Option<&Path> {
        self.socket_path.as_deref()
    }

    // ── HTTP primitives ────────────────────────────────────────────

    /// Send a GET request to `/_db/{database}/_api/{path}`.
    pub async fn get(&self, path: &str) -> Result<Value, ArangoError> {
        self.request(Method::GET, path, None).await
    }

    /// Send a POST request with a JSON body.
    pub async fn post(&self, path: &str, body: &Value) -> Result<Value, ArangoError> {
        self.request(Method::POST, path, Some(body)).await
    }

    /// Send a PUT request with a JSON body.
    pub async fn put(&self, path: &str, body: &Value) -> Result<Value, ArangoError> {
        self.request(Method::PUT, path, Some(body)).await
    }

    /// Send a PATCH request with a JSON body.
    pub async fn patch(&self, path: &str, body: &Value) -> Result<Value, ArangoError> {
        self.request(Method::PATCH, path, Some(body)).await
    }

    /// Send a DELETE request.
    pub async fn delete(&self, path: &str) -> Result<Value, ArangoError> {
        self.request(Method::DELETE, path, None).await
    }

    /// Send a POST request with a raw string body (for NDJSON import).
    pub async fn post_raw(
        &self,
        path: &str,
        body: &str,
        content_type: &str,
    ) -> Result<Value, ArangoError> {
        self.request_raw(Method::POST, path, body.as_bytes(), content_type)
            .await
    }

    /// General-purpose request to `/_db/{database}/_api/{path}`.
    ///
    /// This is the escape hatch for any ArangoDB REST call not covered
    /// by the typed methods above.
    pub async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, ArangoError> {
        let body_bytes = match body {
            Some(v) => serde_json::to_vec(v)?,
            None => Vec::new(),
        };
        self.request_raw(method, path, &body_bytes, "application/json")
            .await
    }

    /// Low-level request with raw bytes body.
    async fn request_raw(
        &self,
        method: Method,
        path: &str,
        body: &[u8],
        content_type: &str,
    ) -> Result<Value, ArangoError> {
        let api_path = format!("/_db/{}/_api/{}", self.database, path);
        let uri = self.build_uri(&api_path)?;

        trace!(%method, %uri, body_len = body.len(), "ArangoDB request");

        let mut builder = Request::builder().method(method).uri(uri);

        builder = builder.header(CONTENT_TYPE, content_type);

        if let Some(ref auth) = self.auth_header {
            builder = builder.header(AUTHORIZATION, auth.as_str());
        }

        let req = builder
            .body(Full::new(Bytes::copy_from_slice(body)))
            .map_err(|e| ArangoError::Request(e.to_string()))?;

        let timeout = self.request_timeout;
        let response_future = if let Some(ref client) = self.unix_client {
            client.request(req)
        } else if let Some(ref client) = self.tcp_client {
            client.request(req)
        } else {
            return Err(ArangoError::Request(
                "no transport configured".to_string(),
            ));
        };

        let response = tokio::time::timeout(timeout, response_future)
            .await
            .map_err(|_| {
                ArangoError::Request(format!(
                    "request timed out after {}s",
                    timeout.as_secs()
                ))
            })??;

        let status = response.status();
        let body_bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|e| ArangoError::Request(e.to_string()))?
            .to_bytes();

        trace!(
            status = status.as_u16(),
            body_len = body_bytes.len(),
            "ArangoDB response"
        );

        // 204 No Content → empty success
        if status == StatusCode::NO_CONTENT {
            return Ok(Value::Object(serde_json::Map::new()));
        }

        // Try to parse as JSON
        let json: Value = if body_bytes.is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_slice(&body_bytes).unwrap_or_else(|_| {
                // Non-JSON response — wrap as raw text
                serde_json::json!({
                    "raw": String::from_utf8_lossy(&body_bytes)
                })
            })
        };

        if status.is_success() {
            return Ok(json);
        }

        // Try to parse ArangoDB's structured error response
        if let Ok(err_body) = serde_json::from_value::<ArangoErrorBody>(json.clone()) {
            return Err(ArangoError::Api {
                status: status.as_u16(),
                error_num: err_body.error_num,
                message: err_body.error_message,
                body: json,
            });
        }

        Err(ArangoError::Http {
            status: status.as_u16(),
            message: json.to_string(),
        })
    }

    /// Build a URI for the given path, using Unix socket or TCP.
    fn build_uri(&self, path: &str) -> Result<Uri, ArangoError> {
        if let Some(ref socket) = self.socket_path {
            Ok(hyperlocal::Uri::new(socket, path).into())
        } else {
            format!("{}{}", self.base_url, path)
                .parse()
                .map_err(|e| ArangoError::Request(format!("invalid URI: {e}")))
        }
    }
}

// ---------------------------------------------------------------------------
// Socket discovery
// ---------------------------------------------------------------------------

/// Direct ArangoDB socket.
const DIRECT_SOCKET: &str = "/run/arangodb3/arangodb.sock";

/// Resolve the best available socket path from config and filesystem probing.
fn resolve_socket(config: &HadesConfig, read_only: bool) -> Option<PathBuf> {
    // 1. Explicit socket from config (set via YAML or env override)
    if let Some(socket) = config.effective_socket(read_only) {
        let path = PathBuf::from(socket);
        if path.exists() {
            debug!(socket = %path.display(), "using configured socket");
            return Some(path);
        }
        debug!(socket = %path.display(), "configured socket not found, probing");
    }

    // 2. Direct ArangoDB socket
    let direct = PathBuf::from(DIRECT_SOCKET);
    if direct.exists() {
        debug!(socket = %direct.display(), "using direct ArangoDB socket");
        return Some(direct);
    }

    // 3. No socket found — caller will use TCP
    debug!("no Unix socket found, will use TCP");
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_uri_tcp() {
        let client = ArangoClient {
            socket_path: None,
            base_url: "http://localhost:8529".into(),
            database: "test_db".into(),
            auth_header: None,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            unix_client: None,
            tcp_client: None,
        };

        let uri = client.build_uri("/_db/test_db/_api/version").unwrap();
        assert_eq!(
            uri.to_string(),
            "http://localhost:8529/_db/test_db/_api/version"
        );
    }

    #[test]
    fn test_build_uri_unix_socket() {
        let client = ArangoClient {
            socket_path: Some(PathBuf::from("/tmp/test.sock")),
            base_url: String::new(),
            database: "test_db".into(),
            auth_header: None,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            unix_client: None,
            tcp_client: None,
        };

        let uri = client.build_uri("/_db/test_db/_api/version").unwrap();
        // hyperlocal encodes the socket path in the URI authority
        let uri_str = uri.to_string();
        assert!(uri_str.contains("_api/version"), "URI: {uri_str}");
    }

    #[test]
    fn test_auth_header_encoding() {
        let client = ArangoClient::with_url(
            "http://localhost:8529",
            "test_db",
            "root",
            "secret",
        );
        let auth = client.auth_header.unwrap();
        assert!(auth.starts_with("Basic "));
        // Decode and verify
        let encoded = auth.strip_prefix("Basic ").unwrap();
        let decoded = String::from_utf8(
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, "root:secret");
    }

    #[test]
    fn test_client_from_config_defaults() {
        let config = HadesConfig::default();
        // This will fall through to TCP since no sockets exist in test env
        let client = ArangoClient::from_config(&config, true).unwrap();
        assert_eq!(client.database(), "NestedLearning");
        // No password in default config → no auth header
        assert!(client.auth_header.is_none());
    }
}
