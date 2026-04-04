//! HADES daemon socket client.
//!
//! Connects to the daemon at `/run/hades/hades.sock` (or a custom path),
//! sends length-prefixed JSON requests, and reads length-prefixed JSON
//! responses.  Auto-reconnects once on broken pipe so callers survive
//! daemon restarts between requests.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::timeout;

use crate::dispatch::{DaemonCommand, DaemonResponse};
#[cfg(test)]
use crate::dispatch::{OrientParams, StatusParams};

/// Default socket path per the daemon protocol spec.
pub const DEFAULT_SOCKET: &str = "/run/hades/hades.sock";
/// Maximum payload size (16 MiB, per protocol spec).
const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;
/// Default timeout for connect and read operations.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from the daemon client.
#[derive(Debug, thiserror::Error)]
pub enum DaemonClientError {
    /// Failed to connect to the daemon socket.
    #[error("failed to connect to {path}")]
    Connect {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A connect or read operation timed out.
    #[error("operation timed out")]
    Timeout,

    /// The serialized request exceeds the 16 MiB protocol limit.
    #[error("payload exceeds 16 MiB limit: {0} bytes")]
    PayloadTooLarge(usize),

    /// Transport-level I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A client for the HADES daemon socket.
///
/// Holds a persistent Unix stream connection.  If a request fails with
/// a broken-pipe or connection-reset error, the client reconnects once
/// and retries automatically.
#[derive(Debug)]
pub struct DaemonClient {
    stream: UnixStream,
    path: PathBuf,
    timeout: Duration,
}

impl DaemonClient {
    /// Connect to the daemon at the given socket path with the default
    /// 10-second timeout.
    pub async fn connect(path: impl AsRef<Path>) -> Result<Self, DaemonClientError> {
        Self::connect_with_timeout(path, DEFAULT_TIMEOUT).await
    }

    /// Connect with a custom timeout for both connect and read operations.
    pub async fn connect_with_timeout(
        path: impl AsRef<Path>,
        timeout_dur: Duration,
    ) -> Result<Self, DaemonClientError> {
        let path = path.as_ref().to_path_buf();
        let stream = timeout(timeout_dur, UnixStream::connect(&path))
            .await
            .map_err(|_| DaemonClientError::Timeout)?
            .map_err(|e| DaemonClientError::Connect {
                path: path.clone(),
                source: e,
            })?;
        Ok(Self {
            stream,
            path,
            timeout: timeout_dur,
        })
    }

    /// Send a command and return the daemon's response.
    pub async fn request(
        &mut self,
        cmd: DaemonCommand,
    ) -> Result<DaemonResponse, DaemonClientError> {
        self.request_inner(None, cmd).await
    }

    /// Send a command with a client-chosen `request_id`.
    pub async fn request_with_id(
        &mut self,
        request_id: impl Into<String>,
        cmd: DaemonCommand,
    ) -> Result<DaemonResponse, DaemonClientError> {
        self.request_inner(Some(request_id.into()), cmd).await
    }

    /// Inner request handler with optional request_id and auto-reconnect.
    async fn request_inner(
        &mut self,
        request_id: Option<String>,
        cmd: DaemonCommand,
    ) -> Result<DaemonResponse, DaemonClientError> {
        // Build the wire-format JSON: DaemonCommand fields + optional request_id.
        let mut frame = serde_json::to_value(&cmd)?;
        if let Some(ref id) = request_id {
            frame["request_id"] = Value::String(id.clone());
        }
        let payload = serde_json::to_vec(&frame)?;

        if payload.len() > MAX_PAYLOAD as usize {
            return Err(DaemonClientError::PayloadTooLarge(payload.len()));
        }

        // Try once; on broken pipe / connection reset, reconnect and retry.
        match self.send_recv(&payload).await {
            Ok(resp) => Ok(resp),
            Err(DaemonClientError::Io(ref e)) if is_connection_lost(e) => {
                tracing::debug!("connection lost, reconnecting");
                self.reconnect().await?;
                self.send_recv(&payload).await
            }
            Err(e) => Err(e),
        }
    }

    /// Write a request frame and read the response frame.
    async fn send_recv(&mut self, payload: &[u8]) -> Result<DaemonResponse, DaemonClientError> {
        // Write length-prefixed request with timeout.
        let len = payload.len() as u32;
        timeout(self.timeout, self.stream.write_all(&len.to_be_bytes()))
            .await
            .map_err(|_| DaemonClientError::Timeout)??;
        timeout(self.timeout, self.stream.write_all(payload))
            .await
            .map_err(|_| DaemonClientError::Timeout)??;
        timeout(self.timeout, self.stream.flush())
            .await
            .map_err(|_| DaemonClientError::Timeout)??;

        // Read length-prefixed response with timeout.
        let mut len_buf = [0u8; 4];
        timeout(self.timeout, self.stream.read_exact(&mut len_buf))
            .await
            .map_err(|_| DaemonClientError::Timeout)??;
        let resp_len = u32::from_be_bytes(len_buf);

        if resp_len > MAX_PAYLOAD {
            return Err(DaemonClientError::PayloadTooLarge(resp_len as usize));
        }

        let mut resp_buf = vec![0u8; resp_len as usize];
        timeout(self.timeout, self.stream.read_exact(&mut resp_buf))
            .await
            .map_err(|_| DaemonClientError::Timeout)??;

        let response: DaemonResponse = serde_json::from_slice(&resp_buf)?;
        Ok(response)
    }

    /// Reconnect to the same socket path.
    async fn reconnect(&mut self) -> Result<(), DaemonClientError> {
        self.stream = timeout(self.timeout, UnixStream::connect(&self.path))
            .await
            .map_err(|_| DaemonClientError::Timeout)?
            .map_err(|e| DaemonClientError::Connect {
                path: self.path.clone(),
                source: e,
            })?;
        Ok(())
    }
}

/// Check if an I/O error indicates a lost connection (retryable).
fn is_connection_lost(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::UnexpectedEof
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::UnixListener;

    /// Spin up a tiny server that reads one request frame and sends a
    /// canned success response.
    async fn mini_server(listener: UnixListener) {
        let (mut stream, _) = listener.accept().await.unwrap();

        // Read request frame.
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        stream.read_exact(&mut buf).await.unwrap();

        // Send response.
        let resp = DaemonResponse::ok(serde_json::json!({"echo": true}));
        let json = serde_json::to_vec(&resp).unwrap();
        stream
            .write_all(&(json.len() as u32).to_be_bytes())
            .await
            .unwrap();
        stream.write_all(&json).await.unwrap();
        stream.flush().await.unwrap();
    }

    fn test_socket_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hades-test-{}-{name}.sock", std::process::id()))
    }

    #[tokio::test]
    async fn test_request_roundtrip() {
        let sock = test_socket_path("roundtrip");
        let _ = std::fs::remove_file(&sock);

        let listener = UnixListener::bind(&sock).unwrap();
        tokio::spawn(mini_server(listener));

        let mut client = DaemonClient::connect(&sock).await.unwrap();
        let resp = client
            .request(DaemonCommand::Orient(OrientParams {
                collection: None,
            }))
            .await
            .unwrap();

        assert!(resp.success);
        assert_eq!(resp.data.unwrap()["echo"], true);

        let _ = std::fs::remove_file(&sock);
    }

    #[tokio::test]
    async fn test_request_with_id() {
        let sock = test_socket_path("reqid");
        let _ = std::fs::remove_file(&sock);

        let listener = UnixListener::bind(&sock).unwrap();
        // Server that echoes request_id.
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut len_buf = [0u8; 4];
            stream.read_exact(&mut len_buf).await.unwrap();
            let len = u32::from_be_bytes(len_buf) as usize;
            let mut buf = vec![0u8; len];
            stream.read_exact(&mut buf).await.unwrap();

            // Extract request_id from the frame.
            let frame: Value = serde_json::from_slice(&buf).unwrap();
            let rid = frame
                .get("request_id")
                .and_then(|v| v.as_str())
                .map(String::from);

            let resp = DaemonResponse::ok(serde_json::json!({})).with_request_id(rid);
            let json = serde_json::to_vec(&resp).unwrap();
            stream
                .write_all(&(json.len() as u32).to_be_bytes())
                .await
                .unwrap();
            stream.write_all(&json).await.unwrap();
            stream.flush().await.unwrap();
        });

        let mut client = DaemonClient::connect(&sock).await.unwrap();
        let resp = client
            .request_with_id("my-req-1", DaemonCommand::Status(StatusParams { verbose: false }))
            .await
            .unwrap();

        assert!(resp.success);
        assert_eq!(resp.request_id.as_deref(), Some("my-req-1"));

        let _ = std::fs::remove_file(&sock);
    }

    #[tokio::test]
    async fn test_connect_nonexistent_socket() {
        let result = DaemonClient::connect("/tmp/hades-nonexistent-test.sock").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DaemonClientError::Connect { .. }));
    }

    #[test]
    fn test_is_connection_lost() {
        assert!(is_connection_lost(&std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "broken"
        )));
        assert!(is_connection_lost(&std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "reset"
        )));
        assert!(!is_connection_lost(&std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied"
        )));
    }
}
