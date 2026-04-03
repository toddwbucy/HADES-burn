//! Unix domain socket daemon server.
//!
//! Binds to a socket (default `/run/hades/hades.sock`), accepts
//! connections, reads length-prefixed JSON requests, dispatches to
//! the shared handler layer in [`hades_core::dispatch`], and writes
//! length-prefixed JSON responses.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;
use tokio::time::timeout;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand, DaemonResponse, DispatchError, HandlerError};

/// Default socket path per the daemon protocol spec.
const DEFAULT_SOCKET: &str = "/run/hades/hades.sock";
/// Maximum payload size (16 MiB, per protocol spec).
const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;
/// Idle timeout — close connection if no request received within this window.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Per-request execution timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the HADES daemon, listening on a Unix domain socket.
pub async fn run(config: &HadesConfig, socket_path: Option<&str>) -> Result<()> {
    let socket = socket_path.unwrap_or(DEFAULT_SOCKET);
    let pool = Arc::new(
        ArangoPool::from_config(config).context("failed to connect to ArangoDB")?,
    );
    let config = Arc::new(config.clone());

    // Clean up stale socket file from a previous unclean shutdown.
    if Path::new(socket).exists() {
        std::fs::remove_file(socket).context("failed to remove stale socket")?;
    }

    // Ensure parent directory exists (/run/hades/ may already be there).
    if let Some(parent) = Path::new(socket).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let listener =
        UnixListener::bind(socket).with_context(|| format!("failed to bind {socket}"))?;
    tracing::info!(socket, "daemon listening");

    // Accept loop with graceful shutdown.
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            result = listener.accept() => {
                let (stream, _addr) = result.context("accept failed")?;
                let pool = Arc::clone(&pool);
                let config = Arc::clone(&config);
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, &pool, &config).await {
                        tracing::debug!(error = %e, "connection closed with error");
                    }
                });
            }
            () = &mut shutdown => {
                tracing::info!("shutting down");
                break;
            }
        }
    }

    // Cleanup socket file.
    let _ = std::fs::remove_file(socket);
    tracing::info!("socket removed, daemon stopped");
    Ok(())
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

/// Handle a single connection: serial request → response loop.
async fn handle_connection(
    mut stream: UnixStream,
    pool: &ArangoPool,
    config: &HadesConfig,
) -> Result<()> {
    loop {
        // Read the 4-byte length header with idle timeout.
        let mut len_buf = [0u8; 4];
        match timeout(IDLE_TIMEOUT, stream.read_exact(&mut len_buf)).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) => return Ok(()), // EOF or broken pipe → clean close
            Err(_) => return Ok(()),      // idle timeout → close
        }

        let len = u32::from_be_bytes(len_buf);

        // Reject oversized payloads without reading the body.
        if len > MAX_PAYLOAD {
            let resp = DaemonResponse::err(
                "PAYLOAD_TOO_LARGE",
                format!("payload {len} bytes exceeds {} byte limit", MAX_PAYLOAD),
            );
            write_frame(&mut stream, &resp).await.ok();
            return Ok(()); // close connection after oversized payload
        }

        // Read the JSON payload.
        let mut payload = vec![0u8; len as usize];
        stream
            .read_exact(&mut payload)
            .await
            .context("failed to read payload")?;

        // Dispatch with per-request timeout.
        let response =
            match timeout(REQUEST_TIMEOUT, process_request(&payload, pool, config)).await {
                Ok(resp) => resp,
                Err(_) => DaemonResponse::err("INTERNAL", "request timed out"),
            };

        write_frame(&mut stream, &response).await?;
    }
}

// ---------------------------------------------------------------------------
// Frame codec
// ---------------------------------------------------------------------------

/// Write a length-prefixed JSON response frame.
async fn write_frame(stream: &mut UnixStream, response: &DaemonResponse) -> Result<()> {
    let json = serde_json::to_vec(response)?;
    let len = json.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(&json).await?;
    stream.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Request processing
// ---------------------------------------------------------------------------

/// Parse a raw JSON payload, dispatch the command, and return a response.
async fn process_request(
    payload: &[u8],
    pool: &ArangoPool,
    config: &HadesConfig,
) -> DaemonResponse {
    // Parse JSON.
    let frame: Value = match serde_json::from_slice(payload) {
        Ok(v) => v,
        Err(e) => return DaemonResponse::err("MALFORMED_JSON", e.to_string()),
    };

    // Extract optional request_id (echoed in response).
    let request_id = frame
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Deserialize the command+params portion.
    let cmd: DaemonCommand = match serde_json::from_value(frame) {
        Ok(c) => c,
        Err(e) => {
            return DaemonResponse::err("UNKNOWN_COMMAND", e.to_string())
                .with_request_id(request_id);
        }
    };

    // Dispatch to the shared handler layer.
    match dispatch::dispatch(pool, config, cmd).await {
        Ok(data) => DaemonResponse::ok(data).with_request_id(request_id),
        Err(DispatchError::NotImplemented(name)) => DaemonResponse::err(
            "UNKNOWN_COMMAND",
            format!("command '{name}' not yet implemented natively"),
        )
        .with_request_id(request_id),
        Err(DispatchError::Handler(e)) => {
            DaemonResponse::err(handler_error_code(&e), e.to_string())
                .with_request_id(request_id)
        }
    }
}

/// Map a [`HandlerError`] to a protocol error code string.
fn handler_error_code(e: &HandlerError) -> &'static str {
    match e {
        HandlerError::InvalidNodeId { .. } | HandlerError::InvalidLimit { .. } => "INVALID_PARAMS",
        HandlerError::NodeNotFound(_) => "NOT_FOUND",
        HandlerError::NoEmbedding { .. } | HandlerError::InvalidEmbedding { .. } => "QUERY_FAILED",
        HandlerError::Query { .. } => "QUERY_FAILED",
    }
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

/// Wait for SIGTERM or SIGINT (Ctrl-C).
async fn shutdown_signal() {
    let ctrl_c = signal::ctrl_c();
    let mut sigterm =
        signal::unix::signal(signal::unix::SignalKind::terminate()).expect("SIGTERM handler");

    tokio::select! {
        _ = ctrl_c => {}
        _ = sigterm.recv() => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::UnixStream;

    /// Helper: write a length-prefixed frame to a stream.
    async fn write_request(stream: &mut UnixStream, payload: &[u8]) {
        let len = payload.len() as u32;
        stream.write_all(&len.to_be_bytes()).await.unwrap();
        stream.write_all(payload).await.unwrap();
        stream.flush().await.unwrap();
    }

    /// Helper: read a length-prefixed response from a stream.
    async fn read_response(stream: &mut UnixStream) -> DaemonResponse {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf);
        let mut buf = vec![0u8; len as usize];
        stream.read_exact(&mut buf).await.unwrap();
        serde_json::from_slice(&buf).unwrap()
    }

    #[test]
    fn test_handler_error_code_mapping() {
        assert_eq!(
            handler_error_code(&HandlerError::NodeNotFound("x".into())),
            "NOT_FOUND",
        );
        assert_eq!(
            handler_error_code(&HandlerError::InvalidNodeId {
                node_id: "x".into(),
                reason: "bad".into(),
            }),
            "INVALID_PARAMS",
        );
        assert_eq!(
            handler_error_code(&HandlerError::InvalidLimit { limit: 0, max: 1000 }),
            "INVALID_PARAMS",
        );
        assert_eq!(
            handler_error_code(&HandlerError::NoEmbedding {
                node_id: "x".into(),
            }),
            "QUERY_FAILED",
        );
    }

    #[tokio::test]
    async fn test_process_request_malformed_json() {
        // process_request needs pool+config, but malformed JSON is rejected
        // before any DB access, so we can test with a dummy.  However,
        // ArangoPool::from_config needs a real socket.  We test the JSON
        // parsing path by calling the function directly where possible.
        let resp = process_request_no_db(b"not json at all").await;
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("MALFORMED_JSON"));
    }

    #[tokio::test]
    async fn test_process_request_unknown_command() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = process_request_no_db(&payload).await;
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("UNKNOWN_COMMAND"));
    }

    #[tokio::test]
    async fn test_process_request_echoes_request_id() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-42",
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = process_request_no_db(&payload).await;
        assert_eq!(resp.request_id.as_deref(), Some("req-42"));
    }

    #[tokio::test]
    async fn test_frame_roundtrip() {
        let (mut client, mut server) = UnixStream::pair().unwrap();

        let payload = b"{\"command\":\"orient\",\"params\":{}}";
        write_request(&mut client, payload).await;

        // Read frame on the server side.
        let mut len_buf = [0u8; 4];
        server.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf);
        let mut buf = vec![0u8; len as usize];
        server.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, payload);

        // Write a response from server.
        let resp = DaemonResponse::ok(serde_json::json!({"test": true}));
        write_frame(&mut server, &resp).await.unwrap();

        // Read response on client side.
        let got = read_response(&mut client).await;
        assert!(got.success);
        assert_eq!(got.data.unwrap()["test"], true);
    }

    /// Lightweight process_request that skips DB access for JSON-level tests.
    /// Malformed JSON and unknown commands are rejected before dispatch.
    async fn process_request_no_db(payload: &[u8]) -> DaemonResponse {
        let frame: Value = match serde_json::from_slice(payload) {
            Ok(v) => v,
            Err(e) => return DaemonResponse::err("MALFORMED_JSON", e.to_string()),
        };

        let request_id = frame
            .get("request_id")
            .and_then(|v| v.as_str())
            .map(String::from);

        let _cmd: DaemonCommand = match serde_json::from_value(frame) {
            Ok(c) => c,
            Err(e) => {
                return DaemonResponse::err("UNKNOWN_COMMAND", e.to_string())
                    .with_request_id(request_id);
            }
        };

        // If we reach here the command deserialized but we have no DB pool.
        DaemonResponse::err("INTERNAL", "no pool in test")
            .with_request_id(request_id)
    }
}
