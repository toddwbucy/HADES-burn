//! Unix domain socket daemon server.
//!
//! Binds to a socket (default `/run/hades/hades.sock`), accepts
//! connections, reads length-prefixed JSON requests, dispatches to
//! the shared handler layer in [`hades_core::dispatch`], and writes
//! length-prefixed JSON responses.

use std::os::unix::fs::FileTypeExt;
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

    // Clean up stale socket — but only after verifying it's actually a socket
    // and that no live daemon is listening on it.
    if Path::new(socket).exists() {
        let meta = std::fs::metadata(socket)
            .with_context(|| format!("failed to stat {socket}"))?;
        if !meta.file_type().is_socket() {
            anyhow::bail!("{socket} exists but is not a Unix socket — refusing to overwrite");
        }
        // Probe for a live daemon.
        if std::os::unix::net::UnixStream::connect(socket).is_ok() {
            anyhow::bail!("{socket} is in use — another daemon is already running");
        }
        // Stale socket from a previous unclean shutdown.
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

        // Read the JSON payload with timeout (client may stall after header).
        let mut payload = vec![0u8; len as usize];
        match timeout(IDLE_TIMEOUT, stream.read_exact(&mut payload)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e).context("failed to read payload"),
            Err(_) => return Ok(()), // client stalled mid-payload → close
        }

        // Parse request_id and command before applying dispatch timeout,
        // so request_id is available for all error responses.
        let (request_id, cmd) = match parse_request(&payload) {
            Ok(parsed) => parsed,
            Err(resp) => {
                write_frame(&mut stream, &resp).await?;
                continue; // keep connection open for next request
            }
        };

        // Preserve command payload for NotImplemented subprocess fallback.
        let cmd_value = serde_json::to_value(&cmd).ok();

        // Dispatch with per-request timeout — request_id always available.
        let response = match timeout(REQUEST_TIMEOUT, dispatch::dispatch(pool, config, cmd)).await
        {
            Ok(Ok(data)) => DaemonResponse::ok(data).with_request_id(request_id),
            Ok(Err(DispatchError::NotImplemented(name))) => {
                let mut resp = DaemonResponse::err(
                    "NOT_IMPLEMENTED",
                    format!("command '{name}' not yet implemented natively"),
                );
                resp.data = cmd_value;
                resp.with_request_id(request_id)
            }
            Ok(Err(DispatchError::Handler(e))) => {
                DaemonResponse::err(handler_error_code(&e), e.to_string())
                    .with_request_id(request_id)
            }
            Err(_) => DaemonResponse::err("INTERNAL", "request timed out")
                .with_request_id(request_id),
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
// Request parsing
// ---------------------------------------------------------------------------

/// Parse a raw JSON payload into a request_id and [`DaemonCommand`].
///
/// Returns `Err(DaemonResponse)` on parse failure — the caller sends it
/// directly to the client.
fn parse_request(payload: &[u8]) -> Result<(Option<String>, DaemonCommand), DaemonResponse> {
    let frame: Value = serde_json::from_slice(payload)
        .map_err(|e| DaemonResponse::err("MALFORMED_JSON", e.to_string()))?;

    let request_id = frame
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    let cmd: DaemonCommand = serde_json::from_value(frame).map_err(|e| {
        DaemonResponse::err("UNKNOWN_COMMAND", e.to_string()).with_request_id(request_id.clone())
    })?;

    Ok((request_id, cmd))
}

/// Map a [`HandlerError`] to a protocol error code string.
fn handler_error_code(e: &HandlerError) -> &'static str {
    match e {
        HandlerError::InvalidNodeId { .. }
        | HandlerError::InvalidLimit { .. }
        | HandlerError::InvalidParameter { .. } => "INVALID_PARAMS",
        HandlerError::NodeNotFound(_) | HandlerError::DocumentNotFound { .. } => "NOT_FOUND",
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

    #[test]
    fn test_parse_request_malformed_json() {
        let resp = parse_request(b"not json at all").unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("MALFORMED_JSON"));
    }

    #[test]
    fn test_parse_request_unknown_command() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("UNKNOWN_COMMAND"));
    }

    #[test]
    fn test_parse_request_echoes_request_id() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-42",
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert_eq!(resp.request_id.as_deref(), Some("req-42"));
    }

    #[test]
    fn test_parse_request_valid_command() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-1",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let (request_id, _cmd) = parse_request(&payload).unwrap();
        assert_eq!(request_id.as_deref(), Some("req-1"));
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

}
