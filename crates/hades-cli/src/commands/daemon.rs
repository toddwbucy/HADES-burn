//! Unix domain socket daemon server.
//!
//! Binds to a socket (default `/run/hades/hades.sock`), accepts
//! connections, reads length-prefixed JSON requests, hands each payload
//! to the transport-agnostic service layer in [`hades_core::service`],
//! and writes length-prefixed JSON responses.
//!
//! This module owns only transport concerns: socket lifecycle, framing,
//! payload limits, and idle timeouts. Parsing, session policy, tier
//! authorization, and dispatch live in `hades_core::service`. Unix
//! socket peers are local operators, so connections here run under
//! [`ConnectionPolicy::local_admin`].

use std::os::unix::fs::FileTypeExt;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;
use tokio::time::timeout;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::DaemonResponse;
use hades_core::service::{self, ConnectionPolicy};

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
    let pool = Arc::new(ArangoPool::from_config(config).context("failed to connect to ArangoDB")?);
    let config = Arc::new(config.clone());

    // Clean up stale socket — but only after verifying it's actually a socket
    // and that no live daemon is listening on it.
    if Path::new(socket).exists() {
        let meta = std::fs::metadata(socket).with_context(|| format!("failed to stat {socket}"))?;
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
///
/// Reads frames, delegates each payload to the service layer under the
/// local-admin policy, and writes response frames back.
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
            Err(_) => return Ok(()),     // idle timeout → close
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

        let response = service::handle_request(
            pool,
            config,
            ConnectionPolicy::local_admin(),
            &payload,
            REQUEST_TIMEOUT,
        )
        .await;

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

    #[tokio::test]
    async fn test_agent_access_denied_wire_protocol() {
        // End-to-end: send an agent-session admin command over the wire
        // and verify the daemon returns ACCESS_DENIED without dispatching.
        let (mut client, server) = UnixStream::pair().unwrap();

        // Dummy pool+config — dispatch is never reached (tier check fires first).
        // The pool just needs to exist; no real DB connection happens in this test.
        let mut config = HadesConfig::default();
        config.database.name = Some("bident_burn".to_string());
        let pool = ArangoPool::from_config(&config).unwrap();

        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "agent",
            "command": "db.aql",
            "params": { "aql": "RETURN 1" }
        }))
        .unwrap();

        // Run client I/O and server handler concurrently.
        let client_task = async {
            write_request(&mut client, &payload).await;
            let resp = read_response(&mut client).await;
            // Close client → handle_connection sees EOF and exits.
            drop(client);
            resp
        };

        let (resp, _) = tokio::join!(client_task, handle_connection(server, &pool, &config),);

        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
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
