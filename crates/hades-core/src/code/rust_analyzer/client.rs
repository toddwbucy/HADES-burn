//! JSON-RPC transport client for LSP servers.
//!
//! Manages a language server subprocess, framing messages with
//! `Content-Length` headers over stdin/stdout per the LSP base protocol.
//!
//! Uses an async reader task for response correlation and notification
//! buffering.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Notify, oneshot};

use super::RustAnalyzerError;

/// A pending request awaiting its response.
type ResponseSender = oneshot::Sender<Result<Value, RustAnalyzerError>>;

/// JSON-RPC LSP transport client.
///
/// Spawns a language server as a child process and communicates via
/// Content-Length framed JSON-RPC over stdin/stdout.
pub struct LspClient {
    /// Handle to the child process.
    child: Child,
    /// Stdin writer (wrapped in Mutex for exclusive access).
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    /// Next request ID to allocate.
    next_id: Arc<Mutex<i64>>,
    /// Pending request map: id → oneshot sender.
    pending: Arc<Mutex<HashMap<i64, ResponseSender>>>,
    /// Buffered server notifications.
    notifications: Arc<Mutex<Vec<Value>>>,
    /// Signalled when any notification arrives.
    notification_signal: Arc<Notify>,
    /// Reader task handle.
    reader_handle: Option<tokio::task::JoinHandle<()>>,
}

impl LspClient {
    /// Spawn the language server and start the reader task.
    pub async fn start(
        command: &str,
        args: &[&str],
        cwd: &std::path::Path,
    ) -> Result<Self, RustAnalyzerError> {
        let mut child = Command::new(command)
            .args(args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    RustAnalyzerError::NotFound(format!("{command}: {e}"))
                } else {
                    RustAnalyzerError::Process(format!("failed to spawn {command}: {e}"))
                }
            })?;

        let stdin = child.stdin.take()
            .ok_or_else(|| RustAnalyzerError::Process("no stdin handle".into()))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| RustAnalyzerError::Process("no stdout handle".into()))?;

        let pending: Arc<Mutex<HashMap<i64, ResponseSender>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let notifications: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let notification_signal = Arc::new(Notify::new());

        // Spawn the reader task.
        let reader_pending = Arc::clone(&pending);
        let reader_notifs = Arc::clone(&notifications);
        let reader_signal = Arc::clone(&notification_signal);
        let reader_stdin = Arc::new(Mutex::new(stdin));

        // Clone stdin Arc for the reader (to respond to server-initiated requests).
        let reader_stdin_clone = Arc::clone(&reader_stdin);

        let reader_handle = tokio::spawn(async move {
            reader_loop(
                stdout,
                reader_pending,
                reader_notifs,
                reader_signal,
                reader_stdin_clone,
            ).await;
        });

        Ok(Self {
            child,
            stdin: reader_stdin,
            next_id: Arc::new(Mutex::new(1)),
            pending,
            notifications,
            notification_signal,
            reader_handle: Some(reader_handle),
        })
    }

    /// Send a JSON-RPC request and await the response.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: std::time::Duration,
    ) -> Result<Value, RustAnalyzerError> {
        let id = {
            let mut next = self.next_id.lock().await;
            let id = *next;
            *next += 1;
            id
        };

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        self.send_message(&message).await?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                // Channel closed — reader task died.
                Err(RustAnalyzerError::Process("reader task closed channel".into()))
            }
            Err(_) => {
                // Timeout — clean up pending entry.
                self.pending.lock().await.remove(&id);
                Err(RustAnalyzerError::Timeout(format!(
                    "{method} (id={id}, timeout={timeout:?})"
                )))
            }
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(
        &self,
        method: &str,
        params: Value,
    ) -> Result<(), RustAnalyzerError> {
        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_message(&message).await
    }

    /// Drain all buffered notifications, optionally filtered by method.
    pub async fn drain_notifications(&self, method: Option<&str>) -> Vec<Value> {
        let mut notifs = self.notifications.lock().await;
        if let Some(m) = method {
            let (matching, remaining): (Vec<_>, Vec<_>) = notifs
                .drain(..)
                .partition(|n| n.get("method").and_then(Value::as_str) == Some(m));
            *notifs = remaining;
            matching
        } else {
            std::mem::take(&mut *notifs)
        }
    }

    /// Wait for a notification to arrive (with timeout).
    pub async fn wait_for_notification(&self, timeout: std::time::Duration) -> bool {
        tokio::time::timeout(timeout, self.notification_signal.notified())
            .await
            .is_ok()
    }

    /// Graceful shutdown: send shutdown request, then exit notification.
    pub async fn shutdown(mut self) -> Result<(), RustAnalyzerError> {
        // Send shutdown request (best effort).
        let _ = self.request(
            "shutdown",
            Value::Null,
            std::time::Duration::from_secs(5),
        ).await;

        // Send exit notification.
        let _ = self.notify("exit", Value::Null).await;

        // Wait for child to exit.
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            self.child.wait(),
        ).await;

        // Abort reader task.
        if let Some(handle) = self.reader_handle.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Check if the child process is still running.
    pub fn is_alive(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_none()
    }

    /// Write a JSON-RPC message with Content-Length framing.
    async fn send_message(&self, message: &Value) -> Result<(), RustAnalyzerError> {
        let body = serde_json::to_vec(message)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(header.as_bytes()).await?;
        stdin.write_all(&body).await?;
        stdin.flush().await?;
        Ok(())
    }
}

/// Background task: reads Content-Length framed messages from stdout
/// and dispatches them to the appropriate handler.
async fn reader_loop(
    stdout: tokio::process::ChildStdout,
    pending: Arc<Mutex<HashMap<i64, ResponseSender>>>,
    notifications: Arc<Mutex<Vec<Value>>>,
    notification_signal: Arc<Notify>,
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
) {
    let mut reader = BufReader::new(stdout);

    loop {
        // Read headers.
        let content_length = match read_headers(&mut reader).await {
            Some(len) => len,
            None => break, // EOF
        };

        // Read body.
        let mut body = vec![0u8; content_length];
        if reader.read_exact(&mut body).await.is_err() {
            break; // EOF or error
        }

        // Parse JSON.
        let message: Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Dispatch.
        if message.get("id").is_some()
            && (message.get("result").is_some() || message.get("error").is_some())
        {
            // Response to a request we sent.
            if let Some(id) = message["id"].as_i64() {
                let sender = pending.lock().await.remove(&id);
                if let Some(tx) = sender {
                    let result = if let Some(err) = message.get("error") {
                        Err(RustAnalyzerError::JsonRpc {
                            code: err["code"].as_i64().unwrap_or(-1),
                            message: err["message"]
                                .as_str()
                                .unwrap_or("unknown")
                                .to_string(),
                        })
                    } else {
                        Ok(message["result"].clone())
                    };
                    let _ = tx.send(result);
                }
            }
        } else if message.get("id").is_some() && message.get("method").is_some() {
            // Server-to-client request — auto-accept with null result.
            if let Some(id) = message.get("id") {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": null,
                });
                let body_bytes = serde_json::to_vec(&response).unwrap_or_default();
                let header = format!("Content-Length: {}\r\n\r\n", body_bytes.len());
                let mut stdin_guard = stdin.lock().await;
                let _ = stdin_guard.write_all(header.as_bytes()).await;
                let _ = stdin_guard.write_all(&body_bytes).await;
                let _ = stdin_guard.flush().await;
            }
        } else {
            // Notification from server.
            notifications.lock().await.push(message);
            notification_signal.notify_waiters();
        }
    }

    // Wake all pending requests so they don't hang.
    let mut pending = pending.lock().await;
    for (_, tx) in pending.drain() {
        let _ = tx.send(Err(RustAnalyzerError::Process(
            "reader task ended (server exited)".into(),
        )));
    }
}

/// Read LSP headers from the stream, return Content-Length.
async fn read_headers<R: AsyncBufReadExt + Unpin>(reader: &mut R) -> Option<usize> {
    let mut content_length: Option<usize> = None;
    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf).await {
            Ok(0) => return None, // EOF
            Ok(_) => {}
            Err(_) => return None,
        }

        let trimmed = line_buf.trim();
        if trimmed.is_empty() {
            break; // End of headers.
        }

        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().ok();
        }
    }

    content_length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_headers_parsing() {
        // Verify the header parser in isolation using a mock stream.
        let header = b"Content-Length: 42\r\n\r\n";
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(async {
            let mut reader = BufReader::new(&header[..]);
            read_headers(&mut reader).await
        });
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_read_headers_eof() {
        let header = b"";
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(async {
            let mut reader = BufReader::new(&header[..]);
            read_headers(&mut reader).await
        });
        assert_eq!(result, None);
    }
}
