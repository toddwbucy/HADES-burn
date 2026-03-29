//! ArangoDB connection pool with split read/write clients.
//!
//! `ArangoPool` holds one or two `ArangoClient` instances — a reader and
//! a writer.  When both resolve to the same socket (common in single-node
//! setups), they share a single underlying hyper client.

use std::time::Instant;

use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use super::error::ArangoError;
use super::transport::ArangoClient;
use crate::config::HadesConfig;

/// Connection pool holding split read/write ArangoDB clients.
///
/// Construct via [`ArangoPool::from_config`] for production or
/// [`ArangoPool::new`] for testing with explicit clients.
#[derive(Clone, Debug)]
pub struct ArangoPool {
    reader: ArangoClient,
    writer: ArangoClient,
    /// True when reader and writer share the same underlying client.
    shared: bool,
}

impl ArangoPool {
    /// Build a pool from the HADES configuration.
    ///
    /// Creates separate RO and RW clients when different sockets are
    /// configured; shares a single client otherwise.
    pub fn from_config(config: &HadesConfig) -> Result<Self, ArangoError> {
        let reader = ArangoClient::from_config(config, true)?;
        let writer = ArangoClient::from_config(config, false)?;

        let shared = reader.socket_path() == writer.socket_path();
        if shared {
            debug!("reader and writer share the same socket");
        } else {
            debug!(
                reader = ?reader.socket_path(),
                writer = ?writer.socket_path(),
                "reader and writer use separate sockets"
            );
        }

        Ok(Self {
            reader,
            writer,
            shared,
        })
    }

    /// Build a pool from explicit clients (for testing).
    pub fn new(reader: ArangoClient, writer: ArangoClient) -> Self {
        let shared = reader.socket_path() == writer.socket_path();
        Self {
            reader,
            writer,
            shared,
        }
    }

    /// The read-only client.
    pub fn reader(&self) -> &ArangoClient {
        &self.reader
    }

    /// The read-write client.
    pub fn writer(&self) -> &ArangoClient {
        &self.writer
    }

    /// The target database name (same for both clients).
    pub fn database(&self) -> &str {
        self.reader.database()
    }

    /// Whether reader and writer share the same underlying connection.
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    /// Check health of all clients by hitting `GET /_api/version`.
    ///
    /// Returns the server version string on success.  Logs latency and
    /// any errors via tracing.
    #[instrument(skip(self), fields(db = %self.reader.database()))]
    pub async fn health_check(&self) -> HealthStatus {
        let start = Instant::now();

        let reader_ok;
        let writer_ok;
        let mut version = String::new();

        match self.check_client(&self.reader, "reader").await {
            Ok(v) => {
                debug!(role = "reader", latency_ms = start.elapsed().as_millis() as u64, "healthy");
                version = v;
                reader_ok = true;
            }
            Err(e) => {
                warn!(role = "reader", error = %e, "health check failed");
                reader_ok = false;
            }
        }

        if self.shared {
            writer_ok = reader_ok;
        } else {
            let writer_start = Instant::now();
            match self.check_client(&self.writer, "writer").await {
                Ok(v) => {
                    debug!(role = "writer", latency_ms = writer_start.elapsed().as_millis() as u64, "healthy");
                    if version.is_empty() {
                        version = v;
                    }
                    writer_ok = true;
                }
                Err(e) => {
                    warn!(role = "writer", error = %e, "health check failed");
                    writer_ok = false;
                }
            }
        }

        if reader_ok && writer_ok {
            info!(
                version = %version,
                shared = self.shared,
                latency_ms = start.elapsed().as_millis() as u64,
                "ArangoDB healthy"
            );
        }

        HealthStatus {
            version,
            reader_ok,
            writer_ok,
            shared: self.shared,
        }
    }

    /// Ping a single client and extract the version string.
    async fn check_client(
        &self,
        client: &ArangoClient,
        role: &str,
    ) -> Result<String, ArangoError> {
        let resp = client.get("version").await?;
        let version = resp
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        debug!(role, version, "client responded");
        Ok(version.to_string())
    }
}

/// Result of a health check.
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// ArangoDB server version string.
    pub version: String,
    /// Whether the reader client is healthy.
    pub reader_ok: bool,
    /// Whether the writer client is healthy.
    pub writer_ok: bool,
    /// Whether reader and writer share the same connection.
    pub shared: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_from_config_shared() {
        let mut config = HadesConfig::default();
        config.database.name = "test_pool_db".to_string();
        let pool = ArangoPool::from_config(&config).unwrap();
        assert!(pool.is_shared());
        assert_eq!(pool.database(), "test_pool_db");
    }

    #[test]
    fn test_pool_accessors() {
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();
        // Both should target the same database
        assert_eq!(pool.reader().database(), pool.writer().database());
    }
}
