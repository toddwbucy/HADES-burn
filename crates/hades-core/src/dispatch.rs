//! Shared command dispatch layer for CLI and daemon.
//!
//! Defines [`DaemonCommand`] — the set of commands the daemon exposes —
//! and [`dispatch`] which routes them to native Rust handlers.
//! Commands not yet ported to native Rust return
//! [`DispatchError::NotImplemented`] so the caller (daemon) can fall
//! back to subprocess invocation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::HadesConfig;
use crate::db::ArangoPool;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Dispatch-level errors.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// The command has no native Rust handler yet.
    #[error("command not yet implemented natively: {0}")]
    NotImplemented(String),

    /// Handler returned an error.
    #[error(transparent)]
    Handler(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Structured response from a dispatched command.
#[derive(Debug, Clone, Serialize)]
pub struct DaemonResponse {
    /// Echoed from request (null if not provided).
    pub request_id: Option<String>,
    /// Whether the command succeeded.
    pub success: bool,
    /// Result payload (null on error).
    pub data: Option<Value>,
    /// Error message (null on success).
    pub error: Option<String>,
    /// Machine-readable error code (null on success).
    pub error_code: Option<String>,
}

impl DaemonResponse {
    /// Build a success response.
    pub fn ok(data: Value) -> Self {
        Self {
            request_id: None,
            success: true,
            data: Some(data),
            error: None,
            error_code: None,
        }
    }

    /// Build an error response.
    pub fn err(code: &str, message: impl Into<String>) -> Self {
        Self {
            request_id: None,
            success: false,
            data: None,
            error: Some(message.into()),
            error_code: Some(code.to_string()),
        }
    }

    /// Set the request_id (echoed from client).
    pub fn with_request_id(mut self, id: Option<String>) -> Self {
        self.request_id = id;
        self
    }
}

// ---------------------------------------------------------------------------
// Command enum
// ---------------------------------------------------------------------------

/// Commands the daemon can dispatch.
///
/// Serializes with `#[serde(tag = "command", content = "params")]` so
/// JSON round-trips match the wire protocol:
/// `{"command": "db.query", "params": {"text": "...", "limit": 10}}`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", content = "params")]
pub enum DaemonCommand {
    // ── System ──────────────────────────────────────────────────────
    #[serde(rename = "orient")]
    Orient {
        collection: Option<String>,
    },

    #[serde(rename = "status")]
    Status {
        #[serde(default)]
        verbose: bool,
    },

    // ── Database ────────────────────────────────────────────────────
    #[serde(rename = "db.query")]
    DbQuery {
        text: String,
        #[serde(default)]
        limit: Option<u32>,
        collection: Option<String>,
        #[serde(default)]
        hybrid: bool,
        #[serde(default)]
        rerank: bool,
        #[serde(default)]
        structural: bool,
    },

    #[serde(rename = "db.aql")]
    DbAql {
        aql: String,
        #[serde(default)]
        bind: Option<Value>,
        limit: Option<u32>,
    },

    #[serde(rename = "db.get")]
    DbGet {
        collection: String,
        key: String,
    },

    #[serde(rename = "db.list")]
    DbList {
        collection: Option<String>,
        #[serde(default)]
        limit: Option<u32>,
        paper: Option<String>,
    },

    #[serde(rename = "db.insert")]
    DbInsert {
        collection: String,
        data: Value,
    },

    #[serde(rename = "db.update")]
    DbUpdate {
        collection: String,
        key: String,
        data: Value,
    },

    #[serde(rename = "db.delete")]
    DbDelete {
        collection: String,
        key: String,
    },

    #[serde(rename = "db.count")]
    DbCount {
        collection: String,
    },

    #[serde(rename = "db.collections")]
    DbCollections {},

    #[serde(rename = "db.stats")]
    DbStats {},

    #[serde(rename = "db.health")]
    DbHealth {
        #[serde(default)]
        verbose: bool,
    },

    #[serde(rename = "db.check")]
    DbCheck {
        document_id: String,
    },

    #[serde(rename = "db.recent")]
    DbRecent {
        #[serde(default)]
        limit: Option<u32>,
    },

    // ── Graph traversal ─────────────────────────────────────────────
    #[serde(rename = "db.graph.traverse")]
    DbGraphTraverse {
        start: String,
        #[serde(default = "default_direction")]
        direction: String,
        #[serde(default = "default_one")]
        min_depth: u32,
        #[serde(default = "default_one")]
        max_depth: u32,
        graph: Option<String>,
    },

    #[serde(rename = "db.graph.shortest_path")]
    DbGraphShortestPath {
        source: String,
        target: String,
        graph: Option<String>,
    },

    #[serde(rename = "db.graph.neighbors")]
    DbGraphNeighbors {
        vertex: String,
        #[serde(default = "default_direction_any")]
        direction: String,
        #[serde(default)]
        limit: Option<u32>,
    },

    #[serde(rename = "db.graph.list")]
    DbGraphList {},

    // ── Embeddings ──────────────────────────────────────────────────
    #[serde(rename = "embed.text")]
    EmbedText {
        text: String,
    },

    #[serde(rename = "graph_embed.embed")]
    GraphEmbedEmbed {
        node_id: String,
    },

    #[serde(rename = "graph_embed.neighbors")]
    GraphEmbedNeighbors {
        node_id: String,
        #[serde(default)]
        limit: Option<u32>,
    },

    // ── Tasks ───────────────────────────────────────────────────────
    #[serde(rename = "task.list")]
    TaskList {
        #[serde(default)]
        status: Option<String>,
        #[serde(rename = "type")]
        task_type: Option<String>,
        parent: Option<String>,
        #[serde(default)]
        limit: Option<u32>,
    },

    #[serde(rename = "task.show")]
    TaskShow {
        key: String,
    },

    #[serde(rename = "task.create")]
    TaskCreate {
        title: String,
        description: Option<String>,
        #[serde(rename = "type", default = "default_task_type")]
        task_type: String,
        parent: Option<String>,
        #[serde(default = "default_priority")]
        priority: String,
        #[serde(default)]
        tags: Vec<String>,
    },

    #[serde(rename = "task.update")]
    TaskUpdate {
        key: String,
        title: Option<String>,
        description: Option<String>,
        priority: Option<String>,
        #[serde(default)]
        add_tags: Vec<String>,
        #[serde(default)]
        remove_tags: Vec<String>,
    },

    #[serde(rename = "task.close")]
    TaskClose {
        key: String,
        message: Option<String>,
    },

    #[serde(rename = "task.context")]
    TaskContext {
        key: String,
    },

    // ── Smell ───────────────────────────────────────────────────────
    #[serde(rename = "smell.check")]
    SmellCheck {
        path: String,
        #[serde(default)]
        verbose: bool,
    },
}

// Serde defaults
fn default_direction() -> String { "outbound".to_string() }
fn default_direction_any() -> String { "any".to_string() }
fn default_one() -> u32 { 1 }
fn default_task_type() -> String { "task".to_string() }
fn default_priority() -> String { "medium".to_string() }

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Route a [`DaemonCommand`] to its native handler.
///
/// Returns the handler's JSON result on success.  Commands without a
/// native Rust implementation return [`DispatchError::NotImplemented`]
/// so the daemon can fall back to subprocess invocation.
pub async fn dispatch(
    pool: &ArangoPool,
    _config: &HadesConfig,
    cmd: DaemonCommand,
) -> Result<Value, DispatchError> {
    match cmd {
        DaemonCommand::GraphEmbedEmbed { node_id } => {
            handlers::graph_embed_embed(pool, &node_id)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::GraphEmbedNeighbors { node_id, limit } => {
            handlers::graph_embed_neighbors(pool, &node_id, limit.unwrap_or(10))
                .await
                .map_err(DispatchError::Handler)
        }

        // All other commands are not yet implemented natively.
        // The daemon (P6.3) will fall back to Python subprocess.
        other => Err(DispatchError::NotImplemented(
            serde_json::to_value(&other)
                .ok()
                .and_then(|v| v.get("command").and_then(|c| c.as_str()).map(String::from))
                .unwrap_or_else(|| format!("{other:?}")),
        )),
    }
}

// ---------------------------------------------------------------------------
// Native handlers
// ---------------------------------------------------------------------------

mod handlers {
    use anyhow::{Context, Result};
    use serde_json::json;

    use crate::db::crud::{list_collections, CollectionInfo};
    use crate::db::query::{self, ExecutionTarget};
    use crate::db::ArangoPool;

    /// Look up the pre-computed structural embedding for a node.
    pub async fn graph_embed_embed(
        pool: &ArangoPool,
        node_id: &str,
    ) -> Result<serde_json::Value> {
        let (col, key) = parse_node_id(node_id)?;

        let aql = "FOR d IN @@col FILTER d._key == @key \
                    RETURN { _id: d._id, structural_embedding: d.structural_embedding, \
                    title: d.title, name: d.name }";

        let node = query::query_single(
            pool,
            aql,
            Some(&json!({ "@col": col, "key": key })),
            ExecutionTarget::Reader,
        )
        .await
        .context("failed to query node")?
        .ok_or_else(|| anyhow::anyhow!("node not found: {node_id}"))?;

        let embedding = node
            .get("structural_embedding")
            .filter(|v| !v.is_null())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no structural embedding for {node_id} — run 'hades graph-embed train' first"
                )
            })?;

        let embed_dim = embedding
            .as_array()
            .map(|a| a.len())
            .ok_or_else(|| {
                anyhow::anyhow!("structural embedding is not an array for {node_id}")
            })?;

        if embed_dim == 0 {
            anyhow::bail!("structural embedding is empty (0 dimensions) for {node_id}");
        }

        let label = node
            .get("title")
            .and_then(|v| v.as_str())
            .or_else(|| node.get("name").and_then(|v| v.as_str()))
            .unwrap_or(key);

        Ok(json!({
            "node_id": node_id,
            "label": label,
            "embedding_dim": embed_dim,
            "embedding": embedding,
        }))
    }

    /// Find k-nearest structural neighbors by dot-product similarity.
    pub async fn graph_embed_neighbors(
        pool: &ArangoPool,
        node_id: &str,
        limit: u32,
    ) -> Result<serde_json::Value> {
        let (col, key) = parse_node_id(node_id)?;

        // Fetch target embedding.
        let emb_aql = "FOR d IN @@col FILTER d._key == @key RETURN d.structural_embedding";

        let emb_value = query::query_single(
            pool,
            emb_aql,
            Some(&json!({ "@col": col, "key": key })),
            ExecutionTarget::Reader,
        )
        .await
        .context("failed to query node embedding")?
        .ok_or_else(|| anyhow::anyhow!("node not found: {node_id}"))?;

        if emb_value.is_null() {
            anyhow::bail!(
                "no structural embedding for {node_id} — run 'hades graph-embed train' first"
            );
        }

        let embed_dim = emb_value
            .as_array()
            .map(|a| a.len())
            .ok_or_else(|| {
                anyhow::anyhow!("structural embedding is not an array for {node_id}")
            })?;

        if embed_dim == 0 {
            anyhow::bail!("structural embedding is empty (0 dimensions) for {node_id}");
        }

        // Discover document collections.
        let collections: Vec<CollectionInfo> = list_collections(pool, true)
            .await
            .context("failed to list collections")?
            .into_iter()
            .filter(|c| c.collection_type == 2)
            .collect();

        // Search each collection.
        let mut all_neighbors: Vec<serde_json::Value> = Vec::new();
        let mut collections_succeeded: usize = 0;
        let mut last_error: Option<String> = None;

        let search_aql =
            "LET te = @target_emb \
             FOR d IN @@col \
               FILTER d.structural_embedding != null \
               FILTER LENGTH(d.structural_embedding) == @dim \
               FILTER d._id != @target_id \
               LET sim = SUM(FOR i IN 0..@dim_minus_1 RETURN te[i] * d.structural_embedding[i]) \
               SORT sim DESC \
               LIMIT @k \
               RETURN { \
                 id: d._id, \
                 label: NOT_NULL(d.title, d.name, d._key), \
                 collection: @col_name, \
                 similarity: ROUND(sim * 10000) / 10000 \
               }";

        for col_info in &collections {
            let bind_vars = json!({
                "@col": col_info.name,
                "col_name": col_info.name,
                "target_emb": emb_value,
                "dim": embed_dim,
                "dim_minus_1": embed_dim - 1,
                "target_id": node_id,
                "k": limit,
            });

            match query::query(
                pool,
                search_aql,
                Some(&bind_vars),
                None,
                false,
                ExecutionTarget::Reader,
            )
            .await
            {
                Ok(result) => {
                    collections_succeeded += 1;
                    all_neighbors.extend(result.results);
                }
                Err(e) => {
                    tracing::debug!(
                        collection = col_info.name,
                        error = %e,
                        "skipping collection"
                    );
                    last_error = Some(e.to_string());
                }
            }
        }

        if collections_succeeded == 0 && !collections.is_empty() {
            anyhow::bail!(
                "all {count} collection queries failed; last error: {err}",
                count = collections.len(),
                err = last_error.as_deref().unwrap_or("unknown"),
            );
        }

        // Sort globally and take top-k.
        all_neighbors.sort_by(|a, b| {
            let sa = a.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let sb = b.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_neighbors.truncate(limit as usize);

        Ok(json!({
            "query_node": node_id,
            "k": limit,
            "neighbors": all_neighbors,
        }))
    }

    /// Parse and validate a `collection/key` node ID.
    fn parse_node_id(node_id: &str) -> Result<(&str, &str)> {
        let (col, key) = node_id
            .split_once('/')
            .ok_or_else(|| {
                anyhow::anyhow!("node ID must be in 'collection/key' format, got: {node_id}")
            })?;

        if col.is_empty() {
            anyhow::bail!("node ID has empty collection name: {node_id}");
        }
        if key.is_empty() {
            anyhow::bail!("node ID has empty key: {node_id}");
        }

        Ok((col, key))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_roundtrip_orient() {
        let cmd = DaemonCommand::Orient { collection: Some("test".into()) };
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(json["command"], "orient");
        assert_eq!(json["params"]["collection"], "test");

        let back: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(back, DaemonCommand::Orient { collection: Some(c) } if c == "test"));
    }

    #[test]
    fn test_command_roundtrip_db_query() {
        let json = serde_json::json!({
            "command": "db.query",
            "params": { "text": "attention", "hybrid": true }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::DbQuery { ref text, hybrid: true, .. } if text == "attention"));
    }

    #[test]
    fn test_command_roundtrip_graph_embed() {
        let json = serde_json::json!({
            "command": "graph_embed.embed",
            "params": { "node_id": "hope_axioms/ax_001" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::GraphEmbedEmbed { ref node_id } if node_id == "hope_axioms/ax_001"));
    }

    #[test]
    fn test_command_defaults() {
        let json = serde_json::json!({
            "command": "graph_embed.neighbors",
            "params": { "node_id": "test/1" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::GraphEmbedNeighbors { limit: None, .. }));
    }

    #[test]
    fn test_response_ok() {
        let resp = DaemonResponse::ok(serde_json::json!({"count": 42}));
        assert!(resp.success);
        assert!(resp.error.is_none());
        assert_eq!(resp.data.unwrap()["count"], 42);
    }

    #[test]
    fn test_response_err() {
        let resp = DaemonResponse::err("NOT_FOUND", "node not found");
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("NOT_FOUND"));
        assert!(resp.data.is_none());
    }

    #[test]
    fn test_response_with_request_id() {
        let resp = DaemonResponse::ok(serde_json::json!({}))
            .with_request_id(Some("req-1".into()));
        assert_eq!(resp.request_id.as_deref(), Some("req-1"));
    }

    #[test]
    fn test_unknown_command_deserialize_fails() {
        let json = serde_json::json!({
            "command": "nonexistent.command",
            "params": {}
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }
}
