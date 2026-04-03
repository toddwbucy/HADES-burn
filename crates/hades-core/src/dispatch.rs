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
use crate::db::{ArangoError, ArangoPool};

/// Maximum value accepted for `limit` parameters in dispatch handlers.
const MAX_LIMIT: u32 = 1000;

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
    Handler(#[from] HandlerError),
}

/// Errors from native command handlers (typed per CLAUDE.md convention).
#[derive(Debug, thiserror::Error)]
pub enum HandlerError {
    /// The node ID is not in `collection/key` format.
    #[error("invalid node ID '{node_id}': {reason}")]
    InvalidNodeId { node_id: String, reason: String },

    /// The requested node does not exist.
    #[error("node not found: {0}")]
    NodeNotFound(String),

    /// The node exists but has no structural embedding.
    #[error("no structural embedding for {node_id} — run 'hades graph-embed train' first")]
    NoEmbedding { node_id: String },

    /// The structural embedding is present but malformed.
    #[error("invalid structural embedding for {node_id}: {reason}")]
    InvalidEmbedding { node_id: String, reason: String },

    /// A `limit` parameter is out of bounds.
    #[error("limit must be 1..={max}, got {limit}")]
    InvalidLimit { limit: u32, max: u32 },

    /// A database query failed.
    #[error("{context}")]
    Query {
        context: String,
        #[source]
        source: ArangoError,
    },
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Structured response from a dispatched command.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

// ---------------------------------------------------------------------------
// Param structs (deny_unknown_fields for implemented commands)
// ---------------------------------------------------------------------------

/// Params for `graph_embed.embed`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphEmbedEmbedParams {
    pub node_id: String,
}

/// Params for `graph_embed.neighbors`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphEmbedNeighborsParams {
    pub node_id: String,
    #[serde(default)]
    pub limit: Option<u32>,
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
    GraphEmbedEmbed(GraphEmbedEmbedParams),

    #[serde(rename = "graph_embed.neighbors")]
    GraphEmbedNeighbors(GraphEmbedNeighborsParams),

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
        DaemonCommand::GraphEmbedEmbed(params) => {
            handlers::graph_embed_embed(pool, &params.node_id)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::GraphEmbedNeighbors(params) => {
            let limit = params.limit.unwrap_or(10);
            if limit == 0 {
                return Err(DispatchError::Handler(HandlerError::InvalidLimit {
                    limit,
                    max: MAX_LIMIT,
                }));
            }
            let limit = limit.min(MAX_LIMIT);
            handlers::graph_embed_neighbors(pool, &params.node_id, limit)
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
    use serde_json::json;

    use super::HandlerError;
    use crate::db::crud::{list_collections, CollectionInfo};
    use crate::db::query::{self, ExecutionTarget};
    use crate::db::ArangoPool;

    /// Look up the pre-computed structural embedding for a node.
    pub async fn graph_embed_embed(
        pool: &ArangoPool,
        node_id: &str,
    ) -> Result<serde_json::Value, HandlerError> {
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
        .map_err(|e| HandlerError::Query {
            context: "failed to query node".into(),
            source: e,
        })?
        .ok_or_else(|| HandlerError::NodeNotFound(node_id.to_string()))?;

        let embedding = node
            .get("structural_embedding")
            .filter(|v| !v.is_null())
            .ok_or_else(|| HandlerError::NoEmbedding {
                node_id: node_id.to_string(),
            })?;

        let embed_dim = embedding
            .as_array()
            .map(|a| a.len())
            .ok_or_else(|| HandlerError::InvalidEmbedding {
                node_id: node_id.to_string(),
                reason: "not an array".into(),
            })?;

        if embed_dim == 0 {
            return Err(HandlerError::InvalidEmbedding {
                node_id: node_id.to_string(),
                reason: "empty (0 dimensions)".into(),
            });
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
    ) -> Result<serde_json::Value, HandlerError> {
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
        .map_err(|e| HandlerError::Query {
            context: "failed to query node embedding".into(),
            source: e,
        })?
        .ok_or_else(|| HandlerError::NodeNotFound(node_id.to_string()))?;

        if emb_value.is_null() {
            return Err(HandlerError::NoEmbedding {
                node_id: node_id.to_string(),
            });
        }

        let embed_dim = emb_value
            .as_array()
            .map(|a| a.len())
            .ok_or_else(|| HandlerError::InvalidEmbedding {
                node_id: node_id.to_string(),
                reason: "not an array".into(),
            })?;

        if embed_dim == 0 {
            return Err(HandlerError::InvalidEmbedding {
                node_id: node_id.to_string(),
                reason: "empty (0 dimensions)".into(),
            });
        }

        // Discover document collections.
        let collections: Vec<CollectionInfo> = list_collections(pool, true)
            .await
            .map_err(|e| HandlerError::Query {
                context: "failed to list collections".into(),
                source: e,
            })?
            .into_iter()
            .filter(|c| c.collection_type == CollectionInfo::DOCUMENT_TYPE)
            .collect();

        // Search each collection — fail fast on first error.
        let mut all_neighbors: Vec<serde_json::Value> = Vec::new();

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
                 similarity: sim \
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

            let result = query::query(
                pool,
                search_aql,
                Some(&bind_vars),
                None,
                false,
                ExecutionTarget::Reader,
            )
            .await
            .map_err(|e| HandlerError::Query {
                context: format!(
                    "neighbor search failed on collection '{}'",
                    col_info.name,
                ),
                source: e,
            })?;

            all_neighbors.extend(result.results);
        }

        // Sort globally and take top-k.
        all_neighbors.sort_by(|a, b| {
            let sa = a.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let sb = b.get("similarity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_neighbors.truncate(limit as usize);

        // Round similarity for presentation after global ordering is final.
        for neighbor in &mut all_neighbors {
            if let Some(sim) = neighbor.get("similarity").and_then(|v| v.as_f64()) {
                neighbor["similarity"] = json!((sim * 10000.0).round() / 10000.0);
            }
        }

        Ok(json!({
            "query_node": node_id,
            "k": limit,
            "neighbors": all_neighbors,
        }))
    }

    /// Parse and validate a `collection/key` node ID.
    pub(super) fn parse_node_id(node_id: &str) -> Result<(&str, &str), HandlerError> {
        let (col, key) = node_id
            .split_once('/')
            .ok_or_else(|| HandlerError::InvalidNodeId {
                node_id: node_id.to_string(),
                reason: "must be in 'collection/key' format".into(),
            })?;

        if col.is_empty() {
            return Err(HandlerError::InvalidNodeId {
                node_id: node_id.to_string(),
                reason: "empty collection name".into(),
            });
        }
        if key.is_empty() {
            return Err(HandlerError::InvalidNodeId {
                node_id: node_id.to_string(),
                reason: "empty key".into(),
            });
        }
        if key.contains('/') {
            return Err(HandlerError::InvalidNodeId {
                node_id: node_id.to_string(),
                reason: "key must not contain '/'".into(),
            });
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
        assert!(matches!(
            cmd,
            DaemonCommand::GraphEmbedEmbed(ref p) if p.node_id == "hope_axioms/ax_001"
        ));
    }

    #[test]
    fn test_command_defaults() {
        let json = serde_json::json!({
            "command": "graph_embed.neighbors",
            "params": { "node_id": "test/1" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::GraphEmbedNeighbors(ref p) if p.limit.is_none()
        ));
    }

    #[test]
    fn test_deny_unknown_fields_graph_embed() {
        let json = serde_json::json!({
            "command": "graph_embed.embed",
            "params": { "node_id": "test/1", "extra_field": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_fields_neighbors() {
        let json = serde_json::json!({
            "command": "graph_embed.neighbors",
            "params": { "node_id": "test/1", "limit": 5, "bogus": "value" }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
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

    // -- parse_node_id -------------------------------------------------------

    #[test]
    fn test_parse_node_id_valid() {
        let (col, key) = handlers::parse_node_id("papers/p123").unwrap();
        assert_eq!(col, "papers");
        assert_eq!(key, "p123");
    }

    #[test]
    fn test_parse_node_id_no_slash() {
        assert!(handlers::parse_node_id("no_slash").is_err());
    }

    #[test]
    fn test_parse_node_id_multiple_slashes() {
        assert!(handlers::parse_node_id("a/b/c").is_err());
    }

    #[test]
    fn test_parse_node_id_empty_collection() {
        assert!(handlers::parse_node_id("/key").is_err());
    }

    #[test]
    fn test_parse_node_id_empty_key() {
        assert!(handlers::parse_node_id("col/").is_err());
    }
}
