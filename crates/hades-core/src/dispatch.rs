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

    /// The requested document does not exist.
    #[error("document not found: {collection}/{key}")]
    DocumentNotFound { collection: String, key: String },

    /// The node exists but has no structural embedding.
    #[error("no structural embedding for {node_id} — run 'hades graph-embed train' first")]
    NoEmbedding { node_id: String },

    /// The structural embedding is present but malformed.
    #[error("invalid structural embedding for {node_id}: {reason}")]
    InvalidEmbedding { node_id: String, reason: String },

    /// A `limit` parameter is out of bounds.
    #[error("limit must be 1..={max}, got {limit}")]
    InvalidLimit { limit: u32, max: u32 },

    /// An invalid parameter was provided.
    #[error("invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

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

        // ── Database read commands ─────────────────────────────────────
        DaemonCommand::DbGet { collection, key } => {
            handlers::db_get(pool, &collection, &key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCount { collection } => {
            handlers::db_count(pool, &collection)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCollections {} => {
            handlers::db_collections(pool)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCheck { document_id } => {
            handlers::db_check(pool, &document_id)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbRecent { limit } => {
            let limit = limit.unwrap_or(10).min(MAX_LIMIT);
            handlers::db_recent(pool, limit)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbList { collection, limit, paper } => {
            let limit = limit.unwrap_or(20).min(MAX_LIMIT);
            handlers::db_list(pool, collection.as_deref(), limit, paper.as_deref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbAql { aql, bind, limit } => {
            let limit = limit.map(|l| l.min(MAX_LIMIT));
            handlers::db_aql(pool, &aql, bind.as_ref(), limit)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbHealth { verbose } => {
            handlers::db_health(pool, verbose)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbStats {} => {
            handlers::db_stats(pool)
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
    use serde_json::{json, Value};

    use super::HandlerError;
    use crate::db::collections::CollectionProfile;
    use crate::db::crud::{self, list_collections, count_collection, CollectionInfo};
    use crate::db::index;
    use crate::db::query::{self, ExecutionTarget};
    use crate::db::ArangoPool;

    // ── Database read handlers ──────────────────────────────────────

    /// Fetch a single document by collection and key.
    pub async fn db_get(
        pool: &ArangoPool,
        collection: &str,
        key: &str,
    ) -> Result<Value, HandlerError> {
        crud::get_document(pool, collection, key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: collection.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to get {collection}/{key}"),
                        source: e,
                    }
                }
            })
    }

    /// Count documents in a collection.
    pub async fn db_count(
        pool: &ArangoPool,
        collection: &str,
    ) -> Result<Value, HandlerError> {
        let count = count_collection(pool, collection)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to count collection '{collection}'"),
                source: e,
            })?;
        Ok(json!({ "collection": collection, "count": count }))
    }

    /// List all non-system collections with document counts.
    pub async fn db_collections(
        pool: &ArangoPool,
    ) -> Result<Value, HandlerError> {
        let collections = list_collections(pool, true)
            .await
            .map_err(|e| HandlerError::Query {
                context: "failed to list collections".into(),
                source: e,
            })?;

        let mut entries = Vec::with_capacity(collections.len());
        for col in &collections {
            let count = match count_collection(pool, &col.name).await {
                Ok(n) => n,
                Err(e) if e.is_not_found() => 0,
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to count collection '{}'", col.name),
                        source: e,
                    });
                }
            };
            let type_name = if col.collection_type == 3 { "edge" } else { "document" };
            entries.push(json!({
                "name": col.name,
                "type": type_name,
                "count": count,
            }));
        }

        Ok(json!({
            "database": pool.database(),
            "collections": entries,
            "total": entries.len(),
        }))
    }

    /// Check if a document exists by its full _id (collection/key).
    pub async fn db_check(
        pool: &ArangoPool,
        document_id: &str,
    ) -> Result<Value, HandlerError> {
        let (col, key) = parse_node_id(document_id)?;
        let exists = match crud::get_document(pool, col, key).await {
            Ok(_) => true,
            Err(e) if e.is_not_found() => false,
            Err(e) => {
                return Err(HandlerError::Query {
                    context: format!("failed to check {document_id}"),
                    source: e,
                })
            }
        };
        Ok(json!({ "document_id": document_id, "exists": exists }))
    }

    /// Return recently created/updated documents across collection profiles.
    pub async fn db_recent(
        pool: &ArangoPool,
        limit: u32,
    ) -> Result<Value, HandlerError> {
        let profile = CollectionProfile::default_profile();
        let aql = "FOR d IN @@col \
                    SORT d.created_at DESC, d._rev DESC \
                    LIMIT @limit \
                    RETURN MERGE(d, { _collection: @col_name })";

        let bind = json!({
            "@col": profile.metadata,
            "col_name": profile.metadata,
            "limit": limit,
        });

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await;

        let result = match result {
            Ok(r) => r,
            Err(e) if e.is_not_found() => {
                return Ok(json!({
                    "collection": profile.metadata,
                    "documents": [],
                    "count": 0,
                    "note": format!("collection '{}' does not exist in this database", profile.metadata),
                }));
            }
            Err(e) => {
                return Err(HandlerError::Query {
                    context: format!("failed to query recent from '{}'", profile.metadata),
                    source: e,
                });
            }
        };

        Ok(json!({
            "collection": profile.metadata,
            "documents": result.results,
            "count": result.results.len(),
        }))
    }

    /// List documents from a collection profile with optional paper filter.
    pub async fn db_list(
        pool: &ArangoPool,
        collection: Option<&str>,
        limit: u32,
        paper: Option<&str>,
    ) -> Result<Value, HandlerError> {
        let profile = match collection {
            Some(name) => CollectionProfile::get(name).ok_or_else(|| {
                HandlerError::InvalidParameter {
                    name: "collection".into(),
                    reason: format!(
                        "unknown profile '{name}' — valid profiles: {}",
                        CollectionProfile::all()
                            .iter()
                            .map(|(n, _)| *n)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }
            })?,
            None => CollectionProfile::default_profile(),
        };

        let (aql, bind) = if let Some(paper_id) = paper {
            (
                "FOR d IN @@col \
                 FILTER d.paper_id == @paper_id || d.document_id == @paper_id || d.arxiv_id == @paper_id \
                 LIMIT @limit \
                 RETURN d",
                json!({
                    "@col": profile.metadata,
                    "paper_id": paper_id,
                    "limit": limit,
                }),
            )
        } else {
            (
                "FOR d IN @@col \
                 LIMIT @limit \
                 RETURN d",
                json!({
                    "@col": profile.metadata,
                    "limit": limit,
                }),
            )
        };

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            true,
            ExecutionTarget::Reader,
        )
        .await;

        // Return empty list if the collection doesn't exist in this database.
        let result = match result {
            Ok(r) => r,
            Err(e) if e.is_not_found() => {
                return Ok(json!({
                    "collection": profile.metadata,
                    "documents": [],
                    "count": 0,
                    "full_count": 0,
                    "note": format!("collection '{}' does not exist in this database", profile.metadata),
                }));
            }
            Err(e) => {
                return Err(HandlerError::Query {
                    context: format!("failed to list from '{}'", profile.metadata),
                    source: e,
                });
            }
        };

        Ok(json!({
            "collection": profile.metadata,
            "documents": result.results,
            "count": result.results.len(),
            "full_count": result.full_count,
        }))
    }

    /// Execute a raw AQL query (read-only enforced).
    pub async fn db_aql(
        pool: &ArangoPool,
        aql: &str,
        bind: Option<&Value>,
        limit: Option<u32>,
    ) -> Result<Value, HandlerError> {
        // Reject mutating AQL — this handler is read-only.
        let upper = aql.to_uppercase();
        for keyword in &["INSERT", "UPDATE", "REPLACE", "REMOVE", "UPSERT"] {
            // Check for standalone keywords (not inside quoted strings).
            // Simple heuristic: keyword preceded by whitespace or at start.
            if upper.split_whitespace().any(|w| w == *keyword) {
                return Err(HandlerError::InvalidParameter {
                    name: "aql".into(),
                    reason: format!(
                        "mutating AQL ({keyword}) not allowed via db.aql — use the write-specific commands"
                    ),
                });
            }
        }

        let full_aql = if let Some(lim) = limit {
            // Wrap in a subquery so LIMIT applies correctly regardless
            // of whether the user's AQL already contains RETURN.
            format!("FOR __r IN ({aql}) LIMIT {lim} RETURN __r")
        } else {
            aql.to_string()
        };

        let result = query::query(
            pool,
            &full_aql,
            bind,
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: "AQL execution failed".into(),
            source: e,
        })?;

        Ok(json!({
            "results": result.results,
            "count": result.results.len(),
        }))
    }

    /// Health check — ArangoDB connectivity + optional integrity checks.
    pub async fn db_health(
        pool: &ArangoPool,
        verbose: bool,
    ) -> Result<Value, HandlerError> {
        let status = pool.health_check().await;

        let mut result = json!({
            "database": pool.database(),
            "arangodb": {
                "version": status.version,
                "reader_ok": status.reader_ok,
                "writer_ok": status.writer_ok,
                "shared_connection": status.shared,
                "status": if status.reader_ok && status.writer_ok { "healthy" } else { "degraded" },
            },
        });

        if verbose {
            // Add per-collection info for verbose mode.
            let collections = list_collections(pool, true)
                .await
                .map_err(|e| HandlerError::Query {
                    context: "failed to list collections for health check".into(),
                    source: e,
                })?;
            let mut col_details = Vec::new();
            for col in &collections {
                let count = match count_collection(pool, &col.name).await {
                    Ok(n) => n,
                    Err(e) if e.is_not_found() => 0,
                    Err(e) => {
                        return Err(HandlerError::Query {
                            context: format!("failed to count '{}' during health check", col.name),
                            source: e,
                        });
                    }
                };
                let indexes = match index::list_indexes(pool, &col.name).await {
                    Ok(idx) => idx,
                    Err(e) if e.is_not_found() => Vec::new(),
                    Err(e) => {
                        return Err(HandlerError::Query {
                            context: format!("failed to list indexes for '{}'", col.name),
                            source: e,
                        });
                    }
                };
                let type_name = if col.collection_type == 3 { "edge" } else { "document" };
                col_details.push(json!({
                    "name": col.name,
                    "type": type_name,
                    "count": count,
                    "indexes": indexes.len(),
                }));
            }
            result["collections"] = json!(col_details);
            result["collection_count"] = json!(collections.len());
        }

        Ok(result)
    }

    /// Aggregate statistics across all collection profiles.
    pub async fn db_stats(
        pool: &ArangoPool,
    ) -> Result<Value, HandlerError> {
        let mut profiles_data = Vec::new();
        let mut total_docs: u64 = 0;
        let mut total_chunks: u64 = 0;
        let mut total_embeddings: u64 = 0;

        for (name, profile) in CollectionProfile::all() {
            let meta_count = match count_collection(pool, profile.metadata).await {
                Ok(n) => n,
                Err(e) if e.is_not_found() => 0,
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to count '{}'", profile.metadata),
                        source: e,
                    });
                }
            };
            let chunk_count = match count_collection(pool, profile.chunks).await {
                Ok(n) => n,
                Err(e) if e.is_not_found() => 0,
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to count '{}'", profile.chunks),
                        source: e,
                    });
                }
            };
            let emb_count = match count_collection(pool, profile.embeddings).await {
                Ok(n) => n,
                Err(e) if e.is_not_found() => 0,
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to count '{}'", profile.embeddings),
                        source: e,
                    });
                }
            };

            total_docs += meta_count;
            total_chunks += chunk_count;
            total_embeddings += emb_count;

            profiles_data.push(json!({
                "name": name,
                "metadata_collection": profile.metadata,
                "chunks_collection": profile.chunks,
                "embeddings_collection": profile.embeddings,
                "documents": meta_count,
                "chunks": chunk_count,
                "embeddings": emb_count,
            }));
        }

        Ok(json!({
            "database": pool.database(),
            "profiles": profiles_data,
            "totals": {
                "documents": total_docs,
                "chunks": total_chunks,
                "embeddings": total_embeddings,
            },
        }))
    }

    // ── Graph embed handlers ──────────────────────────────────────────

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

    // -- DB command serde roundtrips ------------------------------------------

    #[test]
    fn test_command_roundtrip_db_get() {
        let json = serde_json::json!({
            "command": "db.get",
            "params": { "collection": "arxiv_metadata", "key": "2409_04701" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGet { ref collection, ref key }
                if collection == "arxiv_metadata" && key == "2409_04701"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_aql() {
        let json = serde_json::json!({
            "command": "db.aql",
            "params": { "aql": "RETURN 1", "limit": 5 }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbAql { ref aql, limit: Some(5), .. } if aql == "RETURN 1"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_collections() {
        let json = serde_json::json!({
            "command": "db.collections",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::DbCollections {}));
    }

    #[test]
    fn test_command_roundtrip_db_count() {
        let json = serde_json::json!({
            "command": "db.count",
            "params": { "collection": "chunks" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbCount { ref collection } if collection == "chunks"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_check() {
        let json = serde_json::json!({
            "command": "db.check",
            "params": { "document_id": "arxiv_metadata/2409_04701" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbCheck { ref document_id } if document_id == "arxiv_metadata/2409_04701"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_recent_defaults() {
        let json = serde_json::json!({
            "command": "db.recent",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbRecent { limit: None }
        ));
    }

    #[test]
    fn test_command_roundtrip_db_list() {
        let json = serde_json::json!({
            "command": "db.list",
            "params": { "collection": "sync", "limit": 50 }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbList { ref collection, limit: Some(50), .. }
                if collection.as_deref() == Some("sync")
        ));
    }

    #[test]
    fn test_command_roundtrip_db_health() {
        let json = serde_json::json!({
            "command": "db.health",
            "params": { "verbose": true }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbHealth { verbose: true }
        ));
    }

    #[test]
    fn test_command_roundtrip_db_stats() {
        let json = serde_json::json!({
            "command": "db.stats",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::DbStats {}));
    }

    // -- AQL read-only enforcement --------------------------------------------

    #[test]
    fn test_db_aql_rejects_insert() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql {
                aql: "INSERT { foo: 1 } INTO test".to_string(),
                bind: None,
                limit: None,
            },
        ));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DispatchError::Handler(HandlerError::InvalidParameter { .. })));
    }

    #[test]
    fn test_db_aql_rejects_remove() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql {
                aql: "FOR d IN test REMOVE d IN test".to_string(),
                bind: None,
                limit: None,
            },
        ));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DispatchError::Handler(HandlerError::InvalidParameter { .. })));
    }

    #[test]
    fn test_db_aql_rejects_update() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql {
                aql: "FOR d IN test UPDATE d WITH { x: 1 } IN test".to_string(),
                bind: None,
                limit: None,
            },
        ));

        assert!(result.is_err());
    }

    #[test]
    fn test_db_aql_allows_return() {
        // "RETURN 1" should pass the read-only check (will fail at AQL execution
        // if ArangoDB is unreachable, but should not fail at validation).
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql {
                aql: "RETURN 1".to_string(),
                bind: None,
                limit: None,
            },
        ));

        // This will fail with a connection error (ArangoDB not running in test),
        // NOT with InvalidParameter — that's the important assertion.
        match result {
            Err(DispatchError::Handler(HandlerError::InvalidParameter { .. })) => {
                panic!("RETURN 1 should not be rejected as mutating AQL");
            }
            _ => {} // Connection error or success — both acceptable.
        }
    }

    // -- db_list profile validation -------------------------------------------

    #[test]
    fn test_db_list_invalid_profile() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbList {
                collection: Some("nonexistent_profile".into()),
                limit: Some(10),
                paper: None,
            },
        ));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DispatchError::Handler(HandlerError::InvalidParameter { .. })));
    }
}
