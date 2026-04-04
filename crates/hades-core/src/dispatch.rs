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

    /// A write operation was rejected because the target database is read-only.
    #[error("write denied: {0}")]
    WriteDenied(String),

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
// Param structs (deny_unknown_fields for strict wire protocol validation)
// ---------------------------------------------------------------------------

/// Params for `db.aql`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbAqlParams {
    pub aql: String,
    #[serde(default)]
    pub bind: Option<Value>,
    pub limit: Option<u32>,
}

/// Params for `db.get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGetParams {
    pub collection: String,
    pub key: String,
}

/// Params for `db.list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbListParams {
    pub collection: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
    pub paper: Option<String>,
}

/// Params for `db.insert`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbInsertParams {
    pub collection: String,
    pub data: Value,
}

/// Params for `db.update`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbUpdateParams {
    pub collection: String,
    pub key: String,
    pub data: Value,
}

/// Params for `db.delete`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbDeleteParams {
    pub collection: String,
    pub key: String,
}

/// Params for `db.count`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbCountParams {
    pub collection: String,
}

/// Params for `db.health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbHealthParams {
    #[serde(default)]
    pub verbose: bool,
}

/// Params for `db.check`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbCheckParams {
    pub document_id: String,
}

/// Params for `db.recent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbRecentParams {
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Params for `db.purge`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbPurgeParams {
    pub document_id: String,
}

/// Params for `db.create_collection`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbCreateCollectionParams {
    pub name: String,
    /// "document" or "edge" (defaults to "document").
    #[serde(default)]
    pub collection_type: Option<String>,
}

/// Params for `db.create_index`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbCreateIndexParams {
    pub collection: String,
    pub dimension: u32,
    #[serde(default = "default_metric")]
    pub metric: String,
}

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

/// Params for `db.graph.traverse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGraphTraverseParams {
    pub start: String,
    #[serde(default = "default_direction")]
    pub direction: String,
    #[serde(default = "default_one")]
    pub min_depth: u32,
    #[serde(default = "default_one")]
    pub max_depth: u32,
    #[serde(default)]
    pub limit: Option<u32>,
    pub graph: Option<String>,
}

/// Params for `db.graph.shortest_path`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGraphShortestPathParams {
    pub source: String,
    pub target: String,
    #[serde(default = "default_direction_any")]
    pub direction: String,
    pub graph: Option<String>,
}

/// Params for `db.graph.neighbors`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGraphNeighborsParams {
    pub vertex: String,
    #[serde(default = "default_direction_any")]
    pub direction: String,
    #[serde(default)]
    pub limit: Option<u32>,
    pub graph: Option<String>,
}

/// Params for `db.graph.create`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGraphCreateParams {
    pub name: String,
    #[serde(default)]
    pub edge_definitions: Option<Value>,
}

/// Params for `db.graph.drop`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGraphDropParams {
    pub name: String,
    #[serde(default)]
    pub drop_collections: bool,
    /// Confirmation flag — daemon callers must set this to `true`.
    #[serde(default)]
    pub force: bool,
}

/// Params for `status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusParams {
    #[serde(default)]
    pub verbose: bool,
}

/// Params for `orient`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrientParams {
    pub collection: Option<String>,
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
    Orient(OrientParams),

    #[serde(rename = "status")]
    Status(StatusParams),

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
    DbAql(DbAqlParams),

    #[serde(rename = "db.get")]
    DbGet(DbGetParams),

    #[serde(rename = "db.list")]
    DbList(DbListParams),

    #[serde(rename = "db.insert")]
    DbInsert(DbInsertParams),

    #[serde(rename = "db.update")]
    DbUpdate(DbUpdateParams),

    #[serde(rename = "db.delete")]
    DbDelete(DbDeleteParams),

    #[serde(rename = "db.purge")]
    DbPurge(DbPurgeParams),

    #[serde(rename = "db.create_collection")]
    DbCreateCollection(DbCreateCollectionParams),

    #[serde(rename = "db.create_index")]
    DbCreateIndex(DbCreateIndexParams),

    #[serde(rename = "db.count")]
    DbCount(DbCountParams),

    #[serde(rename = "db.collections")]
    DbCollections {},

    #[serde(rename = "db.stats")]
    DbStats {},

    #[serde(rename = "db.health")]
    DbHealth(DbHealthParams),

    #[serde(rename = "db.check")]
    DbCheck(DbCheckParams),

    #[serde(rename = "db.recent")]
    DbRecent(DbRecentParams),

    // ── Graph traversal ─────────────────────────────────────────────
    #[serde(rename = "db.graph.traverse")]
    DbGraphTraverse(DbGraphTraverseParams),

    #[serde(rename = "db.graph.shortest_path")]
    DbGraphShortestPath(DbGraphShortestPathParams),

    #[serde(rename = "db.graph.neighbors")]
    DbGraphNeighbors(DbGraphNeighborsParams),

    #[serde(rename = "db.graph.list")]
    DbGraphList {},

    #[serde(rename = "db.graph.create")]
    DbGraphCreate(DbGraphCreateParams),

    #[serde(rename = "db.graph.drop")]
    DbGraphDrop(DbGraphDropParams),

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
fn default_metric() -> String { "cosine".to_string() }

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Check that the target database is writable, converting the anyhow error
/// into a typed `DispatchError` for the wire protocol.
fn require_writable(config: &HadesConfig) -> Result<(), DispatchError> {
    config
        .require_writable_database()
        .map_err(|e| DispatchError::Handler(HandlerError::WriteDenied(e.to_string())))
}

/// Route a [`DaemonCommand`] to its native handler.
///
/// Returns the handler's JSON result on success.  Commands without a
/// native Rust implementation return [`DispatchError::NotImplemented`]
/// so the daemon can fall back to subprocess invocation.
pub async fn dispatch(
    pool: &ArangoPool,
    config: &HadesConfig,
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
        DaemonCommand::DbGet(params) => {
            handlers::db_get(pool, &params.collection, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCount(params) => {
            handlers::db_count(pool, &params.collection)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCollections {} => {
            handlers::db_collections(pool)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCheck(params) => {
            handlers::db_check(pool, &params.document_id)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbRecent(params) => {
            let limit = params.limit.unwrap_or(10).min(MAX_LIMIT);
            handlers::db_recent(pool, limit)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbList(params) => {
            let limit = params.limit.unwrap_or(20).min(MAX_LIMIT);
            handlers::db_list(pool, params.collection.as_deref(), limit, params.paper.as_deref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbAql(params) => {
            let limit = params.limit.map(|l| l.min(MAX_LIMIT));
            handlers::db_aql(pool, &params.aql, params.bind.as_ref(), limit)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbHealth(params) => {
            handlers::db_health(pool, params.verbose)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbStats {} => {
            handlers::db_stats(pool)
                .await
                .map_err(DispatchError::Handler)
        }

        // ── Database write commands ───────────��───────────────────────
        DaemonCommand::DbInsert(params) => {
            require_writable(config)?;
            handlers::db_insert(pool, &params.collection, &params.data)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbUpdate(params) => {
            require_writable(config)?;
            handlers::db_update(pool, &params.collection, &params.key, &params.data)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbDelete(params) => {
            require_writable(config)?;
            handlers::db_delete(pool, &params.collection, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbPurge(params) => {
            require_writable(config)?;
            handlers::db_purge(pool, &params.document_id)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCreateCollection(params) => {
            require_writable(config)?;
            handlers::db_create_collection(
                pool,
                &params.name,
                params.collection_type.as_deref(),
            )
            .await
            .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbCreateIndex(params) => {
            require_writable(config)?;
            handlers::db_create_index(
                pool,
                &params.collection,
                params.dimension,
                &params.metric,
            )
            .await
            .map_err(DispatchError::Handler)
        }

        // ── Graph read commands ────────────────────────────────────────
        DaemonCommand::DbGraphTraverse(params) => {
            let limit = params.limit.unwrap_or(100).min(MAX_LIMIT);
            handlers::db_graph_traverse(
                pool,
                &params.start,
                &params.direction,
                params.min_depth,
                params.max_depth,
                limit,
                params.graph.as_deref(),
            )
            .await
            .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbGraphShortestPath(params) => {
            handlers::db_graph_shortest_path(
                pool,
                &params.source,
                &params.target,
                &params.direction,
                params.graph.as_deref(),
            )
            .await
            .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbGraphNeighbors(params) => {
            let limit = params.limit.unwrap_or(20).min(MAX_LIMIT);
            handlers::db_graph_neighbors(
                pool,
                &params.vertex,
                &params.direction,
                limit,
                params.graph.as_deref(),
            )
            .await
            .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbGraphList {} => {
            handlers::db_graph_list(pool)
                .await
                .map_err(DispatchError::Handler)
        }

        // ── Graph write commands ──────────────────────────────────────
        DaemonCommand::DbGraphCreate(params) => {
            require_writable(config)?;
            handlers::db_graph_create(pool, &params.name, params.edge_definitions.as_ref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbGraphDrop(params) => {
            require_writable(config)?;
            if !params.force {
                return Err(DispatchError::Handler(HandlerError::InvalidParameter {
                    name: "force".into(),
                    reason: "graph drop requires force=true confirmation".into(),
                }));
            }
            handlers::db_graph_drop(pool, &params.name, params.drop_collections)
                .await
                .map_err(DispatchError::Handler)
        }

        // ── System commands ─────────────────────────────────────────
        DaemonCommand::Status(params) => {
            handlers::status(pool, config, params.verbose)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::Orient(params) => {
            handlers::orient(pool, params.collection.as_deref())
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
    use crate::graph::NL_GRAPH_SCHEMA;
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
        // Reject non-object bind values — AQL bind vars must be a map.
        if let Some(v) = bind.filter(|v| !v.is_object()) {
            return Err(HandlerError::InvalidParameter {
                name: "bind".into(),
                reason: format!(
                    "bind must be an object/map of bind variables, got {}",
                    match v {
                        Value::Array(_) => "array",
                        Value::String(_) => "string",
                        Value::Number(_) => "number",
                        Value::Bool(_) => "bool",
                        Value::Null => "null",
                        _ => "non-object",
                    }
                ),
            });
        }

        // Reject mutating AQL — this handler is read-only.
        // Strip string literals and comments first so keywords inside
        // strings like "INSERT TITLE" don't trigger false positives.
        let stripped = strip_aql_strings_and_comments(aql);
        let upper = stripped.to_uppercase();
        for keyword in &["INSERT", "UPDATE", "REPLACE", "REMOVE", "UPSERT"] {
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

    // ── Database write handlers ──────────────────────────────────────

    /// Insert a single document (or array) into a collection.
    pub async fn db_insert(
        pool: &ArangoPool,
        collection: &str,
        data: &Value,
    ) -> Result<Value, HandlerError> {
        let resp = crud::insert_document(pool, collection, data)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to insert into '{collection}'"),
                source: e,
            })?;
        Ok(resp)
    }

    /// Update (merge-patch) a document's fields.
    pub async fn db_update(
        pool: &ArangoPool,
        collection: &str,
        key: &str,
        data: &Value,
    ) -> Result<Value, HandlerError> {
        crud::update_document(pool, collection, key, data)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: collection.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to update {collection}/{key}"),
                        source: e,
                    }
                }
            })
    }

    /// Delete a single document by collection and key.
    pub async fn db_delete(
        pool: &ArangoPool,
        collection: &str,
        key: &str,
    ) -> Result<Value, HandlerError> {
        crud::delete_document(pool, collection, key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: collection.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to delete {collection}/{key}"),
                        source: e,
                    }
                }
            })
    }

    /// Cascade-delete a document and its related chunks/embeddings.
    pub async fn db_purge(
        pool: &ArangoPool,
        document_id: &str,
    ) -> Result<Value, HandlerError> {
        let (col, key) = parse_node_id(document_id)?;

        let profile = CollectionProfile::find_by_metadata(col).ok_or_else(|| {
            HandlerError::InvalidParameter {
                name: "document_id".into(),
                reason: format!(
                    "collection '{col}' is not a known metadata collection — \
                     valid metadata collections: {}",
                    CollectionProfile::all()
                        .iter()
                        .map(|(_, p)| p.metadata)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        })?;

        let fk = profile.foreign_key;
        let aql = format!(
            "LET meta = (FOR d IN @@meta FILTER d._key == @key REMOVE d IN @@meta RETURN 1) \
             LET chunks = (FOR d IN @@chunks FILTER d.{fk} == @key REMOVE d IN @@chunks RETURN 1) \
             LET embs = (FOR d IN @@embs FILTER d.{fk} == @key REMOVE d IN @@embs RETURN 1) \
             RETURN {{ metadata: LENGTH(meta), chunks: LENGTH(chunks), embeddings: LENGTH(embs) }}"
        );

        let bind = json!({
            "@meta": profile.metadata,
            "@chunks": profile.chunks,
            "@embs": profile.embeddings,
            "key": key,
        });

        let result = query::query_single(
            pool,
            &aql,
            Some(&bind),
            ExecutionTarget::Writer,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("purge AQL failed for '{document_id}'"),
            source: e,
        })?;

        let counts = result.unwrap_or(json!({"metadata": 0, "chunks": 0, "embeddings": 0}));
        let total = counts["metadata"].as_u64().unwrap_or(0)
            + counts["chunks"].as_u64().unwrap_or(0)
            + counts["embeddings"].as_u64().unwrap_or(0);

        Ok(json!({
            "document_id": document_id,
            "deleted": counts,
            "total_deleted": total,
        }))
    }

    /// Create a new collection.
    pub async fn db_create_collection(
        pool: &ArangoPool,
        name: &str,
        collection_type: Option<&str>,
    ) -> Result<Value, HandlerError> {
        let type_code = match collection_type {
            None | Some("document") => None, // ArangoDB defaults to document (2)
            Some("edge") => Some(3u32),
            Some(other) => {
                return Err(HandlerError::InvalidParameter {
                    name: "collection_type".into(),
                    reason: format!(
                        "unknown collection type '{other}' — expected 'document' or 'edge'"
                    ),
                });
            }
        };

        crud::create_collection(pool, name, type_code)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to create collection '{name}'"),
                source: e,
            })
    }

    /// Create a vector index on a collection.
    pub async fn db_create_index(
        pool: &ArangoPool,
        collection: &str,
        dimension: u32,
        metric: &str,
    ) -> Result<Value, HandlerError> {
        let vm = match metric {
            "cosine" => index::VectorMetric::Cosine,
            "l2" | "euclidean" => index::VectorMetric::L2,
            "dotproduct" | "innerProduct" => index::VectorMetric::InnerProduct,
            other => {
                return Err(HandlerError::InvalidParameter {
                    name: "metric".into(),
                    reason: format!(
                        "unknown metric '{other}' — expected 'cosine', 'l2', or 'dotproduct'"
                    ),
                });
            }
        };

        let info = index::create_vector_index(
            pool,
            collection,
            "embedding", // standard field name across all collection profiles
            dimension,
            None, // auto-calculate nLists
            10,   // default nProbe
            vm,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("failed to create vector index on '{collection}'"),
            source: e,
        })?;

        Ok(json!({
            "id": info.id,
            "collection": collection,
            "type": info.index_type,
            "fields": info.fields,
            "params": info.params,
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

    /// Strip string literals and comments from AQL so keyword detection
    /// doesn't trigger on values like `"INSERT TITLE HERE"`.
    ///
    /// Handles single-quoted, double-quoted, and backtick-quoted strings
    /// (with backslash escapes), plus `//` line comments and `/* */` block
    /// comments.  Replaced regions become spaces to preserve token boundaries.
    fn strip_aql_strings_and_comments(aql: &str) -> String {
        let mut out = String::with_capacity(aql.len());
        let bytes = aql.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            let b = bytes[i];

            // Line comment: // ... \n
            if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                while i < len && bytes[i] != b'\n' {
                    out.push(' ');
                    i += 1;
                }
                continue;
            }

            // Block comment: /* ... */
            if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                out.push(' ');
                out.push(' ');
                i += 2;
                while i < len {
                    if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                        out.push(' ');
                        out.push(' ');
                        i += 2;
                        break;
                    }
                    out.push(' ');
                    i += 1;
                }
                continue;
            }

            // String literal: '...', "...", `...` (with backslash escapes)
            if b == b'\'' || b == b'"' || b == b'`' {
                let quote = b;
                out.push(' '); // replace opening quote
                i += 1;
                while i < len {
                    if bytes[i] == b'\\' && i + 1 < len {
                        // Escaped character — skip both bytes.
                        out.push(' ');
                        out.push(' ');
                        i += 2;
                    } else if bytes[i] == quote {
                        out.push(' '); // replace closing quote
                        i += 1;
                        break;
                    } else {
                        out.push(' ');
                        i += 1;
                    }
                }
                continue;
            }

            out.push(b as char);
            i += 1;
        }

        out
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

    // ── Graph handlers ─────────────────────────────────────────────

    /// Validate a graph name — must be non-empty and contain only
    /// alphanumeric, underscore, or hyphen characters (ArangoDB identifier rules).
    fn validate_graph_name(name: &str) -> Result<(), HandlerError> {
        if name.is_empty() {
            return Err(HandlerError::InvalidParameter {
                name: "name".into(),
                reason: "graph name must not be empty".into(),
            });
        }
        if !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(HandlerError::InvalidParameter {
                name: "name".into(),
                reason: format!(
                    "graph name '{name}' contains invalid characters — \
                     only alphanumeric, underscore, and hyphen are allowed"
                ),
            });
        }
        Ok(())
    }

    /// Map lowercase direction to AQL keyword.
    pub(crate) fn validate_direction(direction: &str) -> Result<&'static str, HandlerError> {
        match direction {
            "outbound" => Ok("OUTBOUND"),
            "inbound" => Ok("INBOUND"),
            "any" => Ok("ANY"),
            other => Err(HandlerError::InvalidParameter {
                name: "direction".into(),
                reason: format!(
                    "must be 'outbound', 'inbound', or 'any', got '{other}'"
                ),
            }),
        }
    }

    /// Default graph name used when the caller doesn't specify one.
    const DEFAULT_GRAPH: &str = "nl_concept_map";

    /// Maximum traversal depth to prevent runaway queries.
    const MAX_TRAVERSAL_DEPTH: u32 = 20;

    /// Graph traversal from a starting vertex.
    pub async fn db_graph_traverse(
        pool: &ArangoPool,
        start: &str,
        direction: &str,
        min_depth: u32,
        max_depth: u32,
        limit: u32,
        graph: Option<&str>,
    ) -> Result<Value, HandlerError> {
        // Validate start vertex format.
        parse_node_id(start)?;
        let dir_kw = validate_direction(direction)?;

        if min_depth > MAX_TRAVERSAL_DEPTH {
            return Err(HandlerError::InvalidParameter {
                name: "min_depth".into(),
                reason: format!(
                    "min_depth ({min_depth}) exceeds maximum ({MAX_TRAVERSAL_DEPTH})"
                ),
            });
        }
        let max_depth = max_depth.min(MAX_TRAVERSAL_DEPTH);

        if max_depth < min_depth {
            return Err(HandlerError::InvalidParameter {
                name: "max_depth".into(),
                reason: format!(
                    "max_depth ({max_depth}) must be >= min_depth ({min_depth})"
                ),
            });
        }

        let graph_name = graph.unwrap_or(DEFAULT_GRAPH);

        // Direction is interpolated as a keyword — AQL doesn't support it as
        // a bind variable.  Safe because validate_direction returns only one
        // of three compile-time strings.
        let aql = format!(
            "FOR v, e IN @min_depth..@max_depth {dir_kw} @start GRAPH @graph \
             LIMIT @limit \
             RETURN {{vertex: v, edge: e}}"
        );

        let bind = json!({
            "min_depth": min_depth,
            "max_depth": max_depth,
            "start": start,
            "graph": graph_name,
            "limit": limit,
        });

        let result = query::query(
            pool,
            &aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("graph traverse from '{start}'"),
            source: e,
        })?;

        Ok(json!({
            "results": result.results,
            "start": start,
            "graph": graph_name,
            "direction": direction,
        }))
    }

    /// Find shortest path between two vertices.
    pub async fn db_graph_shortest_path(
        pool: &ArangoPool,
        source: &str,
        target: &str,
        direction: &str,
        graph: Option<&str>,
    ) -> Result<Value, HandlerError> {
        parse_node_id(source)?;
        parse_node_id(target)?;
        let dir_kw = validate_direction(direction)?;
        let graph_name = graph.unwrap_or(DEFAULT_GRAPH);

        let aql = format!(
            "FOR v, e IN {dir_kw} SHORTEST_PATH @from_v TO @to_v GRAPH @graph \
             RETURN {{vertex: v, edge: e}}"
        );

        let bind = json!({
            "from_v": source,
            "to_v": target,
            "graph": graph_name,
        });

        let result = query::query(
            pool,
            &aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("shortest path from '{source}' to '{target}'"),
            source: e,
        })?;

        Ok(json!({
            "results": result.results,
            "from": source,
            "to": target,
            "graph": graph_name,
            "direction": direction,
        }))
    }

    /// Find neighbors of a vertex (depth-1 traversal).
    pub async fn db_graph_neighbors(
        pool: &ArangoPool,
        vertex: &str,
        direction: &str,
        limit: u32,
        graph: Option<&str>,
    ) -> Result<Value, HandlerError> {
        let traverse_result = db_graph_traverse(
            pool, vertex, direction, 1, 1, limit, graph,
        )
        .await?;

        Ok(json!({
            "results": traverse_result["results"],
            "vertex": vertex,
            "direction": direction,
        }))
    }

    /// List all named graphs via the Gharial API.
    pub async fn db_graph_list(
        pool: &ArangoPool,
    ) -> Result<Value, HandlerError> {
        let resp = pool
            .reader()
            .get("gharial")
            .await
            .map_err(|e| HandlerError::Query {
                context: "list graphs via Gharial API".into(),
                source: e,
            })?;

        let graphs = resp
            .get("graphs")
            .and_then(|g| g.as_array())
            .cloned()
            .unwrap_or_default();

        let mapped: Vec<Value> = graphs
            .into_iter()
            .map(|g| {
                json!({
                    "name": g.get("_key").or_else(|| g.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown"),
                    "edge_definitions": g.get("edgeDefinitions")
                        .cloned()
                        .unwrap_or(json!([])),
                })
            })
            .collect();

        Ok(json!({ "graphs": mapped }))
    }

    /// Create a named graph via the Gharial API.
    pub async fn db_graph_create(
        pool: &ArangoPool,
        name: &str,
        edge_definitions: Option<&Value>,
    ) -> Result<Value, HandlerError> {
        validate_graph_name(name)?;

        let body = if let Some(defs) = edge_definitions {
            if !defs.is_array() {
                return Err(HandlerError::InvalidParameter {
                    name: "edge_definitions".into(),
                    reason: "must be a JSON array of edge definition objects".into(),
                });
            }
            json!({
                "name": name,
                "edgeDefinitions": defs,
            })
        } else if let Some(graph_def) = NL_GRAPH_SCHEMA.get_named_graph(name) {
            graph_def.to_gharial_payload()
        } else {
            return Err(HandlerError::InvalidParameter {
                name: "name".into(),
                reason: format!(
                    "unknown graph '{name}' and no --edge-definitions provided — \
                     known graphs: {}",
                    NL_GRAPH_SCHEMA
                        .all_named_graph_names()
                        .join(", ")
                ),
            });
        };

        let edge_defs = body.get("edgeDefinitions").cloned().unwrap_or(json!([]));

        pool.writer()
            .post("gharial", &body)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("create graph '{name}'"),
                source: e,
            })?;

        Ok(json!({
            "graph": name,
            "created": true,
            "edge_definitions": edge_defs,
        }))
    }

    /// Drop a named graph via the Gharial API.
    pub async fn db_graph_drop(
        pool: &ArangoPool,
        name: &str,
        drop_collections: bool,
    ) -> Result<Value, HandlerError> {
        validate_graph_name(name)?;
        let path = format!("gharial/{name}?dropCollections={drop_collections}");

        pool.writer()
            .delete(&path)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("drop graph '{name}'"),
                source: e,
            })?;

        Ok(json!({
            "graph": name,
            "dropped": true,
            "collections_dropped": drop_collections,
        }))
    }

    // ── System handlers ────────────────────────────────────────────

    /// Unified system status — ArangoDB, embedder, sync, config.
    pub async fn status(
        pool: &ArangoPool,
        config: &crate::config::HadesConfig,
        verbose: bool,
    ) -> Result<Value, HandlerError> {
        use std::path::PathBuf;
        use crate::persephone::embedding::{EmbeddingClient, EmbeddingClientConfig, EmbeddingEndpoint};

        // 1. ArangoDB health
        let health = pool.health_check().await;
        let arango_status = if health.reader_ok && health.writer_ok {
            "healthy"
        } else {
            "degraded"
        };

        // 2. Embedder probe — connect + info(), with 5-second timeout
        let socket_path = &config.embedding.service.socket;
        let embedder_info = {
            let emb_config = EmbeddingClientConfig {
                endpoint: EmbeddingEndpoint::Unix(PathBuf::from(socket_path)),
                ..Default::default()
            };
            match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                async {
                    let client = EmbeddingClient::connect(emb_config).await?;
                    client.info().await
                },
            )
            .await
            {
                Ok(Ok(info)) => json!({
                    "status": "running",
                    "socket": socket_path,
                    "model_name": info.model_name,
                    "dimension": info.dimension,
                    "device": info.device,
                    "model_loaded": info.model_loaded,
                }),
                Ok(Err(e)) => json!({
                    "status": "unavailable",
                    "socket": socket_path,
                    "error": e.to_string(),
                }),
                Err(_) => json!({
                    "status": "unavailable",
                    "socket": socket_path,
                    "error": "connection timed out (5s)",
                }),
            }
        };

        // 3. Sync watermark
        let sync_info = match crate::arxiv::sync_metadata::get_sync_status(pool).await {
            Ok(Some(wm)) => json!({
                "last_sync": wm.last_sync,
                "total_synced": wm.total_synced,
            }),
            Ok(None) => json!({ "last_sync": null, "total_synced": 0 }),
            Err(_) => json!({ "last_sync": null, "total_synced": 0, "error": "failed to read sync metadata" }),
        };

        // 4. Config summary
        let writable = config.require_writable_database().is_ok();
        let config_info = json!({
            "database": config.effective_database(),
            "writable": writable,
            "arango_socket_ro": config.effective_socket(true),
            "arango_socket_rw": config.effective_socket(false),
            "embedder_socket": socket_path,
        });

        let mut result = json!({
            "database": pool.database(),
            "arangodb": {
                "version": health.version,
                "reader_ok": health.reader_ok,
                "writer_ok": health.writer_ok,
                "shared_connection": health.shared,
                "status": arango_status,
            },
            "embedder": embedder_info,
            "sync": sync_info,
            "config": config_info,
        });

        // 5. Verbose: per-profile stats (reuses db_stats pattern)
        if verbose {
            let mut profiles_data = Vec::new();
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
                profiles_data.push(json!({
                    "name": name,
                    "metadata_collection": profile.metadata,
                    "documents": meta_count,
                    "chunks": chunk_count,
                    "embeddings": emb_count,
                }));
            }
            result["profiles"] = json!(profiles_data);
        }

        Ok(result)
    }

    /// Metadata-first context orientation for AI agent workspace discovery.
    pub async fn orient(
        pool: &ArangoPool,
        collection: Option<&str>,
    ) -> Result<Value, HandlerError> {
        if let Some(col) = collection {
            // Single-collection detail mode
            return orient_collection(pool, col).await;
        }

        // Overview mode: all profiles with counts + recent papers
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

            // Recent papers for this profile
            let recent = recent_docs(pool, profile.metadata, 5).await;

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
                "recent": recent,
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

    /// Single-collection detail: count, schema sample, recent docs, indexes.
    async fn orient_collection(
        pool: &ArangoPool,
        collection: &str,
    ) -> Result<Value, HandlerError> {
        let count = count_collection(pool, collection)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to count '{collection}'"),
                source: e,
            })?;

        // Schema sample: first document
        let sample_aql = "FOR d IN @@col LIMIT 1 RETURN d";
        let sample_bind = json!({ "@col": collection });
        let sample = query::query(
            pool,
            sample_aql,
            Some(&sample_bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .ok()
        .and_then(|r| r.results.into_iter().next());

        // Extract field names from sample for schema overview
        let schema_fields: Vec<String> = sample
            .as_ref()
            .and_then(|doc| doc.as_object())
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();

        // Recent docs
        let recent = recent_docs(pool, collection, 10).await;

        // Indexes
        let indexes = index::list_indexes(pool, collection)
            .await
            .unwrap_or_default();
        let index_info: Vec<Value> = indexes
            .into_iter()
            .map(|idx| json!({
                "type": idx.index_type,
                "fields": idx.fields,
            }))
            .collect();

        Ok(json!({
            "database": pool.database(),
            "collection": collection,
            "count": count,
            "schema_fields": schema_fields,
            "recent": recent,
            "indexes": index_info,
        }))
    }

    /// Fetch recent documents from a collection, sorted by processing_timestamp.
    /// Silently returns empty vec if the collection doesn't exist.
    async fn recent_docs(pool: &ArangoPool, collection: &str, limit: u32) -> Vec<Value> {
        let aql = "FOR d IN @@col \
                    SORT d.processing_timestamp DESC, d._rev DESC \
                    LIMIT @limit \
                    RETURN { _key: d._key, title: d.title, processing_timestamp: d.processing_timestamp }";
        let bind = json!({
            "@col": collection,
            "limit": limit,
        });
        query::query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader)
            .await
            .map(|r| r.results)
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_roundtrip_orient_with_collection() {
        let cmd = DaemonCommand::Orient(OrientParams { collection: Some("test".into()) });
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(json["command"], "orient");
        assert_eq!(json["params"]["collection"], "test");

        let back: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(back, DaemonCommand::Orient(ref p) if p.collection.as_deref() == Some("test")));
    }

    #[test]
    fn test_command_roundtrip_orient_no_collection() {
        let json = serde_json::json!({
            "command": "orient",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::Orient(ref p) if p.collection.is_none()));
    }

    #[test]
    fn test_command_roundtrip_status_defaults() {
        let json = serde_json::json!({
            "command": "status",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::Status(ref p) if !p.verbose));
    }

    #[test]
    fn test_command_roundtrip_status_verbose() {
        let cmd = DaemonCommand::Status(StatusParams { verbose: true });
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(json["command"], "status");
        assert_eq!(json["params"]["verbose"], true);

        let back: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(back, DaemonCommand::Status(ref p) if p.verbose));
    }

    #[test]
    fn test_deny_unknown_status() {
        let json = serde_json::json!({
            "command": "status",
            "params": { "verbose": false, "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_orient() {
        let json = serde_json::json!({
            "command": "orient",
            "params": { "collection": "test", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
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
            DaemonCommand::DbGet(ref p)
                if p.collection == "arxiv_metadata" && p.key == "2409_04701"
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
            DaemonCommand::DbAql(ref p) if p.aql == "RETURN 1" && p.limit == Some(5)
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
            DaemonCommand::DbCount(ref p) if p.collection == "chunks"
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
            DaemonCommand::DbCheck(ref p) if p.document_id == "arxiv_metadata/2409_04701"
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
            DaemonCommand::DbRecent(ref p) if p.limit.is_none()
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
            DaemonCommand::DbList(ref p)
                if p.collection.as_deref() == Some("sync") && p.limit == Some(50)
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
            DaemonCommand::DbHealth(ref p) if p.verbose
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

    #[test]
    fn test_deny_unknown_fields_db_get() {
        let json = serde_json::json!({
            "command": "db.get",
            "params": { "collection": "test", "key": "k", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_fields_db_aql() {
        let json = serde_json::json!({
            "command": "db.aql",
            "params": { "aql": "RETURN 1", "bogus": 42 }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
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
            DaemonCommand::DbAql(DbAqlParams {
                aql: "INSERT { foo: 1 } INTO test".to_string(),
                bind: None,
                limit: None,
            }),
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
            DaemonCommand::DbAql(DbAqlParams {
                aql: "FOR d IN test REMOVE d IN test".to_string(),
                bind: None,
                limit: None,
            }),
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
            DaemonCommand::DbAql(DbAqlParams {
                aql: "FOR d IN test UPDATE d WITH { x: 1 } IN test".to_string(),
                bind: None,
                limit: None,
            }),
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
            DaemonCommand::DbAql(DbAqlParams {
                aql: "RETURN 1".to_string(),
                bind: None,
                limit: None,
            }),
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

    // -- AQL string stripping -------------------------------------------------

    #[test]
    fn test_db_aql_allows_insert_in_string() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        // The word INSERT appears inside a string literal — should NOT be rejected.
        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql(DbAqlParams {
                aql: r#"FOR d IN test FILTER d.title == "INSERT TITLE HERE" RETURN d"#.to_string(),
                bind: None,
                limit: None,
            }),
        ));

        match result {
            Err(DispatchError::Handler(HandlerError::InvalidParameter { .. })) => {
                panic!("INSERT inside a string literal should not be rejected");
            }
            _ => {} // Connection error or success — both acceptable.
        }
    }

    #[test]
    fn test_db_aql_allows_update_in_comment() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        // The word UPDATE appears inside a comment — should NOT be rejected.
        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql(DbAqlParams {
                aql: "FOR d IN test /* UPDATE note */ RETURN d".to_string(),
                bind: None,
                limit: None,
            }),
        ));

        match result {
            Err(DispatchError::Handler(HandlerError::InvalidParameter { .. })) => {
                panic!("UPDATE inside a comment should not be rejected");
            }
            _ => {}
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
            DaemonCommand::DbList(DbListParams {
                collection: Some("nonexistent_profile".into()),
                limit: Some(10),
                paper: None,
            }),
        ));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DispatchError::Handler(HandlerError::InvalidParameter { .. })));
    }

    // -- db_aql bind validation ------------------------------------------------

    #[test]
    fn test_db_aql_rejects_non_object_bind() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbAql(DbAqlParams {
                aql: "RETURN 1".to_string(),
                bind: Some(serde_json::json!([1, 2, 3])),
                limit: None,
            }),
        ));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DispatchError::Handler(HandlerError::InvalidParameter { ref name, .. }) if name == "bind"));
    }

    // -- Write command serde roundtrips ----------------------------------------

    #[test]
    fn test_command_roundtrip_db_insert() {
        let json = serde_json::json!({
            "command": "db.insert",
            "params": { "collection": "test", "data": {"title": "Hello"} }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::DbInsert(ref p) if p.collection == "test"));
    }

    #[test]
    fn test_command_roundtrip_db_update() {
        let json = serde_json::json!({
            "command": "db.update",
            "params": { "collection": "test", "key": "k1", "data": {"x": 1} }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbUpdate(ref p) if p.collection == "test" && p.key == "k1"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_delete() {
        let json = serde_json::json!({
            "command": "db.delete",
            "params": { "collection": "test", "key": "k1" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbDelete(ref p) if p.collection == "test" && p.key == "k1"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_purge() {
        let json = serde_json::json!({
            "command": "db.purge",
            "params": { "document_id": "arxiv_metadata/2409_04701" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbPurge(ref p) if p.document_id == "arxiv_metadata/2409_04701"
        ));
    }

    #[test]
    fn test_command_roundtrip_db_create_collection() {
        let json = serde_json::json!({
            "command": "db.create_collection",
            "params": { "name": "my_col", "collection_type": "edge" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbCreateCollection(ref p)
                if p.name == "my_col" && p.collection_type.as_deref() == Some("edge")
        ));
    }

    #[test]
    fn test_command_roundtrip_db_create_index() {
        let json = serde_json::json!({
            "command": "db.create_index",
            "params": { "collection": "embeddings", "dimension": 2048 }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbCreateIndex(ref p)
                if p.collection == "embeddings" && p.dimension == 2048 && p.metric == "cosine"
        ));
    }

    // -- deny_unknown_fields for write commands --------------------------------

    #[test]
    fn test_deny_unknown_fields_db_insert() {
        let json = serde_json::json!({
            "command": "db.insert",
            "params": { "collection": "test", "data": {}, "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_fields_db_purge() {
        let json = serde_json::json!({
            "command": "db.purge",
            "params": { "document_id": "test/1", "cascade": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    // -- Write safety guard ---------------------------------------------------

    #[test]
    fn test_write_denied_on_production_db() {
        // Default config targets NestedLearning — writes must be rejected.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbInsert(DbInsertParams {
                collection: "test".into(),
                data: serde_json::json!({"x": 1}),
            }),
        ));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, DispatchError::Handler(HandlerError::WriteDenied(_))),
            "expected WriteDenied, got: {err:?}"
        );
    }

    #[test]
    fn test_write_denied_on_production_db_update() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbUpdate(DbUpdateParams {
                collection: "test".into(),
                key: "k1".into(),
                data: serde_json::json!({"x": 1}),
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }

    #[test]
    fn test_write_denied_on_production_db_delete() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbDelete(DbDeleteParams {
                collection: "test".into(),
                key: "k1".into(),
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }

    #[test]
    fn test_write_denied_on_production_db_purge() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbPurge(DbPurgeParams {
                document_id: "test/k1".into(),
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }

    #[test]
    fn test_write_denied_on_production_db_create_collection() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default();
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbCreateCollection(DbCreateCollectionParams {
                name: "test".into(),
                collection_type: None,
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }

    // -- Collection type parsing ----------------------------------------------

    #[test]
    fn test_create_collection_invalid_type() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut config = HadesConfig::default();
        config.apply_cli_overrides(Some("bident_burn"), None);
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbCreateCollection(DbCreateCollectionParams {
                name: "test".into(),
                collection_type: Some("invalid_type".into()),
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::InvalidParameter { .. })
        ));
    }

    // ── Graph command tests ────────────────────────────────────────

    #[test]
    fn test_command_roundtrip_graph_traverse() {
        let json = serde_json::json!({
            "command": "db.graph.traverse",
            "params": {
                "start": "hope_axioms/ax_001",
                "direction": "outbound",
                "min_depth": 1,
                "max_depth": 3,
                "limit": 50,
                "graph": "nl_core"
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphTraverse(ref p)
            if p.start == "hope_axioms/ax_001"
                && p.direction == "outbound"
                && p.min_depth == 1
                && p.max_depth == 3
                && p.limit == Some(50)
                && p.graph.as_deref() == Some("nl_core")
        ));
    }

    #[test]
    fn test_command_roundtrip_graph_traverse_defaults() {
        let json = serde_json::json!({
            "command": "db.graph.traverse",
            "params": { "start": "hope_axioms/ax_001" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphTraverse(ref p)
            if p.start == "hope_axioms/ax_001"
                && p.direction == "outbound"
                && p.min_depth == 1
                && p.max_depth == 1
                && p.limit.is_none()
                && p.graph.is_none()
        ));
    }

    #[test]
    fn test_command_roundtrip_graph_shortest_path() {
        let json = serde_json::json!({
            "command": "db.graph.shortest_path",
            "params": {
                "source": "hope_axioms/ax_001",
                "target": "hope_axioms/ax_002",
                "direction": "outbound",
                "graph": "nl_core"
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphShortestPath(ref p)
            if p.source == "hope_axioms/ax_001"
                && p.target == "hope_axioms/ax_002"
                && p.direction == "outbound"
                && p.graph.as_deref() == Some("nl_core")
        ));
    }

    #[test]
    fn test_command_roundtrip_graph_shortest_path_defaults() {
        let json = serde_json::json!({
            "command": "db.graph.shortest_path",
            "params": {
                "source": "hope_axioms/ax_001",
                "target": "hope_axioms/ax_002"
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphShortestPath(ref p)
            if p.source == "hope_axioms/ax_001"
                && p.target == "hope_axioms/ax_002"
                && p.direction == "any"
                && p.graph.is_none()
        ));
    }

    #[test]
    fn test_command_roundtrip_graph_neighbors() {
        let json = serde_json::json!({
            "command": "db.graph.neighbors",
            "params": {
                "vertex": "hope_axioms/ax_001",
                "direction": "inbound",
                "limit": 10,
                "graph": "nl_core"
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphNeighbors(ref p)
            if p.vertex == "hope_axioms/ax_001"
                && p.direction == "inbound"
                && p.limit == Some(10)
                && p.graph.as_deref() == Some("nl_core")
        ));
    }

    #[test]
    fn test_command_roundtrip_graph_list() {
        let json = serde_json::json!({
            "command": "db.graph.list",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::DbGraphList {}));
    }

    #[test]
    fn test_command_roundtrip_graph_create() {
        let json = serde_json::json!({
            "command": "db.graph.create",
            "params": {
                "name": "test_graph",
                "edge_definitions": [
                    {"collection": "edges", "from": ["v1"], "to": ["v2"]}
                ]
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphCreate(ref p)
            if p.name == "test_graph" && p.edge_definitions.is_some()
        ));
    }

    #[test]
    fn test_command_roundtrip_graph_drop() {
        let json = serde_json::json!({
            "command": "db.graph.drop",
            "params": { "name": "test_graph", "drop_collections": true, "force": true }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::DbGraphDrop(ref p)
            if p.name == "test_graph" && p.drop_collections && p.force
        ));
    }

    #[test]
    fn test_deny_unknown_fields_graph_traverse() {
        let json = serde_json::json!({
            "command": "db.graph.traverse",
            "params": { "start": "col/key", "bogus": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_fields_graph_create() {
        let json = serde_json::json!({
            "command": "db.graph.create",
            "params": { "name": "test", "bogus": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_fields_graph_drop() {
        let json = serde_json::json!({
            "command": "db.graph.drop",
            "params": { "name": "test", "bogus": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_validate_direction() {
        // Valid directions.
        assert_eq!(
            handlers::validate_direction("outbound").unwrap(),
            "OUTBOUND"
        );
        assert_eq!(
            handlers::validate_direction("inbound").unwrap(),
            "INBOUND"
        );
        assert_eq!(handlers::validate_direction("any").unwrap(), "ANY");

        // Invalid directions.
        assert!(handlers::validate_direction("backwards").is_err());
        assert!(handlers::validate_direction("OUTBOUND").is_err());
        assert!(handlers::validate_direction("").is_err());
    }

    #[test]
    fn test_write_denied_on_production_db_graph_create() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default(); // NestedLearning — production
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbGraphCreate(DbGraphCreateParams {
                name: "test".into(),
                edge_definitions: None,
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }

    #[test]
    fn test_write_denied_on_production_db_graph_drop() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default(); // NestedLearning — production
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::DbGraphDrop(DbGraphDropParams {
                name: "test".into(),
                drop_collections: false,
                force: true,
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }
}
