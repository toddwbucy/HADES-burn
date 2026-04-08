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

    /// An external service call failed.
    #[error("service error: {0}")]
    ServiceError(String),
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

/// Params for `db.schema.init`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbSchemaInitParams {
    /// Seed name: "nl" or "empty".
    pub seed: String,
}

/// Params for `db.schema.show`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbSchemaShowParams {
    /// Name of the edge definition or named graph to show.
    pub name: String,
}

/// Params for `db.graph.materialize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DbGraphMaterializeParams {
    /// Filter to a single edge collection name.
    #[serde(default)]
    pub edge: Option<String>,
    /// Preview mode — count edges without inserting.
    #[serde(default)]
    pub dry_run: bool,
    /// Also create named graphs via the Gharial API.
    #[serde(default)]
    pub register: bool,
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

/// Params for `task.list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskListParams {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(rename = "type")]
    pub task_type: Option<String>,
    pub parent: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Params for `task.show`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskShowParams {
    pub key: String,
}

/// Params for `task.create`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCreateParams {
    pub title: String,
    pub description: Option<String>,
    #[serde(rename = "type", default = "default_task_type")]
    pub task_type: String,
    pub parent: Option<String>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Params for `task.update`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskUpdateParams {
    pub key: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub priority: Option<String>,
    pub status: Option<String>,
    #[serde(default)]
    pub add_tags: Vec<String>,
    #[serde(default)]
    pub remove_tags: Vec<String>,
}

/// Params for `task.close`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCloseParams {
    pub key: String,
    pub message: Option<String>,
}

/// Params for `task.start`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskStartParams {
    pub key: String,
}

/// Params for `task.context`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskContextParams {
    pub key: String,
}

/// Params for `task.review`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskReviewParams {
    pub key: String,
    pub message: Option<String>,
}

/// Params for `task.approve`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskApproveParams {
    pub key: String,
    #[serde(default)]
    pub human: bool,
}

/// Params for `task.block`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskBlockParams {
    pub key: String,
    pub message: Option<String>,
    pub blocker: Option<String>,
}

/// Params for `task.unblock`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskUnblockParams {
    pub key: String,
}

/// Params for `task.handoff`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskHandoffParams {
    pub key: String,
    pub message: Option<String>,
}

/// Params for `task.handoff_show`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskHandoffShowParams {
    pub key: String,
}

/// Params for `task.log`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskLogParams {
    pub key: String,
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Params for `task.sessions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSessionsParams {
    pub key: String,
}

/// Params for `task.dep`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskDepParams {
    pub key: String,
    pub add: Option<String>,
    pub remove: Option<String>,
    #[serde(default)]
    pub graph: bool,
}

/// Params for `task.usage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskUsageParams {}

/// Params for `task.graph_integration`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskGraphIntegrationParams {}

/// Params for `smell.check`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmellCheckParams {
    pub path: String,
    #[serde(default)]
    pub verbose: bool,
}

/// Params for `smell.verify`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmellVerifyParams {
    pub path: String,
    #[serde(default)]
    pub claims: Vec<String>,
}

/// Params for `smell.report`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmellReportParams {
    pub path: String,
}

/// Params for `link_code_smell`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkCodeSmellParams {
    pub source_id: String,
    pub smell_id: String,
    pub enforcement: String,
    #[serde(default)]
    pub methods: Vec<String>,
    pub summary: Option<String>,
}

/// Params for `codebase.stats` (no parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodebaseStatsParams {}

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

    #[serde(rename = "db.graph.materialize")]
    DbGraphMaterialize(DbGraphMaterializeParams),

    // ── Schema management ──────────────────────────────────────────
    #[serde(rename = "db.schema.init")]
    DbSchemaInit(DbSchemaInitParams),

    #[serde(rename = "db.schema.list")]
    DbSchemaList {},

    #[serde(rename = "db.schema.show")]
    DbSchemaShow(DbSchemaShowParams),

    #[serde(rename = "db.schema.version")]
    DbSchemaVersion {},

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
    TaskList(TaskListParams),

    #[serde(rename = "task.show")]
    TaskShow(TaskShowParams),

    #[serde(rename = "task.create")]
    TaskCreate(TaskCreateParams),

    #[serde(rename = "task.update")]
    TaskUpdate(TaskUpdateParams),

    #[serde(rename = "task.close")]
    TaskClose(TaskCloseParams),

    #[serde(rename = "task.start")]
    TaskStart(TaskStartParams),

    #[serde(rename = "task.context")]
    TaskContext(TaskContextParams),

    #[serde(rename = "task.review")]
    TaskReview(TaskReviewParams),

    #[serde(rename = "task.approve")]
    TaskApprove(TaskApproveParams),

    #[serde(rename = "task.block")]
    TaskBlock(TaskBlockParams),

    #[serde(rename = "task.unblock")]
    TaskUnblock(TaskUnblockParams),

    #[serde(rename = "task.handoff")]
    TaskHandoff(TaskHandoffParams),

    #[serde(rename = "task.handoff_show")]
    TaskHandoffShow(TaskHandoffShowParams),

    #[serde(rename = "task.log")]
    TaskLog(TaskLogParams),

    #[serde(rename = "task.sessions")]
    TaskSessions(TaskSessionsParams),

    #[serde(rename = "task.dep")]
    TaskDep(TaskDepParams),

    #[serde(rename = "task.usage")]
    TaskUsage(TaskUsageParams),

    #[serde(rename = "task.graph_integration")]
    TaskGraphIntegration(TaskGraphIntegrationParams),

    // ── Smell ───────────────────────────────────────────────────────
    #[serde(rename = "smell.check")]
    SmellCheck(SmellCheckParams),

    #[serde(rename = "smell.verify")]
    SmellVerify(SmellVerifyParams),

    #[serde(rename = "smell.report")]
    SmellReport(SmellReportParams),

    #[serde(rename = "link_code_smell")]
    LinkCodeSmell(LinkCodeSmellParams),

    // ── Codebase ────────────────────────────────────────────────────
    #[serde(rename = "codebase.stats")]
    CodebaseStats(CodebaseStatsParams),
}

// ---------------------------------------------------------------------------
// Access tier
// ---------------------------------------------------------------------------

/// Access tier for daemon commands.
///
/// Partitions the command vocabulary into three tiers:
///
/// - **Agent**: Safe for model/AI agent use. Bounded operations with
///   predictable cost and no raw query injection. This is the closed
///   vocabulary that models operate within — no AQL, no schema mutation,
///   no unbounded writes.
/// - **Admin**: Requires human authorization. Raw AQL, schema changes,
///   destructive operations. Never exposed to model agents.
/// - **Internal**: System bookkeeping (health, stats). Safe for any
///   caller but not part of the model-facing vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessTier {
    /// Safe for model/AI agents — bounded, predictable operations.
    Agent,
    /// Requires human authorization — raw queries, schema changes.
    Admin,
    /// System bookkeeping — health checks, diagnostics.
    Internal,
}

impl DaemonCommand {
    /// Classify this command's access tier.
    ///
    /// The tier determines whether a model agent is allowed to invoke
    /// the command. The daemon can enforce this boundary at the wire
    /// protocol level, rejecting `Admin` commands from agent sessions.
    ///
    /// # Tier assignments
    ///
    /// **Agent** — bounded reads, task management, semantic search,
    /// graph traversal, code smell checks:
    /// - `db.query`, `db.get`, `db.list`, `db.count`, `db.check`, `db.recent`
    /// - `db.graph.traverse`, `db.graph.neighbors`, `db.graph.shortest_path`
    /// - `embed.text`, `graph_embed.embed`, `graph_embed.neighbors`
    /// - `task.*` (all task management)
    /// - `smell.check`, `smell.verify`, `smell.report`, `link_code_smell`
    /// - `orient`, `codebase.stats`
    ///
    /// **Admin** — raw AQL, schema mutation, destructive operations:
    /// - `db.aql` (arbitrary query injection)
    /// - `db.insert`, `db.update`, `db.delete`, `db.purge` (unbounded writes)
    /// - `db.create_collection`, `db.create_index` (schema changes)
    /// - `db.graph.create`, `db.graph.drop` (graph lifecycle)
    ///
    /// **Internal** — system diagnostics, no model relevance:
    /// - `status`, `db.health`, `db.stats`, `db.collections`, `db.graph.list`
    pub fn access_tier(&self) -> AccessTier {
        match self {
            // ── Agent: bounded reads ──────────────────────────────
            Self::Orient(_) => AccessTier::Agent,
            Self::DbQuery { .. } => AccessTier::Agent,
            Self::DbGet(_) => AccessTier::Agent,
            Self::DbList(_) => AccessTier::Agent,
            Self::DbCount(_) => AccessTier::Agent,
            Self::DbCheck(_) => AccessTier::Agent,
            Self::DbRecent(_) => AccessTier::Agent,

            // ── Agent: graph traversal ────────────────────────────
            Self::DbGraphTraverse(_) => AccessTier::Agent,
            Self::DbGraphShortestPath(_) => AccessTier::Agent,
            Self::DbGraphNeighbors(_) => AccessTier::Agent,

            // ── Agent: embeddings ─────────────────────────────────
            Self::EmbedText { .. } => AccessTier::Agent,
            Self::GraphEmbedEmbed(_) => AccessTier::Agent,
            Self::GraphEmbedNeighbors(_) => AccessTier::Agent,

            // ── Agent: task management ────────────────────────────
            Self::TaskList(_) => AccessTier::Agent,
            Self::TaskShow(_) => AccessTier::Agent,
            Self::TaskCreate(_) => AccessTier::Agent,
            Self::TaskUpdate(_) => AccessTier::Agent,
            Self::TaskClose(_) => AccessTier::Agent,
            Self::TaskStart(_) => AccessTier::Agent,
            Self::TaskContext(_) => AccessTier::Agent,
            Self::TaskReview(_) => AccessTier::Agent,
            Self::TaskApprove(_) => AccessTier::Agent,
            Self::TaskBlock(_) => AccessTier::Agent,
            Self::TaskUnblock(_) => AccessTier::Agent,
            Self::TaskHandoff(_) => AccessTier::Agent,
            Self::TaskHandoffShow(_) => AccessTier::Agent,
            Self::TaskLog(_) => AccessTier::Agent,
            Self::TaskSessions(_) => AccessTier::Agent,
            Self::TaskDep(_) => AccessTier::Agent,
            Self::TaskUsage(_) => AccessTier::Agent,
            Self::TaskGraphIntegration(_) => AccessTier::Agent,

            // ── Agent: code quality ───────────────────────────────
            Self::SmellCheck(_) => AccessTier::Agent,
            Self::SmellVerify(_) => AccessTier::Agent,
            Self::SmellReport(_) => AccessTier::Agent,
            Self::LinkCodeSmell(_) => AccessTier::Agent,

            // ── Agent: codebase ───────────────────────────────────
            Self::CodebaseStats(_) => AccessTier::Agent,

            // ── Admin: raw AQL ────────────────────────────────────
            Self::DbAql(_) => AccessTier::Admin,

            // ── Admin: unbounded writes ───────────────────────────
            Self::DbInsert(_) => AccessTier::Admin,
            Self::DbUpdate(_) => AccessTier::Admin,
            Self::DbDelete(_) => AccessTier::Admin,
            Self::DbPurge(_) => AccessTier::Admin,

            // ── Admin: schema mutation ────────────────────────────
            Self::DbCreateCollection(_) => AccessTier::Admin,
            Self::DbCreateIndex(_) => AccessTier::Admin,
            Self::DbGraphCreate(_) => AccessTier::Admin,
            Self::DbGraphDrop(_) => AccessTier::Admin,
            Self::DbGraphMaterialize(_) => AccessTier::Admin,
            Self::DbSchemaInit(_) => AccessTier::Admin,

            // ── Internal: system diagnostics ──────────────────────
            Self::Status(_) => AccessTier::Internal,
            Self::DbHealth(_) => AccessTier::Internal,
            Self::DbStats {} => AccessTier::Internal,
            Self::DbCollections {} => AccessTier::Internal,
            Self::DbGraphList {} => AccessTier::Internal,
            Self::DbSchemaList {} => AccessTier::Internal,
            Self::DbSchemaShow(_) => AccessTier::Internal,
            Self::DbSchemaVersion {} => AccessTier::Internal,
        }
    }

    /// Whether this command is safe for model/AI agent invocation.
    pub fn is_agent_safe(&self) -> bool {
        self.access_tier() == AccessTier::Agent
    }
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

        DaemonCommand::DbGraphMaterialize(params) => {
            require_writable(config)?;
            handlers::graph_materialize(pool, params.edge.as_deref(), params.dry_run, params.register)
                .await
                .map_err(DispatchError::Handler)
        }

        // ── Schema management ───────────────────────────────────────
        DaemonCommand::DbSchemaInit(params) => {
            require_writable(config)?;
            handlers::schema_init(pool, &params.seed)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbSchemaList {} => {
            handlers::schema_list(pool)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbSchemaShow(params) => {
            handlers::schema_show(pool, &params.name)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::DbSchemaVersion {} => {
            handlers::schema_version(pool)
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

        // ── Task commands ───────────────────────────────────────────
        DaemonCommand::TaskList(params) => {
            let limit = params.limit.unwrap_or(50).min(MAX_LIMIT);
            handlers::task_list(pool, params.status.as_deref(), params.task_type.as_deref(), params.parent.as_deref(), limit)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskShow(params) => {
            handlers::task_show(pool, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskCreate(params) => {
            require_writable(config)?;
            handlers::task_create(pool, &params)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskUpdate(params) => {
            require_writable(config)?;
            handlers::task_update(pool, &params)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskClose(params) => {
            require_writable(config)?;
            handlers::task_close(pool, &params.key, params.message.as_deref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskStart(params) => {
            require_writable(config)?;
            handlers::task_start(pool, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskReview(params) => {
            require_writable(config)?;
            handlers::task_review(pool, &params.key, params.message.as_deref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskApprove(params) => {
            require_writable(config)?;
            handlers::task_approve(pool, &params.key, params.human)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskBlock(params) => {
            require_writable(config)?;
            handlers::task_block(pool, &params.key, params.message.as_deref(), params.blocker.as_deref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskUnblock(params) => {
            require_writable(config)?;
            handlers::task_unblock(pool, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskHandoff(params) => {
            require_writable(config)?;
            handlers::task_handoff(pool, &params.key, params.message.as_deref())
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskHandoffShow(params) => {
            handlers::task_handoff_show(pool, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskContext(params) => {
            handlers::task_context(pool, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskLog(params) => {
            let limit = params.limit.unwrap_or(20).min(MAX_LIMIT);
            handlers::task_log(pool, &params.key, limit)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskSessions(params) => {
            handlers::task_sessions(pool, &params.key)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskDep(params) => {
            if params.add.is_some() || params.remove.is_some() {
                require_writable(config)?;
            }
            handlers::task_dep(
                pool,
                &params.key,
                params.add.as_deref(),
                params.remove.as_deref(),
                params.graph,
            )
            .await
            .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskUsage(_) => {
            handlers::task_usage(pool)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::TaskGraphIntegration(_) => {
            Ok(handlers::task_graph_integration())
        }

        // ── Embedding ─────────────────────────────────────────────────
        DaemonCommand::EmbedText { text } => {
            handlers::embed_text(config, &text)
                .await
                .map_err(DispatchError::Handler)
        }

        // ── Smell & Compliance ────────────────────────────────────────
        DaemonCommand::SmellCheck(params) => {
            handlers::smell_check(pool, &params.path, params.verbose)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::SmellVerify(params) => {
            handlers::smell_verify(pool, &params.path, &params.claims)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::SmellReport(params) => {
            handlers::smell_report(pool, config, &params.path)
                .await
                .map_err(DispatchError::Handler)
        }
        DaemonCommand::LinkCodeSmell(params) => {
            require_writable(config)?;
            handlers::link_code_smell(
                pool,
                &params.source_id,
                &params.smell_id,
                &params.enforcement,
                &params.methods,
                params.summary.as_deref(),
            )
            .await
            .map_err(DispatchError::Handler)
        }

        // ── Codebase ─────────────────────────────────────────────────
        DaemonCommand::CodebaseStats(_) => {
            handlers::codebase_stats(pool)
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
        } else {
            // Look up graph definition from the runtime schema.
            let schema = crate::graph::runtime_schema::RuntimeSchema::load(pool)
                .await
                .map_err(|e| HandlerError::Query {
                    context: "load runtime schema for graph creation".into(),
                    source: crate::db::ArangoError::Request(e.to_string()),
                })?;

            match schema.to_gharial_payload(name) {
                Some(payload) => payload,
                None => {
                    let known: Vec<&str> = schema
                        .named_graphs
                        .iter()
                        .map(|g| g.name.as_str())
                        .collect();
                    return Err(HandlerError::InvalidParameter {
                        name: "name".into(),
                        reason: format!(
                            "unknown graph '{name}' and no --edge-definitions provided — \
                             known graphs: {}",
                            known.join(", ")
                        ),
                    });
                }
            }
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

    // ── Graph materialization ─────────────────────────────────────

    /// Materialize edges from implicit cross-reference fields.
    ///
    /// Loads edge definitions from the database's `hades_schema` collection
    /// (falling back to compile-time NL statics if absent), then scans
    /// source collections for non-null reference fields and creates explicit
    /// edge documents.
    pub async fn graph_materialize(
        pool: &ArangoPool,
        edge_filter: Option<&str>,
        dry_run: bool,
        register: bool,
    ) -> Result<Value, HandlerError> {
        use std::collections::HashSet;
        use crate::db::crud;
        use crate::graph::runtime_schema::RuntimeSchema;

        // 1. Load schema from hades_schema (or NL statics fallback).
        let schema = RuntimeSchema::load(pool).await.map_err(|e| HandlerError::Query {
            context: "load runtime schema for materialization".into(),
            source: crate::db::ArangoError::Request(e.to_string()),
        })?;

        // 2. Fetch existing collections once.
        let existing: HashSet<String> = crud::list_collections(pool, true)
            .await
            .map_err(|e| HandlerError::Query {
                context: "list collections for materialization".into(),
                source: e,
            })?
            .into_iter()
            .map(|c| c.name)
            .collect();

        let mut definitions_output = json!({});
        let mut totals = MaterializeStats::default();
        let start = std::time::Instant::now();

        // 3. Process each edge definition from the runtime schema.
        for edef in &schema.edge_definitions {
            // Apply --edge filter.
            if let Some(filter) = edge_filter
                && edef.name != filter
            {
                continue;
            }

            let def_key = format!("{}:{}", edef.name, edef.source_field);
            tracing::info!(definition = %def_key, "materializing");

            let mut stats = MaterializeStats::default();

            // Determine strategy from schema metadata.
            let edges = match edef.materialize_strategy.as_str() {
                "lineage" => materialize_lineage(pool, edef, &existing, &mut stats).await,
                "cross_paper" => materialize_cross_paper(pool, edef, &existing, &mut stats).await,
                _ => materialize_standard(pool, edef, &existing, &mut stats).await,
            };

            // Insert or count.
            if !dry_run && !edges.is_empty() {
                // Ensure edge collection exists.
                if !existing.contains(&edef.name)
                    && let Err(e) = crud::create_collection(pool, &edef.name, Some(3)).await
                {
                    // 1207 = duplicate name (created concurrently) — not an error.
                    let is_dup = matches!(&e, crate::db::ArangoError::Api { error_num: 1207, .. });
                    if !is_dup {
                        stats.errors.push(format!("create collection {}: {e}", edef.name));
                    }
                }

                // Bulk upsert in chunks.
                for chunk in edges.chunks(5000) {
                    match crud::upsert_documents(pool, &edef.name, chunk).await {
                        Ok(result) => {
                            stats.edges_created += result.created + result.updated;
                        }
                        Err(e) => {
                            stats.errors.push(format!("import {}: {e}", edef.name));
                        }
                    }
                }
            } else {
                stats.edges_created = edges.len() as u64;
            }

            definitions_output[&def_key] = json!({
                "edges_created": stats.edges_created,
                "edges_skipped": stats.edges_skipped,
                "collections_scanned": stats.collections_scanned,
                "collections_missing": stats.collections_missing,
                "errors": stats.errors,
            });

            totals.edges_created += stats.edges_created;
            totals.edges_skipped += stats.edges_skipped;
            totals.collections_scanned += stats.collections_scanned;
            totals.collections_missing += stats.collections_missing;
            totals.errors.extend(stats.errors);
        }

        // 4. Register named graphs if requested.
        let mut graphs_registered = Vec::new();
        if register && !dry_run {
            for ng in &schema.named_graphs {
                if let Some(payload) = schema.to_gharial_payload(&ng.name) {
                    match pool.writer().post("gharial", &payload).await {
                        Ok(_) => graphs_registered.push(ng.name.clone()),
                        Err(e) => {
                            // 1925 = graph already exists.
                            let is_exists = matches!(&e, crate::db::ArangoError::Api { error_num: 1925, .. });
                            if is_exists {
                                graphs_registered.push(ng.name.clone());
                            } else {
                                totals.errors.push(format!("create graph {}: {e}", ng.name));
                            }
                        }
                    }
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(json!({
            "definitions": definitions_output,
            "totals": {
                "edges_created": totals.edges_created,
                "edges_skipped": totals.edges_skipped,
                "collections_scanned": totals.collections_scanned,
                "collections_missing": totals.collections_missing,
                "errors": totals.errors,
                "duration_ms": duration_ms,
            },
            "dry_run": dry_run,
            "schema_source": if schema.from_database { "hades_schema" } else { "nl_statics_fallback" },
            "named_graphs_registered": graphs_registered,
        }))
    }

    /// Stats accumulated during materialization of one edge definition.
    #[derive(Default)]
    struct MaterializeStats {
        edges_created: u64,
        edges_skipped: u64,
        collections_scanned: u64,
        collections_missing: u64,
        errors: Vec<String>,
    }

    /// Resolve a reference string to a valid `collection/key` ID.
    /// Returns `None` for bare keys, empty strings, or non-string values.
    fn resolve_ref(val: &Value) -> Option<&str> {
        let s = val.as_str()?;
        if s.is_empty() || !s.contains('/') {
            return None;
        }
        Some(s)
    }

    /// Build an edge `_key` from `_from` and `_to` IDs.
    fn edge_key(from_id: &str, to_id: &str) -> String {
        format!(
            "{}__{}",
            from_id.replace('/', "_"),
            to_id.replace('/', "_"),
        )
    }

    /// Strategy A: Standard references (single or array field).
    async fn materialize_standard(
        pool: &ArangoPool,
        edef: &crate::graph::runtime_schema::RuntimeEdgeDef,
        existing: &std::collections::HashSet<String>,
        stats: &mut MaterializeStats,
    ) -> Vec<Value> {
        use crate::db::query::{self, ExecutionTarget};

        let mut edges = Vec::new();
        let field = &edef.source_field;

        for from_coll in &edef.from_collections {
            if !existing.contains(from_coll.as_str()) {
                stats.collections_missing += 1;
                continue;
            }
            stats.collections_scanned += 1;

            // Scan for documents with the source field set.
            let aql = format!(
                "FOR d IN `{from_coll}` FILTER d.`{field}` != null RETURN {{ _id: d._id, _key: d._key, ref: d.`{field}` }}"
            );

            let docs = match query::query(pool, &aql, None, None, false, ExecutionTarget::Reader).await {
                Ok(r) => r.results,
                Err(e) => {
                    stats.errors.push(format!("{from_coll}: {e}"));
                    continue;
                }
            };

            for doc in &docs {
                let from_id = match doc["_id"].as_str() {
                    Some(id) => id,
                    None => continue,
                };

                let refs: Vec<&str> = if edef.is_array {
                    doc["ref"]
                        .as_array()
                        .map(|arr| arr.iter().filter_map(resolve_ref).collect())
                        .unwrap_or_default()
                } else {
                    match resolve_ref(&doc["ref"]) {
                        Some(r) => vec![r],
                        None => {
                            stats.edges_skipped += 1;
                            continue;
                        }
                    }
                };

                for to_id in refs {
                    // Validate target collection exists.
                    let to_coll = match to_id.split('/').next() {
                        Some(c) if existing.contains(c) => c,
                        _ => {
                            stats.edges_skipped += 1;
                            continue;
                        }
                    };
                    let _ = to_coll; // used for validation above

                    let mut edge = json!({
                        "_from": from_id,
                        "_to": to_id,
                        "_key": edge_key(from_id, to_id),
                        "source_field": field,
                    });

                    // Copy edge_attributes from source doc.
                    for attr in &edef.edge_attributes {
                        if let Some(val) = doc.get(attr.as_str())
                            && !val.is_null()
                        {
                            edge[attr.as_str()] = val.clone();
                        }
                    }

                    edges.push(edge);
                }
            }
        }

        edges
    }

    /// Strategy B: Lineage chains (sequential + membership edges).
    async fn materialize_lineage(
        pool: &ArangoPool,
        edef: &crate::graph::runtime_schema::RuntimeEdgeDef,
        existing: &std::collections::HashSet<String>,
        stats: &mut MaterializeStats,
    ) -> Vec<Value> {
        use crate::db::query::{self, ExecutionTarget};

        let mut edges = Vec::new();
        let attrs = &edef.edge_attributes;

        for from_coll in &edef.from_collections {
            if !existing.contains(from_coll.as_str()) {
                stats.collections_missing += 1;
                continue;
            }
            stats.collections_scanned += 1;

            // Build RETURN clause with edge_attributes.
            let attr_fields: String = attrs
                .iter()
                .map(|a| format!(", {a}: d.`{a}`"))
                .collect();
            let aql = format!(
                "FOR d IN `{from_coll}` FILTER d.chain != null AND LENGTH(d.chain) >= 2 \
                 RETURN {{ _id: d._id, _key: d._key, chain: d.chain{attr_fields} }}"
            );

            let docs = match query::query(pool, &aql, None, None, false, ExecutionTarget::Reader).await {
                Ok(r) => r.results,
                Err(e) => {
                    stats.errors.push(format!("{from_coll}: {e}"));
                    continue;
                }
            };

            for doc in &docs {
                let lineage_id = match doc["_id"].as_str() {
                    Some(id) => id,
                    None => continue,
                };
                let lineage_key = match doc["_key"].as_str() {
                    Some(k) => k,
                    None => continue,
                };
                let chain: Vec<&str> = match doc["chain"].as_array() {
                    Some(arr) => arr.iter().filter_map(resolve_ref).collect(),
                    None => continue,
                };
                if chain.len() < 2 {
                    continue;
                }

                // Sequential edges: chain[i] → chain[i+1]
                for i in 0..chain.len() - 1 {
                    let mut edge = json!({
                        "_from": chain[i],
                        "_to": chain[i + 1],
                        "_key": format!("{lineage_key}__step_{i}"),
                        "source_field": "chain",
                        "lineage_doc": lineage_id,
                        "chain_position": i,
                    });
                    for attr in attrs {
                        if let Some(val) = doc.get(attr.as_str())
                            && !val.is_null()
                        {
                            edge[attr.as_str()] = val.clone();
                        }
                    }
                    edges.push(edge);
                }

                // Membership edges: lineage_doc → each chain member
                for (i, &member) in chain.iter().enumerate() {
                    let mut edge = json!({
                        "_from": lineage_id,
                        "_to": member,
                        "_key": format!("{lineage_key}__member_{i}"),
                        "source_field": "chain",
                        "chain_position": i,
                    });
                    for attr in attrs {
                        if let Some(val) = doc.get(attr.as_str())
                            && !val.is_null()
                        {
                            edge[attr.as_str()] = val.clone();
                        }
                    }
                    edges.push(edge);
                }
            }
        }

        edges
    }

    /// Strategy C: Cross-paper edges (paired from_node/to_node fields).
    async fn materialize_cross_paper(
        pool: &ArangoPool,
        edef: &crate::graph::runtime_schema::RuntimeEdgeDef,
        existing: &std::collections::HashSet<String>,
        stats: &mut MaterializeStats,
    ) -> Vec<Value> {
        use crate::db::query::{self, ExecutionTarget};

        let mut edges = Vec::new();

        if !existing.contains("paper_edges") {
            stats.collections_missing += 1;
            return edges;
        }
        stats.collections_scanned += 1;

        let aql = "FOR d IN paper_edges FILTER d.from_node != null AND d.to_node != null RETURN d";

        let docs = match query::query(pool, aql, None, None, false, ExecutionTarget::Reader).await {
            Ok(r) => r.results,
            Err(e) => {
                stats.errors.push(format!("paper_edges: {e}"));
                return edges;
            }
        };

        for doc in &docs {
            let from_node = match doc["from_node"].as_str() {
                Some(s) if s.contains('/') => s,
                _ => {
                    stats.edges_skipped += 1;
                    continue;
                }
            };
            let to_node = match doc["to_node"].as_str() {
                Some(s) if s.contains('/') => s,
                _ => {
                    stats.edges_skipped += 1;
                    continue;
                }
            };
            let doc_key = match doc["_key"].as_str() {
                Some(k) => k,
                None => continue,
            };

            let mut edge = json!({
                "_from": from_node,
                "_to": to_node,
                "_key": doc_key,
                "source_field": "from_node/to_node",
            });

            // Copy edge_attributes.
            for attr in &edef.edge_attributes {
                if let Some(val) = doc.get(attr.as_str())
                    && !val.is_null()
                {
                    edge[attr.as_str()] = val.clone();
                }
            }

            edges.push(edge);
        }

        edges
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
        let count = match count_collection(pool, collection).await {
            Ok(n) => n,
            Err(e) if e.is_not_found() => 0,
            Err(e) => {
                return Err(HandlerError::Query {
                    context: format!("failed to count '{collection}'"),
                    source: e,
                });
            }
        };

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

    // ── Task handlers ──────────────────────────────────────────────

    const TASK_COLLECTION: &str = "persephone_tasks";
    const VALID_STATUSES: &[&str] = &["open", "in_progress", "in_review", "closed", "blocked"];
    const VALID_PRIORITIES: &[&str] = &["critical", "high", "medium", "low"];
    const VALID_TYPES: &[&str] = &["task", "bug", "epic"];

    /// Validate a value against an allowed set, returning InvalidParameter on failure.
    fn validate_enum(name: &str, value: &str, allowed: &[&str]) -> Result<(), HandlerError> {
        if allowed.contains(&value) {
            Ok(())
        } else {
            Err(HandlerError::InvalidParameter {
                name: name.to_string(),
                reason: format!("must be one of {:?}, got '{value}'", allowed),
            })
        }
    }

    /// List tasks with optional filters.
    pub async fn task_list(
        pool: &ArangoPool,
        status: Option<&str>,
        task_type: Option<&str>,
        parent: Option<&str>,
        limit: u32,
    ) -> Result<Value, HandlerError> {
        if let Some(s) = status {
            validate_enum("status", s, VALID_STATUSES)?;
        }
        if let Some(t) = task_type {
            validate_enum("type", t, VALID_TYPES)?;
        }

        let mut aql = String::from("FOR doc IN @@col");
        let mut bind = json!({ "@col": TASK_COLLECTION });

        if let Some(s) = status {
            aql.push_str(" FILTER doc.status == @status");
            bind["status"] = json!(s);
        }
        if let Some(t) = task_type {
            aql.push_str(" FILTER doc.type == @task_type");
            bind["task_type"] = json!(t);
        }
        if let Some(p) = parent {
            aql.push_str(" FILTER doc.parent_key == @parent");
            bind["parent"] = json!(p);
        }
        aql.push_str(" SORT doc.created_at DESC LIMIT @limit RETURN doc");
        bind["limit"] = json!(limit);

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
            context: "task list query failed".into(),
            source: e,
        })?;

        Ok(json!({
            "tasks": result.results,
            "count": result.results.len(),
        }))
    }

    /// Get a single task by key.
    pub async fn task_show(
        pool: &ArangoPool,
        key: &str,
    ) -> Result<Value, HandlerError> {
        crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: TASK_COLLECTION.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to get task '{key}'"),
                        source: e,
                    }
                }
            })
    }

    /// Create a new task.
    ///
    /// Key generation retries up to 3 times on duplicate-key conflict
    /// (24-bit key space can collide under load).
    pub async fn task_create(
        pool: &ArangoPool,
        params: &super::TaskCreateParams,
    ) -> Result<Value, HandlerError> {
        use crate::db::ArangoErrorKind;
        use rand::Rng;

        const MAX_KEY_ATTEMPTS: u32 = 3;

        validate_enum("priority", &params.priority, VALID_PRIORITIES)?;
        validate_enum("type", &params.task_type, VALID_TYPES)?;

        if params.title.trim().is_empty() {
            return Err(HandlerError::InvalidParameter {
                name: "title".into(),
                reason: "must not be empty".into(),
            });
        }

        let now = chrono::Utc::now().to_rfc3339();

        for attempt in 0..MAX_KEY_ATTEMPTS {
            let key = format!("task_{:06x}", rand::rng().random::<u32>() & 0xFFFFFF);

            let doc = json!({
                "_key": key,
                "title": params.title.trim(),
                "description": params.description,
                "status": "open",
                "priority": params.priority,
                "type": params.task_type,
                "labels": params.tags,
                "parent_key": params.parent,
                "acceptance": null,
                "minor": false,
                "block_reason": null,
                "created_at": now,
                "updated_at": now,
            });

            match crud::insert_document(pool, TASK_COLLECTION, &doc).await {
                Ok(_) => return Ok(json!({ "task": doc })),
                Err(e) if matches!(e.kind(), ArangoErrorKind::Conflict) => {
                    if attempt == MAX_KEY_ATTEMPTS - 1 {
                        return Err(HandlerError::Query {
                            context: format!(
                                "failed to create task after {MAX_KEY_ATTEMPTS} key-collision retries"
                            ),
                            source: e,
                        });
                    }
                    // Retry with a new key.
                }
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: "failed to create task".into(),
                        source: e,
                    });
                }
            }
        }

        unreachable!("loop always returns")
    }

    /// Update a task's fields.
    pub async fn task_update(
        pool: &ArangoPool,
        params: &super::TaskUpdateParams,
    ) -> Result<Value, HandlerError> {
        // Verify task exists and get current state.
        let existing = crud::get_document(pool, TASK_COLLECTION, &params.key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: TASK_COLLECTION.to_string(),
                        key: params.key.clone(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to get task '{}'", params.key),
                        source: e,
                    }
                }
            })?;

        let mut patch = serde_json::Map::new();

        if let Some(ref title) = params.title {
            if title.trim().is_empty() {
                return Err(HandlerError::InvalidParameter {
                    name: "title".into(),
                    reason: "must not be empty".into(),
                });
            }
            patch.insert("title".into(), json!(title.trim()));
        }
        if let Some(ref desc) = params.description {
            patch.insert("description".into(), json!(desc));
        }
        if let Some(ref pri) = params.priority {
            validate_enum("priority", pri, VALID_PRIORITIES)?;
            patch.insert("priority".into(), json!(pri));
        }
        if let Some(ref status) = params.status {
            validate_enum("status", status, VALID_STATUSES)?;
            patch.insert("status".into(), json!(status));
        }

        // Tag handling: merge existing + add, then subtract remove.
        if !params.add_tags.is_empty() || !params.remove_tags.is_empty() {
            let existing_tags: Vec<String> = existing
                .get("labels")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            let mut tags: Vec<String> = existing_tags;
            for tag in &params.add_tags {
                if !tags.contains(tag) {
                    tags.push(tag.clone());
                }
            }
            tags.retain(|t| !params.remove_tags.contains(t));
            patch.insert("labels".into(), json!(tags));
        }

        if patch.is_empty() {
            // Nothing to update — just return existing.
            return Ok(json!({ "task": existing }));
        }

        let now = chrono::Utc::now().to_rfc3339();
        patch.insert("updated_at".into(), json!(now));

        let patch_value = Value::Object(patch);
        crud::update_document(pool, TASK_COLLECTION, &params.key, &patch_value)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to update task '{}'", params.key),
                source: e,
            })?;

        // Re-fetch to return the merged document.
        let updated = crud::get_document(pool, TASK_COLLECTION, &params.key)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to re-fetch task '{}'", params.key),
                source: e,
            })?;

        Ok(json!({ "task": updated }))
    }

    /// Close a task (set status to closed).
    pub async fn task_close(
        pool: &ArangoPool,
        key: &str,
        message: Option<&str>,
    ) -> Result<Value, HandlerError> {
        let existing = crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: TASK_COLLECTION.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to get task '{key}'"),
                        source: e,
                    }
                }
            })?;

        let current_status = existing.get("status").and_then(Value::as_str).unwrap_or("");
        if current_status == "closed" {
            return Err(HandlerError::InvalidParameter {
                name: "status".into(),
                reason: format!("task '{key}' is already closed"),
            });
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut patch = json!({
            "status": "closed",
            "updated_at": now,
        });
        if let Some(msg) = message {
            patch["close_message"] = json!(msg);
        }

        crud::update_document(pool, TASK_COLLECTION, key, &patch)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to close task '{key}'"),
                source: e,
            })?;

        let updated = crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to re-fetch task '{key}'"),
                source: e,
            })?;

        Ok(json!({ "task": updated }))
    }

    /// Start a task (transition to in_progress).
    pub async fn task_start(
        pool: &ArangoPool,
        key: &str,
    ) -> Result<Value, HandlerError> {
        let existing = crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: TASK_COLLECTION.to_string(),
                        key: key.to_string(),
                    }
                } else {
                    HandlerError::Query {
                        context: format!("failed to get task '{key}'"),
                        source: e,
                    }
                }
            })?;

        let current_status = existing.get("status").and_then(Value::as_str).unwrap_or("");
        if current_status != "open" && current_status != "blocked" {
            return Err(HandlerError::InvalidParameter {
                name: "status".into(),
                reason: format!(
                    "cannot start task '{key}': current status is '{current_status}', \
                     expected 'open' or 'blocked'"
                ),
            });
        }

        let now = chrono::Utc::now().to_rfc3339();
        let patch = json!({
            "status": "in_progress",
            "updated_at": now,
            "block_reason": null,
        });

        crud::update_document(pool, TASK_COLLECTION, key, &patch)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to start task '{key}'"),
                source: e,
            })?;

        let updated = crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to re-fetch task '{key}'"),
                source: e,
            })?;

        Ok(json!({ "task": updated }))
    }

    // ── Workflow handlers ──────────────────────────────────────────

    const EDGE_COLLECTION: &str = "persephone_edges";
    const LOG_COLLECTION: &str = "persephone_logs";
    const HANDOFF_COLLECTION: &str = "persephone_handoffs";

    /// Valid state transitions: (from_status, to_status).
    pub(crate) const VALID_TRANSITIONS: &[(&str, &str)] = &[
        ("open", "in_progress"),
        ("in_progress", "in_review"),
        ("in_progress", "blocked"),
        ("in_progress", "open"),
        ("blocked", "in_progress"),
        ("blocked", "open"),
        ("in_review", "closed"),
        ("in_review", "in_progress"),
        ("closed", "open"),
    ];

    /// Get a task document or return a typed error.
    fn get_task_error(key: &str) -> impl FnOnce(crate::db::ArangoError) -> HandlerError + '_ {
        move |e| {
            if e.is_not_found() {
                HandlerError::DocumentNotFound {
                    collection: TASK_COLLECTION.to_string(),
                    key: key.to_string(),
                }
            } else {
                HandlerError::Query {
                    context: format!("failed to get task '{key}'"),
                    source: e,
                }
            }
        }
    }

    /// Best-effort activity log: insert into persephone_logs, ignoring errors.
    async fn log_activity(pool: &ArangoPool, action: &str, task_key: &str, details: Option<&Value>) {
        use rand::Rng;
        let key = format!("log_{:06x}", rand::rng().random::<u32>() & 0xFFFFFF);
        let now = chrono::Utc::now().to_rfc3339();
        let doc = json!({
            "_key": key,
            "action": action,
            "task_key": task_key,
            "session_key": null,
            "details": details,
            "created_at": now,
        });
        let _ = crud::insert_document(pool, LOG_COLLECTION, &doc).await;
    }

    /// Validate and apply a status transition, returning (updated_doc, from_status).
    ///
    /// `extra_patch` is merged into the update (e.g. `block_reason`).
    async fn transition_task(
        pool: &ArangoPool,
        key: &str,
        to_status: &str,
        extra_patch: Option<&Value>,
    ) -> Result<Value, HandlerError> {
        let existing = crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(get_task_error(key))?;

        let from_status = existing.get("status").and_then(Value::as_str).unwrap_or("");

        if !VALID_TRANSITIONS.contains(&(from_status, to_status)) {
            return Err(HandlerError::InvalidParameter {
                name: "status".into(),
                reason: format!(
                    "invalid transition for task '{key}': '{from_status}' → '{to_status}'"
                ),
            });
        }

        let now = chrono::Utc::now().to_rfc3339();
        let mut patch = json!({
            "status": to_status,
            "updated_at": now,
        });

        // Merge extra fields into the patch.
        if let Some(extra) = extra_patch
            && let Some(obj) = extra.as_object()
        {
            for (k, v) in obj {
                patch[k] = v.clone();
            }
        }

        crud::update_document(pool, TASK_COLLECTION, key, &patch)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to transition task '{key}'"),
                source: e,
            })?;

        // Best-effort activity log.
        let details = json!({
            "from_status": from_status,
            "to_status": to_status,
        });
        log_activity(pool, "task.transitioned", key, Some(&details)).await;

        // Re-fetch to return the merged document.
        let updated = crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to re-fetch task '{key}'"),
                source: e,
            })?;

        Ok(updated)
    }

    /// Submit a task for review (in_progress → in_review).
    pub async fn task_review(
        pool: &ArangoPool,
        key: &str,
        message: Option<&str>,
    ) -> Result<Value, HandlerError> {
        let updated = transition_task(pool, key, "in_review", None).await?;

        // Log review message if provided.
        if let Some(msg) = message {
            let details = json!({ "message": msg });
            log_activity(pool, "task.review_submitted", key, Some(&details)).await;
        }

        Ok(json!({ "task": updated }))
    }

    /// Approve a reviewed task (in_review → closed).
    ///
    /// The `human` flag is accepted for forward-compatibility but the
    /// different-reviewer guard is not enforced until session support
    /// lands in PR C.
    pub async fn task_approve(
        pool: &ArangoPool,
        key: &str,
        _human: bool,
    ) -> Result<Value, HandlerError> {
        let updated = transition_task(pool, key, "closed", None).await?;
        Ok(json!({ "task": updated }))
    }

    /// Block a task with a reason and optional blocker dependency.
    pub async fn task_block(
        pool: &ArangoPool,
        key: &str,
        message: Option<&str>,
        blocker: Option<&str>,
    ) -> Result<Value, HandlerError> {
        use crate::db::ArangoErrorKind;

        // Guard: block_reason is required.
        let reason = message.ok_or_else(|| HandlerError::InvalidParameter {
            name: "message".into(),
            reason: "a --message is required when blocking a task".into(),
        })?;

        // Validate blocker before any state mutation.
        if let Some(blocker_key) = blocker {
            if blocker_key == key {
                return Err(HandlerError::InvalidParameter {
                    name: "blocker".into(),
                    reason: "a task cannot depend on itself".into(),
                });
            }

            crud::get_document(pool, TASK_COLLECTION, blocker_key)
                .await
                .map_err(|e| {
                    if e.is_not_found() {
                        HandlerError::DocumentNotFound {
                            collection: TASK_COLLECTION.to_string(),
                            key: blocker_key.to_string(),
                        }
                    } else {
                        HandlerError::Query {
                            context: format!("failed to verify blocker task '{blocker_key}'"),
                            source: e,
                        }
                    }
                })?;
        }

        // Transition task to blocked.
        let extra = json!({ "block_reason": reason });
        let updated = transition_task(pool, key, "blocked", Some(&extra)).await?;

        // Create blocker dependency edge if specified.
        if let Some(blocker_key) = blocker {
            let edge_key = format!("{key}__blocked_by__{blocker_key}");
            let now = chrono::Utc::now().to_rfc3339();
            let edge = json!({
                "_key": edge_key,
                "_from": format!("{TASK_COLLECTION}/{key}"),
                "_to": format!("{TASK_COLLECTION}/{blocker_key}"),
                "type": "blocked_by",
                "created_at": now,
            });

            // Propagate errors; duplicate-edge conflicts are idempotent success.
            if let Err(e) = crud::insert_document(pool, EDGE_COLLECTION, &edge).await
                && !matches!(e.kind(), ArangoErrorKind::Conflict)
            {
                return Err(HandlerError::Query {
                    context: format!(
                        "failed to create blocked_by edge from '{key}' to '{blocker_key}'"
                    ),
                    source: e,
                });
            }
        }

        Ok(json!({ "task": updated }))
    }

    /// Unblock a task (blocked → in_progress).
    pub async fn task_unblock(
        pool: &ArangoPool,
        key: &str,
    ) -> Result<Value, HandlerError> {
        let extra = json!({ "block_reason": null });
        let updated = transition_task(pool, key, "in_progress", Some(&extra)).await?;
        Ok(json!({ "task": updated }))
    }

    /// Create a handoff context snapshot for a task.
    pub async fn task_handoff(
        pool: &ArangoPool,
        key: &str,
        message: Option<&str>,
    ) -> Result<Value, HandlerError> {
        use crate::db::ArangoErrorKind;
        use rand::Rng;

        const MAX_KEY_ATTEMPTS: u32 = 3;

        // Verify task exists.
        crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(get_task_error(key))?;

        // Require at least one content field (note via --message).
        let note = message.ok_or_else(|| HandlerError::InvalidParameter {
            name: "message".into(),
            reason: "a --message is required for handoff (at least one content field needed)".into(),
        })?;

        let now = chrono::Utc::now().to_rfc3339();

        for attempt in 0..MAX_KEY_ATTEMPTS {
            let handoff_key = format!("handoff_{:06x}", rand::rng().random::<u32>() & 0xFFFFFF);

            let doc = json!({
                "_key": handoff_key,
                "task_key": key,
                "session_key": null,
                "done": [],
                "remaining": [],
                "decisions": [],
                "uncertain": [],
                "note": note,
                "git_branch": null,
                "git_sha": null,
                "git_dirty_files": null,
                "git_changed_files": [],
                "created_at": now,
            });

            match crud::insert_document(pool, HANDOFF_COLLECTION, &doc).await {
                Ok(_) => {
                    // Create handoff_for edge: handoff → task.
                    let edge_key = format!("{handoff_key}__handoff_for__{key}");
                    let edge = json!({
                        "_key": edge_key,
                        "_from": format!("{HANDOFF_COLLECTION}/{handoff_key}"),
                        "_to": format!("{TASK_COLLECTION}/{key}"),
                        "type": "handoff_for",
                        "created_at": now,
                    });

                    if let Err(e) = crud::insert_document(pool, EDGE_COLLECTION, &edge).await {
                        // Clean up the orphaned handoff document.
                        if let Err(del_err) = crud::delete_document(pool, HANDOFF_COLLECTION, &handoff_key).await {
                            tracing::warn!(
                                handoff_key = %handoff_key,
                                "failed to clean up orphaned handoff after edge insert failure: {del_err}"
                            );
                        }
                        return Err(HandlerError::Query {
                            context: format!(
                                "failed to create handoff_for edge for handoff '{handoff_key}'"
                            ),
                            source: e,
                        });
                    }

                    // Best-effort activity log (after both writes succeed).
                    log_activity(pool, "handoff.created", key, None).await;

                    return Ok(json!({ "handoff": doc }));
                }
                Err(e) if matches!(e.kind(), ArangoErrorKind::Conflict) => {
                    if attempt == MAX_KEY_ATTEMPTS - 1 {
                        return Err(HandlerError::Query {
                            context: format!(
                                "failed to create handoff after {MAX_KEY_ATTEMPTS} key-collision retries"
                            ),
                            source: e,
                        });
                    }
                }
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: "failed to create handoff document".into(),
                        source: e,
                    });
                }
            }
        }

        unreachable!("loop always returns")
    }

    /// Show the latest handoff for a task.
    pub async fn task_handoff_show(
        pool: &ArangoPool,
        key: &str,
    ) -> Result<Value, HandlerError> {
        let aql = "\
            FOR e IN @@edges \
                FILTER e._to == @task_id \
                FILTER e.type == \"handoff_for\" \
                FOR h IN @@handoffs \
                    FILTER h._id == e._from \
                    SORT h.created_at DESC \
                    LIMIT 1 \
                    RETURN h";

        let bind = json!({
            "@edges": EDGE_COLLECTION,
            "@handoffs": HANDOFF_COLLECTION,
            "task_id": format!("{TASK_COLLECTION}/{key}"),
        });

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("failed to query handoffs for task '{key}'"),
            source: e,
        })?;

        let handoff = result.results.into_iter().next().unwrap_or(Value::Null);
        Ok(json!({ "handoff": handoff }))
    }

    // ── Query/meta handlers ────────────────────────────────────────

    /// Rich context dump for a task: task + handoff + sessions + deps + logs.
    pub async fn task_context(
        pool: &ArangoPool,
        key: &str,
    ) -> Result<Value, HandlerError> {
        let aql = "\
            LET task = DOCUMENT(@@tasks, @key) \
            LET latest_handoff = FIRST( \
                FOR e IN @@edges \
                    FILTER e._to == task._id \
                    FILTER e.type == \"handoff_for\" \
                    LET h = DOCUMENT(e._from) \
                    SORT h.created_at DESC \
                    LIMIT 1 \
                    RETURN h \
            ) \
            LET sessions = ( \
                FOR e IN @@edges \
                    FILTER e._to == task._id \
                    FILTER e.type IN [\"implements\", \"submitted_review\", \"approved\"] \
                    LET s = DOCUMENT(e._from) \
                    FILTER s != null \
                    SORT s.started_at DESC \
                    RETURN MERGE(s, {edge_type: e.type}) \
            ) \
            LET blockers = ( \
                FOR e IN @@edges \
                    FILTER e._from == task._id \
                    FILTER e.type == \"blocked_by\" \
                    LET blocker = DOCUMENT(e._to) \
                    FILTER blocker != null \
                    FILTER blocker.status != \"closed\" \
                    RETURN blocker \
            ) \
            LET recent_logs = ( \
                FOR log IN @@logs \
                    FILTER log.task_key == @key \
                    SORT log.created_at DESC \
                    LIMIT 5 \
                    RETURN log \
            ) \
            RETURN { \
                task: task, \
                latest_handoff: latest_handoff, \
                sessions: sessions, \
                blockers: blockers, \
                recent_logs: recent_logs \
            }";

        let bind = json!({
            "@tasks": TASK_COLLECTION,
            "@edges": EDGE_COLLECTION,
            "@logs": LOG_COLLECTION,
            "key": key,
        });

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("failed to query context for task '{key}'"),
            source: e,
        })?;

        let ctx = result.results.into_iter().next().unwrap_or(Value::Null);

        // Check if task was found.
        if ctx.get("task").is_none_or(Value::is_null) {
            return Err(HandlerError::DocumentNotFound {
                collection: TASK_COLLECTION.to_string(),
                key: key.to_string(),
            });
        }

        Ok(ctx)
    }

    /// Activity log entries for a task.
    pub async fn task_log(
        pool: &ArangoPool,
        key: &str,
        limit: u32,
    ) -> Result<Value, HandlerError> {
        let aql = "\
            FOR doc IN @@col \
                FILTER doc.task_key == @key \
                SORT doc.created_at DESC \
                LIMIT @limit \
                RETURN doc";

        let bind = json!({
            "@col": LOG_COLLECTION,
            "key": key,
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
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("failed to query logs for task '{key}'"),
            source: e,
        })?;

        Ok(json!({
            "logs": result.results,
            "count": result.results.len(),
        }))
    }

    /// Sessions linked to a task via edge traversal.
    pub async fn task_sessions(
        pool: &ArangoPool,
        key: &str,
    ) -> Result<Value, HandlerError> {
        let aql = "\
            FOR e IN @@edges \
                FILTER e._to == @task_id \
                FILTER e.type IN [\"implements\", \"submitted_review\", \"approved\"] \
                LET s = DOCUMENT(e._from) \
                FILTER s != null \
                SORT s.started_at DESC \
                RETURN MERGE(s, {edge_type: e.type})";

        let bind = json!({
            "@edges": EDGE_COLLECTION,
            "task_id": format!("{TASK_COLLECTION}/{key}"),
        });

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("failed to query sessions for task '{key}'"),
            source: e,
        })?;

        Ok(json!({
            "task_key": key,
            "sessions": result.results,
            "count": result.results.len(),
        }))
    }

    /// Manage task dependencies: show blockers, add, or remove.
    pub async fn task_dep(
        pool: &ArangoPool,
        key: &str,
        add: Option<&str>,
        remove: Option<&str>,
        graph: bool,
    ) -> Result<Value, HandlerError> {
        use crate::db::ArangoErrorKind;

        // Verify the primary task exists.
        crud::get_document(pool, TASK_COLLECTION, key)
            .await
            .map_err(get_task_error(key))?;

        if let Some(dep_key) = add {
            // Add dependency.
            if dep_key == key {
                return Err(HandlerError::InvalidParameter {
                    name: "add".into(),
                    reason: "a task cannot depend on itself".into(),
                });
            }

            crud::get_document(pool, TASK_COLLECTION, dep_key)
                .await
                .map_err(|e| {
                    if e.is_not_found() {
                        HandlerError::DocumentNotFound {
                            collection: TASK_COLLECTION.to_string(),
                            key: dep_key.to_string(),
                        }
                    } else {
                        HandlerError::Query {
                            context: format!("failed to verify dependency task '{dep_key}'"),
                            source: e,
                        }
                    }
                })?;

            let edge_key = format!("{key}__blocked_by__{dep_key}");
            let now = chrono::Utc::now().to_rfc3339();
            let edge = json!({
                "_key": edge_key,
                "_from": format!("{TASK_COLLECTION}/{key}"),
                "_to": format!("{TASK_COLLECTION}/{dep_key}"),
                "type": "blocked_by",
                "created_at": now,
            });

            if let Err(e) = crud::insert_document(pool, EDGE_COLLECTION, &edge).await
                && !matches!(e.kind(), ArangoErrorKind::Conflict)
            {
                return Err(HandlerError::Query {
                    context: format!("failed to create dependency edge from '{key}' to '{dep_key}'"),
                    source: e,
                });
            }

            log_activity(
                pool,
                "task.dependency_added",
                key,
                Some(&json!({ "depends_on": dep_key })),
            )
            .await;

            return Ok(json!({
                "edge": edge_key,
                "message": format!("{key} is now blocked by {dep_key}"),
            }));
        }

        if let Some(dep_key) = remove {
            // Remove dependency.
            let edge_key = format!("{key}__blocked_by__{dep_key}");
            let removed = match crud::delete_document(pool, EDGE_COLLECTION, &edge_key).await {
                Ok(_) => true,
                Err(e) if e.is_not_found() => false,
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to remove dependency edge '{edge_key}'"),
                        source: e,
                    });
                }
            };

            if removed {
                log_activity(
                    pool,
                    "task.dependency_removed",
                    key,
                    Some(&json!({ "depends_on": dep_key })),
                )
                .await;
            }

            return Ok(json!({
                "removed": removed,
                "message": if removed {
                    format!("{key} no longer blocked by {dep_key}")
                } else {
                    format!("no dependency from {key} to {dep_key}")
                },
            }));
        }

        // Default: show blockers.
        let aql = "\
            FOR e IN @@edges \
                FILTER e._from == @task_id \
                FILTER e.type == \"blocked_by\" \
                LET blocker = DOCUMENT(e._to) \
                FILTER blocker != null \
                RETURN blocker";

        let bind = json!({
            "@edges": EDGE_COLLECTION,
            "task_id": format!("{TASK_COLLECTION}/{key}"),
        });

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: format!("failed to query dependencies for task '{key}'"),
            source: e,
        })?;

        let mut response = json!({
            "task_key": key,
            "blocked": !result.results.is_empty(),
            "blockers": result.results,
        });

        // If --graph, add the dependency adjacency list.
        if graph {
            let graph_aql = "\
                FOR e IN @@edges \
                    FILTER e.type == \"blocked_by\" \
                    RETURN { from: e._from, to: e._to }";

            let graph_bind = json!({ "@edges": EDGE_COLLECTION });

            let graph_result = query::query(
                pool,
                graph_aql,
                Some(&graph_bind),
                None,
                false,
                ExecutionTarget::Reader,
            )
            .await
            .map_err(|e| HandlerError::Query {
                context: "failed to query dependency graph".into(),
                source: e,
            })?;

            response["dependency_graph"] = json!(graph_result.results);
        }

        Ok(response)
    }

    /// Aggregation statistics for the task system.
    pub async fn task_usage(
        pool: &ArangoPool,
    ) -> Result<Value, HandlerError> {
        let aql = "\
            LET by_status = ( \
                FOR doc IN @@tasks \
                    COLLECT status = doc.status WITH COUNT INTO c \
                    RETURN { status: status, count: c } \
            ) \
            LET by_priority = ( \
                FOR doc IN @@tasks \
                    COLLECT priority = doc.priority WITH COUNT INTO c \
                    RETURN { priority: priority, count: c } \
            ) \
            LET by_type = ( \
                FOR doc IN @@tasks \
                    COLLECT type = doc.type WITH COUNT INTO c \
                    RETURN { type: type, count: c } \
            ) \
            LET total = LENGTH(@@tasks) \
            RETURN { total: total, by_status: by_status, by_priority: by_priority, by_type: by_type }";

        let bind = json!({ "@tasks": TASK_COLLECTION });

        let result = query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| HandlerError::Query {
            context: "failed to query task usage statistics".into(),
            source: e,
        })?;

        let stats = result.results.into_iter().next().unwrap_or(json!({
            "total": 0,
            "by_status": [],
            "by_priority": [],
            "by_type": [],
        }));

        Ok(stats)
    }

    /// Knowledge graph integration protocol template.
    pub fn task_graph_integration() -> Value {
        json!({
            "protocol": "graph_integration",
            "description": "Knowledge graph integration protocol for NL compliance review",
            "steps": [
                "1. Run 'hades smell check <path>' to audit code compliance",
                "2. Run 'hades smell verify' to check CS-XX references",
                "3. Run 'hades smell report' for full compliance report",
                "4. Review findings and return PASS/REJECT decision"
            ]
        })
    }

    // ── Embedding handlers ────────────────────────────────────────────

    /// Embed a single text via the gRPC embedding service.
    pub async fn embed_text(
        config: &crate::config::HadesConfig,
        text: &str,
    ) -> Result<Value, HandlerError> {
        use crate::persephone::embedding::EmbeddingClient;

        let client = EmbeddingClient::connect_unix_at(&config.embedding.service.socket)
            .await
            .map_err(|e| HandlerError::ServiceError(e.to_string()))?;
        let result = client
            .embed(&[text.to_string()], "retrieval.passage", None)
            .await
            .map_err(|e| HandlerError::ServiceError(e.to_string()))?;

        let embedding = &result.embeddings[0];
        let preview_len = 10.min(embedding.len());
        let text_preview: String = if text.chars().count() > 100 {
            let mut s: String = text.chars().take(100).collect();
            s.push_str("...");
            s
        } else {
            text.to_string()
        };

        Ok(json!({
            "text": text_preview,
            "dimension": result.dimension,
            "model": result.model,
            "embedding": embedding,
            "embedding_preview": &embedding[..preview_len],
            "embedding_truncated": embedding.len() > preview_len,
            "duration_ms": result.duration_ms,
        }))
    }

    // ── Smell & compliance handlers ──────────────────────────────────

    /// STATIC tier smell IDs — violations block ingest.
    const STATIC_SMELL_IDS: &[i64] = &[10, 11, 13, 40];
    /// BEHAVIORAL tier smell IDs — informational.
    const BEHAVIORAL_SMELL_IDS: &[i64] = &[27, 28, 32];
    /// ARCHITECTURAL tier smell IDs — informational.
    const ARCHITECTURAL_SMELL_IDS: &[i64] = &[31];

    /// Source file extensions to scan.
    const SOURCE_EXTENSIONS: &[&str] = &[
        ".py", ".rs", ".cu", ".cpp", ".c", ".ts", ".js",
        ".toml", ".yaml", ".yml", ".sh", ".bash", ".md", ".txt", ".json",
    ];

    /// Allowed enforcement types for compliance edges.
    const ALLOWED_ENFORCEMENT: &[&str] = &[
        "static", "behavioral", "architectural", "review", "documentation",
    ];

    pub(crate) fn smell_tier(smell_id: Option<i64>) -> &'static str {
        match smell_id {
            Some(id) if STATIC_SMELL_IDS.contains(&id) => "static",
            Some(id) if BEHAVIORAL_SMELL_IDS.contains(&id) => "behavioral",
            Some(id) if ARCHITECTURAL_SMELL_IDS.contains(&id) => "architectural",
            _ => "unknown",
        }
    }

    pub(crate) fn comment_prefixes(ext: &str) -> &'static [&'static str] {
        match ext {
            ".py" | ".toml" | ".yaml" | ".yml" | ".sh" | ".bash" => &["#"],
            ".rs" => &["//", "///", "//!"],
            ".cu" | ".cpp" | ".c" | ".ts" | ".js" => &["//"],
            _ => &["#", "//"],
        }
    }

    pub(crate) fn is_comment_line(line: &str, ext: &str) -> bool {
        let stripped = line.trim_start();
        for prefix in comment_prefixes(ext) {
            if stripped.starts_with(prefix) {
                return true;
            }
        }
        false
    }

    /// Collect source files under a path, filtering hidden dirs and noise.
    /// Collect source files under a path, filtering hidden dirs and noise.
    ///
    /// Returns `Err` if the initial path does not exist or is unreadable.
    fn collect_source_files(
        path: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("path does not exist: {}", path.display()),
            ));
        }

        let mut files = Vec::new();
        if path.is_file() {
            files.push(path.to_path_buf());
            return Ok(files);
        }

        fn walk(
            dir: &std::path::Path,
            files: &mut Vec<std::path::PathBuf>,
        ) -> Result<(), std::io::Error> {
            let entries = std::fs::read_dir(dir)?;
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                // Skip hidden dirs, __pycache__, Acheron
                if name_str.starts_with('.') || name_str == "__pycache__" || name_str == "Acheron" {
                    continue;
                }
                if path.is_dir() {
                    // Subdirectory read errors are non-fatal — skip and continue.
                    let _ = walk(&path, files);
                } else if path.is_file()
                    && let Some(ext) = path.extension().and_then(|e| e.to_str())
                {
                    let dot_ext = format!(".{ext}");
                    if SOURCE_EXTENSIONS.contains(&dot_ext.as_str()) {
                        files.push(path);
                    }
                }
            }
            Ok(())
        }

        walk(path, &mut files)?;
        files.sort();
        Ok(files)
    }

    /// Extract CS-NN references from comment lines in a source file.
    fn extract_cs_refs(path: &std::path::Path) -> std::collections::HashMap<String, Vec<usize>> {
        use regex::Regex;
        use std::sync::LazyLock;

        static CS_REF_RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"\bCS-(\d+)\b").unwrap());

        let mut refs = std::collections::HashMap::new();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return refs,
        };

        let ext = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_default();

        for (line_no, line) in content.lines().enumerate() {
            if !is_comment_line(line, &ext) {
                continue;
            }
            for cap in CS_REF_RE.captures_iter(line) {
                let cs_id = format!("CS-{}", &cap[1]);
                refs.entry(cs_id).or_insert_with(Vec::new).push(line_no + 1);
            }
        }
        refs
    }

    /// Convert "CS-32" → "smell-032-" prefix for AQL LEFT match.
    pub(crate) fn smell_key_prefix(cs_number: &str) -> Option<String> {
        use regex::Regex;
        use std::sync::LazyLock;

        static CS_NUM_RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?i)^CS-(\d+)$").unwrap());

        CS_NUM_RE.captures(cs_number).and_then(|cap| {
            cap[1].parse::<u32>().ok().map(|num| format!("smell-{num:03}-"))
        })
    }

    pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Generate candidate document keys for a source file path.
    ///
    /// Given a file like `conductor.rs`, generates:
    /// `["conductor", "conductor-rs", "conductor_rs"]`
    pub(crate) fn candidate_doc_keys(path: &std::path::Path) -> Vec<String> {
        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if stem.is_empty() {
            return Vec::new();
        }
        let ext_label = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let mut keys = vec![stem.clone()];
        if !ext_label.is_empty() {
            keys.push(format!("{stem}-{ext_label}"));
        }
        // underscore → dash variant
        let dashed = stem.replace('_', "-");
        if dashed != stem {
            keys.push(dashed.clone());
            if !ext_label.is_empty() {
                keys.push(format!("{dashed}-{ext_label}"));
            }
        }
        keys
    }

    /// `hades smell check PATH` — scan source files for forbidden patterns.
    pub async fn smell_check(
        pool: &ArangoPool,
        path: &str,
        verbose: bool,
    ) -> Result<Value, HandlerError> {
        use crate::db::query::{self, ExecutionTarget};

        // Load smells with forbidden patterns from the graph.
        let aql = r#"
            FOR doc IN nl_code_smells
                FILTER doc.forbidden_patterns != null AND LENGTH(doc.forbidden_patterns) > 0
                RETURN {
                    _key: doc._key,
                    smell_id: doc.smell_id,
                    name: doc.name,
                    forbidden_patterns: doc.forbidden_patterns,
                    scope: doc.scope
                }
        "#;
        let result = query::query(pool, aql, None, None, false, ExecutionTarget::Reader)
            .await
            .map_err(|e| HandlerError::Query {
                context: "failed to load smell definitions".into(),
                source: e,
            })?;

        let smells = result.results;
        let smells_loaded = smells.len();

        // Collect source files.
        let scan_path = std::path::Path::new(path);
        let files = collect_source_files(scan_path).map_err(|e| {
            HandlerError::InvalidParameter {
                name: "path".into(),
                reason: e.to_string(),
            }
        })?;
        let files_checked = files.len();

        // Scan for forbidden patterns.
        let mut violations = Vec::new();
        let mut has_static_violation = false;

        for file in &files {
            let content = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let ext = file.extension()
                .and_then(|e| e.to_str())
                .map(|e| format!(".{e}"))
                .unwrap_or_default();

            for smell in &smells {
                let patterns = match smell["forbidden_patterns"].as_array() {
                    Some(p) => p,
                    None => continue,
                };
                let smell_id = smell["smell_id"].as_i64();

                for (line_no, line) in content.lines().enumerate() {
                    // CS-13 violations are skipped in comment context
                    if smell_id == Some(13) && is_comment_line(line, &ext) {
                        continue;
                    }

                    for pattern in patterns {
                        let pat = match pattern.as_str() {
                            Some(p) => p,
                            None => continue,
                        };
                        if line.contains(pat) {
                            let tier = smell_tier(smell_id);
                            if tier == "static" {
                                has_static_violation = true;
                            }

                            violations.push(json!({
                                "smell_key": smell["_key"],
                                "smell_id": smell_id,
                                "smell_name": smell["name"],
                                "tier": tier,
                                "file": file.display().to_string(),
                                "line": line_no + 1,
                                "pattern": pat,
                                "content": line.trim_end(),
                            }));
                        }
                    }
                }
            }
        }

        let violation_count = violations.len();
        let passed = !has_static_violation;

        let mut result = json!({
            "passed": passed,
            "violation_count": violation_count,
            "files_checked": files_checked,
            "smells_loaded": smells_loaded,
        });
        if verbose || !violations.is_empty() {
            result["violations"] = json!(violations);
        }
        Ok(result)
    }

    /// `hades smell verify PATH` — verify CS-NN references against graph.
    pub async fn smell_verify(
        pool: &ArangoPool,
        path: &str,
        claims: &[String],
    ) -> Result<Value, HandlerError> {
        use crate::db::query::{self, ExecutionTarget};

        // Build claims filter set — when non-empty, only these CS-NN refs are verified.
        let claims_filter: std::collections::HashSet<&str> =
            claims.iter().map(|s| s.as_str()).collect();

        let scan_path = std::path::Path::new(path);
        let files = collect_source_files(scan_path).map_err(|e| {
            HandlerError::InvalidParameter {
                name: "path".into(),
                reason: e.to_string(),
            }
        })?;

        // Extract all CS-NN refs from comment lines.
        let mut all_refs: Vec<(std::path::PathBuf, String, Vec<usize>)> = Vec::new();
        let mut unique_cs: std::collections::HashSet<String> = std::collections::HashSet::new();

        for file in &files {
            let refs = extract_cs_refs(file);
            for (cs_id, lines) in refs {
                // If claims filter is provided, skip refs not in the filter.
                if !claims_filter.is_empty() && !claims_filter.contains(cs_id.as_str()) {
                    continue;
                }
                unique_cs.insert(cs_id.clone());
                all_refs.push((file.clone(), cs_id, lines));
            }
        }

        // Batch-lookup smell nodes by CS number.
        let mut smell_map: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        for cs_id in &unique_cs {
            // Extract numeric part: "CS-32" → "32"
            let num_str = cs_id.strip_prefix("CS-").unwrap_or(cs_id);

            let aql = r#"
                FOR doc IN nl_code_smells
                    FILTER doc.smell_id == TO_NUMBER(@num) OR STARTS_WITH(doc.name, @prefix)
                    LIMIT 1
                    RETURN {_key: doc._key, _id: doc._id, smell_id: doc.smell_id, name: doc.name}
            "#;
            let bind_vars = json!({ "num": num_str, "prefix": format!("{cs_id}:") });
            let result = query::query(
                pool, aql, Some(&bind_vars), None, false, ExecutionTarget::Reader,
            )
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to look up smell for {cs_id}"),
                source: e,
            })?;

            if let Some(smell) = result.results.into_iter().next() {
                smell_map.insert(cs_id.clone(), smell);
            }
        }

        // Verify compliance edges.
        let mut verified_refs = Vec::new();
        let mut missing_from_graph = Vec::new();
        let mut unlinked_claims = Vec::new();

        for (file, cs_id, lines) in &all_refs {
            let smell = match smell_map.get(cs_id) {
                Some(s) => s,
                None => {
                    missing_from_graph.push(json!({
                        "cs_id": cs_id,
                        "file": file.display().to_string(),
                        "lines": lines,
                        "reason": "smell node not found in graph",
                    }));
                    continue;
                }
            };

            let smell_id = smell["_id"].as_str().unwrap_or("");
            let candidate_keys = candidate_doc_keys(file);

            // Try each candidate key to find a compliance edge.
            let mut found_edge = None;
            for key in &candidate_keys {
                let from_id = format!("arxiv_metadata/{key}");
                let aql = r#"
                    FOR e IN nl_smell_compliance_edges
                        FILTER e._from == @from AND e._to == @to
                        LIMIT 1
                        RETURN e
                "#;
                let bind_vars = json!({ "from": from_id, "to": smell_id });
                let result = query::query(
                    pool, aql, Some(&bind_vars), None, false, ExecutionTarget::Reader,
                )
                .await
                .map_err(|e| HandlerError::Query {
                    context: "failed to check compliance edge".into(),
                    source: e,
                })?;

                if let Some(edge) = result.results.into_iter().next() {
                    found_edge = Some(edge);
                    break;
                }
            }

            match found_edge {
                Some(edge) => {
                    verified_refs.push(json!({
                        "cs_id": cs_id,
                        "file": file.display().to_string(),
                        "lines": lines,
                        "smell_key": smell["_key"],
                        "smell_name": smell["name"],
                        "edge": {
                            "enforcement_type": edge["enforcement_type"],
                            "claim_summary": edge.get("summary").or(edge.get("claim_summary")),
                            "claiming_methods": edge.get("methods").or(edge.get("claiming_methods")),
                        },
                    }));
                }
                None => {
                    unlinked_claims.push(json!({
                        "cs_id": cs_id,
                        "file": file.display().to_string(),
                        "lines": lines,
                        "smell_key": smell["_key"],
                        "smell_name": smell["name"],
                        "reason": "no compliance edge",
                    }));
                }
            }
        }

        Ok(json!({
            "refs_found": all_refs.len(),
            "verified": verified_refs.len(),
            "missing": missing_from_graph.len(),
            "unlinked": unlinked_claims.len(),
            "verified_refs": verified_refs,
            "missing_from_graph": missing_from_graph,
            "unlinked_claims": unlinked_claims,
        }))
    }

    /// `hades smell report PATH` — combined check + verify + embedding probe.
    pub async fn smell_report(
        pool: &ArangoPool,
        config: &crate::config::HadesConfig,
        path: &str,
    ) -> Result<Value, HandlerError> {
        use crate::persephone::embedding::EmbeddingClient;

        // Run check and verify.
        let check_result = smell_check(pool, path, true).await?;
        let verify_result = smell_verify(pool, path, &[]).await?;

        // Embedding probe for each verified ref.
        let client = EmbeddingClient::connect_unix_at(&config.embedding.service.socket)
            .await
            .map_err(|e| HandlerError::ServiceError(format!("embedding connect: {e}")))?;
        let mut probes = Vec::new();

        if let Some(verified) = verify_result["verified_refs"].as_array() {
            for vref in verified {
                let cs_id = vref["cs_id"].as_str().unwrap_or("");
                let smell_name = vref["smell_name"].as_str().unwrap_or("");
                let file_path = vref["file"].as_str().unwrap_or("");

                // Read file text, truncated to 8000 chars.
                let file_text = match std::fs::read_to_string(file_path) {
                    Ok(text) => {
                        if text.chars().count() > 8000 {
                            text.chars().take(8000).collect::<String>()
                        } else {
                            text
                        }
                    }
                    Err(_) => {
                        probes.push(json!({
                            "cs_id": cs_id,
                            "smell_name": smell_name,
                            "file": file_path,
                            "error": "failed to read file",
                            "pass": null,
                        }));
                        continue;
                    }
                };

                // Embed both file text and smell name.
                match (
                    client.embed_one(&file_text, "retrieval.passage").await,
                    client.embed_one(smell_name, "retrieval.query").await,
                ) {
                    (Ok(file_emb), Ok(smell_emb)) => {
                        let sim = cosine_similarity(&file_emb, &smell_emb);
                        probes.push(json!({
                            "cs_id": cs_id,
                            "smell_name": smell_name,
                            "file": file_path,
                            "cosine_similarity": (sim * 10000.0).round() / 10000.0,
                            "pass": sim >= 0.5,
                        }));
                    }
                    (Err(e), _) | (_, Err(e)) => {
                        probes.push(json!({
                            "cs_id": cs_id,
                            "smell_name": smell_name,
                            "file": file_path,
                            "error": e.to_string(),
                            "pass": null,
                        }));
                    }
                }
            }
        }

        let static_passed = check_result["passed"].as_bool().unwrap_or(true);
        let has_unlinked = verify_result["unlinked"].as_u64().unwrap_or(0) > 0;
        let any_probe_failed = probes.iter().any(|p| p["pass"] == json!(false));
        let passed = static_passed && !has_unlinked && !any_probe_failed;

        Ok(json!({
            "path": path,
            "passed": passed,
            "has_unlinked_claims": has_unlinked,
            "static_check": {
                "passed": check_result["passed"],
                "violations": check_result["violations"],
                "violation_count": check_result["violation_count"],
                "files_checked": check_result["files_checked"],
            },
            "ref_verification": {
                "refs_found": verify_result["refs_found"],
                "verified_refs": verify_result["verified_refs"],
                "missing_from_graph": verify_result["missing_from_graph"],
                "unlinked_claims": verify_result["unlinked_claims"],
            },
            "embedding_probe": probes,
        }))
    }

    /// `hades link` — create compliance edge linking document to smell.
    pub async fn link_code_smell(
        pool: &ArangoPool,
        source_id: &str,
        smell_id: &str,
        enforcement: &str,
        methods: &[String],
        summary: Option<&str>,
    ) -> Result<Value, HandlerError> {
        use crate::db::crud;
        use crate::db::ArangoErrorKind;
        use crate::db::query::{self, ExecutionTarget};

        // Validate enforcement type.
        if !ALLOWED_ENFORCEMENT.contains(&enforcement) {
            return Err(HandlerError::InvalidParameter {
                name: "enforcement".into(),
                reason: format!(
                    "must be one of: {}",
                    ALLOWED_ENFORCEMENT.join(", "),
                ),
            });
        }

        // Verify source document exists in arxiv_metadata.
        let source_key = source_id.trim();
        crud::get_document(pool, "arxiv_metadata", source_key)
            .await
            .map_err(|e| {
                if e.is_not_found() {
                    HandlerError::DocumentNotFound {
                        collection: "arxiv_metadata".into(),
                        key: source_key.into(),
                    }
                } else {
                    HandlerError::Query {
                        context: "failed to verify source document".into(),
                        source: e,
                    }
                }
            })?;

        // Find smell node — try CS-NN prefix, then exact key.
        let smell_doc = if let Some(prefix) = smell_key_prefix(smell_id) {
            let prefix_len = prefix.len() as i64;
            let aql = r#"
                FOR s IN nl_code_smells
                    FILTER LEFT(s._key, @prefix_len) == @prefix
                    LIMIT 1
                    RETURN s
            "#;
            let bind_vars = json!({ "prefix": prefix, "prefix_len": prefix_len });
            let result = query::query(
                pool, aql, Some(&bind_vars), None, false, ExecutionTarget::Reader,
            )
            .await
            .map_err(|e| HandlerError::Query {
                context: format!("failed to find smell for {smell_id}"),
                source: e,
            })?;
            result.results.into_iter().next()
        } else {
            // Treat as exact key.
            match crud::get_document(pool, "nl_code_smells", smell_id).await {
                Ok(doc) => Some(doc),
                Err(e) if e.is_not_found() => None,
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to look up smell {smell_id}"),
                        source: e,
                    });
                }
            }
        };

        let smell_doc = smell_doc.ok_or_else(|| HandlerError::DocumentNotFound {
            collection: "nl_code_smells".into(),
            key: smell_id.into(),
        })?;

        let smell_key = smell_doc["_key"].as_str().unwrap_or(smell_id);
        let smell_name = smell_doc["name"].as_str().unwrap_or("");

        // Construct edge.
        let edge_key = format!(
            "arxiv_metadata_{source_key}__nl_code_smells_{smell_key}"
        );
        let from_id = format!("arxiv_metadata/{source_key}");
        let to_id = format!("nl_code_smells/{smell_key}");

        let mut edge_doc = json!({
            "_key": edge_key,
            "_from": from_id,
            "_to": to_id,
            "enforcement_type": enforcement,
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        if !methods.is_empty() {
            edge_doc["methods"] = json!(methods);
        }
        if let Some(s) = summary {
            edge_doc["summary"] = json!(s);
        }

        // Insert edge — treat 409 Conflict as idempotent success.
        let already_exists = match crud::insert_document(
            pool, "nl_smell_compliance_edges", &edge_doc,
        ).await {
            Ok(_) => false,
            Err(e) if e.kind() == ArangoErrorKind::Conflict => true,
            Err(e) => {
                return Err(HandlerError::Query {
                    context: "failed to insert compliance edge".into(),
                    source: e,
                });
            }
        };

        Ok(json!({
            "edge_key": edge_key,
            "from": from_id,
            "to": to_id,
            "smell": smell_name,
            "enforcement": enforcement,
            "summary": summary,
            "methods": methods,
            "already_exists": already_exists,
        }))
    }

    // ── Codebase stats ─────────────────────────────────────────────

    /// `hades codebase stats` — document counts for all codebase collections.
    pub async fn codebase_stats(pool: &ArangoPool) -> Result<Value, HandlerError> {
        use crate::db::collections::CODEBASE;
        use crate::db::crud;

        let collections = [
            ("files", CODEBASE.files),
            ("symbols", CODEBASE.symbols),
            ("chunks", CODEBASE.chunks),
            ("embeddings", CODEBASE.embeddings),
            ("defines_edges", CODEBASE.defines_edges),
            ("calls_edges", CODEBASE.calls_edges),
            ("implements_edges", CODEBASE.implements_edges),
            ("imports_edges", CODEBASE.imports_edges),
        ];

        let mut counts = serde_json::Map::new();
        for (label, name) in &collections {
            let count = match crud::count_collection(pool, name).await {
                Ok(n) => json!(n),
                Err(e) if e.is_not_found() => json!(0),
                Err(e) => {
                    return Err(HandlerError::Query {
                        context: format!("failed to count {name}"),
                        source: e,
                    });
                }
            };
            counts.insert((*label).to_string(), count);
        }

        Ok(json!({
            "command": "codebase stats",
            "collections": counts,
        }))
    }

    // ── Schema management handlers ──────────────────────────────────────

    /// Initialize the `_schema` collection with a seed.
    pub async fn schema_init(
        pool: &ArangoPool,
        seed: &str,
    ) -> Result<Value, HandlerError> {
        use crate::db::crud;
        use crate::graph::runtime_schema;

        // Validate seed name.
        let docs = match seed {
            "nl" => runtime_schema::nl_seed_documents(),
            "empty" => runtime_schema::empty_seed_documents(),
            other => {
                return Err(HandlerError::InvalidParameter {
                    name: "seed".to_string(),
                    reason: format!("unknown seed \"{other}\"; expected \"nl\" or \"empty\""),
                });
            }
        };

        // Create _schema collection if not exists.
        let collections = crud::list_collections(pool, false)
            .await
            .map_err(|e| HandlerError::Query {
                context: "list collections for schema init".into(),
                source: e,
            })?;

        if !collections.iter().any(|c| c.name == "hades_schema") {
            crud::create_collection(pool, "hades_schema", None)
                .await
                .map_err(|e| HandlerError::Query {
                    context: "create _schema collection".into(),
                    source: e,
                })?;
        }

        // Truncate existing schema so init is a clean reset.
        let truncate_path = "collection/hades_schema/truncate";
        let _ = pool.writer().put(truncate_path, &json!({}))
            .await; // Ignore errors (collection may be empty/new).

        // Insert seed documents.
        let result = crud::upsert_documents(pool, "hades_schema", &docs)
            .await
            .map_err(|e| HandlerError::Query {
                context: "seed _schema documents".into(),
                source: e,
            })?;

        Ok(json!({
            "command": "db.schema.init",
            "seed": seed,
            "documents_written": result.created + result.updated,
            "errors": result.errors,
        }))
    }

    /// List all edge definitions and named graphs in `_schema`.
    pub async fn schema_list(
        pool: &ArangoPool,
    ) -> Result<Value, HandlerError> {
        use crate::graph::runtime_schema::RuntimeSchema;

        let schema = RuntimeSchema::load(pool).await.map_err(|e| {
            HandlerError::Query {
                context: "load runtime schema".into(),
                source: crate::db::ArangoError::Request(e.to_string()),
            }
        })?;

        let edge_defs: Vec<Value> = schema
            .edge_definitions
            .iter()
            .map(|e| {
                json!({
                    "name": e.name,
                    "source_field": e.source_field,
                    "strategy": e.materialize_strategy,
                    "is_array": e.is_array,
                    "from_count": e.from_collections.len(),
                    "to_count": e.to_collections.len(),
                })
            })
            .collect();

        let graphs: Vec<Value> = schema
            .named_graphs
            .iter()
            .map(|g| {
                json!({
                    "name": g.name,
                    "edge_count": g.edge_definitions.len(),
                    "description": g.description,
                })
            })
            .collect();

        Ok(json!({
            "command": "db.schema.list",
            "edge_definitions": edge_defs,
            "named_graphs": graphs,
            "source": if schema.from_database { "database" } else { "fallback" },
        }))
    }

    /// Show a single edge definition or named graph by name.
    pub async fn schema_show(
        pool: &ArangoPool,
        name: &str,
    ) -> Result<Value, HandlerError> {
        use crate::graph::runtime_schema::RuntimeSchema;

        let schema = RuntimeSchema::load(pool).await.map_err(|e| {
            HandlerError::Query {
                context: "load runtime schema".into(),
                source: crate::db::ArangoError::Request(e.to_string()),
            }
        })?;

        // Try edge definition first, then named graph.
        if let Some(edef) = schema.get_edge_def(name) {
            return Ok(json!({
                "command": "db.schema.show",
                "type": "edge_definition",
                "definition": edef,
            }));
        }

        if let Some(ng) = schema.get_named_graph(name) {
            return Ok(json!({
                "command": "db.schema.show",
                "type": "named_graph",
                "definition": ng,
            }));
        }

        Err(HandlerError::InvalidParameter {
            name: "name".to_string(),
            reason: format!("no edge definition or named graph \"{name}\" found in schema"),
        })
    }

    /// Show schema version and checksum.
    pub async fn schema_version(
        pool: &ArangoPool,
    ) -> Result<Value, HandlerError> {
        use crate::graph::runtime_schema::RuntimeSchema;

        let schema = RuntimeSchema::load(pool).await.map_err(|e| {
            HandlerError::Query {
                context: "load runtime schema".into(),
                source: crate::db::ArangoError::Request(e.to_string()),
            }
        })?;

        Ok(json!({
            "command": "db.schema.version",
            "schema_version": schema.meta.schema_version,
            "schema_checksum": schema.meta.schema_checksum,
            "seed_name": schema.meta.seed_name,
            "num_relations": schema.meta.num_relations,
            "feature_dim": schema.meta.feature_dim,
        }))
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

    #[test]
    fn test_write_denied_on_production_db_link_code_smell() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = HadesConfig::default(); // NestedLearning — production
        let pool = ArangoPool::from_config(&config).unwrap();

        let result = rt.block_on(dispatch(
            &pool,
            &config,
            DaemonCommand::LinkCodeSmell(LinkCodeSmellParams {
                source_id: "test-doc".into(),
                smell_id: "CS-32".into(),
                enforcement: "static".into(),
                methods: Vec::new(),
                summary: None,
            }),
        ));

        assert!(matches!(
            result.unwrap_err(),
            DispatchError::Handler(HandlerError::WriteDenied(_))
        ));
    }

    // ── Task command tests ─────────────────────────────────────────

    #[test]
    fn test_command_roundtrip_task_list_defaults() {
        let json = serde_json::json!({
            "command": "task.list",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskList(ref p) if p.status.is_none() && p.limit.is_none()
        ));
    }

    #[test]
    fn test_command_roundtrip_task_list_filtered() {
        let json = serde_json::json!({
            "command": "task.list",
            "params": { "status": "open", "type": "bug", "limit": 20 }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskList(ref p)
                if p.status.as_deref() == Some("open")
                && p.task_type.as_deref() == Some("bug")
                && p.limit == Some(20)
        ));
    }

    #[test]
    fn test_command_roundtrip_task_show() {
        let json = serde_json::json!({
            "command": "task.show",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskShow(ref p) if p.key == "task_abc123"
        ));
    }

    #[test]
    fn test_command_roundtrip_task_create_defaults() {
        let json = serde_json::json!({
            "command": "task.create",
            "params": { "title": "Test task" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskCreate(ref p)
                if p.title == "Test task"
                && p.task_type == "task"
                && p.priority == "medium"
                && p.tags.is_empty()
        ));
    }

    #[test]
    fn test_command_roundtrip_task_update_tags() {
        let json = serde_json::json!({
            "command": "task.update",
            "params": {
                "key": "task_abc123",
                "add_tags": ["phase-7"],
                "remove_tags": ["draft"]
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskUpdate(ref p)
                if p.key == "task_abc123"
                && p.add_tags == vec!["phase-7"]
                && p.remove_tags == vec!["draft"]
        ));
    }

    #[test]
    fn test_command_roundtrip_task_close() {
        let json = serde_json::json!({
            "command": "task.close",
            "params": { "key": "task_abc123", "message": "Done" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskClose(ref p)
                if p.key == "task_abc123" && p.message.as_deref() == Some("Done")
        ));
    }

    #[test]
    fn test_command_roundtrip_task_start() {
        let json = serde_json::json!({
            "command": "task.start",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskStart(ref p) if p.key == "task_abc123"
        ));
    }

    #[test]
    fn test_command_roundtrip_task_context() {
        let json = serde_json::json!({
            "command": "task.context",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskContext(ref p) if p.key == "task_abc123"
        ));
    }

    #[test]
    fn test_deny_unknown_task_create() {
        let json = serde_json::json!({
            "command": "task.create",
            "params": { "title": "Test", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_task_start() {
        let json = serde_json::json!({
            "command": "task.start",
            "params": { "key": "task_abc123", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_task_list() {
        let json = serde_json::json!({
            "command": "task.list",
            "params": { "status": "open", "extra_field": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    // ── P7.5b: Workflow command roundtrip tests ─────────────────────

    #[test]
    fn test_command_roundtrip_task_review() {
        let json = serde_json::json!({
            "command": "task.review",
            "params": { "key": "task_abc123", "message": "ready for review" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskReview(ref p) if p.key == "task_abc123"
                && p.message.as_deref() == Some("ready for review")
        ));

        // Without message.
        let json2 = serde_json::json!({
            "command": "task.review",
            "params": { "key": "task_abc123" }
        });
        let cmd2: DaemonCommand = serde_json::from_value(json2).unwrap();
        assert!(matches!(
            cmd2,
            DaemonCommand::TaskReview(ref p) if p.message.is_none()
        ));
    }

    #[test]
    fn test_command_roundtrip_task_approve() {
        let json = serde_json::json!({
            "command": "task.approve",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskApprove(ref p) if p.key == "task_abc123" && !p.human
        ));

        // With human override.
        let json2 = serde_json::json!({
            "command": "task.approve",
            "params": { "key": "task_abc123", "human": true }
        });
        let cmd2: DaemonCommand = serde_json::from_value(json2).unwrap();
        assert!(matches!(
            cmd2,
            DaemonCommand::TaskApprove(ref p) if p.human
        ));
    }

    #[test]
    fn test_command_roundtrip_task_block() {
        let json = serde_json::json!({
            "command": "task.block",
            "params": {
                "key": "task_abc123",
                "message": "blocked by infra",
                "blocker": "task_def456"
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskBlock(ref p) if p.key == "task_abc123"
                && p.message.as_deref() == Some("blocked by infra")
                && p.blocker.as_deref() == Some("task_def456")
        ));

        // Without blocker.
        let json2 = serde_json::json!({
            "command": "task.block",
            "params": { "key": "task_abc123", "message": "waiting" }
        });
        let cmd2: DaemonCommand = serde_json::from_value(json2).unwrap();
        assert!(matches!(
            cmd2,
            DaemonCommand::TaskBlock(ref p) if p.blocker.is_none()
        ));
    }

    #[test]
    fn test_command_roundtrip_task_unblock() {
        let json = serde_json::json!({
            "command": "task.unblock",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskUnblock(ref p) if p.key == "task_abc123"
        ));
    }

    #[test]
    fn test_command_roundtrip_task_handoff() {
        let json = serde_json::json!({
            "command": "task.handoff",
            "params": { "key": "task_abc123", "message": "context for next session" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskHandoff(ref p) if p.key == "task_abc123"
                && p.message.as_deref() == Some("context for next session")
        ));
    }

    #[test]
    fn test_command_roundtrip_task_handoff_show() {
        let json = serde_json::json!({
            "command": "task.handoff_show",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskHandoffShow(ref p) if p.key == "task_abc123"
        ));
    }

    #[test]
    fn test_deny_unknown_task_review() {
        let json = serde_json::json!({
            "command": "task.review",
            "params": { "key": "task_abc123", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_task_handoff() {
        let json = serde_json::json!({
            "command": "task.handoff",
            "params": { "key": "task_abc123", "message": "hi", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_valid_transitions_contains_review() {
        use super::handlers::VALID_TRANSITIONS;
        assert!(VALID_TRANSITIONS.contains(&("in_progress", "in_review")));
    }

    #[test]
    fn test_valid_transitions_rejects_open_to_in_review() {
        use super::handlers::VALID_TRANSITIONS;
        assert!(!VALID_TRANSITIONS.contains(&("open", "in_review")));
    }

    #[test]
    fn test_valid_transitions_contains_block() {
        use super::handlers::VALID_TRANSITIONS;
        assert!(VALID_TRANSITIONS.contains(&("in_progress", "blocked")));
    }

    #[test]
    fn test_valid_transitions_rejects_closed_to_blocked() {
        use super::handlers::VALID_TRANSITIONS;
        assert!(!VALID_TRANSITIONS.contains(&("closed", "blocked")));
    }

    // ── P7.5c: Query/meta command roundtrip tests ───────────────────

    #[test]
    fn test_command_roundtrip_task_log() {
        let json = serde_json::json!({
            "command": "task.log",
            "params": { "key": "task_abc123", "limit": 10 }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskLog(ref p) if p.key == "task_abc123" && p.limit == Some(10)
        ));

        // Without limit (defaults to None).
        let json2 = serde_json::json!({
            "command": "task.log",
            "params": { "key": "task_abc123" }
        });
        let cmd2: DaemonCommand = serde_json::from_value(json2).unwrap();
        assert!(matches!(
            cmd2,
            DaemonCommand::TaskLog(ref p) if p.limit.is_none()
        ));
    }

    #[test]
    fn test_command_roundtrip_task_sessions() {
        let json = serde_json::json!({
            "command": "task.sessions",
            "params": { "key": "task_abc123" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskSessions(ref p) if p.key == "task_abc123"
        ));
    }

    #[test]
    fn test_command_roundtrip_task_dep_add() {
        let json = serde_json::json!({
            "command": "task.dep",
            "params": { "key": "task_abc123", "add": "task_def456" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskDep(ref p) if p.key == "task_abc123"
                && p.add.as_deref() == Some("task_def456")
                && p.remove.is_none()
                && !p.graph
        ));
    }

    #[test]
    fn test_command_roundtrip_task_dep_remove() {
        let json = serde_json::json!({
            "command": "task.dep",
            "params": { "key": "task_abc123", "remove": "task_def456" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskDep(ref p) if p.remove.as_deref() == Some("task_def456")
        ));
    }

    #[test]
    fn test_command_roundtrip_task_dep_show_graph() {
        let json = serde_json::json!({
            "command": "task.dep",
            "params": { "key": "task_abc123", "graph": true }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(
            cmd,
            DaemonCommand::TaskDep(ref p) if p.add.is_none() && p.remove.is_none() && p.graph
        ));
    }

    #[test]
    fn test_command_roundtrip_task_usage() {
        let json = serde_json::json!({
            "command": "task.usage",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::TaskUsage(_)));
    }

    #[test]
    fn test_command_roundtrip_task_graph_integration() {
        let json = serde_json::json!({
            "command": "task.graph_integration",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::TaskGraphIntegration(_)));
    }

    #[test]
    fn test_deny_unknown_task_log() {
        let json = serde_json::json!({
            "command": "task.log",
            "params": { "key": "task_abc123", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    #[test]
    fn test_deny_unknown_task_dep() {
        let json = serde_json::json!({
            "command": "task.dep",
            "params": { "key": "task_abc123", "extra": true }
        });
        assert!(serde_json::from_value::<DaemonCommand>(json).is_err());
    }

    // ── Embedding command tests ───────────────────────────────────────

    #[test]
    fn test_command_roundtrip_embed_text() {
        let json = serde_json::json!({
            "command": "embed.text",
            "params": { "text": "hello world" }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        match cmd {
            DaemonCommand::EmbedText { ref text } => assert_eq!(text, "hello world"),
            other => panic!("expected EmbedText, got {other:?}"),
        }
        // Round-trip
        let serialized = serde_json::to_value(&cmd).unwrap();
        let _: DaemonCommand = serde_json::from_value(serialized).unwrap();
    }

    #[test]
    fn test_command_roundtrip_smell_check() {
        let json = serde_json::json!({
            "command": "smell.check",
            "params": { "path": "/tmp/test", "verbose": true }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        match cmd {
            DaemonCommand::SmellCheck(ref params) => {
                assert_eq!(params.path, "/tmp/test");
                assert!(params.verbose);
            }
            other => panic!("expected SmellCheck, got {other:?}"),
        }
        let serialized = serde_json::to_value(&cmd).unwrap();
        let _: DaemonCommand = serde_json::from_value(serialized).unwrap();
    }

    #[test]
    fn test_command_roundtrip_smell_verify() {
        let json = serde_json::json!({
            "command": "smell.verify",
            "params": { "path": "/tmp/test", "claims": ["CS-32", "CS-10"] }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        match cmd {
            DaemonCommand::SmellVerify(ref params) => {
                assert_eq!(params.path, "/tmp/test");
                assert_eq!(params.claims, &["CS-32", "CS-10"]);
            }
            other => panic!("expected SmellVerify, got {other:?}"),
        }
        let serialized = serde_json::to_value(&cmd).unwrap();
        let _: DaemonCommand = serde_json::from_value(serialized).unwrap();
    }

    #[test]
    fn test_command_roundtrip_link_code_smell() {
        let json = serde_json::json!({
            "command": "link_code_smell",
            "params": {
                "source_id": "conductor-rs",
                "smell_id": "CS-32",
                "enforcement": "static",
                "methods": ["method_a"],
                "summary": "test summary",
            }
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        match cmd {
            DaemonCommand::LinkCodeSmell(ref params) => {
                assert_eq!(params.source_id, "conductor-rs");
                assert_eq!(params.smell_id, "CS-32");
                assert_eq!(params.enforcement, "static");
                assert_eq!(params.methods, &["method_a"]);
                assert_eq!(params.summary.as_deref(), Some("test summary"));
            }
            other => panic!("expected LinkCodeSmell, got {other:?}"),
        }
        let serialized = serde_json::to_value(&cmd).unwrap();
        let _: DaemonCommand = serde_json::from_value(serialized).unwrap();
    }

    #[test]
    fn test_command_roundtrip_codebase_stats() {
        let json = serde_json::json!({
            "command": "codebase.stats",
            "params": {}
        });
        let cmd: DaemonCommand = serde_json::from_value(json).unwrap();
        assert!(matches!(cmd, DaemonCommand::CodebaseStats(_)));
        let serialized = serde_json::to_value(&cmd).unwrap();
        let _: DaemonCommand = serde_json::from_value(serialized).unwrap();
    }

    #[test]
    fn test_smell_tier_classification() {
        use super::handlers;
        assert_eq!(handlers::smell_tier(Some(10)), "static");
        assert_eq!(handlers::smell_tier(Some(11)), "static");
        assert_eq!(handlers::smell_tier(Some(13)), "static");
        assert_eq!(handlers::smell_tier(Some(40)), "static");
        assert_eq!(handlers::smell_tier(Some(27)), "behavioral");
        assert_eq!(handlers::smell_tier(Some(28)), "behavioral");
        assert_eq!(handlers::smell_tier(Some(32)), "behavioral");
        assert_eq!(handlers::smell_tier(Some(31)), "architectural");
        assert_eq!(handlers::smell_tier(Some(99)), "unknown");
        assert_eq!(handlers::smell_tier(None), "unknown");
    }

    #[test]
    fn test_smell_key_prefix() {
        use super::handlers;
        assert_eq!(handlers::smell_key_prefix("CS-32"), Some("smell-032-".into()));
        assert_eq!(handlers::smell_key_prefix("CS-1"), Some("smell-001-".into()));
        assert_eq!(handlers::smell_key_prefix("CS-100"), Some("smell-100-".into()));
        assert_eq!(handlers::smell_key_prefix("cs-10"), Some("smell-010-".into()));
        assert_eq!(handlers::smell_key_prefix("not-a-smell"), None);
        assert_eq!(handlers::smell_key_prefix("CS-"), None);
        assert_eq!(handlers::smell_key_prefix(""), None);
        // Overflow: u32::MAX + 1 should return None, not "smell-000-"
        assert_eq!(handlers::smell_key_prefix("CS-99999999999"), None);
    }

    #[test]
    fn test_cosine_similarity() {
        use super::handlers;
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        assert!((handlers::cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        // Orthogonal
        let b = vec![0.0, 1.0, 0.0];
        assert!(handlers::cosine_similarity(&a, &b).abs() < 1e-6);
        // Zero vector
        let z = vec![0.0, 0.0, 0.0];
        assert!(handlers::cosine_similarity(&a, &z).abs() < 1e-6);
    }

    #[test]
    fn test_is_comment_line() {
        use super::handlers;
        assert!(handlers::is_comment_line("  # comment", ".py"));
        assert!(handlers::is_comment_line("  // comment", ".rs"));
        assert!(handlers::is_comment_line("  /// doc comment", ".rs"));
        assert!(handlers::is_comment_line("  //! module doc", ".rs"));
        assert!(!handlers::is_comment_line("  let x = 1;", ".rs"));
        assert!(!handlers::is_comment_line("  x = 1", ".py"));
        assert!(handlers::is_comment_line("  // comment", ".ts"));
        assert!(handlers::is_comment_line("  # comment", ".yaml"));
    }

    #[test]
    fn test_candidate_doc_keys() {
        use super::handlers;
        use std::path::Path;
        let keys = handlers::candidate_doc_keys(Path::new("conductor.rs"));
        assert!(keys.contains(&"conductor".into()));
        assert!(keys.contains(&"conductor-rs".into()));

        let keys2 = handlers::candidate_doc_keys(Path::new("my_module.py"));
        assert!(keys2.contains(&"my_module".into()));
        assert!(keys2.contains(&"my_module-py".into()));
        assert!(keys2.contains(&"my-module".into()));
        assert!(keys2.contains(&"my-module-py".into()));
    }

    #[test]
    fn test_access_tier_agent_safe() {
        // Spot-check representative agent-safe commands.
        let cmd = DaemonCommand::Orient(OrientParams { collection: None });
        assert_eq!(cmd.access_tier(), AccessTier::Agent);
        assert!(cmd.is_agent_safe());

        let cmd = DaemonCommand::DbGet(DbGetParams {
            collection: "test".into(),
            key: "k".into(),
        });
        assert_eq!(cmd.access_tier(), AccessTier::Agent);

        let cmd = DaemonCommand::DbGraphTraverse(DbGraphTraverseParams {
            start: "c/k".into(),
            direction: "outbound".into(),
            min_depth: 1,
            max_depth: 1,
            limit: None,
            graph: None,
        });
        assert_eq!(cmd.access_tier(), AccessTier::Agent);

        let cmd = DaemonCommand::TaskCreate(TaskCreateParams {
            title: "test".into(),
            description: None,
            task_type: "task".into(),
            parent: None,
            priority: "medium".into(),
            tags: vec![],
        });
        assert_eq!(cmd.access_tier(), AccessTier::Agent);
    }

    #[test]
    fn test_access_tier_admin() {
        let cmd = DaemonCommand::DbAql(DbAqlParams {
            aql: "RETURN 1".into(),
            bind: None,
            limit: None,
        });
        assert_eq!(cmd.access_tier(), AccessTier::Admin);
        assert!(!cmd.is_agent_safe());

        let cmd = DaemonCommand::DbInsert(DbInsertParams {
            collection: "test".into(),
            data: serde_json::json!({}),
        });
        assert_eq!(cmd.access_tier(), AccessTier::Admin);

        let cmd = DaemonCommand::DbGraphDrop(DbGraphDropParams {
            name: "g".into(),
            drop_collections: false,
            force: false,
        });
        assert_eq!(cmd.access_tier(), AccessTier::Admin);
    }

    #[test]
    fn test_access_tier_internal() {
        let cmd = DaemonCommand::Status(StatusParams { verbose: false });
        assert_eq!(cmd.access_tier(), AccessTier::Internal);

        let cmd = DaemonCommand::DbHealth(DbHealthParams { verbose: false });
        assert_eq!(cmd.access_tier(), AccessTier::Internal);

        let cmd = DaemonCommand::DbStats {};
        assert_eq!(cmd.access_tier(), AccessTier::Internal);
    }
}
