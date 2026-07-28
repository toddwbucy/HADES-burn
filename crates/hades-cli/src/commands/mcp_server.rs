//! MCP streamable-HTTP endpoint for the trusted LAN (#155).
//!
//! Serves a curated agent-tier tool surface over the Model Context
//! Protocol so remote agent sessions consume HADES without a local
//! binary. Every tool call is serialized to the daemon frame format and
//! routed through [`hades_core::service::handle_request`] under
//! [`ConnectionPolicy::agent_only`] — the identical parse/authorize/
//! dispatch path the Unix socket uses, with the network transport's tier
//! ceiling. No tool here can reach an Admin-tier command even if one
//! were mounted by mistake.
//!
//! Security boundary (per #155): bearer-token auth on `/mcp`, explicit
//! bind to a loopback or RFC1918 address only, fail-closed startup when
//! the token file is missing or empty. ArangoDB ACLs on the `hades` user
//! remain the authoritative per-database write gate.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{StatusCode, header::AUTHORIZATION};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ErrorData as McpError, ServerHandler, tool, tool_handler, tool_router};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{
    DaemonCommand, DbCountParams, DbGetParams, DbGraphNeighborsParams, DbGraphTraverseParams,
    DbListParams, OrientParams, SmellReportParams, TaskCreateParams, TaskListParams,
    TaskShowParams, TaskUpdateParams,
};
use hades_core::service::{self, ConnectionPolicy};

/// Per-request execution timeout — matches the Unix daemon transport.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Maximum request body size (16 MiB, matching the daemon protocol limit).
const MAX_BODY: usize = 16 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Bearer tokens
// ---------------------------------------------------------------------------

/// The set of bearer tokens accepted on the MCP endpoint.
#[derive(Debug, Clone)]
pub struct TokenSet {
    tokens: Vec<String>,
}

impl TokenSet {
    /// Load tokens from a file: one token per line, blank lines and
    /// `#`-comments ignored. Fails closed on a missing or empty file —
    /// the endpoint must never start unauthenticated.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read MCP token file {}", path.display()))?;
        let tokens: Vec<String> = raw
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect();
        anyhow::ensure!(
            !tokens.is_empty(),
            "MCP token file {} contains no tokens — refusing to serve unauthenticated",
            path.display()
        );
        Ok(Self { tokens })
    }

    #[cfg(test)]
    fn from_tokens(tokens: Vec<String>) -> Self {
        Self { tokens }
    }

    /// Check a candidate token, comparing each stored token in constant
    /// time so a match position is not observable through timing.
    pub fn contains(&self, candidate: &str) -> bool {
        let mut found = false;
        for token in &self.tokens {
            found |= ct_eq(token.as_bytes(), candidate.as_bytes());
        }
        found
    }
}

/// Constant-time byte-slice equality. Length differences return early —
/// token lengths are not secret, their contents are.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Axum middleware: require a valid `Authorization: Bearer <token>` header.
async fn require_bearer(
    State(tokens): State<Arc<TokenSet>>,
    request: Request,
    next: Next,
) -> Response {
    let authorized = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| tokens.contains(t))
        .unwrap_or(false);

    if authorized {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [("www-authenticate", "Bearer")],
            "missing or invalid bearer token\n",
        )
            .into_response()
    }
}

// ---------------------------------------------------------------------------
// Bind policy
// ---------------------------------------------------------------------------

/// Refuse to serve on anything but a loopback, RFC1918, or IPv6
/// unique-local (fc00::/7) address. The LAN endpoint is plain HTTP with
/// bearer tokens — acceptable only inside the explicitly trusted
/// network, so a public or wildcard bind is a configuration error, not
/// a choice.
pub fn ensure_private_bind(addr: &SocketAddr) -> Result<()> {
    let ip = addr.ip();
    let ok = match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback() || v4.is_private(),
        // `Ipv6Addr::is_unique_local` is unstable in std; check fc00::/7
        // directly.
        std::net::IpAddr::V6(v6) => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00,
    };
    anyhow::ensure!(
        ok,
        "refusing to bind MCP endpoint to {ip}: bind a loopback, RFC1918, or IPv6 \
         unique-local address explicitly (wildcard and public addresses are not \
         served — see #155 security boundary)"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-database pool cache
// ---------------------------------------------------------------------------

/// One database's dispatch context: the config and pool handed to
/// `service::handle_request` for requests targeting that database.
type DbEntry = (Arc<HadesConfig>, Arc<ArangoPool>);

/// Lazily-built (config, pool) pairs per database name, scoped by an
/// explicit allowlist.
///
/// The Unix daemon serves its one configured database; the MCP tools
/// take an optional `db` so one endpoint can serve both the knowledge
/// graph and the task board. The allowlist defaults to the configured
/// database alone — parity with the Unix socket — and widens only via
/// `--mcp-dbs`. Without it, an authenticated LAN agent could read every
/// database the `hades` ArangoDB user can reach, including production
/// research databases that are ro-readable by design. ArangoDB ACLs
/// remain the write gate; the allowlist scopes remote *reads*.
struct PoolCache {
    base_config: HadesConfig,
    default_db: String,
    allowed: std::collections::HashSet<String>,
    pools: RwLock<HashMap<String, DbEntry>>,
}

impl PoolCache {
    fn new(base_config: HadesConfig, extra_dbs: &[String]) -> Result<Self> {
        let default_db = base_config
            .database
            .name
            .clone()
            .context("MCP endpoint requires a configured default database")?;
        let mut allowed: std::collections::HashSet<String> = extra_dbs.iter().cloned().collect();
        allowed.insert(default_db.clone());
        for name in &allowed {
            anyhow::ensure!(
                is_valid_db_name(name),
                "invalid database name '{name}' in MCP allowlist: \
                 expected [A-Za-z0-9_-], max 64 chars"
            );
        }
        Ok(Self {
            base_config,
            default_db,
            allowed,
            pools: RwLock::new(HashMap::new()),
        })
    }

    async fn entry_for(&self, db: Option<&str>) -> Result<DbEntry, String> {
        let name = db.unwrap_or(&self.default_db);
        if !is_valid_db_name(name) {
            return Err(format!(
                "invalid database name '{name}': expected [A-Za-z0-9_-], max 64 chars"
            ));
        }
        if !self.allowed.contains(name) {
            let mut served: Vec<&str> = self.allowed.iter().map(String::as_str).collect();
            served.sort_unstable();
            return Err(format!(
                "database '{name}' is not served by this endpoint (served: {}); \
                 add it to --mcp-dbs on the daemon to expose it",
                served.join(", ")
            ));
        }
        if let Some(entry) = self.pools.read().await.get(name) {
            return Ok(entry.clone());
        }
        // Re-check under the write lock so concurrent first-hits for the
        // same database build one pool, not two.
        let mut pools = self.pools.write().await;
        if let Some(entry) = pools.get(name) {
            return Ok(entry.clone());
        }
        let mut config = self.base_config.clone();
        config.apply_cli_overrides(Some(name), None);
        let pool = ArangoPool::from_config(&config)
            .map_err(|e| format!("failed to open database '{name}': {e}"))?;
        let entry = (Arc::new(config), Arc::new(pool));
        pools.insert(name.to_string(), entry.clone());
        Ok(entry)
    }
}

fn is_valid_db_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

// ---------------------------------------------------------------------------
// Tool argument schemas
// ---------------------------------------------------------------------------
//
// Thin schemars-typed mirrors of the dispatch param structs. Kept separate
// so hades-core stays schema-free and so each description is written for
// the consuming agent, not the wire format.

/// Shared doc text for the optional `db` field.
macro_rules! db_field_doc {
    () => {
        "Database to run against (default: the endpoint's configured database)"
    };
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct OrientArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Optional collection to orient on; omit for a whole-graph overview")]
    collection: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct DbQueryArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Natural-language query text to search for")]
    text: String,
    #[schemars(
        description = "Collection *profile* to search (`default`, `codebase`) — not an \
                       arbitrary collection name. Omit for `default`."
    )]
    collection: Option<String>,
    #[schemars(description = "Maximum results to return (default 10)")]
    limit: Option<u32>,
    #[schemars(
        description = "Blend vector similarity with keyword term-coverage (the \
                       fraction of distinct query terms present; not BM25 — no term \
                       frequency, IDF, or length normalization)"
    )]
    hybrid: Option<bool>,
    // Accepted but not advertised. The cross-encoder it needs does not ship, so
    // the shared handler rejects it and putting it in the schema would hand an
    // agent a schema-legal argument that always errors. Dropping it silently is
    // worse still: a client written against `docs/daemon-protocol.md`, or one
    // holding a schema cached across a daemon upgrade, would send `rerank: true`
    // and get `success` with results it believes were reranked. Passing it
    // through lets the handler answer with its own "use hybrid and/or
    // structural instead" guidance, which is what the CLI does at the argument
    // boundary.
    #[schemars(skip)]
    rerank: Option<bool>,
    #[schemars(description = "Blend in structural (graph-topology) embedding similarity")]
    structural: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct DbGetArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Collection holding the document")]
    collection: String,
    #[schemars(description = "Document _key to fetch")]
    key: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct DbListArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Collection to list; omit to list collections instead of documents")]
    collection: Option<String>,
    #[schemars(description = "Maximum documents to return")]
    limit: Option<u32>,
    #[schemars(description = "Filter by source paper/document identifier")]
    paper: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct DbCountArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Collection to count documents in")]
    collection: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct GraphNeighborsArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Start vertex as collection/_key, e.g. 'wt_assertions/trace-01'")]
    vertex: String,
    #[schemars(description = "Edge direction: 'any' (default), 'inbound', or 'outbound'")]
    direction: Option<String>,
    #[schemars(description = "Maximum neighbors to return")]
    limit: Option<u32>,
    #[schemars(description = "Named graph to traverse; omit for the default graph")]
    graph: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct GraphTraverseArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Start vertex as collection/_key")]
    start: String,
    #[schemars(description = "Edge direction: 'outbound' (default), 'inbound', or 'any'")]
    direction: Option<String>,
    #[schemars(description = "Minimum traversal depth (default 1)")]
    min_depth: Option<u32>,
    #[schemars(description = "Maximum traversal depth (default 1)")]
    max_depth: Option<u32>,
    #[schemars(description = "Maximum vertices to return")]
    limit: Option<u32>,
    #[schemars(description = "Named graph to traverse; omit for the default graph")]
    graph: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct SmellReportArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(
        description = "Repo-root-relative file path (or file key) to report code smells for"
    )]
    path: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct TaskListArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Filter by status, e.g. 'open', 'in_progress', 'done'")]
    status: Option<String>,
    #[schemars(description = "Filter by task type")]
    r#type: Option<String>,
    #[schemars(description = "Filter to children of this parent task key")]
    parent: Option<String>,
    #[schemars(description = "Maximum tasks to return")]
    limit: Option<u32>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct TaskShowArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Task _key to show")]
    key: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct TaskCreateArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Short task title")]
    title: String,
    #[schemars(description = "Longer task description")]
    description: Option<String>,
    #[schemars(description = "Task type (default 'task')")]
    r#type: Option<String>,
    #[schemars(description = "Parent task key for subtasks")]
    parent: Option<String>,
    #[schemars(description = "Priority: 'low', 'medium' (default), 'high', 'critical'")]
    priority: Option<String>,
    #[schemars(description = "Tags to attach")]
    tags: Option<Vec<String>>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct TaskUpdateArgs {
    #[schemars(description = db_field_doc!())]
    db: Option<String>,
    #[schemars(description = "Task _key to update")]
    key: String,
    #[schemars(description = "New title")]
    title: Option<String>,
    #[schemars(description = "New description")]
    description: Option<String>,
    #[schemars(description = "New priority")]
    priority: Option<String>,
    #[schemars(description = "New status")]
    status: Option<String>,
    #[schemars(description = "Tags to add")]
    add_tags: Option<Vec<String>>,
    #[schemars(description = "Tags to remove")]
    remove_tags: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// The MCP server
// ---------------------------------------------------------------------------

/// The HADES MCP tool server: a curated agent-tier surface over the
/// shared daemon service layer.
pub struct HadesMcpServer {
    pools: Arc<PoolCache>,
    tool_router: ToolRouter<Self>,
}

impl HadesMcpServer {
    fn new(pools: Arc<PoolCache>) -> Self {
        Self {
            pools,
            tool_router: Self::tool_router(),
        }
    }

    /// Route a command through the shared service layer under the
    /// network policy and wrap the response envelope as a tool result.
    async fn run(
        &self,
        db: Option<String>,
        cmd: DaemonCommand,
    ) -> Result<CallToolResult, McpError> {
        let (config, pool) = match self.pools.entry_for(db.as_deref()).await {
            Ok(entry) => entry,
            Err(msg) => return Ok(CallToolResult::error(vec![ContentBlock::text(msg)])),
        };
        let payload =
            serde_json::to_vec(&cmd).map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let resp = service::handle_request(
            &pool,
            &config,
            ConnectionPolicy::agent_only(),
            &payload,
            REQUEST_TIMEOUT,
        )
        .await;
        let success = resp.success;
        let text = serde_json::to_string(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        if success {
            Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
        } else {
            Ok(CallToolResult::error(vec![ContentBlock::text(text)]))
        }
    }
}

#[tool_router]
impl HadesMcpServer {
    #[tool(
        description = "Orient yourself in a HADES knowledge graph: collections, counts, and schema hints. Call this first in a new session."
    )]
    async fn orient(
        &self,
        Parameters(a): Parameters<OrientArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::Orient(OrientParams {
                collection: a.collection,
            }),
        )
        .await
    }

    #[tool(
        description = "Semantic search over a HADES knowledge graph. Embeds the query text and returns the closest chunks/nodes, optionally blended with keyword term-coverage (not BM25) or structural graph similarity. `collection` names a profile (default, codebase), not an arbitrary collection."
    )]
    async fn db_query(
        &self,
        Parameters(a): Parameters<DbQueryArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::DbQuery {
                text: a.text,
                limit: a.limit,
                collection: a.collection,
                hybrid: a.hybrid.unwrap_or(false),
                // Forwarded, not forced: an unadvertised `rerank: true` must
                // reach the handler's guard and be refused out loud.
                rerank: a.rerank.unwrap_or(false),
                structural: a.structural.unwrap_or(false),
            },
        )
        .await
    }

    #[tool(description = "Fetch a single document by collection and _key.")]
    async fn db_get(
        &self,
        Parameters(a): Parameters<DbGetArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::DbGet(DbGetParams {
                collection: a.collection,
                key: a.key,
            }),
        )
        .await
    }

    #[tool(
        description = "List documents in a collection (bounded), or list all collections when no collection is given."
    )]
    async fn db_list(
        &self,
        Parameters(a): Parameters<DbListArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::DbList(DbListParams {
                collection: a.collection,
                limit: a.limit,
                paper: a.paper,
            }),
        )
        .await
    }

    #[tool(description = "Count documents in a collection.")]
    async fn db_count(
        &self,
        Parameters(a): Parameters<DbCountArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::DbCount(DbCountParams {
                collection: a.collection,
            }),
        )
        .await
    }

    #[tool(
        description = "List direct graph neighbors of a vertex (edges in, out, or both), with the connecting edges."
    )]
    async fn graph_neighbors(
        &self,
        Parameters(a): Parameters<GraphNeighborsArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::DbGraphNeighbors(DbGraphNeighborsParams {
                vertex: a.vertex,
                direction: a.direction.unwrap_or_else(|| "any".to_string()),
                limit: a.limit,
                graph: a.graph,
            }),
        )
        .await
    }

    #[tool(
        description = "Traverse the graph from a start vertex between min and max depth, returning reached vertices and paths."
    )]
    async fn graph_traverse(
        &self,
        Parameters(a): Parameters<GraphTraverseArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::DbGraphTraverse(DbGraphTraverseParams {
                start: a.start,
                direction: a.direction.unwrap_or_else(|| "outbound".to_string()),
                min_depth: a.min_depth.unwrap_or(1),
                max_depth: a.max_depth.unwrap_or(1),
                limit: a.limit,
                graph: a.graph,
            }),
        )
        .await
    }

    #[tool(description = "Report recorded code smells for a file in the codebase graph.")]
    async fn smell_report(
        &self,
        Parameters(a): Parameters<SmellReportArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::SmellReport(SmellReportParams { path: a.path }),
        )
        .await
    }

    #[tool(description = "List tasks on the kanban board, filterable by status, type, or parent.")]
    async fn task_list(
        &self,
        Parameters(a): Parameters<TaskListArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::TaskList(TaskListParams {
                status: a.status,
                task_type: a.r#type,
                parent: a.parent,
                limit: a.limit,
            }),
        )
        .await
    }

    #[tool(description = "Show one task in full, including its log and relationships.")]
    async fn task_show(
        &self,
        Parameters(a): Parameters<TaskShowArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(a.db, DaemonCommand::TaskShow(TaskShowParams { key: a.key }))
            .await
    }

    #[tool(description = "Create a task on the kanban board.")]
    async fn task_create(
        &self,
        Parameters(a): Parameters<TaskCreateArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::TaskCreate(TaskCreateParams {
                title: a.title,
                description: a.description,
                task_type: a.r#type.unwrap_or_else(|| "task".to_string()),
                parent: a.parent,
                priority: a.priority.unwrap_or_else(|| "medium".to_string()),
                tags: a.tags.unwrap_or_default(),
            }),
        )
        .await
    }

    #[tool(description = "Update a task's title, description, status, priority, or tags.")]
    async fn task_update(
        &self,
        Parameters(a): Parameters<TaskUpdateArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(
            a.db,
            DaemonCommand::TaskUpdate(TaskUpdateParams {
                key: a.key,
                title: a.title,
                description: a.description,
                priority: a.priority,
                status: a.status,
                add_tags: a.add_tags.unwrap_or_default(),
                remove_tags: a.remove_tags.unwrap_or_default(),
            }),
        )
        .await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for HadesMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                rmcp::model::Implementation::new("hades", env!("CARGO_PKG_VERSION"))
                    .with_title("HADES knowledge graph"),
            )
            .with_instructions(
                "HADES knowledge-graph tools. Results are the HADES JSON envelope: \
             {success, data, error, error_code}. Start with `orient` to discover \
             collections. Pass `db` to target a specific database (e.g. the \
             knowledge graph vs the task board); writes are governed by \
             database ACLs and the agent access tier.",
            )
    }
}

// ---------------------------------------------------------------------------
// Router / serve
// ---------------------------------------------------------------------------

/// Build the axum application: bearer-gated `/mcp`, open `/healthz`.
///
/// `allowed_hosts` feeds the transport's Host-header validation (MCP
/// DNS-rebinding protection). The SDK default only allows localhost
/// forms, which would 400 every LAN client — the bind address must be
/// included explicitly.
fn build_router(
    config: HadesConfig,
    tokens: Arc<TokenSet>,
    allowed_hosts: Vec<String>,
    extra_dbs: &[String],
) -> Result<Router> {
    let pools = Arc::new(PoolCache::new(config, extra_dbs)?);
    let mcp_service: StreamableHttpService<HadesMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(HadesMcpServer::new(pools.clone())),
            Default::default(),
            StreamableHttpServerConfig::default().with_allowed_hosts(allowed_hosts),
        );

    let mcp_router = Router::new()
        .nest_service("/mcp", mcp_service)
        .layer(middleware::from_fn_with_state(tokens, require_bearer))
        .layer(DefaultBodyLimit::max(MAX_BODY));

    Ok(Router::new()
        .route(
            "/healthz",
            get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
        )
        .merge(mcp_router))
}

/// Build the full application for a bound listener: host allowlist from
/// the bind address, bearer gate, pool cache. Fallible — call this
/// BEFORE spawning the serve task so a bad configuration stops daemon
/// startup instead of dying silently in a background task.
pub fn build_app(
    addr: &SocketAddr,
    config: HadesConfig,
    tokens: TokenSet,
    extra_dbs: &[String],
) -> Result<Router> {
    // Host allowlist: localhost forms for on-box clients, plus the bind
    // address itself for LAN clients (a port-less entry matches any port).
    let allowed_hosts = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
        addr.ip().to_string(),
    ];
    build_router(config, Arc::new(tokens), allowed_hosts, extra_dbs)
}

/// Serve a pre-built application until the cancellation token fires.
pub async fn serve_app(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: CancellationToken,
) -> Result<()> {
    let addr = listener.local_addr().context("mcp listener local_addr")?;
    tracing::info!(%addr, "mcp endpoint listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled_owned().await })
        .await
        .context("mcp server error")?;
    tracing::info!("mcp endpoint stopped");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use tower::ServiceExt;

    fn test_config() -> HadesConfig {
        let mut config = HadesConfig::default();
        config.database.name = Some("bident_burn".to_string());
        config
    }

    fn test_router() -> Router {
        build_router(
            test_config(),
            Arc::new(TokenSet::from_tokens(vec!["tok".into()])),
            vec!["127.0.0.1".to_string(), "localhost".to_string()],
            &[],
        )
        .unwrap()
    }

    // --- tokens ------------------------------------------------------------

    #[test]
    fn token_file_parses_and_ignores_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens");
        std::fs::write(&path, "# comment\n\nalpha-token\n  beta-token  \n").unwrap();
        let set = TokenSet::load(&path).unwrap();
        assert!(set.contains("alpha-token"));
        assert!(set.contains("beta-token"));
        assert!(!set.contains("gamma-token"));
    }

    #[test]
    fn empty_token_file_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens");
        std::fs::write(&path, "# only comments\n\n").unwrap();
        assert!(TokenSet::load(&path).is_err());
    }

    #[test]
    fn missing_token_file_fails_closed() {
        assert!(TokenSet::load(Path::new("/nonexistent/hades-tokens")).is_err());
    }

    #[test]
    fn ct_eq_basics() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secre"));
        assert!(ct_eq(b"", b""));
    }

    // --- bind policy -------------------------------------------------------

    #[test]
    fn bind_policy_allows_loopback_and_rfc1918() {
        for good in [
            "127.0.0.1:8088",
            "192.168.0.10:8088",
            "10.1.2.3:8088",
            "[::1]:8088",
            "[fd00::1]:8088",
        ] {
            let addr: SocketAddr = good.parse().unwrap();
            assert!(
                ensure_private_bind(&addr).is_ok(),
                "{good} should be allowed"
            );
        }
    }

    #[test]
    fn bind_policy_refuses_public_and_wildcard() {
        for bad in [
            "0.0.0.0:8088",
            "8.8.8.8:8088",
            "[::]:8088",
            "203.0.113.7:80",
        ] {
            let addr: SocketAddr = bad.parse().unwrap();
            assert!(
                ensure_private_bind(&addr).is_err(),
                "{bad} should be refused"
            );
        }
    }

    // --- database allowlist ------------------------------------------------

    #[tokio::test]
    async fn db_allowlist_defaults_to_configured_database_only() {
        let cache = PoolCache::new(test_config(), &[]).unwrap();
        // Default database resolves.
        assert!(cache.entry_for(None).await.is_ok());
        assert!(cache.entry_for(Some("bident_burn")).await.is_ok());
        // Anything else — including a ro-readable production name — is
        // refused before any pool is built.
        let err = cache.entry_for(Some("NestedLearning")).await.unwrap_err();
        assert!(err.contains("not served"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn db_allowlist_widens_only_via_explicit_list() {
        let cache = PoolCache::new(test_config(), &["weavertools_v2".to_string()]).unwrap();
        assert!(cache.entry_for(Some("weavertools_v2")).await.is_ok());
        assert!(cache.entry_for(Some("bident_burn")).await.is_ok());
        assert!(cache.entry_for(Some("NestedLearning")).await.is_err());
    }

    // --- database names ----------------------------------------------------

    #[test]
    fn db_name_validation() {
        assert!(is_valid_db_name("weavertools_v2"));
        assert!(is_valid_db_name("bident_burn"));
        assert!(!is_valid_db_name(""));
        assert!(!is_valid_db_name("db/../etc"));
        assert!(!is_valid_db_name(&"x".repeat(65)));
    }

    // --- tool surface ------------------------------------------------------

    #[test]
    fn tool_surface_is_the_curated_twelve() {
        let router = HadesMcpServer::tool_router();
        let mut names: Vec<String> = router
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "db_count",
                "db_get",
                "db_list",
                "db_query",
                "graph_neighbors",
                "graph_traverse",
                "orient",
                "smell_report",
                "task_create",
                "task_list",
                "task_show",
                "task_update",
            ]
        );
    }

    // --- http surface ------------------------------------------------------

    #[tokio::test]
    async fn healthz_is_open() {
        let app = test_router();
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn mcp_requires_bearer_token() {
        let app = test_router();
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn mcp_rejects_wrong_token() {
        let app = test_router();
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("authorization", "Bearer wrong")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn lan_host_header_accepted_when_allowlisted() {
        // The SDK's default allowlist is localhost-only, which would 400
        // every LAN client. This pins the fix: the bind address entry
        // admits LAN Host headers (port-less entries match any port).
        let app = build_router(
            test_config(),
            Arc::new(TokenSet::from_tokens(vec!["tok".into()])),
            vec!["192.168.0.10".to_string()],
            &[],
        )
        .unwrap();
        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#;
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("authorization", "Bearer tok")
                    .header("host", "192.168.0.10:8088")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .body(Body::from(init))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unallowlisted_host_header_rejected() {
        let app = test_router(); // allows 127.0.0.1 + localhost only
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("authorization", "Bearer tok")
                    .header("host", "evil.example.com")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn mcp_initialize_succeeds_with_token() {
        let app = test_router();
        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#;
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("authorization", "Bearer tok")
                    .header("host", "127.0.0.1")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .body(Body::from(init))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = http_body_util::BodyExt::collect(resp.into_body())
            .await
            .unwrap()
            .to_bytes();
        assert_eq!(
            status,
            StatusCode::OK,
            "body: {}",
            String::from_utf8_lossy(&body)
        );
    }
}
