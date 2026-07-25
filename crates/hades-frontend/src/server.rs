//! Local HTTP server: serves the viewer assets and a graph-data API.
//!
//! The viewer (a WebGL single-page app) fetches `/api/graphs` to populate a
//! graph picker, `/api/graph-data?graph=<name>&limit=<n>` to load a (optionally
//! capped) snapshot, and `/api/expand?graph=&node=` to grow a node's
//! neighborhood on demand. Every route delegates to the [`Backend`] assembler,
//! so the server is a thin shell over the same coupling layer the `dump`
//! subcommand uses.
//!
//! Binding is guarded to loopback or RFC1918/ULA private ranges only — the viewer
//! reads a database and has no auth, so it must never land on a public address.
//! The pinned database is fixed at startup and not client-selectable, containing
//! blast radius to the one graph store this server was pointed at.

use crate::assemble::{Backend, discover_databases};
use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    extract::{Query, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use base64::Engine;
use serde::Deserialize;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    bin: String,
    limit: Option<u32>,
    /// Databases the client may select. Exact-match validated per request, so the
    /// UI's database picker can never reach a database outside this set.
    databases: Arc<Vec<String>>,
    /// When set, every request must present HTTP Basic credentials whose password
    /// matches. Protects the viewer when it's exposed on a LAN address.
    password: Option<Arc<String>>,
    /// Host header values this server will answer to. A browser enforces
    /// same-origin by *name*, not address, so without this an attacker page can
    /// point a short-TTL hostname at 127.0.0.1 (DNS rebinding), reach an
    /// unauthenticated loopback bind as a same-origin request, and read every
    /// served database. Checking Host closes that: the rebound name never matches.
    allowed_hosts: Arc<Vec<String>>,
}

impl AppState {
    /// Resolve a request's target database to a [`Backend`], defaulting to the
    /// first allowlisted database and rejecting anything outside the allowlist.
    #[allow(clippy::result_large_err)] // Err is an axum Response by design (one call site each)
    fn backend_for(&self, db: Option<&str>) -> std::result::Result<Backend, Response> {
        let db = db.unwrap_or(&self.databases[0]);
        if !self.databases.iter().any(|d| d == db) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("database `{db}` is not in the allowlist") })),
            )
                .into_response());
        }
        Ok(Backend {
            bin: self.bin.clone(),
            database: db.to_string(),
            limit: self.limit,
        })
    }
}

pub async fn serve(
    bin: String,
    limit: Option<u32>,
    databases: Vec<String>,
    password: Option<String>,
    bind: &str,
) -> Result<()> {
    // No explicit --db list => serve every database the hades user can access.
    let databases = if databases.is_empty() {
        let all = discover_databases(&bin)
            .await
            .context("failed to discover databases (pass --db to list them explicitly)")?;
        tracing::info!(count = all.len(), "serving all accessible databases");
        all
    } else {
        databases
    };
    if databases.is_empty() {
        return Err(anyhow!("no databases to serve"));
    }

    let addr = ensure_private(bind)?;
    // Loud refusal to expose an unauthenticated viewer beyond loopback: with no
    // password, anyone on the LAN could read every served database.
    if password.is_none() && !addr.ip().is_loopback() {
        return Err(anyhow!(
            "refusing to serve {} databases unauthenticated on {} — pass --password <PASS> \
             (browser prompts once), or bind loopback (127.0.0.1)",
            databases.len(),
            addr.ip()
        ));
    }

    let state = AppState {
        bin,
        limit,
        databases: Arc::new(databases),
        password: password.map(Arc::new),
        allowed_hosts: Arc::new(allowed_hosts_for(&addr)),
    };
    let app = Router::new()
        .route(
            "/",
            get(|| async { asset(INDEX_HTML, "text/html; charset=utf-8") }),
        )
        .route("/viewer.js", get(|| async { asset(VIEWER_JS, JS) }))
        .route(
            "/vendor/graphology.min.js",
            get(|| async { asset(GRAPHOLOGY_JS, JS) }),
        )
        .route(
            "/vendor/sigma.min.js",
            get(|| async { asset(SIGMA_JS, JS) }),
        )
        .route(
            "/vendor/fa2.bundle.min.js",
            get(|| async { asset(FA2_JS, JS) }),
        )
        .route("/api/databases", get(list_databases))
        .route("/api/graphs", get(list_graphs))
        .route("/api/graph-data", get(graph_data))
        .route("/api/expand", get(expand))
        .route("/api/node", get(node))
        .route("/api/content", get(content))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_known_host,
        ))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    tracing::info!(%addr, "hades-viewer serving");
    axum::serve(listener, app).await.context("server error")?;
    Ok(())
}

/// The Host header values this server answers to: the literal bind address, plus
/// the loopback spellings when bound to loopback.
fn allowed_hosts_for(addr: &SocketAddr) -> Vec<String> {
    let port = addr.port();
    let mut hosts = vec![addr.to_string(), format!("{}:{}", addr.ip(), port)];
    if addr.ip().is_loopback() {
        hosts.push(format!("localhost:{port}"));
        hosts.push(format!("127.0.0.1:{port}"));
        hosts.push(format!("[::1]:{port}"));
    }
    hosts.sort();
    hosts.dedup();
    hosts
}

/// Reject requests whose Host header isn't one this server was bound as, which
/// defeats DNS rebinding against the unauthenticated loopback bind.
async fn require_known_host(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    if state.allowed_hosts.iter().any(|h| h == host) {
        return next.run(req).await;
    }
    tracing::warn!(%host, "rejected request with unrecognized Host header");
    (
        StatusCode::MISDIRECTED_REQUEST,
        Json(serde_json::json!({ "error": "unrecognized Host header" })),
    )
        .into_response()
}

/// HTTP Basic auth gate (no-op when no password is configured). The username is
/// ignored; only the password must match, compared in constant time.
async fn require_auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Some(expected) = state.password.as_ref() else {
        return next.run(req).await; // auth disabled
    };
    let supplied = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Basic "))
        .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok())
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|creds| creds.split_once(':').map(|(_, p)| p.to_string()));

    if supplied.map(|p| constant_time_eq(p.as_bytes(), expected.as_bytes())) == Some(true) {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Basic realm=\"hades-viewer\"")],
            "authentication required",
        )
            .into_response()
    }
}

/// Length-independent, content-constant-time byte comparison for the password.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Parse and gate the bind address: loopback or a private LAN range only. The
/// viewer is unauthenticated, so a public bind is refused outright rather than
/// silently exposing a database to the internet.
fn ensure_private(bind: &str) -> Result<SocketAddr> {
    let addr: SocketAddr = bind
        .parse()
        .with_context(|| format!("invalid bind address `{bind}` (want IP:PORT)"))?;
    let private = match addr.ip() {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        // Stable-Rust equivalents of ULA (fc00::/7) and link-local (fe80::/10),
        // since Ipv6Addr::is_unique_local / is_unicast_link_local are unstable.
        IpAddr::V6(v6) => {
            let o = v6.octets();
            v6.is_loopback() || (o[0] & 0xfe) == 0xfc || (o[0] == 0xfe && (o[1] & 0xc0) == 0x80)
        }
    };
    if !private {
        return Err(anyhow!(
            "refusing to bind non-private address {} — use loopback or a private LAN IP \
             (10/8, 172.16/12, 192.168/16)",
            addr.ip()
        ));
    }
    Ok(addr)
}

// Frontend assets, embedded at compile time so the binary is self-contained
// (no CDN, no external files at runtime). Vendored libs are pinned; see
// web/vendor/PROVENANCE.md.
const INDEX_HTML: &str = include_str!("../web/index.html");
const VIEWER_JS: &str = include_str!("../web/viewer.js");
const GRAPHOLOGY_JS: &str = include_str!("../web/vendor/graphology.min.js");
const SIGMA_JS: &str = include_str!("../web/vendor/sigma.min.js");
const FA2_JS: &str = include_str!("../web/vendor/fa2.bundle.min.js");
const JS: &str = "text/javascript; charset=utf-8";

fn asset(body: &'static str, content_type: &'static str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

/// The databases this server may serve; the first is the client default.
async fn list_databases(State(state): State<AppState>) -> Response {
    Json(serde_json::json!({ "databases": &*state.databases })).into_response()
}

#[derive(Deserialize)]
struct DbQuery {
    db: Option<String>,
}

async fn list_graphs(State(state): State<AppState>, Query(q): Query<DbQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend.list_graphs().await {
        Ok(graphs) => {
            let names: Vec<&str> = graphs.iter().map(|g| g.name.as_str()).collect();
            Json(serde_json::json!({ "graphs": names })).into_response()
        }
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct GraphQuery {
    db: Option<String>,
    graph: String,
    /// Optional per-collection cap for server-side slicing of large graphs.
    limit: Option<u32>,
}

async fn graph_data(State(state): State<AppState>, Query(q): Query<GraphQuery>) -> Response {
    let mut backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    // Per-request limit may only narrow the operator's cap, never widen it.
    // An unclamped override would let any client erase `--limit` and force a
    // full in-memory export of every collection.
    backend.limit = match (state.limit, q.limit) {
        (Some(op), Some(client)) => Some(op.min(client)),
        (Some(op), None) => Some(op),
        (None, client) => client,
    };
    match backend.assemble(&q.graph).await {
        Ok(snapshot) => Json(snapshot).into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct ExpandQuery {
    db: Option<String>,
    graph: String,
    node: String,
    #[serde(default = "default_direction")]
    direction: String,
    #[serde(default = "default_expand_limit")]
    limit: u32,
}

fn default_direction() -> String {
    "any".to_string()
}
fn default_expand_limit() -> u32 {
    50
}

/// Grow one node's immediate neighborhood — the interactive click-to-expand path
/// for graphs too large to load whole.
async fn expand(State(state): State<AppState>, Query(q): Query<ExpandQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend
        .neighbors(&q.graph, &q.node, &q.direction, q.limit)
        .await
    {
        Ok(delta) => Json(delta).into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct NodeQuery {
    db: Option<String>,
    /// Full ArangoDB `_id` (`collection/key`).
    id: String,
}

/// Fetch one node by `_id` — lets a deep link resolve a node that isn't part of
/// the currently loaded slice.
async fn node(State(state): State<AppState>, Query(q): Query<NodeQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend.fetch_node(&q.id).await {
        Ok(Some(n)) => Json(n).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("node `{}` not found", q.id) })),
        )
            .into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct ContentQuery {
    db: Option<String>,
    /// The `file_key` whose source chunks to fetch (codebase-schema graphs).
    file_key: String,
}

/// Fetch a node's associated source text (ordered chunks). Codebase-schema-aware;
/// returns an empty list for graphs without a `codebase_chunks` collection.
async fn content(State(state): State<AppState>, Query(q): Query<ContentQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend.file_content(&q.file_key).await {
        Ok(chunks) => Json(serde_json::json!({ "chunks": chunks })).into_response(),
        Err(e) => error_response(e),
    }
}

fn error_response(e: anyhow::Error) -> Response {
    tracing::error!(error = %e, "request failed");
    (
        StatusCode::BAD_GATEWAY,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
        .into_response()
}
