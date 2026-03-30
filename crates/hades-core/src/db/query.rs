//! AQL query execution with cursor-based pagination.
//!
//! By default, the initial cursor request goes through `pool.reader()`.
//! Use [`ExecutionTarget::Writer`] for mutating AQL (INSERT, UPDATE,
//! REMOVE).  Cursor continuation is always routed through the writer
//! client because the read-only proxy socket does not support cursor
//! state endpoints.

use serde_json::Value;
use tracing::{debug, instrument, trace};

use super::error::ArangoError;
use super::pool::ArangoPool;

/// Default batch size for AQL queries.
const DEFAULT_BATCH_SIZE: u32 = 1000;

/// Controls which pool endpoint receives the initial cursor request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionTarget {
    /// Route through `pool.reader()` — appropriate for read-only AQL.
    #[default]
    Reader,
    /// Route through `pool.writer()` — required for mutating AQL
    /// (INSERT, UPDATE, REPLACE, REMOVE, UPSERT).
    Writer,
}

/// Result of an AQL query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// All result documents accumulated across pages.
    pub results: Vec<Value>,
    /// Total matching documents (only set when `full_count` was requested).
    pub full_count: Option<u64>,
    /// Extra metadata from ArangoDB (stats, profile, warnings).
    pub extra: Option<Value>,
}

/// Execute an AQL query with optional bind variables and pagination.
///
/// The initial cursor request is routed based on `target`: reader for
/// read-only queries, writer for mutating AQL.  Cursor continuation
/// always goes through `pool.writer()` because the read-only proxy
/// socket does not support `POST /_api/cursor/{id}`.
///
/// All pages are accumulated into a single `Vec<Value>`.
#[instrument(skip(pool, bind_vars), fields(db = %pool.database()))]
pub async fn query(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
    batch_size: Option<u32>,
    full_count: bool,
    target: ExecutionTarget,
) -> Result<QueryResult, ArangoError> {
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

    let mut body = serde_json::json!({
        "query": aql,
        "batchSize": batch_size,
    });

    if let Some(vars) = bind_vars {
        body["bindVars"] = vars.clone();
    }

    if full_count {
        body["options"] = serde_json::json!({"fullCount": true});
    }

    debug!(batch_size, full_count, ?target, "executing AQL query");
    trace!(aql, "query text");

    // Initial cursor request through the selected endpoint
    let initial_client = match target {
        ExecutionTarget::Reader => pool.reader(),
        ExecutionTarget::Writer => pool.writer(),
    };
    let resp = initial_client.post("cursor", &body).await?;

    let mut results = extract_results(&resp)?;
    let extra = resp.get("extra").cloned();
    let full_count_val = extra
        .as_ref()
        .and_then(|e| e["stats"]["fullCount"].as_u64());

    let has_more = resp["hasMore"].as_bool().unwrap_or(false);
    let cursor_id = resp["id"].as_str().map(String::from);

    debug!(
        first_batch = results.len(),
        has_more,
        cursor_id = cursor_id.as_deref().unwrap_or("none"),
        "initial cursor response"
    );

    // Paginate through remaining results via writer (proxy limitation)
    if has_more {
        let id = cursor_id.as_deref().ok_or_else(|| {
            ArangoError::Request(
                "hasMore=true but no cursor ID in response".to_string(),
            )
        })?;

        let pagination_result = paginate(pool, id, &mut results).await;

        // Always attempt cursor cleanup, even if pagination failed
        let delete_path = format!("cursor/{id}");
        if let Err(e) = pool.writer().delete(&delete_path).await {
            trace!(error = %e, "cursor cleanup failed (non-fatal)");
        }

        // Propagate pagination error after cleanup
        pagination_result?;
    }

    debug!(total_results = results.len(), "query complete");

    Ok(QueryResult {
        results,
        full_count: full_count_val,
        extra,
    })
}

/// Fetch remaining cursor pages, appending to `results`.
async fn paginate(
    pool: &ArangoPool,
    cursor_id: &str,
    results: &mut Vec<Value>,
) -> Result<(), ArangoError> {
    loop {
        let path = format!("cursor/{cursor_id}");
        let resp = pool.writer().post(&path, &serde_json::json!({})).await?;

        let page = extract_results(&resp)?;
        trace!(page_size = page.len(), "cursor page");
        results.extend(page);

        if !resp["hasMore"].as_bool().unwrap_or(false) {
            break;
        }
    }
    Ok(())
}

/// Execute an AQL query and return the first result, or `None` if empty.
#[instrument(skip(pool, bind_vars), fields(db = %pool.database()))]
pub async fn query_single(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
) -> Result<Option<Value>, ArangoError> {
    let result = query(pool, aql, bind_vars, Some(1), false, ExecutionTarget::Reader).await?;
    Ok(result.results.into_iter().next())
}

/// Extract the `result` array from a cursor response.
///
/// Errors if the response has no `result` key or if it is not an array,
/// which indicates a malformed cursor payload from ArangoDB.
fn extract_results(resp: &Value) -> Result<Vec<Value>, ArangoError> {
    match resp.get("result") {
        Some(v) if v.is_array() => Ok(v.as_array().unwrap().clone()),
        Some(_) => Err(ArangoError::Request(
            "cursor response 'result' field is not an array".to_string(),
        )),
        None => Err(ArangoError::Request(
            "cursor response missing 'result' field".to_string(),
        )),
    }
}
