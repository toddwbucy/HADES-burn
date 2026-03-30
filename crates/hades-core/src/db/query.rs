//! AQL query execution with cursor-based pagination.
//!
//! Queries are sent to the reader client.  Cursor continuation (for
//! paginated results) is routed through the writer client because the
//! read-only proxy socket does not support cursor state endpoints.

use serde_json::Value;
use tracing::{debug, instrument, trace, warn};

use super::error::ArangoError;
use super::pool::ArangoPool;

/// Default batch size for AQL queries.
const DEFAULT_BATCH_SIZE: u32 = 1000;

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
/// The initial cursor request goes through `pool.reader()`.  If the
/// result set spans multiple pages (`hasMore`), cursor continuation
/// requests are routed through `pool.writer()` because the read-only
/// proxy socket does not support `POST /_api/cursor/{id}`.
///
/// All pages are accumulated into a single `Vec<Value>`.
#[instrument(skip(pool, bind_vars), fields(db = %pool.database()))]
pub async fn query(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
    batch_size: Option<u32>,
    full_count: bool,
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

    debug!(batch_size, full_count, "executing AQL query");
    trace!(aql, "query text");

    // Initial cursor request through reader
    let resp = pool.reader().post("cursor", &body).await?;

    let mut results = extract_results(&resp);
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
        if let Some(ref id) = cursor_id {
            loop {
                let path = format!("cursor/{id}");
                let resp = pool.writer().post(&path, &serde_json::json!({})).await?;

                let page = extract_results(&resp);
                trace!(page_size = page.len(), "cursor page");
                results.extend(page);

                if !resp["hasMore"].as_bool().unwrap_or(false) {
                    break;
                }
            }

            // Clean up cursor (best-effort, ignore errors)
            let delete_path = format!("cursor/{id}");
            if let Err(e) = pool.writer().delete(&delete_path).await {
                trace!(error = %e, "cursor cleanup failed (non-fatal)");
            }
        } else {
            warn!("hasMore=true but no cursor ID in response");
        }
    }

    debug!(total_results = results.len(), "query complete");

    Ok(QueryResult {
        results,
        full_count: full_count_val,
        extra,
    })
}

/// Execute an AQL query and return the first result, or `None` if empty.
#[instrument(skip(pool, bind_vars), fields(db = %pool.database()))]
pub async fn query_single(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
) -> Result<Option<Value>, ArangoError> {
    let result = query(pool, aql, bind_vars, Some(1), false).await?;
    Ok(result.results.into_iter().next())
}

/// Extract the `result` array from a cursor response.
fn extract_results(resp: &Value) -> Vec<Value> {
    resp["result"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}
