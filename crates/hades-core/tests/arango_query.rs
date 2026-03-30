//! Integration tests for AQL query execution.
//!
//! Prerequisites: see arango_crud.rs module docs.

use std::path::PathBuf;

use hades_core::db::query::{self, ExecutionTarget};
use hades_core::db::{ArangoClient, ArangoPool};
use tracing::warn;

fn arango_socket() -> PathBuf {
    PathBuf::from(
        std::env::var("ARANGO_SOCKET")
            .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
    )
}

fn arango_password() -> String {
    std::env::var("ARANGO_PASSWORD").expect(
        "ARANGO_PASSWORD must be set for integration tests.",
    )
}

fn test_pool() -> Option<ArangoPool> {
    let socket = arango_socket();
    if !socket.exists() {
        if std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true") {
            panic!("ARANGO_TESTS is set but socket not found at {}", socket.display());
        }
        warn!("skipping: ArangoDB socket not found at {}", socket.display());
        return None;
    }

    let client = ArangoClient::with_socket(
        socket,
        "bident_burn",
        "root",
        &arango_password(),
    );
    Some(ArangoPool::new(client.clone(), client))
}

#[tokio::test]
async fn test_simple_query() {
    let Some(pool) = test_pool() else { return };

    let result = query::query(&pool, "RETURN 1 + 1", None, None, false, ExecutionTarget::Reader)
        .await
        .unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0], 2);
}

#[tokio::test]
async fn test_query_with_bind_vars() {
    let Some(pool) = test_pool() else { return };

    let vars = serde_json::json!({"value": 42});
    let result = query::query(&pool, "RETURN @value", Some(&vars), None, false, ExecutionTarget::Reader)
        .await
        .unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0], 42);
}

#[tokio::test]
async fn test_query_collection() {
    let Some(pool) = test_pool() else { return };

    let result = query::query(
        &pool,
        "FOR t IN persephone_tasks LIMIT 3 RETURN t._key",
        None,
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    .unwrap();
    assert!(!result.results.is_empty(), "expected at least one task key");
    assert!(result.results.len() <= 3);
}

#[tokio::test]
async fn test_query_full_count() {
    let Some(pool) = test_pool() else { return };

    let result = query::query(
        &pool,
        "FOR t IN persephone_tasks LIMIT 2 RETURN t._key",
        None,
        None,
        true,
        ExecutionTarget::Reader,
    )
    .await
    .unwrap();
    assert!(result.full_count.is_some(), "expected full_count with fullCount=true");
    // full_count should be >= the number of results
    assert!(result.full_count.unwrap() >= result.results.len() as u64);
}

#[tokio::test]
async fn test_query_pagination() {
    let Some(pool) = test_pool() else { return };

    // Use a very small batch_size to force pagination
    let result = query::query(
        &pool,
        "FOR t IN persephone_tasks LIMIT 5 RETURN t._key",
        None,
        Some(2), // batch_size=2, should paginate across 3 pages
        false,
        ExecutionTarget::Reader,
    )
    .await
    .unwrap();
    assert_eq!(result.results.len(), 5, "expected 5 results after pagination");
}

#[tokio::test]
async fn test_query_single() {
    let Some(pool) = test_pool() else { return };

    let result = query::query_single(&pool, "RETURN 'hello'", None)
        .await
        .unwrap();
    assert_eq!(result, Some(serde_json::json!("hello")));
}

#[tokio::test]
async fn test_query_single_empty() {
    let Some(pool) = test_pool() else { return };

    let result = query::query_single(
        &pool,
        "FOR x IN [] RETURN x",
        None,
    )
    .await
    .unwrap();
    assert_eq!(result, None);
}

#[tokio::test]
async fn test_query_syntax_error() {
    let Some(pool) = test_pool() else { return };

    let result = query::query(
        &pool,
        "THIS IS NOT VALID AQL",
        None,
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await;
    assert!(result.is_err(), "expected error for invalid AQL");
}
