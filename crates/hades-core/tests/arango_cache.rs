//! Integration tests for the cache layer.
//!
//! Prerequisites: ArangoDB running with socket at /run/arangodb3/arangodb.sock
//! and ARANGO_PASSWORD set. Tests use bident_burn database.

use std::path::PathBuf;

use hades_core::db::cache::CachedPool;
use hades_core::db::crud;
use hades_core::db::query::ExecutionTarget;
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
async fn test_cached_get_document() {
    let Some(pool) = test_pool() else { return };

    let col = format!("test_cache_get_{}", std::process::id());
    crud::create_collection(&pool, &col, Some(2)).await.unwrap();

    let doc = serde_json::json!({"_key": "doc1", "value": 42});
    crud::insert_documents(&pool, &col, &[doc], false).await.unwrap();

    let cached = CachedPool::with_defaults(pool.clone());

    // First call: cache miss
    let result = cached.get_document(&col, "doc1").await.unwrap();
    assert_eq!(result["value"], 42);

    let m = cached.metrics();
    assert_eq!(m.doc_hits, 0);
    assert_eq!(m.doc_misses, 1);

    // Second call: cache hit
    let result2 = cached.get_document(&col, "doc1").await.unwrap();
    assert_eq!(result2["value"], 42);

    let m = cached.metrics();
    assert_eq!(m.doc_hits, 1);
    assert_eq!(m.doc_misses, 1);

    crud::drop_collection(&pool, &col, true).await.unwrap();
}

#[tokio::test]
async fn test_cached_delete_invalidates() {
    let Some(pool) = test_pool() else { return };

    let col = format!("test_cache_del_{}", std::process::id());
    crud::create_collection(&pool, &col, Some(2)).await.unwrap();

    let doc = serde_json::json!({"_key": "doc1", "value": 99});
    crud::insert_documents(&pool, &col, &[doc], false).await.unwrap();

    let cached = CachedPool::with_defaults(pool.clone());

    // Populate cache
    cached.get_document(&col, "doc1").await.unwrap();
    assert_eq!(cached.metrics().doc_misses, 1);

    // Delete should invalidate
    cached.delete_document(&col, "doc1").await.unwrap();

    // Next get should be a miss (and fail with 404 since doc is deleted)
    let result = cached.get_document(&col, "doc1").await;
    assert!(result.is_err());
    assert_eq!(cached.metrics().doc_misses, 2);

    crud::drop_collection(&pool, &col, true).await.unwrap();
}

#[tokio::test]
async fn test_cached_query() {
    let Some(pool) = test_pool() else { return };
    let cached = CachedPool::with_defaults(pool);

    // First query: miss
    let r1 = cached
        .query("RETURN 42", None, None, false, ExecutionTarget::Reader)
        .await
        .unwrap();
    assert_eq!(r1.results[0], 42);
    assert_eq!(cached.metrics().query_misses, 1);
    assert_eq!(cached.metrics().query_hits, 0);

    // Same query: hit
    let r2 = cached
        .query("RETURN 42", None, None, false, ExecutionTarget::Reader)
        .await
        .unwrap();
    assert_eq!(r2.results[0], 42);
    assert_eq!(cached.metrics().query_hits, 1);
}

#[tokio::test]
async fn test_cached_query_writer_bypasses_cache() {
    let Some(pool) = test_pool() else { return };
    let cached = CachedPool::with_defaults(pool);

    // Writer queries should not be cached
    cached
        .query("RETURN 1", None, None, false, ExecutionTarget::Writer)
        .await
        .unwrap();

    cached
        .query("RETURN 1", None, None, false, ExecutionTarget::Writer)
        .await
        .unwrap();

    // No cache activity for writer queries
    assert_eq!(cached.metrics().query_hits, 0);
    assert_eq!(cached.metrics().query_misses, 0);
}

#[tokio::test]
async fn test_invalidate_all() {
    let Some(pool) = test_pool() else { return };
    let cached = CachedPool::with_defaults(pool);

    // Populate query cache
    cached
        .query("RETURN 1", None, None, false, ExecutionTarget::Reader)
        .await
        .unwrap();

    assert_eq!(cached.metrics().query_misses, 1);

    // Verify it's cached (hit)
    cached
        .query("RETURN 1", None, None, false, ExecutionTarget::Reader)
        .await
        .unwrap();
    assert_eq!(cached.metrics().query_hits, 1);

    cached.invalidate_all();

    // After invalidation, next call should be a miss
    cached
        .query("RETURN 1", None, None, false, ExecutionTarget::Reader)
        .await
        .unwrap();

    assert_eq!(cached.metrics().query_misses, 2);
}
