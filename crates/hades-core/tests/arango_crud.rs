//! Integration tests for ArangoDB CRUD operations.
//!
//! These tests require a running ArangoDB instance accessible via
//! Unix socket. They use bident_burn (the writable project database).
//!
//! Prerequisites:
//! - ArangoDB running with socket at /run/arangodb3/arangodb.sock
//! - `bident_burn` database with `persephone_tasks` collection (the
//!   permanent HADES kanban table, managed by `hades --db bident_burn`)
//! - ARANGO_PASSWORD env var set for root authentication
//! - Set ARANGO_TESTS=1 to require tests to run (fail instead of skip)
//!
//! Each mutating test uses its own temporary collection (with PID suffix)
//! to avoid parallel conflicts.

use std::path::PathBuf;

use hades_core::db::{crud, ArangoClient, ArangoPool};

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

/// Whether integration tests are required to run (ARANGO_TESTS=1).
fn arango_tests_required() -> bool {
    std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true")
}

fn test_pool() -> Option<ArangoPool> {
    let socket = arango_socket();
    if !socket.exists() {
        if arango_tests_required() {
            panic!(
                "ARANGO_TESTS is set but socket not found at {}",
                socket.display()
            );
        }
        eprintln!("skipping: ArangoDB socket not found at {}", socket.display());
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

/// Helper to ensure a collection exists, dropping first if present.
async fn setup(pool: &ArangoPool, name: &str) {
    let _ = crud::drop_collection(pool, name, true).await;
    crud::create_collection(pool, name, None)
        .await
        .expect("failed to create test collection");
}

/// Helper to clean up a collection.
async fn teardown(pool: &ArangoPool, name: &str) {
    let _ = crud::drop_collection(pool, name, true).await;
}

// ---------------------------------------------------------------------------
// Collection operations (read-only, no temp collection needed)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_list_collections() {
    let Some(pool) = test_pool() else { return };

    let collections = crud::list_collections(&pool, true).await.unwrap();
    let names: Vec<&str> = collections.iter().map(|c| c.name.as_str()).collect();

    assert!(names.contains(&"persephone_tasks"), "collections: {names:?}");
    assert!(!names.iter().any(|n| n.starts_with('_')), "system collections not excluded");
}

#[tokio::test]
async fn test_list_collections_include_system() {
    let Some(pool) = test_pool() else { return };

    let collections = crud::list_collections(&pool, false).await.unwrap();
    let has_system = collections.iter().any(|c| c.name.starts_with('_'));
    assert!(has_system, "expected system collections when exclude_system=false");
}

#[tokio::test]
async fn test_drop_collection_ignore_missing() {
    let Some(pool) = test_pool() else { return };

    // Use a unique name to avoid any collision
    let name = format!("nonexistent_{}", std::process::id());
    let resp = crud::drop_collection(&pool, &name, true)
        .await
        .unwrap();
    assert_eq!(resp["dropped"], false);
}

#[tokio::test]
async fn test_count_collection() {
    let Some(pool) = test_pool() else { return };

    let count = crud::count_collection(&pool, "persephone_tasks").await.unwrap();
    assert!(count > 0, "expected at least one task, got {count}");
}

// ---------------------------------------------------------------------------
// Collection create/drop (own collection)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_create_and_drop_collection() {
    let Some(pool) = test_pool() else { return };
    let col = format!("test_crud_create_drop_{}", std::process::id());

    let _ = crud::drop_collection(&pool, &col, true).await;

    let resp = crud::create_collection(&pool, &col, None).await.unwrap();
    assert_eq!(resp["name"].as_str(), Some(col.as_str()));

    let collections = crud::list_collections(&pool, false).await.unwrap();
    assert!(collections.iter().any(|c| c.name == col));

    crud::drop_collection(&pool, &col, false).await.unwrap();

    let collections = crud::list_collections(&pool, false).await.unwrap();
    assert!(!collections.iter().any(|c| c.name == col));
}

// ---------------------------------------------------------------------------
// Document CRUD (each test gets its own collection)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_document_crud_lifecycle() {
    let Some(pool) = test_pool() else { return };
    let col = "test_crud_lifecycle";
    setup(&pool, col).await;

    // Insert
    let docs = vec![serde_json::json!({
        "_key": "test_doc_1",
        "title": "Test Document",
        "value": 42
    })];
    let result = crud::insert_documents(&pool, col, &docs, false).await.unwrap();
    assert_eq!(result.created, 1);

    // Get
    let doc = crud::get_document(&pool, col, "test_doc_1").await.unwrap();
    assert_eq!(doc["title"], "Test Document");
    assert_eq!(doc["value"], 42);

    // Update (merge-patch)
    let update = serde_json::json!({"value": 99, "extra": "field"});
    crud::update_document(&pool, col, "test_doc_1", &update).await.unwrap();

    let doc = crud::get_document(&pool, col, "test_doc_1").await.unwrap();
    assert_eq!(doc["value"], 99);
    assert_eq!(doc["extra"], "field");
    assert_eq!(doc["title"], "Test Document"); // preserved by PATCH

    // Delete
    crud::delete_document(&pool, col, "test_doc_1").await.unwrap();

    let result = crud::get_document(&pool, col, "test_doc_1").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().is_not_found());

    teardown(&pool, col).await;
}

#[tokio::test]
async fn test_replace_document() {
    let Some(pool) = test_pool() else { return };
    let col = "test_crud_replace";
    setup(&pool, col).await;

    let docs = vec![serde_json::json!({
        "_key": "replace_test",
        "title": "Original",
        "extra": "will_be_removed"
    })];
    crud::insert_documents(&pool, col, &docs, false).await.unwrap();

    let replacement = serde_json::json!({"title": "Replaced"});
    crud::replace_document(&pool, col, "replace_test", &replacement).await.unwrap();

    let doc = crud::get_document(&pool, col, "replace_test").await.unwrap();
    assert_eq!(doc["title"], "Replaced");
    assert!(doc.get("extra").is_none() || doc["extra"].is_null());

    teardown(&pool, col).await;
}

// ---------------------------------------------------------------------------
// Bulk import
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_bulk_insert() {
    let Some(pool) = test_pool() else { return };
    let col = "test_crud_bulk";
    setup(&pool, col).await;

    let docs: Vec<serde_json::Value> = (0..25)
        .map(|i| serde_json::json!({"_key": format!("bulk_{i}"), "index": i}))
        .collect();

    let result = crud::bulk_insert(&pool, col, &docs, Some(10), false).await.unwrap();
    assert_eq!(result.created, 25);

    let count = crud::count_collection(&pool, col).await.unwrap();
    assert_eq!(count, 25);

    teardown(&pool, col).await;
}

#[tokio::test]
async fn test_insert_with_overwrite() {
    let Some(pool) = test_pool() else { return };
    let col = "test_crud_overwrite";
    setup(&pool, col).await;

    let docs = vec![serde_json::json!({"_key": "ow_test", "version": 1})];
    crud::insert_documents(&pool, col, &docs, false).await.unwrap();

    let docs = vec![serde_json::json!({"_key": "ow_test", "version": 2})];
    let result = crud::insert_documents(&pool, col, &docs, true).await.unwrap();
    assert!(result.created > 0 || result.updated > 0);

    let doc = crud::get_document(&pool, col, "ow_test").await.unwrap();
    assert_eq!(doc["version"], 2);

    teardown(&pool, col).await;
}

#[tokio::test]
async fn test_bulk_insert_zero_chunk_size() {
    let Some(pool) = test_pool() else { return };
    let col = "test_crud_zero_chunk";
    setup(&pool, col).await;

    let docs = vec![serde_json::json!({"_key": "x"})];
    let err = crud::bulk_insert(&pool, col, &docs, Some(0), false)
        .await
        .unwrap_err();
    assert_eq!(err.committed.created, 0);
    assert!(err.error.to_string().contains("chunk_size must be > 0"));

    teardown(&pool, col).await;
}
