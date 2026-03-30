//! Integration tests for ArangoDB index management.
//!
//! Prerequisites: ArangoDB running with socket at /run/arangodb3/arangodb.sock
//! and ARANGO_PASSWORD set. Tests use bident_burn database.

use std::path::PathBuf;

use hades_core::db::index::{self, VectorMetric};
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
async fn test_list_indexes() {
    let Some(pool) = test_pool() else { return };

    // persephone_tasks is known to exist in bident_burn
    let indexes = index::list_indexes(&pool, "persephone_tasks")
        .await
        .unwrap();

    // Every collection has at least a primary index
    assert!(!indexes.is_empty(), "expected at least primary index");

    let primary = indexes.iter().find(|i| i.index_type == "primary");
    assert!(primary.is_some(), "expected a primary index");
}

#[tokio::test]
async fn test_list_indexes_nonexistent_collection() {
    let Some(pool) = test_pool() else { return };

    let result = index::list_indexes(&pool, "nonexistent_collection_xyz").await;
    assert!(result.is_err(), "expected error for nonexistent collection");
}

#[tokio::test]
async fn test_create_and_drop_vector_index() {
    use hades_core::db::crud;

    let Some(pool) = test_pool() else { return };

    let col_name = format!("test_vec_idx_{}", std::process::id());

    // Create a test collection
    crud::create_collection(&pool, &col_name, Some(2)).await.unwrap();

    // Insert a few documents with embedding fields so the index has something
    let docs: Vec<serde_json::Value> = (0..5)
        .map(|i| {
            serde_json::json!({
                "_key": format!("doc_{i}"),
                "embedding": vec![0.0_f64; 128],
                "chunk_key": format!("chunk_{i}"),
            })
        })
        .collect();
    crud::insert_documents(&pool, &col_name, &docs, false)
        .await
        .unwrap();

    // Create a vector index (requires --experimental-vector-index flag)
    let idx = match index::create_vector_index(
        &pool,
        &col_name,
        "embedding",
        128,
        Some(1), // explicit nLists for small collection
        10,
        VectorMetric::Cosine,
    )
    .await
    {
        Ok(idx) => idx,
        Err(e) if e.to_string().contains("vector index feature is not enabled") => {
            warn!("skipping: ArangoDB vector index feature not enabled");
            crud::drop_collection(&pool, &col_name, true).await.unwrap();
            return;
        }
        Err(e) => panic!("unexpected error creating vector index: {e}"),
    };

    assert_eq!(idx.index_type, "vector");
    assert!(idx.fields.contains(&"embedding".to_string()));

    // Verify it shows up in list
    let indexes = index::list_indexes(&pool, &col_name).await.unwrap();
    let vector_idx = indexes.iter().find(|i| i.index_type == "vector");
    assert!(vector_idx.is_some(), "expected vector index in list");

    // Detect it
    let metric = index::detect_vector_index(&pool, &col_name).await.unwrap();
    assert_eq!(metric, Some(VectorMetric::Cosine));

    // Drop it
    index::drop_index(&pool, &idx.id).await.unwrap();

    // Verify it's gone
    let metric_after = index::detect_vector_index(&pool, &col_name).await.unwrap();
    assert_eq!(metric_after, None);

    // Cleanup
    crud::drop_collection(&pool, &col_name, true).await.unwrap();
}

#[tokio::test]
async fn test_detect_vector_index_none() {
    let Some(pool) = test_pool() else { return };

    // persephone_tasks should not have a vector index
    let metric = index::detect_vector_index(&pool, "persephone_tasks")
        .await
        .unwrap();
    assert_eq!(metric, None);
}
