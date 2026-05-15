//! Integration tests for the graph loader.
//!
//! Prerequisites:
//! - ArangoDB running with socket at /run/arangodb3/arangodb.sock
//! - ARANGO_PASSWORD environment variable set
//! - NestedLearning database with populated edge and vertex collections
//!
//! Tests are skipped gracefully when the socket is not available.

use std::path::PathBuf;

use hades_core::db::{ArangoClient, ArangoPool};
use hades_core::graph::{self, RuntimeSchema};
use tracing::warn;

/// Load the runtime schema for the integration tests.
///
/// Returns `None` if `hades_schema` is not seeded in the test database —
/// the test should skip gracefully in that case, matching the existing
/// pattern for missing-socket skips in [`nl_pool`].
async fn load_test_schema(pool: &ArangoPool) -> Option<RuntimeSchema> {
    match RuntimeSchema::load(pool).await {
        Ok(schema) => Some(schema),
        Err(e) => {
            warn!(error = %e, "skipping: hades_schema not available in test database");
            None
        }
    }
}

fn arango_socket() -> PathBuf {
    PathBuf::from(
        std::env::var("ARANGO_SOCKET")
            .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
    )
}

fn arango_user() -> String {
    std::env::var("ARANGO_USER").unwrap_or_else(|_| "root".to_string())
}

fn arango_password() -> String {
    std::env::var("ARANGO_PASSWORD").expect("ARANGO_PASSWORD must be set for integration tests.")
}

/// Pool targeting NestedLearning (read-only — no writes).
///
/// Uses `ARANGO_USER` env var (default: `"root"`). Set to a read-only
/// user (e.g. `"nl_readonly"`) to enforce least-privilege access.
fn nl_pool() -> Option<ArangoPool> {
    let socket = arango_socket();
    if !socket.exists() {
        if std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true") {
            panic!(
                "ARANGO_TESTS is set but socket not found at {}",
                socket.display()
            );
        }
        warn!(
            "skipping: ArangoDB socket not found at {}",
            socket.display()
        );
        return None;
    }

    let client =
        ArangoClient::with_socket(socket, "NestedLearning", &arango_user(), &arango_password());
    Some(ArangoPool::new(client.clone(), client))
}

#[tokio::test]
async fn test_load_full_graph() {
    let Some(pool) = nl_pool() else { return };

    let Some(schema) = load_test_schema(&pool).await else {
        return;
    };
    let (graph, id_map) = graph::load(&pool, &schema)
        .await
        .expect("graph load failed");

    // Structural invariants — graph dimensions must match the loaded schema.
    assert!(graph.num_nodes > 0, "graph should have nodes");
    assert!(graph.num_edges > 0, "graph should have edges");
    assert_eq!(graph.num_relations, schema.meta.num_relations);
    assert_eq!(graph.feature_dim, schema.meta.feature_dim);

    // Validate passes (already called internally, but double-check)
    graph.validate().unwrap();

    // Edge arrays match counts
    assert_eq!(graph.edge_src.len(), graph.num_edges);
    assert_eq!(graph.edge_dst.len(), graph.num_edges);
    assert_eq!(graph.edge_type.len(), graph.num_edges);

    // Node features are sized correctly
    assert_eq!(
        graph.node_features.len(),
        graph.num_nodes * graph.feature_dim
    );
    assert_eq!(graph.has_embedding.len(), graph.num_nodes);
    assert_eq!(graph.node_collections.len(), graph.num_nodes);

    // IDMap is consistent with graph
    assert_eq!(id_map.len(), graph.num_nodes);

    // Collection names are sorted and non-empty
    assert!(!graph.collection_names.is_empty());
    let mut sorted = graph.collection_names.clone();
    sorted.sort();
    assert_eq!(
        graph.collection_names, sorted,
        "collection_names must be sorted"
    );

    // Some nodes should have embeddings (NestedLearning has real data)
    assert!(
        graph.embedded_count() > 0,
        "expected some nodes with Jina embeddings"
    );

    // All edge relation types should be valid indices
    for &rel in &graph.edge_type {
        assert!(
            (rel as usize) < schema.meta.num_relations,
            "edge relation type {rel} out of bounds"
        );
    }

    // All edge node indices should be valid
    for (&src, &dst) in graph.edge_src.iter().zip(&graph.edge_dst) {
        assert!(
            (src as usize) < graph.num_nodes,
            "edge src {src} out of bounds"
        );
        assert!(
            (dst as usize) < graph.num_nodes,
            "edge dst {dst} out of bounds"
        );
    }

    // Print summary for manual inspection
    tracing::info!(
        num_nodes = graph.num_nodes,
        num_edges = graph.num_edges,
        num_relations = graph.num_relations,
        num_collections = graph.collection_names.len(),
        embedded = graph.embedded_count(),
        total = graph.num_nodes,
        coverage_pct = format_args!("{:.1}", graph.embedding_coverage() * 100.0),
        "graph load summary"
    );
}

#[tokio::test]
async fn test_idmap_collection_grouping() {
    let Some(pool) = nl_pool() else { return };

    let Some(schema) = load_test_schema(&pool).await else {
        return;
    };
    let (_graph, id_map) = graph::load(&pool, &schema)
        .await
        .expect("graph load failed");

    let groups = id_map.nodes_by_collection();

    // Every node should be in exactly one group
    let total: usize = groups.values().map(|v| v.len()).sum();
    assert_eq!(total, id_map.len());

    // Every node's _id prefix (before '/') must exactly match its group key
    for (col, nodes) in &groups {
        assert!(!col.is_empty(), "collection name should not be empty");
        for (arango_id, _) in nodes {
            let (prefix, _key) = arango_id
                .split_once('/')
                .unwrap_or_else(|| panic!("arango_id {arango_id} missing '/' separator"));
            assert_eq!(
                prefix, *col,
                "node {arango_id} has prefix {prefix}, expected {col}"
            );
        }
    }
}
