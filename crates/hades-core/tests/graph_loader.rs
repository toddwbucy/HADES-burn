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
use hades_core::graph::{self, EDGE_COLLECTION_NAMES, JINA_DIM, NUM_RELATIONS};
use tracing::warn;

fn arango_socket() -> PathBuf {
    PathBuf::from(
        std::env::var("ARANGO_SOCKET")
            .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
    )
}

fn arango_password() -> String {
    std::env::var("ARANGO_PASSWORD").expect("ARANGO_PASSWORD must be set for integration tests.")
}

/// Pool targeting NestedLearning (read-only — no writes).
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
        ArangoClient::with_socket(socket, "NestedLearning", "root", &arango_password());
    Some(ArangoPool::new(client.clone(), client))
}

#[tokio::test]
async fn test_load_full_graph() {
    let Some(pool) = nl_pool() else { return };

    let (graph, id_map) = graph::load(&pool).await.expect("graph load failed");

    // Structural invariants
    assert!(graph.num_nodes > 0, "graph should have nodes");
    assert!(graph.num_edges > 0, "graph should have edges");
    assert_eq!(graph.num_relations, NUM_RELATIONS);
    assert_eq!(graph.feature_dim, JINA_DIM);

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
    assert_eq!(graph.collection_names, sorted, "collection_names must be sorted");

    // Some nodes should have embeddings (NestedLearning has real data)
    assert!(
        graph.embedded_count() > 0,
        "expected some nodes with Jina embeddings"
    );

    // All edge relation types should be valid indices
    for &rel in &graph.edge_type {
        assert!(
            (rel as usize) < NUM_RELATIONS,
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
    eprintln!(
        "\n=== Graph Load Summary ===\n\
         Nodes:        {}\n\
         Edges:        {}\n\
         Relations:    {}\n\
         Collections:  {}\n\
         Embedded:     {}/{} ({:.1}%)\n",
        graph.num_nodes,
        graph.num_edges,
        graph.num_relations,
        graph.collection_names.len(),
        graph.embedded_count(),
        graph.num_nodes,
        graph.embedding_coverage() * 100.0,
    );
}

#[test]
fn test_edge_collection_names_match_schema() {
    // Verify the edge collection names used by the loader match the schema constant
    assert_eq!(EDGE_COLLECTION_NAMES.len(), NUM_RELATIONS);

    // First collection should be nl_axiom_basis_edges (index 0)
    assert_eq!(EDGE_COLLECTION_NAMES[0], "nl_axiom_basis_edges");
    // Last collection should be nl_smell_spec_edges (index 21)
    assert_eq!(EDGE_COLLECTION_NAMES[21], "nl_smell_spec_edges");
}

#[tokio::test]
async fn test_idmap_collection_grouping() {
    let Some(pool) = nl_pool() else { return };

    let (_graph, id_map) = graph::load(&pool).await.expect("graph load failed");

    let groups = id_map.nodes_by_collection();

    // Every node should be in exactly one group
    let total: usize = groups.values().map(|v| v.len()).sum();
    assert_eq!(total, id_map.len());

    // Every node should have a valid collection prefix
    for (col, nodes) in &groups {
        assert!(!col.is_empty(), "collection name should not be empty");
        for (arango_id, _) in nodes {
            assert!(
                arango_id.starts_with(col),
                "node {arango_id} should start with collection {col}"
            );
        }
    }
}
