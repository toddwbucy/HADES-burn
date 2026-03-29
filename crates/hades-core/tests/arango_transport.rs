//! Integration tests for ArangoDB transport layer.
//!
//! These tests require a running ArangoDB instance accessible via
//! Unix socket at /run/arangodb3/arangodb.sock.
//!
//! Set ARANGO_PASSWORD env var to authenticate.
//! Tests only perform read-only operations — no data is modified.

use std::path::PathBuf;

fn arango_socket() -> PathBuf {
    PathBuf::from(
        std::env::var("ARANGO_SOCKET")
            .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
    )
}

fn arango_password() -> String {
    std::env::var("ARANGO_PASSWORD").expect(
        "ARANGO_PASSWORD must be set for integration tests. \
         These tests are read-only and safe to run against any database.",
    )
}

/// Verify we can connect and get the server version.
#[tokio::test]
async fn get_server_version() {
    let socket = arango_socket();
    if !socket.exists() {
        eprintln!("skipping: ArangoDB socket not found at {}", socket.display());
        return;
    }

    let client = hades_core::db::ArangoClient::with_socket(
        socket,
        "_system",
        "root",
        &arango_password(),
    );

    // /_api/version doesn't need a database prefix, but our client
    // always prepends /_db/{db}/ — _system works for version endpoint
    let result = client.get("version").await.unwrap();
    assert!(result.get("server").is_some(), "response: {result}");
    assert_eq!(result["server"], "arango");
}

/// Verify we can list databases (read-only).
#[tokio::test]
async fn list_databases() {
    let socket = arango_socket();
    if !socket.exists() {
        eprintln!("skipping: ArangoDB socket not found at {}", socket.display());
        return;
    }

    let client = hades_core::db::ArangoClient::with_socket(
        socket,
        "_system",
        "root",
        &arango_password(),
    );

    let result = client.get("database").await.unwrap();
    let databases = result["result"].as_array().expect("expected array of databases");

    // _system should always exist
    let names: Vec<&str> = databases.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"_system"), "databases: {names:?}");
    // bident_burn should exist (our project management DB)
    assert!(names.contains(&"bident_burn"), "databases: {names:?}");
}

/// Verify we can query bident_burn collections (read-only).
#[tokio::test]
async fn list_bident_burn_collections() {
    let socket = arango_socket();
    if !socket.exists() {
        eprintln!("skipping: ArangoDB socket not found at {}", socket.display());
        return;
    }

    let client = hades_core::db::ArangoClient::with_socket(
        socket,
        "bident_burn",
        "root",
        &arango_password(),
    );

    let result = client.get("collection").await.unwrap();
    let collections = result["result"].as_array().expect("expected array");

    let names: Vec<&str> = collections
        .iter()
        .filter_map(|c| c["name"].as_str())
        .filter(|n| !n.starts_with('_')) // skip system collections
        .collect();

    assert!(
        names.contains(&"persephone_tasks"),
        "collections: {names:?}"
    );
}

/// Verify that a 404 returns a proper ArangoError.
#[tokio::test]
async fn get_nonexistent_document_returns_error() {
    let socket = arango_socket();
    if !socket.exists() {
        eprintln!("skipping: ArangoDB socket not found at {}", socket.display());
        return;
    }

    let client = hades_core::db::ArangoClient::with_socket(
        socket,
        "bident_burn",
        "root",
        &arango_password(),
    );

    let result = client.get("document/persephone_tasks/nonexistent_key_xyz").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.is_not_found(),
        "expected NotFound, got: {err}"
    );
}

/// Verify AQL query execution (read-only).
#[tokio::test]
async fn execute_aql_query() {
    let socket = arango_socket();
    if !socket.exists() {
        eprintln!("skipping: ArangoDB socket not found at {}", socket.display());
        return;
    }

    let client = hades_core::db::ArangoClient::with_socket(
        socket,
        "bident_burn",
        "root",
        &arango_password(),
    );

    let body = serde_json::json!({
        "query": "FOR t IN persephone_tasks LIMIT 3 RETURN t._key",
        "batchSize": 10
    });

    let result = client.post("cursor", &body).await.unwrap();
    let keys = result["result"].as_array().expect("expected result array");
    assert!(!keys.is_empty(), "expected at least one task key");
}
