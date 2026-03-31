//! Tests for the Persephone embedding client.
//!
//! Connection tests require a running Persephone embedding service.
//! Type and config tests run without a service.

use std::path::PathBuf;

use hades_core::persephone::embedding::{
    EmbedResult, EmbeddingClientConfig, EmbeddingEndpoint,
};

#[test]
fn test_config_defaults() {
    let config = EmbeddingClientConfig::default();
    match &config.endpoint {
        EmbeddingEndpoint::Unix(path) => {
            assert_eq!(path, &PathBuf::from("/run/hades/embedder.sock"));
        }
        EmbeddingEndpoint::Tcp(_) => panic!("expected Unix endpoint by default"),
    }
    assert_eq!(config.timeout.as_secs(), 300);
    assert_eq!(config.connect_timeout.as_secs(), 10);
}

#[test]
fn test_embed_result_struct() {
    let result = EmbedResult {
        embeddings: vec![vec![0.1, 0.2, 0.3]],
        model: "test-model".to_string(),
        dimension: 3,
        duration_ms: 42,
    };
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 3);
    assert_eq!(result.dimension, 3);
}

#[test]
fn test_endpoint_variants() {
    let unix = EmbeddingEndpoint::Unix(PathBuf::from("/tmp/test.sock"));
    let tcp = EmbeddingEndpoint::Tcp("http://localhost:50051".to_string());

    // Just verify they construct without panic
    assert!(matches!(unix, EmbeddingEndpoint::Unix(_)));
    assert!(matches!(tcp, EmbeddingEndpoint::Tcp(_)));
}

#[tokio::test]
async fn test_connect_nonexistent_socket_fails() {
    use hades_core::persephone::embedding::EmbeddingClient;

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let socket_path = dir.path().join("nonexistent.sock");

    let config = EmbeddingClientConfig {
        endpoint: EmbeddingEndpoint::Unix(socket_path),
        connect_timeout: std::time::Duration::from_millis(500),
        ..Default::default()
    };

    let result = EmbeddingClient::connect(config).await;
    assert!(result.is_err(), "expected connection error for nonexistent socket");
}
