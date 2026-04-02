//! Tests for the Persephone training client.
//!
//! Connection tests require a running Persephone training service.
//! Type and config tests run without a service.

use std::path::PathBuf;

use hades_core::persephone::training::{
    CheckpointResult, EmbeddingsResult, EvalResult, InitResult, LoadGraphResult, StepResult,
    TrainingClientConfig, TrainingEndpoint,
};

#[test]
fn test_config_defaults() {
    let config = TrainingClientConfig::default();
    match &config.endpoint {
        TrainingEndpoint::Unix(path) => {
            assert_eq!(path, &PathBuf::from("/run/hades/training.sock"));
        }
        TrainingEndpoint::Tcp(_) => panic!("expected Unix endpoint by default"),
    }
    assert_eq!(config.timeout.as_secs(), 60);
    assert_eq!(config.slow_timeout.as_secs(), 600);
    assert_eq!(config.connect_timeout.as_secs(), 10);
}

#[test]
fn test_result_types() {
    let init = InitResult {
        num_parameters: 1_000_000,
        device: "cuda:2".to_string(),
    };
    assert_eq!(init.num_parameters, 1_000_000);

    let load = LoadGraphResult {
        num_nodes: 50_000,
        num_edges: 200_000,
        feature_dim: 2048,
        gpu_memory_bytes: 512 * 1024 * 1024,
    };
    assert_eq!(load.feature_dim, 2048);

    let step = StepResult {
        loss: 0.693,
        accuracy: 0.5,
    };
    assert!(step.loss > 0.0);

    let eval = EvalResult {
        loss: 0.4,
        accuracy: 0.85,
        auc: 0.92,
    };
    assert!(eval.auc > eval.accuracy);

    let emb = EmbeddingsResult {
        num_nodes: 50_000,
        embed_dim: 128,
        embeddings: vec![],
        output_path: "/tmp/embeddings.safetensors".to_string(),
    };
    assert!(emb.embeddings.is_empty()); // file output mode

    let ckpt = CheckpointResult {
        path: "/tmp/model.pt".to_string(),
        size_bytes: 4 * 1024 * 1024,
    };
    assert!(ckpt.size_bytes > 0);
}

#[test]
fn test_endpoint_variants() {
    let unix = TrainingEndpoint::Unix(PathBuf::from("/run/hades/training.sock"));
    let tcp = TrainingEndpoint::Tcp("http://localhost:50052".to_string());

    assert!(matches!(unix, TrainingEndpoint::Unix(_)));
    assert!(matches!(tcp, TrainingEndpoint::Tcp(_)));
}

#[tokio::test]
async fn test_connect_nonexistent_socket_fails() {
    use hades_core::persephone::training::TrainingClient;

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let socket_path = dir.path().join("nonexistent.sock");

    let config = TrainingClientConfig {
        endpoint: TrainingEndpoint::Unix(socket_path),
        connect_timeout: std::time::Duration::from_millis(500),
        ..Default::default()
    };

    let result = TrainingClient::connect(config).await;
    assert!(result.is_err(), "expected connection error for nonexistent socket");
}
