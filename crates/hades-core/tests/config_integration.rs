//! Integration test: load the actual production hades.yaml
//! and verify it parses correctly with matching values.

use std::path::PathBuf;

/// Resolve the path to production hades.yaml.
///
/// Uses HADES_TEST_CONFIG env var if set, otherwise falls back to
/// the known location relative to the HADES repo.
fn production_yaml_path() -> PathBuf {
    if let Ok(p) = std::env::var("HADES_TEST_CONFIG") {
        return PathBuf::from(p);
    }
    // Default: known location on this machine
    PathBuf::from("/home/todd/git/HADES/core/config/hades.yaml")
}

#[test]
fn load_production_hades_yaml() {
    let path = production_yaml_path();
    assert!(
        path.exists(),
        "production hades.yaml not found at {path:?}. \
         Set HADES_TEST_CONFIG env var to override the path."
    );

    let contents = std::fs::read_to_string(&path).expect("failed to read hades.yaml");
    let config: hades_core::config::HadesConfig =
        serde_yaml::from_str(&contents).expect("failed to parse hades.yaml");

    // Database
    assert_eq!(config.database.host, "localhost");
    assert_eq!(config.database.port, 8529);
    assert_eq!(config.database.name, "NestedLearning");
    assert_eq!(config.database.username, "root");
    assert_eq!(
        config.database.sockets.readonly.as_deref(),
        Some("/run/hades/readonly/arangod.sock")
    );
    assert_eq!(
        config.database.sockets.readwrite.as_deref(),
        Some("/run/hades/readwrite/arangod.sock")
    );

    // Embedding service
    assert_eq!(config.embedding.service.socket, "/run/hades/embedder.sock");
    assert!(config.embedding.service.fallback_to_local);
    assert_eq!(config.embedding.service.timeout_ms, 30000);
    assert_eq!(config.embedding.service.idle_timeout, 0);

    // Embedding model
    assert_eq!(config.embedding.model.name, "jinaai/jina-embeddings-v4");
    assert_eq!(config.embedding.model.dimension, 2048);
    assert_eq!(config.embedding.model.max_tokens, 32768);
    assert!(config.embedding.model.use_fp16);
    assert!(config.embedding.model.normalize);

    // Batch
    assert_eq!(config.embedding.batch.size, 48);
    assert_eq!(config.embedding.batch.size_small, 8);

    // Chunking
    assert_eq!(config.embedding.chunking.size_tokens, 500);
    assert_eq!(config.embedding.chunking.overlap_tokens, 200);

    // GPU
    assert_eq!(config.gpu.device, "cuda:2");
    assert!(config.gpu.enabled);

    // Vector index
    assert_eq!(config.vector_index.default_n_probe, 10);
    assert_eq!(config.vector_index.metric, "cosine");
    assert!(config.vector_index.auto_n_lists);

    // Search
    assert_eq!(config.search.limit, 10);
    assert_eq!(config.search.max_limit, 100);
    assert_eq!(config.search.hybrid.vector_weight, 0.7);
    assert_eq!(config.search.hybrid.keyword_weight, 0.3);

    // Rocchio
    assert_eq!(config.rocchio.alpha, 1.0);
    assert_eq!(config.rocchio.beta, 0.75);
    assert_eq!(config.rocchio.gamma, 0.15);

    // Sync
    assert_eq!(config.sync.default_lookback_days, 7);
    assert_eq!(config.sync.batch_size, 8);
    assert_eq!(config.sync.max_results, 1000);

    // ArXiv paths
    assert_eq!(
        config.arxiv.pdf_base_path.to_str().unwrap(),
        "/bulk-store/arxiv-data/pdf"
    );
    assert_eq!(
        config.arxiv.latex_base_path.to_str().unwrap(),
        "/bulk-store/arxiv-data/src"
    );

    // Logging
    assert_eq!(config.logging.level, "INFO");
}

#[test]
fn load_production_yaml_then_apply_overrides() {
    let path = production_yaml_path();
    assert!(
        path.exists(),
        "production hades.yaml not found at {path:?}. \
         Set HADES_TEST_CONFIG env var to override the path."
    );

    let contents = std::fs::read_to_string(&path).unwrap();
    let mut config: hades_core::config::HadesConfig =
        serde_yaml::from_str(&contents).unwrap();

    // Simulate --db bident_burn
    config.apply_cli_overrides(Some("bident_burn"), None);
    assert_eq!(config.effective_database(), "bident_burn");

    // Other values unchanged
    assert_eq!(config.database.host, "localhost");
    assert_eq!(config.embedding.model.dimension, 2048);
}
