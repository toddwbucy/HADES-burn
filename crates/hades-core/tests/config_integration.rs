//! Integration test: load hades.yaml test fixture
//! and verify it parses correctly with matching values.

use std::path::PathBuf;

/// Resolve the path to the hades.yaml test fixture.
///
/// Uses HADES_TEST_CONFIG env var if set, otherwise uses the
/// fixture checked into the repo at tests/fixtures/hades.yaml.
fn fixture_yaml_path() -> PathBuf {
    if let Ok(p) = std::env::var("HADES_TEST_CONFIG") {
        return PathBuf::from(p);
    }
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set — must run via cargo test");
    PathBuf::from(manifest_dir)
        .join("tests/fixtures/hades.yaml")
}

/// Load and parse the test fixture into a HadesConfig.
fn load_fixture_config() -> hades_core::config::HadesConfig {
    let path = fixture_yaml_path();
    assert!(
        path.exists(),
        "hades.yaml fixture not found at {path:?}. \
         Set HADES_TEST_CONFIG env var to override the path."
    );
    let contents = std::fs::read_to_string(&path).expect("failed to read hades.yaml");
    serde_yaml::from_str(&contents).expect("failed to parse hades.yaml")
}

#[test]
fn load_fixture_hades_yaml() {
    let config = load_fixture_config();

    // Database
    assert_eq!(config.database.host, "localhost");
    assert_eq!(config.database.port, 8529);
    assert_eq!(config.database.name, "NestedLearning");
    assert_eq!(config.database.username, "root");
    assert_eq!(
        config.database.sockets.readonly.as_deref(),
        Some("/run/arangodb3/arangodb.sock")
    );
    assert_eq!(
        config.database.sockets.readwrite.as_deref(),
        Some("/run/arangodb3/arangodb.sock")
    );

    // Embedding service — OpenAI-compatible HTTP endpoint (vLLM-style).
    assert_eq!(
        config.embedding.service.socket,
        "http://localhost:8000/v1"
    );
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

    // Batch processing
    assert_eq!(config.batch_processing.concurrency, 1);
    assert_eq!(config.batch_processing.progress_interval_secs, 1.0);
    assert_eq!(config.batch_processing.rate_limit_rps, 0.0);
    assert_eq!(config.batch_processing.rate_limit_retries, 3);

    // Logging
    assert_eq!(config.logging.level, "INFO");
}

#[test]
fn load_fixture_yaml_then_apply_overrides() {
    let mut config = load_fixture_config();

    // Simulate --db bident_burn
    config.apply_cli_overrides(Some("bident_burn"), None);
    assert_eq!(config.effective_database(), "bident_burn");

    // Other values unchanged
    assert_eq!(config.database.host, "localhost");
    assert_eq!(config.embedding.model.dimension, 2048);
}
