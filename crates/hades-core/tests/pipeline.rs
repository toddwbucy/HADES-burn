//! Tests for the pipeline orchestrator types and configuration.
//!
//! Integration tests require running Persephone services and ArangoDB.
//! These tests validate types, config defaults, and summary logic.

use hades_core::pipeline::{DocumentResult, PipelineConfig, PipelineSummary};

#[test]
fn test_pipeline_config_defaults() {
    let config = PipelineConfig::default();
    assert_eq!(config.embed_task, "retrieval.passage");
    assert!(config.embed_batch_size.is_none());
    assert!(config.overwrite);
    assert_eq!(config.profile.metadata, "arxiv_metadata");
}

#[test]
fn test_document_result_success() {
    let result = DocumentResult {
        doc_key: "2501_12345".to_string(),
        success: true,
        chunk_count: 10,
        duration_ms: 500,
        error: None,
    };
    assert!(result.success);
    assert_eq!(result.chunk_count, 10);
    assert!(result.error.is_none());
}

#[test]
fn test_document_result_failure() {
    let result = DocumentResult {
        doc_key: "bad_doc".to_string(),
        success: false,
        chunk_count: 0,
        duration_ms: 100,
        error: Some("extraction: connection error".to_string()),
    };
    assert!(!result.success);
    assert!(result.error.is_some());
}

#[test]
fn test_pipeline_summary_counts() {
    let results = vec![
        DocumentResult {
            doc_key: "doc_a".to_string(),
            success: true,
            chunk_count: 5,
            duration_ms: 200,
            error: None,
        },
        DocumentResult {
            doc_key: "doc_b".to_string(),
            success: false,
            chunk_count: 0,
            duration_ms: 50,
            error: Some("failed".to_string()),
        },
        DocumentResult {
            doc_key: "doc_c".to_string(),
            success: true,
            chunk_count: 8,
            duration_ms: 300,
            error: None,
        },
    ];

    let summary = PipelineSummary {
        total: 3,
        succeeded: 2,
        failed: 1,
        results,
        total_duration_ms: 1000,
    };

    assert_eq!(summary.total, 3);
    assert_eq!(summary.succeeded, 2);
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.results.len(), 3);
}

#[test]
fn test_collection_profile_routing() {
    use hades_core::db::collections::CollectionProfile;

    let arxiv = CollectionProfile::get("arxiv").unwrap();
    let config = PipelineConfig {
        profile: arxiv,
        ..PipelineConfig::default()
    };
    assert_eq!(config.profile.metadata, "arxiv_metadata");
    assert_eq!(config.profile.chunks, "arxiv_abstract_chunks");
    assert_eq!(config.profile.embeddings, "arxiv_abstract_embeddings");

    let sync = CollectionProfile::get("sync").unwrap();
    let config = PipelineConfig {
        profile: sync,
        ..PipelineConfig::default()
    };
    assert_eq!(config.profile.metadata, "arxiv_papers");
}
