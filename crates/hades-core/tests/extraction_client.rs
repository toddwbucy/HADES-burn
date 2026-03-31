//! Tests for the Persephone extraction client.
//!
//! Connection tests require a running Persephone extraction service.
//! Type and config tests run without a service.

use std::path::PathBuf;

use hades_core::persephone::extraction::{
    ExtractOptions, ExtractResult, ExtractionClientConfig, ExtractionEndpoint,
};
use hades_proto::extraction::SourceType;

#[test]
fn test_config_defaults() {
    let config = ExtractionClientConfig::default();
    match &config.endpoint {
        ExtractionEndpoint::Unix(path) => {
            assert_eq!(path, &PathBuf::from("/run/hades/extractor.sock"));
        }
        ExtractionEndpoint::Tcp(_) => panic!("expected Unix endpoint by default"),
    }
    assert_eq!(config.timeout.as_secs(), 600);
    assert_eq!(config.connect_timeout.as_secs(), 10);
}

#[test]
fn test_endpoint_variants() {
    let unix = ExtractionEndpoint::Unix(PathBuf::from("/tmp/test.sock"));
    let tcp = ExtractionEndpoint::Tcp("http://localhost:50052".to_string());

    assert!(matches!(unix, ExtractionEndpoint::Unix(_)));
    assert!(matches!(tcp, ExtractionEndpoint::Tcp(_)));
}

#[test]
fn test_extract_options_default() {
    let opts = ExtractOptions::default();
    assert!(!opts.extract_tables);
    assert!(!opts.extract_equations);
    assert!(!opts.extract_images);
    assert!(!opts.use_ocr);
    assert!(opts.source_type.is_none());
}

#[test]
fn test_extract_options_all() {
    let opts = ExtractOptions::all();
    assert!(opts.extract_tables);
    assert!(opts.extract_equations);
    assert!(opts.extract_images);
    assert!(!opts.use_ocr);
    assert!(opts.source_type.is_none());
}

#[test]
fn test_extract_result_struct() {
    let result = ExtractResult {
        full_text: "Hello world".to_string(),
        tables: vec![],
        equations: vec![],
        images: vec![],
        metadata: std::collections::HashMap::new(),
        source_type: SourceType::Pdf,
    };
    assert_eq!(result.full_text, "Hello world");
    assert!(result.tables.is_empty());
    assert!(matches!(result.source_type, SourceType::Pdf));
}

#[test]
fn test_source_type_variants() {
    // Verify all proto enum variants are accessible
    let _unknown = SourceType::Unknown;
    let _pdf = SourceType::Pdf;
    let _latex = SourceType::Latex;
    let _code = SourceType::Code;
    let _markdown = SourceType::Markdown;
    let _text = SourceType::Text;
}

#[tokio::test]
async fn test_connect_nonexistent_socket_fails() {
    use hades_core::persephone::extraction::ExtractionClient;

    let dir = tempfile::tempdir().expect("failed to create tempdir");
    let socket_path = dir.path().join("nonexistent.sock");

    let config = ExtractionClientConfig {
        endpoint: ExtractionEndpoint::Unix(socket_path),
        connect_timeout: std::time::Duration::from_millis(500),
        ..Default::default()
    };

    let result = ExtractionClient::connect(config).await;
    assert!(result.is_err(), "expected connection error for nonexistent socket");
}
