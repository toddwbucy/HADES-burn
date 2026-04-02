//! Verify generated protobuf types compile and are accessible.

use hades_proto::common::{ChunkMetadata, ChunkingStrategy};
use hades_proto::embedding::{EmbedRequest, EmbedResponse, Embedding, ProviderInfo};
use hades_proto::extraction::{
    ExtractRequest, ExtractResponse, ExtractorInfo, SourceType, Table,
};
use hades_proto::training::{
    CheckpointRequest, EvaluateRequest, GetEmbeddingsRequest, InitModelRequest,
    LoadCheckpointRequest, LoadGraphRequest, ModelConfig, OptimizerConfig, TrainStepRequest,
};

#[test]
fn test_common_types() {
    let meta = ChunkMetadata {
        chunk_index: 0,
        total_chunks: 5,
        start_char: 0,
        end_char: 500,
    };
    assert_eq!(meta.chunk_index, 0);
    assert_eq!(meta.total_chunks, 5);

    // Enum values
    assert_eq!(ChunkingStrategy::Late as i32, 0);
    assert_eq!(ChunkingStrategy::Semantic as i32, 1);
    assert_eq!(ChunkingStrategy::Sliding as i32, 2);
    assert_eq!(ChunkingStrategy::Token as i32, 3);
}

#[test]
fn test_embedding_types() {
    let req = EmbedRequest {
        texts: vec!["hello world".to_string()],
        task: "retrieval.passage".to_string(),
        batch_size: 32,
    };
    assert_eq!(req.texts.len(), 1);
    assert_eq!(req.task, "retrieval.passage");

    let embedding = Embedding {
        values: vec![0.1, 0.2, 0.3],
    };
    assert_eq!(embedding.values.len(), 3);

    let resp = EmbedResponse {
        embeddings: vec![embedding],
        model: "jinaai/jina-embeddings-v4".to_string(),
        dimension: 2048,
        duration_ms: 42,
    };
    assert_eq!(resp.embeddings.len(), 1);
    assert_eq!(resp.dimension, 2048);

    let info = ProviderInfo {
        model_name: "test".to_string(),
        dimension: 2048,
        max_seq_length: 32768,
        supported_tasks: vec!["retrieval.passage".to_string()],
        device: "cuda:0".to_string(),
        model_loaded: true,
    };
    assert!(info.model_loaded);
}

#[test]
fn test_extraction_types() {
    let req = ExtractRequest {
        file_path: "/tmp/paper.pdf".to_string(),
        content: vec![],
        source_type: SourceType::Pdf.into(),
        extract_tables: true,
        extract_equations: true,
        extract_images: false,
        use_ocr: false,
    };
    assert_eq!(req.source_type, SourceType::Pdf as i32);
    assert!(req.extract_tables);

    let table = Table {
        content: "col1 | col2".to_string(),
        caption: "Table 1".to_string(),
        index: 0,
    };

    let resp = ExtractResponse {
        full_text: "Some extracted text".to_string(),
        tables: vec![table],
        equations: vec![],
        images: vec![],
        metadata: [("pages".to_string(), "10".to_string())].into(),
        source_type: SourceType::Pdf.into(),
    };
    assert_eq!(resp.tables.len(), 1);
    assert_eq!(resp.metadata.get("pages").unwrap(), "10");

    let info = ExtractorInfo {
        supported_extensions: vec![".pdf".to_string(), ".tex".to_string()],
        supported_types: vec![SourceType::Pdf.into(), SourceType::Latex.into()],
        features: vec!["ocr".to_string(), "tables".to_string()],
        gpu_available: true,
    };
    assert_eq!(info.supported_extensions.len(), 2);
}

#[test]
fn test_training_types() {
    let model_config = ModelConfig {
        num_relations: 22,
        num_collection_types: 62,
        hidden_dim: 256,
        embed_dim: 128,
        num_bases: 21,
        dropout: 0.2,
    };
    assert_eq!(model_config.num_relations, 22);
    assert_eq!(model_config.embed_dim, 128);

    let optimizer = OptimizerConfig {
        learning_rate: 0.01,
        weight_decay: 5e-4,
    };
    assert!(optimizer.learning_rate > 0.0);

    let init_req = InitModelRequest {
        model: Some(model_config),
        optimizer: Some(optimizer),
        device: "cuda:2".to_string(),
    };
    assert!(init_req.model.is_some());
    assert_eq!(init_req.device, "cuda:2");

    let load_req = LoadGraphRequest {
        safetensors_path: "/tmp/graph.safetensors".to_string(),
    };
    assert!(!load_req.safetensors_path.is_empty());

    let step_req = TrainStepRequest {
        train_edge_indices: vec![0, 1, 2, 3],
        neg_src: vec![5, 6],
        neg_dst: vec![7, 8],
    };
    assert_eq!(step_req.train_edge_indices.len(), 4);
    assert_eq!(step_req.neg_src.len(), step_req.neg_dst.len());

    let eval_req = EvaluateRequest {
        edge_indices: vec![10, 11],
        neg_src: vec![0],
        neg_dst: vec![1],
    };
    assert_eq!(eval_req.edge_indices.len(), 2);

    let emb_req = GetEmbeddingsRequest {
        output_path: "/tmp/embeddings.safetensors".to_string(),
    };
    assert!(!emb_req.output_path.is_empty());

    let ckpt_req = CheckpointRequest {
        path: "/tmp/model.pt".to_string(),
    };
    assert!(!ckpt_req.path.is_empty());

    let load_ckpt = LoadCheckpointRequest {
        path: "/tmp/model.pt".to_string(),
        device: "cuda:0".to_string(),
    };
    assert_eq!(load_ckpt.device, "cuda:0");
}
