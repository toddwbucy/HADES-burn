//! Pipeline orchestrator — the core extract → chunk → embed → store flow.

use std::path::Path;
use std::time::Instant;

use serde_json::{json, Value};
use tracing::{debug, error, info, instrument, warn};

use crate::chunking::{ChunkingStrategy, TextChunk};
use crate::db::collections::CollectionProfile;
use crate::db::crud;
use crate::db::keys;
use crate::db::query::{self, ExecutionTarget};
use crate::db::ArangoPool;
use crate::persephone::embedding::{EmbedResult, EmbeddingClient, EmbeddingError};
use crate::persephone::extraction::{
    ExtractOptions, ExtractResult, ExtractionClient, ExtractionError,
};

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Collection profile to store results in.
    pub profile: &'static CollectionProfile,
    /// Embedding task parameter (e.g. "retrieval.passage").
    pub embed_task: String,
    /// Embedding batch size (None = server default).
    pub embed_batch_size: Option<u32>,
    /// Extraction options.
    pub extract_options: ExtractOptions,
    /// Whether to overwrite existing documents.
    pub overwrite: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            profile: CollectionProfile::default_profile(),
            embed_task: "retrieval.passage".to_string(),
            embed_batch_size: None,
            extract_options: ExtractOptions::all(),
            overwrite: true,
        }
    }
}

/// Error type for pipeline operations.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("extraction failed: {0}")]
    Extraction(#[from] ExtractionError),

    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("database error: {0}")]
    Database(#[from] crate::db::ArangoError),

    #[error("chunking produced no content")]
    EmptyChunks,

    #[error("pipeline error: {0}")]
    Other(String),
}

/// Result of processing a single document.
#[derive(Debug)]
pub struct DocumentResult {
    /// Document key (normalized from source identifier).
    pub doc_key: String,
    /// Whether processing succeeded.
    pub success: bool,
    /// Number of chunks produced.
    pub chunk_count: usize,
    /// Wall-clock time for this document.
    pub duration_ms: u64,
    /// Error message if processing failed.
    pub error: Option<String>,
}

/// Summary of a batch pipeline run.
#[derive(Debug)]
pub struct PipelineSummary {
    /// Per-document results.
    pub results: Vec<DocumentResult>,
    /// Total documents processed.
    pub total: usize,
    /// Successful documents.
    pub succeeded: usize,
    /// Failed documents.
    pub failed: usize,
    /// Total wall-clock time in milliseconds.
    pub total_duration_ms: u64,
}

impl PipelineSummary {
    /// Build a summary from a list of document results.
    pub fn from_results(results: Vec<DocumentResult>, total_duration_ms: u64) -> Self {
        let total = results.len();
        let succeeded = results.iter().filter(|r| r.success).count();
        Self {
            total,
            succeeded,
            failed: total - succeeded,
            results,
            total_duration_ms,
        }
    }
}

/// The document processing pipeline.
///
/// Orchestrates extraction, chunking, embedding, and storage for
/// documents flowing through the HADES knowledge graph system.
pub struct Pipeline {
    extractor: ExtractionClient,
    embedder: EmbeddingClient,
    db: ArangoPool,
    config: PipelineConfig,
}

impl Pipeline {
    /// Create a new pipeline with the given service clients and config.
    pub fn new(
        extractor: ExtractionClient,
        embedder: EmbeddingClient,
        db: ArangoPool,
        config: PipelineConfig,
    ) -> Self {
        Self {
            extractor,
            embedder,
            db,
            config,
        }
    }

    /// Process a single document through the full pipeline.
    ///
    /// Extracts content, chunks text, embeds chunks, and stores everything
    /// in the configured ArangoDB collections.
    #[instrument(skip(self, chunker), fields(doc_id))]
    pub async fn process_document(
        &self,
        file_path: &Path,
        doc_id: &str,
        chunker: &(dyn ChunkingStrategy + Send + Sync),
    ) -> DocumentResult {
        let start = Instant::now();
        let doc_key = keys::normalize_document_key(doc_id);

        match self.process_inner(file_path, &doc_key, chunker).await {
            Ok(chunk_count) => {
                let duration = start.elapsed().as_millis() as u64;
                info!(doc_key, chunk_count, duration_ms = duration, "document processed");
                DocumentResult {
                    doc_key,
                    success: true,
                    chunk_count,
                    duration_ms: duration,
                    error: None,
                }
            }
            Err(e) => {
                let duration = start.elapsed().as_millis() as u64;
                error!(doc_key, error = %e, "document processing failed");
                DocumentResult {
                    doc_key,
                    success: false,
                    chunk_count: 0,
                    duration_ms: duration,
                    error: Some(e.to_string()),
                }
            }
        }
    }

    /// Process a batch of documents with two-phase GPU optimization.
    ///
    /// **Phase 1** — Extract all documents (VLM stays loaded on GPU).
    /// **Phase 2** — Chunk and embed all extracted content (embedder loaded).
    ///
    /// Per-document errors are isolated: one failure does not abort the batch.
    #[instrument(skip(self, chunker, documents), fields(batch_size = documents.len()))]
    pub async fn process_batch(
        &self,
        documents: &[(&Path, &str)],
        chunker: &(dyn ChunkingStrategy + Send + Sync),
    ) -> PipelineSummary {
        let batch_start = Instant::now();
        let mut results = Vec::with_capacity(documents.len());

        // -- Phase 1: Extract all documents --------------------------------
        info!(count = documents.len(), "phase 1: extracting documents");
        // (doc_key, extract_result, extraction_duration_ms)
        let mut extractions: Vec<Option<(String, ExtractResult, u64)>> =
            Vec::with_capacity(documents.len());

        for &(path, doc_id) in documents {
            let doc_key = keys::normalize_document_key(doc_id);
            let extract_start = Instant::now();
            match self
                .extractor
                .extract_file(path, self.config.extract_options.clone())
                .await
            {
                Ok(result) => {
                    let extraction_ms = extract_start.elapsed().as_millis() as u64;
                    debug!(doc_key, text_len = result.full_text.len(), extraction_ms, "extracted");
                    extractions.push(Some((doc_key, result, extraction_ms)));
                }
                Err(e) => {
                    let extraction_ms = extract_start.elapsed().as_millis() as u64;
                    error!(doc_key, error = %e, "extraction failed");
                    results.push(DocumentResult {
                        doc_key,
                        success: false,
                        chunk_count: 0,
                        duration_ms: extraction_ms,
                        error: Some(format!("extraction: {e}")),
                    });
                    extractions.push(None);
                }
            }
        }

        // -- Phase 2: Chunk + Embed + Store --------------------------------
        info!("phase 2: chunking, embedding, and storing");
        for extraction in extractions {
            let Some((doc_key, extract_result, extraction_ms)) = extraction else {
                continue; // already recorded as failed
            };

            let phase2_start = Instant::now();
            match self
                .chunk_embed_store(&doc_key, &extract_result, chunker)
                .await
            {
                Ok(chunk_count) => {
                    let duration = extraction_ms + phase2_start.elapsed().as_millis() as u64;
                    info!(doc_key, chunk_count, duration_ms = duration, "stored");
                    results.push(DocumentResult {
                        doc_key,
                        success: true,
                        chunk_count,
                        duration_ms: duration,
                        error: None,
                    });
                }
                Err(e) => {
                    let duration = extraction_ms + phase2_start.elapsed().as_millis() as u64;
                    error!(doc_key, error = %e, "chunk/embed/store failed");
                    results.push(DocumentResult {
                        doc_key,
                        success: false,
                        chunk_count: 0,
                        duration_ms: duration,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let total_duration = batch_start.elapsed().as_millis() as u64;
        let summary = PipelineSummary::from_results(results, total_duration);
        info!(
            total = summary.total,
            succeeded = summary.succeeded,
            failed = summary.failed,
            duration_ms = summary.total_duration_ms,
            "batch complete"
        );
        summary
    }

    // -----------------------------------------------------------------------
    // Internal pipeline steps
    // -----------------------------------------------------------------------

    /// Full single-document pipeline: extract → chunk → embed → store.
    async fn process_inner(
        &self,
        file_path: &Path,
        doc_key: &str,
        chunker: &(dyn ChunkingStrategy + Send + Sync),
    ) -> Result<usize, PipelineError> {
        // 1. Extract
        let extract_result = self
            .extractor
            .extract_file(file_path, self.config.extract_options.clone())
            .await?;

        // 2-4. Chunk, embed, store
        self.chunk_embed_store(doc_key, &extract_result, chunker)
            .await
    }

    /// Chunk extracted text, embed chunks, store in ArangoDB.
    async fn chunk_embed_store(
        &self,
        doc_key: &str,
        extract_result: &ExtractResult,
        chunker: &(dyn ChunkingStrategy + Send + Sync),
    ) -> Result<usize, PipelineError> {
        // 2. Chunk
        let chunks = chunker.chunk(&extract_result.full_text);
        if chunks.is_empty() {
            warn!(doc_key, "chunking produced no text chunks");
            return Err(PipelineError::EmptyChunks);
        }

        // 3. Embed
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embed_result = self
            .embedder
            .embed(&texts, &self.config.embed_task, self.config.embed_batch_size)
            .await?;

        if embed_result.embeddings.len() != chunks.len() {
            return Err(PipelineError::Other(format!(
                "embedding count mismatch: expected {} (chunks), got {} (embeddings)",
                chunks.len(),
                embed_result.embeddings.len()
            )));
        }

        // 4. Store
        self.store(doc_key, extract_result, &chunks, &embed_result)
            .await?;

        Ok(chunks.len())
    }

    /// Store metadata, chunks, and embeddings in ArangoDB.
    async fn store(
        &self,
        doc_key: &str,
        extract_result: &ExtractResult,
        chunks: &[TextChunk],
        embed_result: &EmbedResult,
    ) -> Result<(), PipelineError> {
        let profile = self.config.profile;

        // -- Delete stale chunks/embeddings for this doc_key -----------------
        // When overwriting, a rerun with fewer chunks would leave orphans.
        if self.config.overwrite {
            self.delete_doc_chunks(profile, doc_key).await?;
        }

        // -- Metadata document ---------------------------------------------
        let metadata_doc = json!({
            "_key": doc_key,
            "full_text": extract_result.full_text,
            "tables": extract_result.tables.len(),
            "equations": extract_result.equations.len(),
            "images": extract_result.images.len(),
            "chunk_count": chunks.len(),
            "embedding_model": embed_result.model,
            "embedding_dimension": embed_result.dimension,
            "extractor_metadata": extract_result.metadata,
        });

        let meta_res = crud::insert_documents(
            &self.db,
            profile.metadata,
            &[metadata_doc],
            self.config.overwrite,
        )
        .await?;
        if meta_res.errors > 0 {
            return Err(PipelineError::Other(format!(
                "metadata import had {} errors",
                meta_res.errors
            )));
        }

        // -- Chunk documents -----------------------------------------------
        let chunk_docs: Vec<Value> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                json!({
                    "_key": keys::chunk_key(doc_key, i),
                    "doc_key": doc_key,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                })
            })
            .collect();

        let chunk_res = crud::insert_documents(
            &self.db,
            profile.chunks,
            &chunk_docs,
            self.config.overwrite,
        )
        .await?;
        if chunk_res.errors > 0 {
            return Err(PipelineError::Other(format!(
                "chunk import had {} errors (created {})",
                chunk_res.errors, chunk_res.created
            )));
        }

        // -- Embedding documents -------------------------------------------
        let embedding_docs: Vec<Value> = embed_result
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let ck = keys::chunk_key(doc_key, i);
                json!({
                    "_key": keys::embedding_key(&ck),
                    "chunk_key": ck,
                    "doc_key": doc_key,
                    "embedding": emb,
                })
            })
            .collect();

        let emb_res = crud::insert_documents(
            &self.db,
            profile.embeddings,
            &embedding_docs,
            self.config.overwrite,
        )
        .await?;
        if emb_res.errors > 0 {
            return Err(PipelineError::Other(format!(
                "embedding import had {} errors (created {})",
                emb_res.errors, emb_res.created
            )));
        }

        debug!(
            doc_key,
            chunks = chunks.len(),
            "stored metadata, chunks, and embeddings"
        );
        Ok(())
    }

    /// Delete existing chunk and embedding documents for a doc_key.
    ///
    /// Prevents stale rows when a rerun produces fewer chunks than before.
    async fn delete_doc_chunks(
        &self,
        profile: &CollectionProfile,
        doc_key: &str,
    ) -> Result<(), PipelineError> {
        let bind_vars = json!({
            "doc_key": doc_key,
            "@chunks": profile.chunks,
            "@embeddings": profile.embeddings,
        });

        // Delete chunks by doc_key
        query::query(
            &self.db,
            "FOR c IN @@chunks FILTER c.doc_key == @doc_key REMOVE c IN @@chunks",
            Some(&bind_vars),
            None,
            false,
            ExecutionTarget::Writer,
        )
        .await?;

        // Delete embeddings by doc_key
        query::query(
            &self.db,
            "FOR e IN @@embeddings FILTER e.doc_key == @doc_key REMOVE e IN @@embeddings",
            Some(&bind_vars),
            None,
            false,
            ExecutionTarget::Writer,
        )
        .await?;

        debug!(doc_key, "deleted stale chunks and embeddings");
        Ok(())
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("profile", &self.config.profile)
            .field("embed_task", &self.config.embed_task)
            .finish()
    }
}
