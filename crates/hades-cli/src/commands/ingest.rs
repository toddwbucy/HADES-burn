//! Native Rust implementation of the `hades ingest` command.
//!
//! Replaces the Python dispatch for document ingestion.  Supports:
//! - ArXiv paper ingest (download PDF + extract + chunk + embed + store)
//! - Local file ingest (extract + chunk + embed + store)
//! - Mixed inputs (arXiv IDs and file paths in one invocation)
//! - Batch mode with per-document error isolation
//! - Resumable checkpointing with progress reporting
//! - Bounded concurrency and rate limiting
//! - Custom metadata merging
//! - Collection profile selection
//! - Force re-processing (surgical delete + re-insert)

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{json, Value};
use tracing::{info, warn};

use hades_core::arxiv::{ArxivClient, is_arxiv_id, normalize_arxiv_id};
use hades_core::batch::{BatchProcessor, BatchProcessorConfig, RateLimiter};
use hades_core::chunking::{ChunkingStrategy, TokenChunking};
use hades_core::db::collections::CollectionProfile;
use hades_core::db::keys;
use hades_core::db::ArangoPool;
use hades_core::persephone::embedding::EmbeddingClient;
use hades_core::persephone::extraction::ExtractionClient;
use hades_core::pipeline::{Pipeline, PipelineConfig};
use hades_core::HadesConfig;

/// Code-file extensions for auto-detecting the `code` embedding task.
const CODE_EXTENSIONS: &[&str] = &[
    "py", "rs", "cu", "cuh", "cpp", "c", "h", "hpp", "js", "ts",
    "go", "java", "rb", "swift", "kt",
];

/// Keys that user-provided metadata cannot override.
const PROTECTED_KEYS: &[&str] = &["_key", "status", "source", "arxiv_id"];

/// Classify a single input as either an arXiv ID or a file path.
enum InputKind {
    Arxiv(String),
    File(PathBuf),
}

/// Ingest command failed with partial results.
///
/// Returned instead of calling `process::exit` so callers control the exit code.
#[derive(Debug, thiserror::Error)]
#[error("{failed} of {total} documents failed to ingest")]
pub struct IngestFailure {
    pub total: usize,
    pub failed: usize,
}

/// Run the ingest command.
///
/// This is the entry point called from `main.rs` when the user runs
/// `hades-burn ingest ...`.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: &HadesConfig,
    inputs: Vec<PathBuf>,
    batch: bool,
    metadata_json: Option<&str>,
    _claims: &[String],
    collection: Option<&str>,
    force: bool,
    task: Option<&str>,
    id: Option<&str>,
    resume: bool,
    reset: bool,
    concurrency: Option<usize>,
) -> Result<()> {
    let cmd_start = Instant::now();

    // -- Validate inputs -------------------------------------------------------
    if inputs.is_empty() && !resume {
        bail!("no inputs provided. Supply arXiv IDs or file paths.");
    }

    if id.is_some() && inputs.len() > 1 {
        bail!("--id can only be used with a single input");
    }

    // Parse custom metadata if provided.
    let extra_metadata: Option<Value> = match metadata_json {
        Some(s) => {
            let val: Value =
                serde_json::from_str(s).context("--metadata must be valid JSON")?;
            if !val.is_object() {
                bail!("--metadata must be a JSON object, got: {}", val);
            }
            Some(val)
        }
        None => None,
    };

    // Resolve collection profile.
    let profile = match collection {
        Some(name) => CollectionProfile::get(name)
            .ok_or_else(|| anyhow::anyhow!("unknown collection profile: {name}"))?,
        None => CollectionProfile::default_profile(),
    };

    // -- Classify inputs -------------------------------------------------------
    let classified: Vec<InputKind> = inputs
        .iter()
        .map(|input| {
            let s = input.to_string_lossy();
            if is_arxiv_id(&s) {
                InputKind::Arxiv(s.into_owned())
            } else {
                InputKind::File(input.clone())
            }
        })
        .collect();

    // Guard: refuse to write to production databases.
    config.require_writable_database()?;

    // -- Connect to services ---------------------------------------------------
    let db = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    let extractor = ExtractionClient::connect_default()
        .await
        .context("failed to connect to extraction service")?;

    let embedder = EmbeddingClient::connect_default()
        .await
        .context("failed to connect to embedding service")?;

    // -- Build pipeline --------------------------------------------------------
    let embed_task = determine_embed_task(task, &classified);
    let pipeline_config = PipelineConfig {
        profile,
        embed_task,
        embed_batch_size: Some(config.embedding.batch.size),
        extract_options: Default::default(),
        overwrite: force,
    };

    let pipeline = Arc::new(Pipeline::new(extractor, embedder, db.clone(), pipeline_config));
    let chunker = Arc::new(TokenChunking {
        chunk_size: config.embedding.chunking.size_tokens as usize,
        overlap: config.embedding.chunking.overlap_tokens as usize,
    });

    // -- ArXiv client (only created if we have arXiv inputs) -------------------
    let arxiv_client = if classified.iter().any(|k| matches!(k, InputKind::Arxiv(_))) {
        Some(Arc::new(
            ArxivClient::new().context("failed to create ArXiv client")?,
        ))
    } else {
        None
    };

    // -- Configure batch processor ---------------------------------------------
    let batch_concurrency = concurrency
        .unwrap_or(config.batch_processing.concurrency)
        .max(1);

    let rate_limiter = if config.batch_processing.rate_limit_rps > 0.0 {
        Some(Arc::new(RateLimiter::new(
            config.batch_processing.rate_limit_rps,
            config.batch_processing.rate_limit_retries,
        )))
    } else {
        None
    };

    let batch_config = BatchProcessorConfig {
        concurrency: batch_concurrency,
        state_file: if batch || resume || classified.len() > 1 {
            Some(PathBuf::from(".hades-batch-state.json"))
        } else {
            None
        },
        resume,
        reset,
        progress_interval: Duration::from_secs_f64(
            config.batch_processing.progress_interval_secs,
        ),
        rate_limiter,
    };

    let processor = BatchProcessor::new(batch_config);

    // -- Build items for batch processor ---------------------------------------
    let config = Arc::new(config.clone());
    let custom_id: Option<Arc<str>> = id.map(Arc::from);
    let extra_metadata = extra_metadata.map(Arc::new);

    let items: Vec<(String, InputKind)> = classified
        .into_iter()
        .map(|kind| {
            let item_id = match &kind {
                InputKind::Arxiv(s) => s.clone(),
                InputKind::File(p) => p.display().to_string(),
            };
            (item_id, kind)
        })
        .collect();

    // -- Process batch ---------------------------------------------------------
    let summary = processor
        .process(items, move |_item_id, kind| {
            let pipeline = pipeline.clone();
            let chunker = chunker.clone();
            let db = db.clone();
            let arxiv_client = arxiv_client.clone();
            let config = config.clone();
            let custom_id = custom_id.clone();
            let extra_metadata = extra_metadata.clone();

            async move {
                match kind {
                    InputKind::Arxiv(arxiv_id) => {
                        ingest_arxiv_paper(
                            arxiv_client.as_deref().unwrap(),
                            &pipeline,
                            chunker.as_ref(),
                            &db,
                            profile,
                            &arxiv_id,
                            &config,
                            force,
                            extra_metadata.as_deref(),
                            custom_id.as_deref(),
                        )
                        .await
                    }
                    InputKind::File(path) => {
                        ingest_file(
                            &pipeline,
                            chunker.as_ref(),
                            &db,
                            profile,
                            &path,
                            force,
                            extra_metadata.as_deref(),
                            custom_id.as_deref(),
                        )
                        .await
                    }
                }
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("batch processing error: {e}"))?;

    // -- Output summary --------------------------------------------------------
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let result_values: Vec<Value> = summary
        .results
        .iter()
        .map(|r| {
            let mut val = json!({
                "input": r.item_id,
                "success": r.success,
                "duration_ms": r.duration_ms,
            });
            if r.skipped == Some(true) {
                val["skipped"] = json!(true);
            }
            // Merge domain-specific fields from the process_fn result.
            if let Some(ref data) = r.data
                && let Some(obj) = data.as_object()
            {
                for (k, v) in obj {
                    val[k] = v.clone();
                }
            }
            if let Some(ref err) = r.error {
                val["error"] = json!(err.message);
            }
            val
        })
        .collect();

    // Count skipped from both checkpoint resume and database-exists checks.
    let skipped = summary.skipped
        + summary
            .results
            .iter()
            .filter(|r| {
                r.skipped != Some(true)
                    && r.data
                        .as_ref()
                        .and_then(|d| d.get("skipped"))
                        .and_then(|v| v.as_bool())
                        == Some(true)
            })
            .count();

    let output = json!({
        "success": summary.failed == 0,
        "command": "ingest",
        "data": {
            "total": summary.total,
            "completed": summary.completed,
            "failed": summary.failed,
            "skipped": skipped,
            "results": result_values,
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "duration_ms": duration_ms,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    if summary.failed > 0 {
        return Err(IngestFailure {
            total: summary.total,
            failed: summary.failed,
        }
        .into());
    }
    Ok(())
}

// ── ArXiv paper ingest ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn ingest_arxiv_paper(
    arxiv: &ArxivClient,
    pipeline: &Pipeline,
    chunker: &(dyn ChunkingStrategy + Send + Sync),
    db: &ArangoPool,
    profile: &CollectionProfile,
    arxiv_id: &str,
    config: &HadesConfig,
    force: bool,
    extra_metadata: Option<&Value>,
    custom_id: Option<&str>,
) -> Result<Value> {
    let normalized = normalize_arxiv_id(arxiv_id);
    let doc_key = custom_id
        .map(keys::normalize_document_key)
        .unwrap_or_else(|| keys::normalize_document_key(&normalized));

    // Check if already exists (unless --force).
    if !force && document_exists(db, profile.metadata, &doc_key).await? {
        info!(doc_key, "already ingested, skipping (use --force to re-process)");
        return Ok(json!({"skipped": true}));
    }

    // Fetch metadata from arXiv API.
    let paper = arxiv
        .get_paper_metadata(arxiv_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("paper not found on arXiv: {arxiv_id}"))?;

    // Download PDF.
    let download = arxiv
        .download_paper(
            arxiv_id,
            &config.arxiv.pdf_base_path,
            Some(&config.arxiv.latex_base_path),
            force,
        )
        .await;

    if !download.success {
        bail!(
            "download failed: {}",
            download.error_message.unwrap_or_default()
        );
    }

    let pdf_path = download
        .pdf_path
        .ok_or_else(|| anyhow::anyhow!("download succeeded but no PDF path returned"))?;

    // Process through pipeline.
    let result = pipeline.process_document(&pdf_path, &doc_key, chunker).await;

    if !result.success {
        bail!(
            "{}",
            result
                .error
                .unwrap_or_else(|| "unknown pipeline error".into())
        );
    }

    // Store arXiv-specific metadata (title, authors, abstract, etc.) as a merge.
    let mut arxiv_meta = json!({
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract_text,
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "published": paper.published.to_rfc3339(),
        "updated": paper.updated.to_rfc3339(),
        "arxiv_id": paper.arxiv_id,
        "source": "arxiv",
        "status": "PROCESSED",
    });
    if let Some(doi) = &paper.doi {
        arxiv_meta["doi"] = json!(doi);
    }
    if let Some(jr) = &paper.journal_ref {
        arxiv_meta["journal_ref"] = json!(jr);
    }
    // Merge user-provided extra metadata, skipping protected keys.
    merge_extra_metadata(&mut arxiv_meta, extra_metadata);
    // Identity fields are sacrosanct — reassert after merge.
    arxiv_meta["_key"] = json!(doc_key);

    // Update the metadata document with arXiv-specific fields.
    if let Err(e) = hades_core::db::crud::update_document(
        db,
        profile.metadata,
        &doc_key,
        &arxiv_meta,
    )
    .await
    {
        warn!(doc_key, error = %e, "failed to update arxiv metadata (document was still stored)");
    }

    Ok(json!({
        "num_chunks": result.chunk_count,
        "title": paper.title,
    }))
}

// ── Local file ingest ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn ingest_file(
    pipeline: &Pipeline,
    chunker: &(dyn ChunkingStrategy + Send + Sync),
    db: &ArangoPool,
    profile: &CollectionProfile,
    path: &Path,
    force: bool,
    extra_metadata: Option<&Value>,
    custom_id: Option<&str>,
) -> Result<Value> {
    if !path.exists() {
        bail!("file not found: {}", path.display());
    }

    let doc_key = custom_id
        .map(keys::normalize_document_key)
        .unwrap_or_else(|| {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            keys::normalize_document_key(stem)
        });

    // Check if already exists (unless --force).
    if !force && document_exists(db, profile.metadata, &doc_key).await? {
        info!(doc_key, "already ingested, skipping (use --force to re-process)");
        return Ok(json!({"skipped": true}));
    }

    // Process through pipeline.
    let result = pipeline.process_document(path, &doc_key, chunker).await;

    if !result.success {
        bail!(
            "{}",
            result
                .error
                .unwrap_or_else(|| "unknown pipeline error".into())
        );
    }

    // Update metadata with source info + extra metadata.
    let mut file_meta = json!({
        "source": "local",
        "source_path": path.display().to_string(),
        "status": "PROCESSED",
    });

    // Detect code files and tag them.
    if is_code_file(path) {
        file_meta["pipeline"] = json!("code");
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            file_meta["file_type"] = json!(format!("{ext}_source"));
        }
    }

    // Merge user-provided extra metadata, skipping protected keys.
    merge_extra_metadata(&mut file_meta, extra_metadata);
    file_meta["_key"] = json!(doc_key);

    if let Err(e) =
        hades_core::db::crud::update_document(db, profile.metadata, &doc_key, &file_meta).await
    {
        warn!(doc_key, error = %e, "failed to update file metadata");
    }

    Ok(json!({
        "num_chunks": result.chunk_count,
    }))
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Merge user-provided extra metadata into a document, skipping protected keys.
fn merge_extra_metadata(doc: &mut Value, extra: Option<&Value>) {
    if let Some(extra) = extra
        && let Some(obj) = extra.as_object()
    {
        for (k, v) in obj {
            if PROTECTED_KEYS.contains(&k.as_str()) {
                warn!(key = k, "ignoring protected key in user metadata");
                continue;
            }
            doc[k] = v.clone();
        }
    }
}

/// Check if a document key already exists in a collection.
async fn document_exists(
    db: &ArangoPool,
    collection: &str,
    doc_key: &str,
) -> Result<bool> {
    match hades_core::db::crud::get_document(db, collection, doc_key).await {
        Ok(_) => Ok(true),
        Err(e) if e.is_not_found() => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Determine the embedding task based on explicit --task flag or file extensions.
fn determine_embed_task(task: Option<&str>, inputs: &[InputKind]) -> String {
    if let Some(t) = task {
        if t == "code" {
            return "retrieval.code".to_string();
        }
        return t.to_string();
    }

    // Auto-detect: if all file inputs are code files, use code task.
    let file_inputs: Vec<&PathBuf> = inputs
        .iter()
        .filter_map(|k| match k {
            InputKind::File(p) => Some(p),
            _ => None,
        })
        .collect();

    if !file_inputs.is_empty() && file_inputs.iter().all(|p| is_code_file(p)) {
        return "retrieval.code".to_string();
    }

    "retrieval.passage".to_string()
}

/// Check if a file path looks like a code file by extension.
fn is_code_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| CODE_EXTENSIONS.contains(&ext))
}
