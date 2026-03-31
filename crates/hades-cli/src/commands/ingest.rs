//! Native Rust implementation of the `hades ingest` command.
//!
//! Replaces the Python dispatch for document ingestion.  Supports:
//! - ArXiv paper ingest (download PDF + extract + chunk + embed + store)
//! - Local file ingest (extract + chunk + embed + store)
//! - Mixed inputs (arXiv IDs and file paths in one invocation)
//! - Batch mode with per-document error isolation
//! - Custom metadata merging
//! - Collection profile selection
//! - Force re-processing (surgical delete + re-insert)

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::{json, Value};
use tracing::{error, info, warn};

use hades_core::arxiv::{ArxivClient, is_arxiv_id, normalize_arxiv_id};
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

/// Per-document result for JSON output.
#[derive(serde::Serialize)]
struct ItemResult {
    /// The input identifier (arXiv ID or file path).
    input: String,
    /// Whether this item was processed successfully.
    success: bool,
    /// Number of chunks produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    num_chunks: Option<usize>,
    /// Paper title (arXiv papers).
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    /// Whether we skipped because it already exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    skipped: Option<bool>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// Processing duration in milliseconds.
    duration_ms: u64,
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

    // Auto-activate batch mode for large input sets.
    let batch_mode = batch || resume || classified.len() > 5;

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

    let pipeline = Pipeline::new(extractor, embedder, db.clone(), pipeline_config);
    let chunker = TokenChunking {
        chunk_size: config.embedding.chunking.size_tokens as usize,
        overlap: config.embedding.chunking.overlap_tokens as usize,
    };

    // -- ArXiv client (only created if we have arXiv inputs) -------------------
    let arxiv_client = if classified.iter().any(|k| matches!(k, InputKind::Arxiv(_))) {
        Some(ArxivClient::new().context("failed to create ArXiv client")?)
    } else {
        None
    };

    // -- Process each input with per-document error isolation -------------------
    let mut results: Vec<ItemResult> = Vec::with_capacity(classified.len());

    for (idx, kind) in classified.iter().enumerate() {
        if batch_mode {
            let progress = json!({
                "type": "progress",
                "current": idx + 1,
                "total": classified.len(),
                "percent": ((idx + 1) as f64 / classified.len() as f64 * 100.0),
            });
            eprintln!("{}", serde_json::to_string(&progress).unwrap_or_default());
        }

        let item_start = Instant::now();
        match kind {
            InputKind::Arxiv(arxiv_id) => {
                let result = ingest_arxiv_paper(
                    arxiv_client.as_ref().unwrap(),
                    &pipeline,
                    &chunker,
                    &db,
                    profile,
                    arxiv_id,
                    config,
                    force,
                    extra_metadata.as_ref(),
                    id,
                )
                .await;
                let duration = item_start.elapsed().as_millis() as u64;
                match result {
                    Ok(r) => results.push(ItemResult {
                        input: arxiv_id.clone(),
                        duration_ms: duration,
                        ..r
                    }),
                    Err(e) => {
                        error!(arxiv_id, error = %e, "ingest failed");
                        results.push(ItemResult {
                            input: arxiv_id.clone(),
                            success: false,
                            num_chunks: None,
                            title: None,
                            skipped: None,
                            error: Some(e.to_string()),
                            duration_ms: duration,
                        });
                    }
                }
            }
            InputKind::File(path) => {
                let result = ingest_file(
                    &pipeline,
                    &chunker,
                    &db,
                    profile,
                    path,
                    force,
                    extra_metadata.as_ref(),
                    id,
                )
                .await;
                let duration = item_start.elapsed().as_millis() as u64;
                let display = path.display().to_string();
                match result {
                    Ok(r) => results.push(ItemResult {
                        input: display,
                        duration_ms: duration,
                        ..r
                    }),
                    Err(e) => {
                        error!(path = %path.display(), error = %e, "ingest failed");
                        results.push(ItemResult {
                            input: display,
                            success: false,
                            num_chunks: None,
                            title: None,
                            skipped: None,
                            error: Some(e.to_string()),
                            duration_ms: duration,
                        });
                    }
                }
            }
        }
    }

    // -- Output summary --------------------------------------------------------
    let total = results.len();
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = total - succeeded;
    let skipped = results.iter().filter(|r| r.skipped == Some(true)).count();
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let output = json!({
        "success": failed == 0,
        "command": "ingest",
        "data": {
            "total": total,
            "completed": succeeded,
            "failed": failed,
            "skipped": skipped,
            "results": results,
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "duration_ms": duration_ms,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    if failed > 0 {
        return Err(IngestFailure { total, failed }.into());
    }
    Ok(())
}

// ── ArXiv paper ingest ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn ingest_arxiv_paper(
    arxiv: &ArxivClient,
    pipeline: &Pipeline,
    chunker: &dyn ChunkingStrategy,
    db: &ArangoPool,
    profile: &CollectionProfile,
    arxiv_id: &str,
    config: &HadesConfig,
    force: bool,
    extra_metadata: Option<&Value>,
    custom_id: Option<&str>,
) -> Result<ItemResult> {
    let normalized = normalize_arxiv_id(arxiv_id);
    let doc_key = custom_id
        .map(keys::normalize_document_key)
        .unwrap_or_else(|| keys::normalize_document_key(&normalized));

    // Check if already exists (unless --force).
    if !force && document_exists(db, profile.metadata, &doc_key).await? {
        info!(doc_key, "already ingested, skipping (use --force to re-process)");
        return Ok(ItemResult {
            input: String::new(), // filled by caller
            success: true,
            num_chunks: None,
            title: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
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

    // Store arXiv-specific metadata (title, authors, abstract, etc.) as a merge.
    if result.success {
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
    }

    Ok(ItemResult {
        input: String::new(), // filled by caller
        success: result.success,
        num_chunks: if result.success {
            Some(result.chunk_count)
        } else {
            None
        },
        title: Some(paper.title.clone()),
        skipped: None,
        error: result.error,
        duration_ms: 0, // filled by caller
    })
}

// ── Local file ingest ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn ingest_file(
    pipeline: &Pipeline,
    chunker: &dyn ChunkingStrategy,
    db: &ArangoPool,
    profile: &CollectionProfile,
    path: &Path,
    force: bool,
    extra_metadata: Option<&Value>,
    custom_id: Option<&str>,
) -> Result<ItemResult> {
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
        return Ok(ItemResult {
            input: String::new(),
            success: true,
            num_chunks: None,
            title: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }

    // Process through pipeline.
    let result = pipeline.process_document(path, &doc_key, chunker).await;

    // Update metadata with source info + extra metadata.
    if result.success {
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
            hades_core::db::crud::update_document(db, profile.metadata, &doc_key, &file_meta)
                .await
        {
            warn!(doc_key, error = %e, "failed to update file metadata");
        }
    }

    Ok(ItemResult {
        input: String::new(),
        success: result.success,
        num_chunks: if result.success {
            Some(result.chunk_count)
        } else {
            None
        },
        title: None,
        skipped: None,
        error: result.error,
        duration_ms: 0,
    })
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
