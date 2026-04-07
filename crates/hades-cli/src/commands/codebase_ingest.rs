//! Native Rust implementation of the `hades codebase ingest` command.
//!
//! Walks a directory (or single file), detects language, runs AST
//! analysis, chunks at function/class boundaries, embeds chunks via
//! the Persephone embedder, and stores everything in dedicated codebase
//! collections.
//!
//! Supports:
//! - Recursive directory traversal (respects common ignore patterns)
//! - Language auto-detection from file extension
//! - Incremental ingestion via symbol_hash comparison
//! - Python import graph resolution (file→file edges)
//! - Per-file error isolation in batch mode

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

use hades_core::chunking::ChunkingStrategy;
use hades_core::code::{self, AstChunking, CodeAnalysisError, Language, Symbol, SymbolKind};
use hades_core::code::rust_analyzer::{
    EdgeKind, RustAnalyzerSession, RustEdgeResolver, RustSymbolExtractor, group_files_by_crate,
};
use hades_core::code::rust_imports;
use hades_core::db::collections::CODEBASE;
use hades_core::db::crud;
use hades_core::db::keys;
use hades_core::db::query::ExecutionTarget;
use hades_core::db::{ArangoErrorKind, ArangoPool};
use hades_core::persephone::embedding::EmbeddingClient;
use hades_core::HadesConfig;

use super::output::{self, OutputFormat};

/// Directories to skip during recursive traversal.
const SKIP_DIRS: &[&str] = &[
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "target", ".tox", ".venv", "venv", ".eggs",
    "dist", "build", ".cargo",
];

/// Per-file result for JSON output.
#[derive(serde::Serialize)]
struct FileResult {
    path: String,
    success: bool,
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_symbols: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_embeddings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skipped: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    duration_ms: u64,
}

/// Codebase ingest command failed with partial results.
#[derive(Debug, thiserror::Error)]
#[error("{failed} of {total} files failed to ingest")]
pub struct CodebaseIngestFailure {
    pub total: usize,
    pub failed: usize,
}

/// Accumulators for cross-file import resolution.
///
/// Collects per-file import data during the ingest loop so that
/// import edges can be resolved in a batch pass after all files are processed.
struct ImportContext {
    /// Python: rel_path → list of import symbols (with metadata for resolution).
    python_imports: HashMap<String, Vec<Symbol>>,
    /// Python: rel_path → all definition symbols (for building the resolution index).
    python_file_symbols: HashMap<String, Vec<Symbol>>,
    /// Rust: rel_path → list of expanded use-paths.
    rust_imports: HashMap<String, Vec<String>>,
    /// Rust: rel_path → all symbols (for building the resolution index).
    rust_file_symbols: HashMap<String, Vec<Symbol>>,
}

/// Run the codebase ingest command.
// TODO: support --batch to enable parallel/batched ingestion
pub async fn run(
    config: &HadesConfig,
    path: PathBuf,
    language: Option<&str>,
    batch: bool,
) -> Result<()> {
    let cmd_start = Instant::now();

    // Validate path exists.
    if !path.exists() {
        bail!("path not found: {}", path.display());
    }

    // Parse language override if provided.
    let lang_override = match language {
        Some(l) => {
            let lang = match l.to_lowercase().as_str() {
                "python" | "py" => Language::Python,
                "rust" | "rs" => Language::Rust,
                other => bail!("unsupported language: {other}. Supported: python, rust"),
            };
            Some(lang)
        }
        None => None,
    };

    // Guard: refuse to write to production databases.
    config.require_writable_database()?;

    // Connect to services.
    let db = ArangoPool::from_config(config)
        .context("failed to connect to ArangoDB")?;

    // Embedding is optional — ingest proceeds without vectors if the service is unavailable.
    let embedder = match EmbeddingClient::connect_default().await {
        Ok(client) => {
            info!("connected to embedding service");
            Some(client)
        }
        Err(e) => {
            warn!(error = %e, "embedding service unavailable — ingesting without vectors");
            None
        }
    };

    // Ensure codebase collections exist.
    ensure_collections(&db).await?;

    // Discover source files.
    let files = discover_files(&path, lang_override)?;
    if files.is_empty() {
        output::print_output(
            "codebase.ingest",
            json!({ "total": 0, "message": "no supported source files found" }),
            &OutputFormat::Json,
        );
        return Ok(());
    }

    info!(file_count = files.len(), "discovered source files");

    // Compute base path for relative paths.
    let base = if path.is_dir() {
        path.canonicalize().unwrap_or(path.clone())
    } else {
        path.parent()
            .map(|p| p.canonicalize().unwrap_or(p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    };

    // Process each file with per-file error isolation.
    let mut results: Vec<FileResult> = Vec::with_capacity(files.len());
    // Accumulators for cross-file import resolution.
    let mut imports = ImportContext {
        python_imports: HashMap::new(),
        python_file_symbols: HashMap::new(),
        rust_imports: HashMap::new(),
        rust_file_symbols: HashMap::new(),
    };
    // Collect absolute paths for Rust files — used for rust-analyzer post-loop phase.
    let mut rust_abs_paths: Vec<PathBuf> = Vec::new();

    // Auto-activate batch mode for large input sets.
    let batch_mode = batch || files.len() > 5;

    let total_files = files.len();
    for (idx, file_path) in files.iter().enumerate() {
        if batch_mode {
            let progress = json!({
                "type": "progress",
                "current": idx + 1,
                "total": total_files,
                "percent": ((idx + 1) as f64 / total_files as f64 * 100.0),
            });
            eprintln!("{}", serde_json::to_string(&progress).unwrap_or_default());
        }

        let item_start = Instant::now();
        let rel_path = file_path
            .strip_prefix(&base)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();

        // Track Rust files for rust-analyzer post-loop enrichment.
        let is_rust = lang_override == Some(Language::Rust)
            || file_path.extension().and_then(|e| e.to_str()) == Some("rs");
        if is_rust {
            rust_abs_paths.push(file_path.clone());
        }

        let result = ingest_file(
            &db,
            embedder.as_ref(),
            config,
            file_path,
            &rel_path,
            lang_override,
            &mut imports,
        )
        .await;

        let duration = item_start.elapsed().as_millis() as u64;
        match result {
            Ok(r) => results.push(FileResult { duration_ms: duration, ..r }),
            Err(e) => {
                error!(path = %rel_path, error = %e, "ingest failed");
                results.push(FileResult {
                    path: rel_path,
                    success: false,
                    language: None,
                    num_symbols: None,
                    num_chunks: None,
                    num_embeddings: None,
                    skipped: None,
                    error: Some(e.to_string()),
                    duration_ms: duration,
                });
            }
        }
    }

    // Resolve Python import graph edges (file→symbol where possible, file→file fallback).
    let py_symbol_index = build_python_symbol_index(&imports.python_file_symbols);
    let py_import_edges = resolve_python_imports(
        &imports.python_imports,
        &imports.python_file_symbols,
        &py_symbol_index,
    );
    if !py_import_edges.is_empty() {
        info!(edge_count = py_import_edges.len(), "resolved Python import edges");
        if let Err(e) = crud::insert_documents(&db, CODEBASE.imports_edges, &py_import_edges, true).await {
            warn!(error = %e, "failed to store Python import edges");
        }
    }

    // Resolve Rust import graph edges (file → symbol).
    let rust_symbol_index = rust_imports::build_symbol_index(&imports.rust_file_symbols);
    let rs_import_edges = rust_imports::resolve_rust_imports(&imports.rust_imports, &rust_symbol_index);
    if !rs_import_edges.is_empty() {
        info!(edge_count = rs_import_edges.len(), "resolved Rust import edges");
        if let Err(e) = crud::insert_documents(&db, CODEBASE.imports_edges, &rs_import_edges, true).await {
            warn!(error = %e, "failed to store Rust import edges");
        }
    }

    let total_import_edges = py_import_edges.len() + rs_import_edges.len();

    // ── rust-analyzer deep analysis ────────────────────────────────────
    // When Rust files were ingested, optionally use rust-analyzer for richer
    // symbol extraction: qualified names, call hierarchy, impl-trait edges,
    // PyO3/FFI detection. This enrichment phase runs after the syn-based loop.
    let ra_stats = if !rust_abs_paths.is_empty() {
        match run_rust_analyzer_phase(&db, &base, &rust_abs_paths).await {
            Ok(stats) => {
                info!(
                    symbols = stats.symbols,
                    edges = stats.edges,
                    crates = stats.crates,
                    "rust-analyzer enrichment complete"
                );
                stats
            }
            Err(e) => {
                warn!(error = %e, "rust-analyzer enrichment failed, syn-based data retained");
                RustAnalyzerStats::default()
            }
        }
    } else {
        RustAnalyzerStats::default()
    };

    // Output summary.
    let total = results.len();
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = results.iter().filter(|r| !r.success && r.skipped != Some(true)).count();
    let skipped = results.iter().filter(|r| r.skipped == Some(true)).count();
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let files_embedded = results.iter()
        .filter(|r| r.num_embeddings.is_some_and(|n| n > 0))
        .count();
    let total_embeddings: usize = results.iter()
        .filter_map(|r| r.num_embeddings)
        .sum();

    let result_data = json!({
        "total": total,
        "completed": succeeded,
        "failed": failed,
        "skipped": skipped,
        "embedding": {
            "service_connected": embedder.is_some(),
            "files_embedded": files_embedded,
            "total_embeddings": total_embeddings,
        },
        "import_edges": total_import_edges,
        "python_import_edges": py_import_edges.len(),
        "rust_import_edges": rs_import_edges.len(),
        "rust_analyzer": {
            "symbols": ra_stats.symbols,
            "edges": ra_stats.edges,
            "crates_analyzed": ra_stats.crates,
        },
        "results": results,
        "duration_ms": duration_ms,
    });

    output::print_output("codebase.ingest", result_data, &OutputFormat::Json);

    if failed > 0 {
        return Err(CodebaseIngestFailure { total, failed }.into());
    }
    Ok(())
}

// ── Collection setup ────────────────────────────────────────────────────

/// Ensure all codebase collections, named graph, and indices exist.
///
/// Creation order (per ontology spec §7.1):
/// 1. Document collections (files, chunks, embeddings, symbols)
/// 2. Edge collections (defines, calls, implements, imports)
/// 3. Named graph `codebase_graph` via Gharial API
/// 4. Persistent indices on document collections
async fn ensure_collections(db: &ArangoPool) -> Result<()> {
    // Step 1–2: Create collections.
    let existing = crud::list_collections(db, false)
        .await
        .context("failed to list collections")?;
    let existing_names: Vec<&str> = existing.iter().map(|c| c.name.as_str()).collect();

    for (name, col_type) in CODEBASE.all_collections() {
        if !existing_names.contains(&name) {
            info!(collection = name, col_type, "creating collection");
            crud::create_collection(db, name, Some(col_type))
                .await
                .with_context(|| format!("failed to create collection: {name}"))?;
        }
    }

    // Step 3: Create named graph (idempotent — 409 means it already exists).
    ensure_named_graph(db).await?;

    // Step 4: Ensure persistent indices.
    ensure_indices(db).await?;

    Ok(())
}

/// The named graph name.
const CODEBASE_GRAPH: &str = "codebase_graph";

/// Create the `codebase_graph` named graph via the Gharial API.
///
/// The named graph enforces `_from`/`_to` vertex constraints at insert
/// time — an edge with `_from` pointing to the wrong collection is
/// rejected by ArangoDB rather than silently corrupting the graph.
async fn ensure_named_graph(db: &ArangoPool) -> Result<()> {
    let body = json!({
        "name": CODEBASE_GRAPH,
        "edgeDefinitions": [
            {
                "collection": CODEBASE.defines_edges,
                "from": [CODEBASE.files],
                "to": [CODEBASE.symbols],
            },
            {
                "collection": CODEBASE.calls_edges,
                "from": [CODEBASE.symbols],
                "to": [CODEBASE.symbols],
            },
            {
                "collection": CODEBASE.implements_edges,
                "from": [CODEBASE.symbols],
                "to": [CODEBASE.symbols],
            },
            {
                "collection": CODEBASE.imports_edges,
                "from": [CODEBASE.files],
                "to": [CODEBASE.files, CODEBASE.symbols],
            },
        ],
        "orphanCollections": [CODEBASE.chunks, CODEBASE.embeddings],
    });

    match db.writer().post("gharial", &body).await {
        Ok(_) => {
            info!(graph = CODEBASE_GRAPH, "created named graph");
        }
        Err(e) if e.kind() == ArangoErrorKind::Conflict => {
            debug!(graph = CODEBASE_GRAPH, "named graph already exists");
        }
        Err(e) => {
            return Err(anyhow::anyhow!(e).context("failed to create named graph"));
        }
    }
    Ok(())
}

/// Ensure persistent indices exist on codebase document collections.
///
/// ArangoDB's `ensureIndex` is idempotent — if an index with the same
/// fields and type already exists, it returns the existing index.
async fn ensure_indices(db: &ArangoPool) -> Result<()> {
    let indices: &[(&str, &[&str])] = &[
        (CODEBASE.chunks, &["file_key"]),
        (CODEBASE.chunks, &["symbols[*]"]),
        (CODEBASE.embeddings, &["file_key"]),
        (CODEBASE.embeddings, &["chunk_key"]),
        (CODEBASE.symbols, &["file_key"]),
        (CODEBASE.symbols, &["kind"]),
    ];

    for (collection, fields) in indices {
        let path = format!("index?collection={collection}");
        let body = json!({
            "type": "persistent",
            "fields": fields,
        });
        db.writer()
            .post(&path, &body)
            .await
            .with_context(|| {
                format!("failed to ensure index on {collection} {fields:?}")
            })?;
    }

    debug!("ensured {} persistent indices", indices.len());
    Ok(())
}

// ── File discovery ──────────────────────────────────────────────────────

/// Discover source files to ingest from a path.
///
/// If `path` is a file, returns just that file (if it matches the language
/// filter). If a directory, walks recursively, skipping common non-source
/// directories.
fn discover_files(path: &Path, lang_override: Option<Language>) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        let path_str = path.to_string_lossy();
        if lang_override.is_some() || Language::from_path(&path_str).is_some() {
            return Ok(vec![path.to_path_buf()]);
        }
        bail!(
            "unsupported file type: {}. Use --language to override.",
            path.display()
        );
    }

    let mut files = Vec::new();
    let walker = WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            if entry.file_type().is_dir() {
                let name = entry.file_name().to_string_lossy();
                return !SKIP_DIRS.contains(&name.as_ref());
            }
            true
        });

    for entry in walker {
        let entry = entry.context("error walking directory")?;
        if !entry.file_type().is_file() {
            continue;
        }
        let entry_path = entry.path();
        let path_str = entry_path.to_string_lossy();
        let include = if Language::from_path(&path_str).is_some() {
            // File has a recognized source extension — always include.
            true
        } else if lang_override.is_some() {
            // Language override active: include extensionless files only
            // (skip .md, .json, images, etc.).
            entry_path.extension().is_none()
        } else {
            false
        };
        if include {
            files.push(entry_path.to_path_buf());
        }
    }

    files.sort();
    Ok(files)
}

// ── Per-file ingest ─────────────────────────────────────────────────────

/// Ingest a single source file: analyze → chunk → embed → store.
async fn ingest_file(
    db: &ArangoPool,
    embedder: Option<&EmbeddingClient>,
    config: &HadesConfig,
    file_path: &Path,
    rel_path: &str,
    lang_override: Option<Language>,
    imports: &mut ImportContext,
) -> Result<FileResult> {
    // Read source.
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;

    // Detect language.
    let lang = lang_override
        .or_else(|| Language::from_path(rel_path))
        .ok_or_else(|| anyhow::anyhow!("cannot detect language for {rel_path}"))?;

    // Analyze.
    let mut analysis = match code::analyze_with_language(&source, lang) {
        Ok(a) => a,
        Err(CodeAnalysisError::ParseError(msg)) => {
            warn!(path = rel_path, error = %msg, "parse error, skipping");
            return Ok(FileResult {
                path: rel_path.to_string(),
                success: true,
                language: Some(lang.name().to_string()),
                num_symbols: None,
                num_chunks: None,
                num_embeddings: None,
                skipped: Some(true),
                error: Some(format!("parse error: {msg}")),
                duration_ms: 0,
            });
        }
        Err(e) => return Err(e.into()),
    };

    // Check for incremental skip via symbol_hash.
    // Only skip if the code is unchanged AND embeddings aren't needed (either
    // already present or no embedder available to backfill).
    let fkey = keys::file_key(rel_path);
    if let Some(true) = check_unchanged(db, &fkey, &analysis.symbol_hash, embedder.is_some()).await? {
        debug!(path = rel_path, "unchanged (same symbol_hash, embeddings present), skipping");
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang.name().to_string()),
            num_symbols: Some(analysis.symbols.len()),
            num_chunks: None,
            num_embeddings: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }

    // Collect Python import symbols for later edge resolution.
    if lang == Language::Python {
        let py_import_syms: Vec<Symbol> = analysis
            .symbols
            .iter()
            .filter(|s| s.kind == hades_core::code::SymbolKind::Import)
            .cloned()
            .collect();
        if !py_import_syms.is_empty() {
            imports.python_imports.insert(rel_path.to_string(), py_import_syms);
        }
    }

    // Collect Rust use-paths for later import edge resolution.
    // Symbol transfer into the index is deferred until after all uses of analysis.symbols.
    if lang == Language::Rust {
        let use_paths = rust_imports::collect_use_paths(&analysis.symbols);
        if !use_paths.is_empty() {
            imports.rust_imports.insert(rel_path.to_string(), use_paths);
        }
    }

    // Chunk with AST-aligned chunking.
    let chunker = AstChunking::new(analysis.top_level_defs.clone());
    let chunks = chunker.chunk(&source);

    // Build file document (embedding_count populated after embed step below).
    let num_sym = analysis.symbols.len();
    let num_chk = chunks.len();

    // Build line→byte offset table for symbol-chunk interval intersection.
    let line_offsets = build_line_offsets(&source);

    // Build chunk documents with symbol context.
    let chunk_docs: Vec<Value> = chunks
        .iter()
        .map(|c| {
            // Find symbols whose span overlaps this chunk (interval intersection).
            let overlapping_symbols: Vec<String> = analysis
                .symbols
                .iter()
                .filter(|s| s.kind.is_primitive())
                .filter_map(|s| {
                    let sym_start = line_offsets
                        .get(s.start_line.saturating_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let sym_end = line_offsets
                        .get(s.end_line)
                        .copied()
                        .unwrap_or(source.len());
                    if c.start_char < sym_end && sym_start < c.end_char {
                        Some(keys::symbol_key(&fkey, &s.name))
                    } else {
                        None
                    }
                })
                .collect();

            let ckey = keys::chunk_key(&fkey, c.chunk_index);
            json!({
                "_key": ckey,
                "file_key": fkey,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "symbols": overlapping_symbols,
            })
        })
        .collect();

    // Build symbol documents (primitives only — imports and impl blocks are not vertices).
    let symbol_docs: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| s.kind.is_primitive())
        .map(|s| {
            let skey = keys::symbol_key(&fkey, &s.name);
            json!({
                "_key": skey,
                "file_key": fkey,
                "file_path": rel_path,
                "name": s.name,
                "qualified_name": s.name,
                "kind": s.kind.universal_kind().unwrap(),
                "lang_kind": s.kind.lang_kind(),
                "start_line": s.start_line,
                "end_line": s.end_line,
                "metadata": s.metadata,
            })
        })
        .collect();

    // Build defines edges (file → symbol) for primitives only.
    let define_edges: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| s.kind.is_primitive())
        .map(|s| {
            let skey = keys::symbol_key(&fkey, &s.name);
            let edge_key = keys::edge_key(&fkey, "defines", &skey);
            json!({
                "_key": edge_key,
                "_from": format!("{}/{}", CODEBASE.files, fkey),
                "_to": format!("{}/{}", CODEBASE.symbols, skey),
                "file_path": rel_path,
                "symbol_name": s.name,
            })
        })
        .collect();

    // Remove stale embeddings for this file before (re-)embedding.
    // This ensures that if embedding is skipped or fails, old vectors
    // from a previous run (which embed outdated text) don't linger.
    delete_file_embeddings(db, &fkey).await;

    // Embed chunks (skipped if embedder is unavailable).
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embedding_docs = match embedder {
        Some(emb) if !chunk_texts.is_empty() => {
            match emb
                .embed(
                    &chunk_texts,
                    "retrieval.code",
                    Some(config.embedding.batch.size),
                )
                .await
            {
                Ok(embed_result) => {
                    embed_result
                        .embeddings
                        .iter()
                        .enumerate()
                        .map(|(i, vec)| {
                            let ckey = keys::chunk_key(&fkey, i);
                            let ekey = keys::embedding_key(&ckey);
                            json!({
                                "_key": ekey,
                                "chunk_key": ckey,
                                "file_key": fkey,
                                "embedding": vec,
                                "model": embed_result.model,
                                "model_hash": keys::model_hash(&embed_result.model),
                                "dimension": embed_result.dimension,
                            })
                        })
                        .collect::<Vec<Value>>()
                }
                Err(e) => {
                    warn!(path = rel_path, error = %e, "embedding failed, storing without vectors");
                    Vec::new()
                }
            }
        }
        _ => Vec::new(),
    };
    let num_embeddings_written = embedding_docs.len();

    // Build file document (after embedding so we can record embedding_count).
    // symbol_count reflects primitives only (no imports, no impl blocks).
    let primitive_count = analysis.symbols.iter().filter(|s| s.kind.is_primitive()).count();
    let file_doc = json!({
        "_key": fkey,
        "path": rel_path,
        "kind": "file",
        "language": lang.name(),
        "metrics": analysis.metrics,
        "symbol_hash": analysis.symbol_hash,
        "symbol_count": primitive_count,
        "chunk_count": num_chk,
        "embedding_count": num_embeddings_written,
        "total_lines": analysis.metrics.total_lines,
        "status": "PROCESSED",
        "ingested_at": chrono::Utc::now().to_rfc3339(),
    });

    // Store to ArangoDB. Embeddings are persisted BEFORE the file document
    // so that embedding_count is only recorded once the vectors are durable.
    // This prevents check_unchanged() from skipping future backfills if
    // embedding persistence fails partway through.
    if !chunk_docs.is_empty() {
        crud::insert_documents(db, CODEBASE.chunks, &chunk_docs, true)
            .await
            .context("failed to store chunk documents")?;
    }

    if !symbol_docs.is_empty() {
        crud::insert_documents(db, CODEBASE.symbols, &symbol_docs, true)
            .await
            .context("failed to store symbol documents")?;
    }

    if !embedding_docs.is_empty() {
        crud::insert_documents(db, CODEBASE.embeddings, &embedding_docs, true)
            .await
            .context("failed to store embedding documents")?;
    }

    if !define_edges.is_empty() {
        crud::insert_documents(db, CODEBASE.defines_edges, &define_edges, true)
            .await
            .context("failed to store define edges")?;
    }

    // File document stored last — embedding_count is only recorded after
    // vectors are durable, so check_unchanged() won't wrongly skip backfills.
    crud::insert_documents(db, CODEBASE.files, &[file_doc], true)
        .await
        .context("failed to store file document")?;

    // Transfer symbols into the import index (avoids cloning).
    match lang {
        Language::Rust => {
            imports.rust_file_symbols.insert(
                rel_path.to_string(),
                std::mem::take(&mut analysis.symbols),
            );
        }
        Language::Python => {
            imports.python_file_symbols.insert(
                rel_path.to_string(),
                std::mem::take(&mut analysis.symbols),
            );
        }
        _ => {}
    }

    info!(
        path = rel_path,
        language = lang.name(),
        symbols = num_sym,
        chunks = num_chk,
        embeddings = num_embeddings_written,
        "ingested"
    );

    Ok(FileResult {
        path: rel_path.to_string(),
        success: true,
        language: Some(lang.name().to_string()),
        num_symbols: Some(num_sym),
        num_chunks: Some(num_chk),
        num_embeddings: Some(num_embeddings_written),
        skipped: None,
        error: None,
        duration_ms: 0,
    })
}

// ── Line-offset table ─────────────────────────────────────────────────

/// Build a byte-offset table for each line in `source`.
///
/// `offsets[i]` is the byte position where line `i` starts (0-based line
/// numbering). An extra sentinel entry for `offsets[line_count]` equals
/// `source.len()`, so callers can use `offsets[end_line]` to get the byte
/// position just past the last line of a span without bounds checks.
fn build_line_offsets(source: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            offsets.push(i + 1);
        }
    }
    offsets.push(source.len());
    offsets
}

// ── Stale embedding cleanup ────────────────────────────────────────────

/// Remove existing embedding documents for a file.
///
/// Called before (re-)embedding to ensure stale vectors from a previous
/// run don't linger when the embedder is unavailable or fails.
async fn delete_file_embeddings(db: &ArangoPool, file_key: &str) {
    let aql = "FOR e IN @@col FILTER e.file_key == @fk REMOVE e IN @@col";
    let bind = json!({ "@col": CODEBASE.embeddings, "fk": file_key });
    if let Err(e) = hades_core::db::query::query(db, aql, Some(&bind), None, false, ExecutionTarget::Writer).await {
        debug!(file_key, error = %e, "failed to clean up old embeddings (non-fatal)");
    }
}

// ── Incremental check ───────────────────────────────────────────────────

/// Check if a file can be skipped during incremental ingest.
///
/// Returns `Some(true)` if the file should be skipped:
/// - Symbol hash matches AND (embeddings already exist OR no embedder to backfill)
///
/// Returns `Some(false)` if re-processing is needed:
/// - Symbol hash differs, OR
/// - Symbol hash matches but embeddings are missing and embedder is available
///
/// Returns `None` if the file is not in the database (first ingest).
async fn check_unchanged(
    db: &ArangoPool,
    file_key: &str,
    new_hash: &str,
    embedder_available: bool,
) -> Result<Option<bool>> {
    match crud::get_document(db, CODEBASE.files, file_key).await {
        Ok(doc) => {
            let stored_hash = doc["symbol_hash"].as_str().unwrap_or("");
            if stored_hash != new_hash {
                return Ok(Some(false)); // code changed, must re-process
            }
            // Code unchanged. Skip only if embeddings aren't needed or already present.
            if embedder_available {
                // Files with no chunks have nothing to embed — always skip.
                let chunk_count = doc["chunk_count"].as_u64().unwrap_or(0);
                if chunk_count == 0 {
                    return Ok(Some(true));
                }
                let has_embeddings = doc["embedding_count"]
                    .as_u64()
                    .is_some_and(|n| n > 0);
                Ok(Some(has_embeddings)) // skip only if embeddings exist
            } else {
                Ok(Some(true)) // no embedder → nothing to backfill → skip
            }
        }
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ── rust-analyzer enrichment ───────────────────────────────────────────

/// Stats returned from the rust-analyzer enrichment phase.
#[derive(Default)]
struct RustAnalyzerStats {
    symbols: usize,
    edges: usize,
    crates: usize,
}

/// Run rust-analyzer over ingested Rust files to produce rich symbols and edges.
///
/// Groups files by crate root, spawns a `RustAnalyzerSession` per crate,
/// extracts qualified symbols with call hierarchy and impl-trait info, then
/// stores the enriched symbol documents and edges to ArangoDB.
///
/// This phase is additive: it overwrites top-level symbol documents (same
/// keys as syn) and adds new method-level symbols and cross-file edges
/// that syn cannot produce.
async fn run_rust_analyzer_phase(
    db: &ArangoPool,
    base: &Path,
    rust_files: &[PathBuf],
) -> Result<RustAnalyzerStats> {
    let groups = group_files_by_crate(rust_files);
    if groups.is_empty() {
        return Ok(RustAnalyzerStats::default());
    }

    info!(
        crate_count = groups.len(),
        file_count = rust_files.len(),
        "starting rust-analyzer enrichment"
    );

    let mut all_extractions = HashMap::new();
    let mut crates_analyzed = 0;

    for (crate_root, crate_files) in &groups {
        info!(
            crate_root = %crate_root.display(),
            file_count = crate_files.len(),
            "analyzing crate with rust-analyzer"
        );

        let session = match RustAnalyzerSession::start(crate_root).await {
            Ok(s) => s,
            Err(e) => {
                // If rust-analyzer isn't installed or fails to start,
                // skip this crate but try others.
                warn!(
                    crate_root = %crate_root.display(),
                    error = %e,
                    "failed to start rust-analyzer session, skipping crate"
                );
                continue;
            }
        };

        let extractor = RustSymbolExtractor::new(&session, true);
        let file_refs: Vec<&Path> = crate_files.iter().map(|p| p.as_path()).collect();
        let extractions = extractor.extract_crate(&file_refs).await;

        // Convert absolute path keys to relative paths (matching file_key convention).
        for (abs_path_str, extraction) in extractions {
            let abs = Path::new(&abs_path_str);
            let rel = abs
                .strip_prefix(base)
                .unwrap_or(abs)
                .to_string_lossy()
                .to_string();
            all_extractions.insert(rel, extraction);
        }

        crates_analyzed += 1;

        // Graceful shutdown — non-fatal if it fails.
        if let Err(e) = session.shutdown().await {
            debug!(error = %e, "rust-analyzer shutdown warning (non-fatal)");
        }
    }

    if all_extractions.is_empty() {
        return Ok(RustAnalyzerStats {
            crates: crates_analyzed,
            ..Default::default()
        });
    }

    // Collect per-file stats before the resolver takes ownership.
    let file_patches: Vec<(String, usize, String)> = all_extractions
        .iter()
        .map(|(rel_path, extraction)| {
            (
                rel_path.clone(),
                extraction.symbols.len(),
                extraction.analyzed_at.clone(),
            )
        })
        .collect();

    // Build rich symbol documents and edges via RustEdgeResolver.
    let resolver = RustEdgeResolver::new(all_extractions);
    let symbol_docs = resolver.build_symbol_documents();
    let crate_edges = resolver.build_edges();

    let sym_count = symbol_docs.len();
    let edge_count = crate_edges.len();

    // Store enriched symbol documents (overwrite=true for idempotent re-runs).
    if !symbol_docs.is_empty() {
        let docs: Vec<Value> = symbol_docs
            .iter()
            .filter_map(|s| serde_json::to_value(s).ok())
            .collect();
        crud::insert_documents(db, CODEBASE.symbols, &docs, true)
            .await
            .context("failed to store rust-analyzer symbol documents")?;
        info!(count = docs.len(), "stored rust-analyzer symbol documents");
    }

    // Store edges grouped by collection (collection-per-relation).
    if !crate_edges.is_empty() {
        // Build edge documents with deterministic keys.
        let edge_docs: Vec<(EdgeKind, Value)> = crate_edges
            .iter()
            .map(|e| {
                let from_suffix = e.from.rsplit('/').next().unwrap_or(&e.from);
                let to_suffix = e.to.rsplit('/').next().unwrap_or(&e.to);
                let edge_key = keys::edge_key(from_suffix, e.kind.as_str(), to_suffix);
                let mut doc = json!({
                    "_key": edge_key,
                    "_from": e.from,
                    "_to": e.to,
                });
                // Merge edge metadata.
                if let Value::Object(meta) = &e.metadata
                    && let Value::Object(ref mut obj) = doc
                {
                    for (k, v) in meta {
                        obj.insert(k.clone(), v.clone());
                    }
                }
                (e.kind, doc)
            })
            .collect();

        // Group by edge kind and insert into the appropriate collection.
        for kind in [EdgeKind::Defines, EdgeKind::Calls, EdgeKind::Implements] {
            let docs: Vec<Value> = edge_docs
                .iter()
                .filter(|(k, _)| *k == kind)
                .map(|(_, d)| d.clone())
                .collect();
            if !docs.is_empty() {
                crud::insert_documents(db, kind.collection(), &docs, true)
                    .await
                    .with_context(|| format!("failed to store {} edges", kind.as_str()))?;
            }
        }

        info!(count = edge_docs.len(), "stored rust-analyzer edges");
    }

    // Patch file documents with rust-analyzer metadata (partial update — preserves existing fields).
    let mut patched_count = 0;
    for (rel_path, sym_count, analyzed_at) in &file_patches {
        let fkey = keys::file_key(rel_path);
        let patch = json!({
            "ra_analyzed": true,
            "ra_symbol_count": sym_count,
            "ra_analyzed_at": analyzed_at,
        });
        match crud::update_document(db, CODEBASE.files, &fkey, &patch).await {
            Ok(_) => patched_count += 1,
            Err(e) => {
                debug!(file_key = %fkey, error = %e, "failed to patch file document (non-fatal)");
            }
        }
    }
    if patched_count > 0 {
        info!(count = patched_count, "patched file documents with rust-analyzer metadata");
    }

    Ok(RustAnalyzerStats {
        symbols: sym_count,
        edges: edge_count,
        crates: crates_analyzed,
    })
}

// ── Python import graph resolution ──────────────────────────────────────

/// Resolve Python import statements to file→file edges.
///
/// Only creates edges for imports that resolve to files within the
/// ingested set. External package imports are silently skipped.
/// Build a symbol index for Python files: bare name → vec of (rel_path, symbol_key).
fn build_python_symbol_index(
    file_symbols: &HashMap<String, Vec<Symbol>>,
) -> HashMap<String, Vec<(String, String)>> {
    let mut index: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (rel_path, symbols) in file_symbols {
        let fkey = keys::file_key(rel_path);
        for sym in symbols {
            // Only index definitions, not imports.
            if sym.kind == SymbolKind::Import {
                continue;
            }
            let skey = keys::symbol_key(&fkey, &sym.name);
            index
                .entry(sym.name.clone())
                .or_default()
                .push((rel_path.clone(), skey));
        }
    }
    index
}

/// Build a mapping from Python module name → relative file path.
fn build_python_module_map(all_files: &HashMap<String, Vec<Symbol>>) -> HashMap<String, String> {
    let mut module_to_file: HashMap<String, String> = HashMap::new();
    for rel_path in all_files.keys() {
        let p = Path::new(rel_path);
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let mut parts: Vec<&str> = p
            .parent()
            .map(|parent| {
                parent
                    .components()
                    .filter_map(|c| c.as_os_str().to_str())
                    .collect()
            })
            .unwrap_or_default();
        parts.push(stem);
        let module = parts.join(".");
        let module = module
            .strip_suffix(".__init__")
            .unwrap_or(&module)
            .to_string();
        module_to_file.insert(module, rel_path.clone());
    }
    module_to_file
}

/// Resolve a Python module name to a file path, trying exact then prefix match.
fn resolve_module_to_file<'a>(
    module: &str,
    module_to_file: &'a HashMap<String, String>,
) -> Option<&'a String> {
    module_to_file.get(module).or_else(|| {
        let mut parts: Vec<&str> = module.split('.').collect();
        while parts.len() > 1 {
            parts.pop();
            let prefix = parts.join(".");
            if let Some(path) = module_to_file.get(&prefix) {
                return Some(path);
            }
        }
        None
    })
}

fn resolve_python_imports(
    python_imports: &HashMap<String, Vec<Symbol>>,
    python_file_symbols: &HashMap<String, Vec<Symbol>>,
    symbol_index: &HashMap<String, Vec<(String, String)>>,
) -> Vec<Value> {
    // Build module→file mapping from all known Python files.
    let module_to_file = build_python_module_map(python_file_symbols);

    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (source_path, import_syms) in python_imports {
        let source_fkey = keys::file_key(source_path);

        for sym in import_syms {
            let import_type = sym.metadata.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let module = sym.metadata.get("module").and_then(|v| v.as_str()).unwrap_or("");

            if module.is_empty() {
                continue;
            }

            match import_type {
                "from_import" => {
                    // `from module import Name` — try to resolve Name to a specific symbol.
                    let original_name = sym
                        .metadata
                        .get("original_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&sym.name);

                    // First, find the target file.
                    let target_file = resolve_module_to_file(module, &module_to_file);

                    // Try symbol-level resolution: look up the imported name in the symbol index.
                    let mut resolved = false;
                    if let Some(targets) = symbol_index.get(original_name) {
                        // If we know the target file, prefer symbols from that file.
                        let target = if let Some(tf) = target_file {
                            targets.iter().find(|(path, _)| path == tf)
                        } else {
                            None
                        }
                        .or_else(|| targets.first());

                        if let Some((target_path, target_skey)) = target
                            && target_path != source_path {
                                let edge_key =
                                    keys::edge_key(&source_fkey, "imports", target_skey);
                                if seen.insert(edge_key.clone()) {
                                    edges.push(json!({
                                        "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                        "_to": format!("{}/{}", CODEBASE.symbols, target_skey),
                                        "_key": edge_key,
                                        "type": "imports",
                                        "source_path": source_path,
                                        "target_path": target_path,
                                        "symbol_name": original_name,
                                        "module_path": module,
                                    }));
                                    resolved = true;
                                }
                            }
                    }

                    // Fall back to file→file if symbol not found (external package or
                    // symbol not in our index).
                    if !resolved
                        && let Some(target_path) = target_file
                            && target_path != source_path {
                                let target_fkey = keys::file_key(target_path);
                                let edge_key =
                                    keys::edge_key(&source_fkey, "imports", &target_fkey);
                                if seen.insert(edge_key.clone()) {
                                    edges.push(json!({
                                        "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                        "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                                        "_key": edge_key,
                                        "type": "imports",
                                        "source_path": source_path,
                                        "target_path": target_path,
                                        "module_path": module,
                                    }));
                                }
                            }
                }

                "import" => {
                    // `import module` — file-level edge (no specific symbol target).
                    if let Some(target_path) = resolve_module_to_file(module, &module_to_file)
                        && target_path != source_path {
                            let target_fkey = keys::file_key(target_path);
                            let edge_key = keys::edge_key(&source_fkey, "imports", &target_fkey);
                            if seen.insert(edge_key.clone()) {
                                edges.push(json!({
                                    "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                    "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                                    "_key": edge_key,
                                    "type": "imports",
                                    "source_path": source_path,
                                    "target_path": target_path,
                                    "module_path": module,
                                }));
                            }
                        }
                }

                _ => {}
            }
        }
    }

    edges
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_discover_files_single() {
        let dir = TempDir::new().unwrap();
        let py_file = dir.path().join("test.py");
        fs::write(&py_file, "x = 1\n").unwrap();

        let files = discover_files(&py_file, None).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("b.rs"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap();

        let files = discover_files(dir.path(), None).unwrap();
        assert_eq!(files.len(), 2); // .py + .rs, not .md
    }

    #[test]
    fn test_discover_files_skips_dirs() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        let git_dir = dir.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("config.py"), "x = 1\n").unwrap();
        let pycache = dir.path().join("__pycache__");
        fs::create_dir(&pycache).unwrap();
        fs::write(pycache.join("mod.py"), "x = 1\n").unwrap();

        let files = discover_files(dir.path(), None).unwrap();
        assert_eq!(files.len(), 1); // only a.py
    }

    #[test]
    fn test_discover_files_language_override() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("script"), "x = 1\n").unwrap(); // no extension

        // Without override: no files found.
        let files = discover_files(dir.path(), None).unwrap();
        assert_eq!(files.len(), 0);

        // With override: extensionless file is included.
        let files = discover_files(dir.path(), Some(Language::Python)).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_override_excludes_non_source() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("script"), "x = 1\n").unwrap(); // no extension — included
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap(); // has extension — excluded
        fs::write(dir.path().join("data.json"), "{}").unwrap(); // has extension — excluded
        fs::write(dir.path().join("real.py"), "x = 1\n").unwrap(); // recognized — included

        let files = discover_files(dir.path(), Some(Language::Python)).unwrap();
        assert_eq!(files.len(), 2); // script + real.py, not readme.md or data.json
    }

    /// Helper to create a Python import symbol for tests.
    fn make_import_sym(name: &str, import_type: &str, module: &str) -> Symbol {
        let mut metadata = json!({ "type": import_type, "module": module });
        if import_type == "from_import" {
            metadata["original_name"] = json!(name);
        }
        Symbol {
            name: name.to_string(),
            kind: SymbolKind::Import,
            start_line: 1,
            end_line: 1,
            metadata,
        }
    }

    /// Helper to create a definition symbol for tests.
    fn make_def_sym(name: &str, kind: SymbolKind) -> Symbol {
        Symbol {
            name: name.to_string(),
            kind,
            start_line: 1,
            end_line: 10,
            metadata: json!({}),
        }
    }

    #[test]
    fn test_resolve_python_imports_basic() {
        // core/models.py does `from core.utils import helper`
        let mut imports = HashMap::new();
        imports.insert(
            "core/models.py".to_string(),
            vec![make_import_sym("helper", "from_import", "core.utils")],
        );
        imports.insert("core/utils.py".to_string(), vec![]);

        // utils.py defines a function called `helper`
        let mut file_symbols = HashMap::new();
        file_symbols.insert(
            "core/models.py".to_string(),
            vec![make_def_sym("Model", SymbolKind::Class)],
        );
        file_symbols.insert(
            "core/utils.py".to_string(),
            vec![make_def_sym("helper", SymbolKind::Function)],
        );

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["type"], "imports");
        assert_eq!(edges[0]["source_path"], "core/models.py");
        assert_eq!(edges[0]["symbol_name"], "helper");
        // Should be file→symbol edge
        assert!(edges[0]["_to"].as_str().unwrap().contains("codebase_symbols"));
    }

    #[test]
    fn test_resolve_python_imports_no_self_edge() {
        let mut imports = HashMap::new();
        imports.insert(
            "core/models.py".to_string(),
            vec![make_import_sym("core.models", "import", "core.models")],
        );

        let file_symbols: HashMap<String, Vec<Symbol>> = HashMap::new();
        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_resolve_python_imports_init_package() {
        let mut imports = HashMap::new();
        imports.insert("core/__init__.py".to_string(), vec![]);
        imports.insert(
            "app.py".to_string(),
            vec![make_import_sym("core", "import", "core")],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("core/__init__.py".to_string(), vec![]);
        file_symbols.insert("app.py".to_string(), vec![]);

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["target_path"], "core/__init__.py");
    }

    #[test]
    fn test_resolve_python_imports_dedup() {
        let mut imports = HashMap::new();
        imports.insert(
            "a.py".to_string(),
            vec![
                make_import_sym("b", "import", "b"),
                make_import_sym("b", "import", "b"), // duplicate
            ],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("a.py".to_string(), vec![]);
        file_symbols.insert("b.py".to_string(), vec![]);

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_resolve_python_imports_from_import_symbol_level() {
        // server.py does `from config import EmbeddingConfig`
        let mut imports = HashMap::new();
        imports.insert(
            "server.py".to_string(),
            vec![make_import_sym("EmbeddingConfig", "from_import", "config")],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("server.py".to_string(), vec![]);
        file_symbols.insert(
            "config.py".to_string(),
            vec![make_def_sym("EmbeddingConfig", SymbolKind::Class)],
        );

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        // Should target the symbol, not the file
        let to = edges[0]["_to"].as_str().unwrap();
        assert!(to.starts_with("codebase_symbols/"), "expected symbol edge, got: {to}");
        assert!(to.contains("EmbeddingConfig"));
    }

    #[test]
    fn test_resolve_python_imports_fallback_to_file() {
        // server.py does `from config import SomethingUnknown`
        let mut imports = HashMap::new();
        imports.insert(
            "server.py".to_string(),
            vec![make_import_sym("SomethingUnknown", "from_import", "config")],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("server.py".to_string(), vec![]);
        file_symbols.insert("config.py".to_string(), vec![]); // no symbols defined

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        // Should fall back to file→file
        let to = edges[0]["_to"].as_str().unwrap();
        assert!(to.starts_with("codebase_files/"), "expected file edge fallback, got: {to}");
    }
}
