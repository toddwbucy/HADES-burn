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
//! - Rust and Go semantic enrichment through language servers
//! - Per-file error isolation in batch mode

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ignore::WalkBuilder;
use serde_json::{Value, json};
use tracing::{debug, error, info, warn};

use hades_core::HadesConfig;
use hades_core::chunking::ChunkingStrategy;
use hades_core::code::lsp::go_symbols::GoSymbolExtractor;
use hades_core::code::lsp::symbols::FileExtraction;
use hades_core::code::lsp::{
    EdgeKind, GoplsSession, LspEdgeResolver, RustAnalyzerSession, RustSymbolExtractor,
    group_files_by_crate, group_files_by_go_module,
};
use hades_core::code::{
    self, AnalysisOptions, AnalysisTier, AnalyzerOutcome, AstChunking, Language, Symbol, SymbolKind,
};
use hades_core::code::{cpp_edges, python_calls, rust_imports, tree_sitter_edges};
use hades_core::db::collections::CODEBASE;
use hades_core::db::crud;
use hades_core::db::keys;
use hades_core::db::query::ExecutionTarget;
use hades_core::db::{ArangoErrorKind, ArangoPool};
use hades_core::persephone::embedding::EmbeddingClient;

use super::output::{self, OutputFormat};

/// Hard floor for directory exclusions — applied even when the project has
/// no `.gitignore`, no `.ignore`, and no `.hadesignore`. The `ignore` crate's
/// standard filters (gitignore + hidden-file skipping) already handle most
/// real repos; this list catches the unfortunate case of a flat directory
/// dropped onto disk without any ignore files.
const SKIP_DIRS: &[&str] = &[
    "__pycache__",
    "node_modules",
    "target",
    "venv",
    "dist",
    "build",
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
    /// Set when the chunks were stored but the embed call failed. The file
    /// is structurally ingested (symbols, chunks, edges all present), but
    /// the chunks have no associated embeddings — typically a transient
    /// embedder failure (OOM on the embedding service GPU, request timeout,
    /// etc.). The user-facing summary counts these as
    /// `partial_embedding_failures` so they don't get lost in a green run.
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_error: Option<String>,
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
    /// C/C++/CUDA: rel_path → semantic symbols and resolved call metadata.
    cpp_file_symbols: HashMap<String, Vec<Symbol>>,
    /// Lower-fidelity files used for syntax-only relationship resolution.
    structural_file_symbols: HashMap<String, Vec<Symbol>>,
}

/// Run the codebase ingest command.
// TODO: support --batch to enable parallel/batched ingestion
#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: &HadesConfig,
    path: PathBuf,
    language: Option<&str>,
    batch: bool,
    unparsed_ext: &[String],
    compile_commands: Option<&Path>,
    force: bool,
    allow_analysis_downgrade: bool,
) -> Result<()> {
    let cmd_start = Instant::now();

    // Validate path exists.
    if !path.exists() {
        bail!("path not found: {}", path.display());
    }

    // Normalize the unparsed-extension allowlist: strip leading dots, lowercase.
    // Files matching these extensions are embedded without a parser (#121).
    let unparsed_set: std::collections::HashSet<String> = unparsed_ext
        .iter()
        .map(|e| e.trim().trim_start_matches('.').to_lowercase())
        .filter(|e| !e.is_empty())
        .collect();

    // Parse language override if provided.
    let lang_override = match language {
        Some(l) => {
            let lang = match l.to_lowercase().as_str() {
                "python" | "py" => Language::Python,
                "rust" | "rs" => Language::Rust,
                "c" | "cpp" | "c++" | "cuda" | "cu" => Language::Cpp,
                "go" | "golang" => Language::Go,
                other => {
                    bail!("unsupported language: {other}. Supported: python, rust, go, c/c++/cuda")
                }
            };
            Some(lang)
        }
        None => None,
    };

    // Connect to services.
    let db = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // Embedding is optional — ingest proceeds without vectors if the service is unavailable.
    let embedder = match EmbeddingClient::connect_at(&config.embedding.service.socket).await {
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
    let files = discover_files(&path, lang_override, &unparsed_set)?;
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
        cpp_file_symbols: HashMap::new(),
        structural_file_symbols: HashMap::new(),
    };
    // Collect absolute paths for Rust files — used for rust-analyzer post-loop phase.
    let mut rust_abs_paths: Vec<PathBuf> = Vec::new();
    // Go starts with Tree-sitter and is semantically enriched by gopls after
    // the full module file set is available.
    let mut go_abs_paths: Vec<PathBuf> = Vec::new();

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

        // Route unparsed-allowlisted files (e.g. CUDA `.cu`) through the
        // parser-free fallback: line/size chunk + embed, no AST (#121).
        let file_ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        // The unparsed allowlist is orthogonal to `--language`: an allowlisted
        // extension with no recognized parser (e.g. `.cu`) always takes the
        // parser-free path, even when `--language` is set for other files.
        // `discover_files` likewise includes these regardless of the override.
        let is_unparsed = Language::from_path(&rel_path).is_none()
            && file_ext
                .as_deref()
                .is_some_and(|e| unparsed_set.contains(e));

        // Track Rust files for rust-analyzer post-loop enrichment (parsed only).
        let is_rust = !is_unparsed
            && (lang_override == Some(Language::Rust) || file_ext.as_deref() == Some("rs"));
        if is_rust {
            rust_abs_paths.push(file_path.clone());
        }
        let is_go = !is_unparsed
            && (lang_override == Some(Language::Go) || file_ext.as_deref() == Some("go"));
        if is_go {
            go_abs_paths.push(file_path.clone());
        }

        let result = if is_unparsed {
            ingest_unparsed_file(
                &db,
                embedder.as_ref(),
                config,
                file_path,
                &rel_path,
                None,
                "no registered language or grammar",
                force,
                allow_analysis_downgrade,
            )
            .await
        } else {
            ingest_file(
                &db,
                embedder.as_ref(),
                config,
                file_path,
                &rel_path,
                lang_override,
                &mut imports,
                compile_commands,
                force,
                allow_analysis_downgrade,
            )
            .await
        };

        let duration = item_start.elapsed().as_millis() as u64;
        match result {
            Ok(r) => results.push(FileResult {
                duration_ms: duration,
                ..r
            }),
            Err(e) => {
                error!(path = %rel_path, error = %e, "ingest failed");
                results.push(FileResult {
                    path: rel_path,
                    success: false,
                    language: None,
                    num_symbols: None,
                    num_chunks: None,
                    num_embeddings: None,
                    embedding_error: None,
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
        info!(
            edge_count = py_import_edges.len(),
            "resolved Python import edges"
        );
        if let Err(e) =
            crud::insert_documents(&db, CODEBASE.imports_edges, &py_import_edges, true).await
        {
            warn!(error = %e, "failed to store Python import edges");
        }
    }

    // Resolve Python call graph edges (symbol → symbol). Uses the calls + parent_symbol
    // metadata attached during AST extraction; reuses the bare-name index already built
    // for imports as the Strategy-3 fallback.
    let py_qualified_index = python_calls::build_qualified_index(&imports.python_file_symbols);
    let py_call_edges = python_calls::resolve_python_calls(
        &imports.python_file_symbols,
        &py_qualified_index,
        &py_symbol_index,
    );
    if !py_call_edges.is_empty() {
        info!(
            edge_count = py_call_edges.len(),
            "resolved Python call edges"
        );
        if let Err(e) =
            crud::insert_documents(&db, CODEBASE.calls_edges, &py_call_edges, true).await
        {
            warn!(error = %e, "failed to store Python call edges");
        }
    }

    // Resolve compiler-grade C/C++/CUDA calls, including CUDA kernel launches.
    // libclang records target USRs and definition spans during each file parse;
    // this batch phase maps them onto HADES's cross-file span keys.
    let cpp_call_edges = cpp_edges::resolve_cpp_calls(&base, &imports.cpp_file_symbols);
    if !cpp_call_edges.is_empty() {
        info!(
            edge_count = cpp_call_edges.len(),
            "resolved C/C++/CUDA call edges"
        );
        if let Err(e) =
            crud::insert_documents(&db, CODEBASE.calls_edges, &cpp_call_edges, true).await
        {
            warn!(error = %e, "failed to store C/C++/CUDA call edges");
        }
    }

    let structural_edges = tree_sitter_edges::resolve(&imports.structural_file_symbols);
    if !structural_edges.calls.is_empty() {
        crud::insert_documents(&db, CODEBASE.calls_edges, &structural_edges.calls, true)
            .await
            .context("failed to store Tree-sitter call edges")?;
    }
    if !structural_edges.imports.is_empty() {
        crud::insert_documents(&db, CODEBASE.imports_edges, &structural_edges.imports, true)
            .await
            .context("failed to store Tree-sitter import edges")?;
    }

    // Resolve Rust import graph edges (file → symbol).
    let rust_symbol_index = rust_imports::build_symbol_index(&imports.rust_file_symbols);
    let rs_import_edges =
        rust_imports::resolve_rust_imports(&imports.rust_imports, &rust_symbol_index);
    if !rs_import_edges.is_empty() {
        info!(
            edge_count = rs_import_edges.len(),
            "resolved Rust import edges"
        );
        if let Err(e) =
            crud::insert_documents(&db, CODEBASE.imports_edges, &rs_import_edges, true).await
        {
            warn!(error = %e, "failed to store Rust import edges");
        }
    }

    let total_import_edges =
        py_import_edges.len() + rs_import_edges.len() + structural_edges.imports.len();

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
                    crates = stats.workspaces,
                    "rust-analyzer enrichment complete"
                );
                stats
            }
            Err(e) => {
                warn!(error = %e, "rust-analyzer enrichment failed, syn-based data retained");
                SemanticLspStats::default()
            }
        }
    } else {
        SemanticLspStats::default()
    };

    let gopls_stats = if !go_abs_paths.is_empty() {
        match run_gopls_phase(&db, &base, &go_abs_paths).await {
            Ok(stats) => {
                info!(
                    symbols = stats.symbols,
                    edges = stats.edges,
                    modules = stats.workspaces,
                    "gopls semantic enrichment complete"
                );
                stats
            }
            Err(error) => {
                warn!(%error, "gopls enrichment failed; Tree-sitter Go data retained");
                SemanticLspStats::default()
            }
        }
    } else {
        SemanticLspStats::default()
    };

    // Output summary.
    let total = results.len();
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = results
        .iter()
        .filter(|r| !r.success && r.skipped != Some(true))
        .count();
    let skipped = results.iter().filter(|r| r.skipped == Some(true)).count();
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let files_embedded = results
        .iter()
        .filter(|r| r.num_embeddings.is_some_and(|n| n > 0))
        .count();
    let total_embeddings: usize = results.iter().filter_map(|r| r.num_embeddings).sum();

    // Files whose chunks were stored but whose embed call failed (e.g. embedder
    // GPU OOM, timeout). Their per-file `embedding_error` carries the message;
    // we surface a count + the affected paths here so a green-looking "completed"
    // doesn't hide silent vector loss.
    let embedding_failures: Vec<&str> = results
        .iter()
        .filter(|r| r.embedding_error.is_some())
        .map(|r| r.path.as_str())
        .collect();

    let result_data = json!({
        "total": total,
        "completed": succeeded,
        "failed": failed,
        "skipped": skipped,
        "embedding": {
            "service_connected": embedder.is_some(),
            "files_embedded": files_embedded,
            "total_embeddings": total_embeddings,
            "files_with_embedding_failures": embedding_failures.len(),
            "embedding_failure_paths": embedding_failures,
        },
        "import_edges": total_import_edges,
        "python_import_edges": py_import_edges.len(),
        "rust_import_edges": rs_import_edges.len(),
        "python_call_edges": py_call_edges.len(),
        "cpp_call_edges": cpp_call_edges.len(),
        "structural_call_edges": structural_edges.calls.len(),
        "structural_import_edges": structural_edges.imports.len(),
        "rust_analyzer": {
            "symbols": ra_stats.symbols,
            "edges": ra_stats.edges,
            "crates_analyzed": ra_stats.workspaces,
        },
        "gopls": {
            "symbols": gopls_stats.symbols,
            "edges": gopls_stats.edges,
            "modules_analyzed": gopls_stats.workspaces,
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
        Err(e)
            if matches!(
                e.kind(),
                ArangoErrorKind::Forbidden
                    | ArangoErrorKind::NotFound
                    | ArangoErrorKind::Unavailable
            ) =>
        {
            // Non-fatal: edge collections work for AQL traversals without
            // a named graph wrapper. Forbidden/NotFound typically mean the
            // Metis proxy or RBAC blocks the gharial management endpoint;
            // Unavailable means the endpoint is temporarily down.
            warn!(
                graph = CODEBASE_GRAPH,
                error = %e,
                kind = ?e.kind(),
                "failed to create named graph (non-fatal — edges still work)"
            );
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
            .with_context(|| format!("failed to ensure index on {collection} {fields:?}"))?;
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
fn discover_files(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<Vec<PathBuf>> {
    // Whether a path's (lowercased) extension is in the unparsed allowlist.
    let ext_allowed = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| unparsed_set.contains(&e.to_lowercase()))
    };

    if path.is_file() {
        let path_str = path.to_string_lossy();
        if lang_override.is_some() || Language::from_path(&path_str).is_some() || ext_allowed(path)
        {
            return Ok(vec![path.to_path_buf()]);
        }
        bail!(
            "unsupported file type: {}. Use --language or --unparsed-ext to override.",
            path.display()
        );
    }

    let mut files = Vec::new();
    let walker = WalkBuilder::new(path)
        .follow_links(false)
        .add_custom_ignore_filename(".hadesignore")
        .filter_entry(|entry| {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && let Some(name) = entry.file_name().to_str()
            {
                return !SKIP_DIRS.contains(&name);
            }
            true
        })
        .build();

    for entry in walker {
        let entry = entry.context("error walking directory")?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let entry_path = entry.path();
        let path_str = entry_path.to_string_lossy();
        let include = if Language::from_path(&path_str).is_some() {
            // File has a recognized source extension — always include.
            true
        } else if ext_allowed(entry_path) {
            // Extension is in the unparsed allowlist (e.g. cu,cuh) — include
            // for the parser-free embedding fallback (#121).
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
#[allow(clippy::too_many_arguments)]
async fn ingest_file(
    db: &ArangoPool,
    embedder: Option<&EmbeddingClient>,
    config: &HadesConfig,
    file_path: &Path,
    rel_path: &str,
    lang_override: Option<Language>,
    imports: &mut ImportContext,
    compile_commands: Option<&Path>,
    force: bool,
    allow_analysis_downgrade: bool,
) -> Result<FileResult> {
    // Read source.
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;

    // Detect language.
    let lang = lang_override
        .or_else(|| Language::from_path(rel_path))
        .ok_or_else(|| anyhow::anyhow!("cannot detect language for {rel_path}"))?;

    // Analyze.
    // Pass the absolute path to the analyzer so libclang resolves C/C++ includes
    // consistently regardless of cwd. Keys and logging still use rel_path.
    let options = AnalysisOptions {
        compilation_database: compile_commands.map(Path::to_path_buf),
    };
    let mut analysis =
        match code::analyze_with_fallback(&source, lang, &file_path.to_string_lossy(), &options) {
            AnalyzerOutcome::Success(analysis) => analysis,
            AnalyzerOutcome::Failed { analyzer, reason } => {
                warn!(
                    path = rel_path,
                    analyzer,
                    reason,
                    "semantic and structural analysis unavailable; using raw text fallback"
                );
                return ingest_unparsed_file(
                    db,
                    embedder,
                    config,
                    file_path,
                    rel_path,
                    Some(lang.name()),
                    &reason,
                    force,
                    allow_analysis_downgrade,
                )
                .await;
            }
        };
    info!(
        path = rel_path,
        analyzer = analysis.analyzer,
        analysis_tier = %analysis.analysis_tier,
        fallback_reason = analysis.fallback_reason.as_deref(),
        "selected code analyzer"
    );

    // Check for incremental skip via symbol_hash.
    // Only skip if the code is unchanged AND embeddings aren't needed (either
    // already present or no embedder available to backfill). `--force` bypasses
    // this entirely, re-ingesting in place (#145) — the per-file purge below
    // touches only this file's own symbols and outbound edges, so inbound
    // authored bridge edges are preserved (unlike a cascading `db purge`).
    let fkey = keys::file_key(rel_path);
    if preserve_higher_fidelity(db, &fkey, analysis.analysis_tier, allow_analysis_downgrade).await?
    {
        warn!(
            path = rel_path,
            incoming_tier = %analysis.analysis_tier,
            "preserving higher-fidelity stored analysis"
        );
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang.name().to_string()),
            num_symbols: None,
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: Some("higher-fidelity stored analysis preserved".to_string()),
            duration_ms: 0,
        });
    }
    if !force
        && check_unchanged(db, &fkey, &analysis.symbol_hash, embedder.is_some()).await?
            == Some(true)
    {
        debug!(
            path = rel_path,
            "unchanged (same symbol_hash, embeddings present), skipping"
        );
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang.name().to_string()),
            num_symbols: Some(analysis.symbols.len()),
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }

    // Purge this file's existing symbols and incident edges before re-writing.
    // Symbol/edge inserts are overwrite-by-key only, so without this a renamed
    // or deleted symbol would leave an orphaned row that later inflates
    // `symbol_count` and dangles in the graph (#126). We only reach here when
    // the file actually changed (the unchanged-skip returned above), so an
    // unchanged file — which has no orphans — is never needlessly purged.
    // RA enrichment runs afterward and *augments* the freshly-written syn set.
    purge_file_symbols_and_edges(db, &fkey).await;

    // Collect Python import symbols for later edge resolution.
    if lang == Language::Python {
        let py_import_syms: Vec<Symbol> = analysis
            .symbols
            .iter()
            .filter(|s| s.kind == hades_core::code::SymbolKind::Import)
            .cloned()
            .collect();
        if !py_import_syms.is_empty() {
            imports
                .python_imports
                .insert(rel_path.to_string(), py_import_syms);
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
                        Some(keys::symbol_key(&fkey, &s.qualified_name(), s.start_line))
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
                "analysis_tier": analysis.analysis_tier.as_str(),
                "analyzer": analysis.analyzer,
            })
        })
        .collect();

    // Build symbol documents (primitives only — imports and impl blocks are not vertices).
    let symbol_docs: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| s.kind.is_primitive())
        .map(|s| {
            let qname = s.qualified_name();
            let skey = keys::symbol_key(&fkey, &qname, s.start_line);
            json!({
                "_key": skey,
                "file_key": fkey,
                "file_path": rel_path,
                "name": s.name,
                "qualified_name": qname,
                "kind": s.kind.universal_kind().unwrap(),
                "lang_kind": s.kind.lang_kind(),
                "start_line": s.start_line,
                "end_line": s.end_line,
                "metadata": s.metadata,
                "analysis_tier": analysis.analysis_tier.as_str(),
                "analyzer": analysis.analyzer,
            })
        })
        .collect();

    // Build defines edges (file → symbol) for primitives only.
    let define_edges: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| s.kind.is_primitive())
        .map(|s| {
            let skey = keys::symbol_key(&fkey, &s.qualified_name(), s.start_line);
            let edge_key = keys::edge_key(&fkey, "defines", &skey);
            json!({
                "_key": edge_key,
                "_from": format!("{}/{}", CODEBASE.files, fkey),
                "_to": format!("{}/{}", CODEBASE.symbols, skey),
                "file_path": rel_path,
                "symbol_name": s.name,
                "analysis_tier": analysis.analysis_tier.as_str(),
                "analyzer": analysis.analyzer,
                "resolution": if analysis.analysis_tier == AnalysisTier::Semantic { "semantic" } else { "syntactic" },
            })
        })
        .collect();

    // Remove stale embeddings for this file before (re-)embedding.
    // This ensures that if embedding is skipped or fails, old vectors
    // from a previous run (which embed outdated text) don't linger.
    delete_file_embeddings(db, &fkey).await;

    // Embed chunks (skipped if embedder is unavailable).
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let (embedding_docs, embedding_error): (Vec<Value>, Option<String>) = match embedder {
        Some(emb) if !chunk_texts.is_empty() => {
            match emb
                .embed(&chunk_texts, "code", Some(config.embedding.batch.size))
                .await
            {
                Ok(embed_result) => {
                    let docs = embed_result
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
                        .collect::<Vec<Value>>();
                    (docs, None)
                }
                Err(e) => {
                    warn!(path = rel_path, error = %e, "embedding failed, storing without vectors");
                    (Vec::new(), Some(e.to_string()))
                }
            }
        }
        _ => (Vec::new(), None),
    };
    let num_embeddings_written = embedding_docs.len();

    // Build file document (after embedding so we can record embedding_count).
    // symbol_count reflects primitives only (no imports, no impl blocks), and is
    // counted from the DEDUPLICATED stored set (distinct `_key`) rather than the
    // pre-dedup primitive list — the upsert collapses any same-keyed symbols, so
    // counting the input would over-report and break `symbol_count_consistency`
    // (#113). With qualified-name keying collisions should not occur, but this
    // keeps the denorm honest regardless.
    let primitive_count = symbol_docs
        .iter()
        .filter_map(|d| d["_key"].as_str())
        .collect::<std::collections::HashSet<_>>()
        .len();
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
        "analysis_tier": analysis.analysis_tier.as_str(),
        "analyzer": analysis.analyzer,
        "fallback_reason": analysis.fallback_reason,
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

    // Transfer symbols into relationship indexes.
    if analysis.analysis_tier == AnalysisTier::Structural {
        imports
            .structural_file_symbols
            .insert(rel_path.to_string(), analysis.symbols.clone());
    }
    match lang {
        Language::Rust if uses_semantic_relationship_resolver(lang, analysis.analysis_tier) => {
            imports
                .rust_file_symbols
                .insert(rel_path.to_string(), std::mem::take(&mut analysis.symbols));
        }
        Language::Python if uses_semantic_relationship_resolver(lang, analysis.analysis_tier) => {
            imports
                .python_file_symbols
                .insert(rel_path.to_string(), std::mem::take(&mut analysis.symbols));
        }
        Language::Cpp if uses_semantic_relationship_resolver(lang, analysis.analysis_tier) => {
            imports
                .cpp_file_symbols
                .insert(rel_path.to_string(), std::mem::take(&mut analysis.symbols));
        }
        Language::Cpp => {}
        Language::Go => {}
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
        embedding_error,
        skipped: None,
        error: None,
        duration_ms: 0,
    })
}

fn uses_semantic_relationship_resolver(language: Language, tier: AnalysisTier) -> bool {
    matches!(language, Language::Rust | Language::Python | Language::Cpp)
        && tier == AnalysisTier::Semantic
}

// ── Unparsed-language fallback (#121) ────────────────────────────────────

/// Map an unparsed file extension to a language label for the file node.
fn unparsed_language_label(rel_path: &str) -> &'static str {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    match ext.as_deref() {
        Some("cu") | Some("cuh") => "cuda",
        Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hh") | Some("hxx") => "cpp",
        // `.h` is shared by C and C++; labelled "c" as an intentional
        // simplification. The label is for human display / RGCN features only,
        // not parsing, so the ambiguity is harmless here.
        Some("c") | Some("h") => "c",
        Some("go") => "go",
        _ => "other",
    }
}

/// Ingest a file whose language has no parser: size-chunk the raw text, embed
/// the chunks as node features, and attach them to the file node — WITHOUT
/// symbol/edge extraction. The file node is *merged* (existing fields
/// preserved), not overwritten, so a pre-existing node's metadata survives
/// (e.g. an externally-created stub's `note`/`source`). See #121.
#[allow(clippy::too_many_arguments)]
async fn ingest_unparsed_file(
    db: &ArangoPool,
    embedder: Option<&EmbeddingClient>,
    config: &HadesConfig,
    file_path: &Path,
    rel_path: &str,
    language_label: Option<&str>,
    fallback_reason: &str,
    force: bool,
    allow_analysis_downgrade: bool,
) -> Result<FileResult> {
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;
    let fkey = keys::file_key(rel_path);
    let lang_label = language_label.unwrap_or_else(|| unparsed_language_label(rel_path));

    if preserve_higher_fidelity(db, &fkey, AnalysisTier::Text, allow_analysis_downgrade).await? {
        warn!(
            path = rel_path,
            fallback_reason, "raw fallback skipped to preserve higher-fidelity stored analysis"
        );
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang_label.to_string()),
            num_symbols: None,
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: Some("higher-fidelity stored analysis preserved".to_string()),
            duration_ms: 0,
        });
    }
    let content_hash = code::compute_content_hash(&source);
    if !force && check_unchanged(db, &fkey, &content_hash, embedder.is_some()).await? == Some(true)
    {
        debug!(path = rel_path, "unchanged raw text, skipping");
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang_label.to_string()),
            num_symbols: Some(0),
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }
    if allow_analysis_downgrade {
        purge_file_symbols_and_edges(db, &fkey).await;
    }

    // Parser-free chunking: empty defs => whole file, split at line boundaries
    // to stay under the max chunk size.
    let chunker = AstChunking::new(Vec::new());
    let chunks = chunker.chunk(&source);
    let num_chk = chunks.len();

    // Chunk documents — no symbol overlap (unparsed files have no symbols).
    let chunk_docs: Vec<Value> = chunks
        .iter()
        .map(|c| {
            let ckey = keys::chunk_key(&fkey, c.chunk_index);
            json!({
                "_key": ckey,
                "file_key": fkey,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "symbols": [],
                "analysis_tier": "text",
                "analyzer": "raw-text",
            })
        })
        .collect();

    // Clear stale chunks AND embeddings before re-writing, so a re-ingest that
    // produces fewer chunks leaves no orphaned high-index chunk/vector docs.
    delete_file_chunks(db, &fkey).await;
    delete_file_embeddings(db, &fkey).await;

    // Embed chunks (skipped if embedder unavailable). Same path/task as parsed.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let (embedding_docs, embedding_error): (Vec<Value>, Option<String>) = match embedder {
        Some(emb) if !chunk_texts.is_empty() => {
            match emb
                .embed(&chunk_texts, "code", Some(config.embedding.batch.size))
                .await
            {
                Ok(embed_result) => {
                    let docs = embed_result
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
                        .collect::<Vec<Value>>();
                    (docs, None)
                }
                Err(e) => {
                    warn!(path = rel_path, error = %e, "embedding failed, storing without vectors");
                    (Vec::new(), Some(e.to_string()))
                }
            }
        }
        _ => (Vec::new(), None),
    };
    let num_embeddings_written = embedding_docs.len();

    // Persist chunks + embeddings (ours — overwrite-by-key is fine).
    if !chunk_docs.is_empty() {
        crud::insert_documents(db, CODEBASE.chunks, &chunk_docs, true)
            .await
            .context("failed to store chunk documents")?;
    }
    if !embedding_docs.is_empty() {
        crud::insert_documents(db, CODEBASE.embeddings, &embedding_docs, true)
            .await
            .context("failed to store embedding documents")?;
    }

    // Merge the file node — preserve any pre-existing fields, set only ours,
    // create if absent.
    let total_lines = source.lines().count();
    let fields = json!({
        "path": rel_path,
        "rel_path": rel_path,
        "kind": "file",
        "language": lang_label,
        "symbol_hash": content_hash,
        "symbol_count": 0,
        "chunk_count": num_chk,
        "embedding_count": num_embeddings_written,
        "total_lines": total_lines,
        "status": "PROCESSED",
        "analysis_tier": "text",
        "analyzer": "raw-text",
        "fallback_reason": fallback_reason,
        "ingested_at": chrono::Utc::now().to_rfc3339(),
    });
    upsert_merge_file_node(db, &fkey, fields).await?;

    info!(
        path = rel_path,
        language = lang_label,
        chunks = num_chk,
        embeddings = num_embeddings_written,
        "ingested (unparsed)"
    );

    Ok(FileResult {
        path: rel_path.to_string(),
        success: true,
        language: Some(lang_label.to_string()),
        num_symbols: Some(0),
        num_chunks: Some(num_chk),
        num_embeddings: Some(num_embeddings_written),
        embedding_error,
        skipped: None,
        error: None,
        duration_ms: 0,
    })
}

/// Merge-write a `codebase_files` node: PATCH (preserving existing fields) when
/// it exists, otherwise insert. Lets the unparsed fallback attach to a
/// pre-existing file node without clobbering its metadata (#121).
async fn upsert_merge_file_node(db: &ArangoPool, fkey: &str, fields: Value) -> Result<()> {
    match crud::update_document(db, CODEBASE.files, fkey, &fields).await {
        Ok(_) => Ok(()),
        Err(e) if e.is_not_found() => {
            let mut doc = fields;
            doc["_key"] = json!(fkey);
            crud::insert_documents(db, CODEBASE.files, &[doc], true)
                .await
                .context("failed to insert file document")?;
            Ok(())
        }
        Err(e) => Err(anyhow::Error::new(e).context("failed to merge file document")),
    }
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
    if let Err(e) =
        hades_core::db::query::query(db, aql, Some(&bind), None, false, ExecutionTarget::Writer)
            .await
    {
        debug!(file_key, error = %e, "failed to clean up old embeddings (non-fatal)");
    }
}

/// Delete all chunk documents for a file.
///
/// Called before re-chunking on the unparsed path so that a re-ingest which
/// produces fewer chunks leaves no orphaned high-index chunk docs behind
/// (overwrite-by-key only updates the chunks that still exist).
async fn delete_file_chunks(db: &ArangoPool, file_key: &str) {
    let aql = "FOR c IN @@col FILTER c.file_key == @fk REMOVE c IN @@col";
    let bind = json!({ "@col": CODEBASE.chunks, "fk": file_key });
    if let Err(e) =
        hades_core::db::query::query(db, aql, Some(&bind), None, false, ExecutionTarget::Writer)
            .await
    {
        debug!(file_key, error = %e, "failed to clean up old chunks (non-fatal)");
    }
}

/// Purge a file's existing symbols and the **source-owned (outgoing) edges**
/// the file's own ingest will recreate, before re-writing. Symbol/edge inserts
/// are overwrite-by-key only, so without this a renamed/deleted symbol leaves
/// an orphaned row — which inflates `symbol_count` and dangles in the graph
/// (#126). This makes `codebase_symbols` authoritative on re-ingest.
///
/// Only edges with `_from` in this file (the file node for `defines`, or one of
/// its symbols for `calls`/`implements`/`imports`) are removed — those are
/// rebuilt by this file's ingest. **Incoming** edges (`_to` in this file) are
/// owned by *other* source files and are NOT touched here: deleting them would
/// drop valid edges that a skipped source file never rebuilds. Incoming edges
/// left dangling by a rename/delete are cleaned by `hades codebase prune`.
///
/// The `ids` list is snapshotted in-query from the current symbols plus the
/// file `_id`, so the edge filter is consistent even under concurrent inserts.
async fn purge_file_symbols_and_edges(db: &ArangoPool, file_key: &str) {
    let aql = "\
        LET ids = APPEND( \
            (FOR s IN @@symbols FILTER s.file_key == @key RETURN s._id), \
            [CONCAT(@files_name, '/', @key)]) \
        LET syms = (FOR d IN @@symbols FILTER d.file_key == @key REMOVE d IN @@symbols RETURN 1) \
        LET defs = (FOR e IN @@defines FILTER e._from IN ids REMOVE e IN @@defines RETURN 1) \
        LET calls = (FOR e IN @@calls FILTER e._from IN ids REMOVE e IN @@calls RETURN 1) \
        LET impls = (FOR e IN @@implements FILTER e._from IN ids REMOVE e IN @@implements RETURN 1) \
        LET imps = (FOR e IN @@imports FILTER e._from IN ids REMOVE e IN @@imports RETURN 1) \
        RETURN 1";
    let bind = json!({
        "@symbols": CODEBASE.symbols,
        "@defines": CODEBASE.defines_edges,
        "@calls": CODEBASE.calls_edges,
        "@implements": CODEBASE.implements_edges,
        "@imports": CODEBASE.imports_edges,
        "files_name": CODEBASE.files,
        "key": file_key,
    });
    if let Err(e) =
        hades_core::db::query::query(db, aql, Some(&bind), None, false, ExecutionTarget::Writer)
            .await
    {
        warn!(file_key, error = %e, "failed to purge stale symbols/edges before re-ingest");
    }
}

// ── Incremental check ───────────────────────────────────────────────────

/// Return true when replacing the stored artifacts would lower fidelity.
fn should_preserve_tier(
    stored: Option<AnalysisTier>,
    incoming: AnalysisTier,
    allow_downgrade: bool,
) -> bool {
    !allow_downgrade && stored.is_some_and(|tier| tier > incoming)
}

/// Enforce monotonic analyzer fidelity before any destructive per-file write.
async fn preserve_higher_fidelity(
    db: &ArangoPool,
    file_key: &str,
    incoming: AnalysisTier,
    allow_downgrade: bool,
) -> Result<bool> {
    match crud::get_document(db, CODEBASE.files, file_key).await {
        Ok(doc) => {
            let stored = doc["analysis_tier"].as_str().and_then(AnalysisTier::parse);
            Ok(should_preserve_tier(stored, incoming, allow_downgrade))
        }
        Err(e) if e.is_not_found() => Ok(false),
        Err(e) => Err(e.into()),
    }
}

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
            // Code unchanged. Skip only if embeddings aren't needed or already complete.
            if embedder_available {
                // Files with no chunks have nothing to embed — always skip.
                let chunk_count = doc["chunk_count"].as_u64().unwrap_or(0);
                if chunk_count == 0 {
                    return Ok(Some(true));
                }
                // Skip only if every chunk has an embedding. A partial-success
                // state (e.g. from a prior ingest that hit embedder OOM on some
                // batches) must NOT be treated as "done" — the missing chunks
                // need backfill on this run. Previously this checked only
                // `embedding_count > 0`, which left partial-success files stuck.
                let embedding_count = doc["embedding_count"].as_u64().unwrap_or(0);
                Ok(Some(embedding_count >= chunk_count))
            } else {
                Ok(Some(true)) // no embedder → nothing to backfill → skip
            }
        }
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ── semantic language-server enrichment ───────────────────────────────

/// Stats returned from the rust-analyzer enrichment phase.
#[derive(Default)]
struct SemanticLspStats {
    symbols: usize,
    edges: usize,
    workspaces: usize,
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
) -> Result<SemanticLspStats> {
    let groups = group_files_by_crate(rust_files);
    if groups.is_empty() {
        return Ok(SemanticLspStats::default());
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

        let extractor = RustSymbolExtractor::new(&session, true).with_path_root(base);
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

    store_lsp_extractions(db, all_extractions, crates_analyzed, "rust-analyzer", "ra").await
}

/// Run gopls over each discovered Go module. Tree-sitter artifacts remain in
/// place when gopls is absent or a module fails, satisfying the #152 fallback
/// contract without a database-wide language mode.
async fn run_gopls_phase(
    db: &ArangoPool,
    base: &Path,
    go_files: &[PathBuf],
) -> Result<SemanticLspStats> {
    let groups = group_files_by_go_module(go_files);
    if groups.is_empty() {
        return Ok(SemanticLspStats::default());
    }
    info!(
        module_count = groups.len(),
        file_count = go_files.len(),
        "starting gopls semantic enrichment"
    );
    let mut all_extractions = HashMap::new();
    let mut modules_analyzed = 0;
    for (module_root, module_files) in &groups {
        let session = match GoplsSession::start(module_root).await {
            Ok(session) => session,
            Err(error) => {
                warn!(
                    module_root = %module_root.display(),
                    %error,
                    "gopls unavailable for module; Tree-sitter data retained"
                );
                continue;
            }
        };
        let extractor = GoSymbolExtractor::new(&session, true).with_path_root(base);
        let file_refs: Vec<&Path> = module_files.iter().map(PathBuf::as_path).collect();
        for (absolute, extraction) in extractor.extract_module(&file_refs).await {
            let absolute = Path::new(&absolute);
            let relative = absolute
                .strip_prefix(base)
                .unwrap_or(absolute)
                .to_string_lossy()
                .into_owned();
            all_extractions.insert(relative, extraction);
        }
        modules_analyzed += 1;
        if let Err(error) = session.shutdown().await {
            debug!(%error, "gopls shutdown warning (non-fatal)");
        }
    }
    store_lsp_extractions(db, all_extractions, modules_analyzed, "gopls", "gopls").await
}

/// Persist language-neutral LSP symbol documents and semantic edges.
async fn store_lsp_extractions(
    db: &ArangoPool,
    all_extractions: HashMap<String, FileExtraction>,
    workspaces: usize,
    analyzer: &'static str,
    metadata_prefix: &'static str,
) -> Result<SemanticLspStats> {
    if all_extractions.is_empty() {
        return Ok(SemanticLspStats {
            workspaces,
            ..Default::default()
        });
    }
    let file_patches: Vec<(String, usize, String)> = all_extractions
        .iter()
        .map(|(path, extraction)| {
            (
                path.clone(),
                extraction.symbols.len(),
                extraction.analyzed_at.clone(),
            )
        })
        .collect();
    let resolver = LspEdgeResolver::new(all_extractions, analyzer);
    let symbol_docs = resolver.build_symbol_documents();
    let semantic_edges = resolver.build_edges();

    let sym_count = symbol_docs.len();
    let edge_count = semantic_edges.len();

    // Store enriched symbol documents (overwrite=true for idempotent re-runs).
    if !symbol_docs.is_empty() {
        let docs: Vec<Value> = symbol_docs
            .iter()
            .filter_map(|s| serde_json::to_value(s).ok())
            .collect();
        crud::insert_documents(db, CODEBASE.symbols, &docs, true)
            .await
            .with_context(|| format!("failed to store {analyzer} symbol documents"))?;
        info!(count = docs.len(), analyzer, "stored LSP symbol documents");
    }

    // Store edges grouped by collection (collection-per-relation).
    if !semantic_edges.is_empty() {
        // Build edge documents with deterministic keys.
        let edge_docs: Vec<(EdgeKind, Value)> = semantic_edges
            .iter()
            .map(|e| {
                let from_suffix = e.from.rsplit('/').next().unwrap_or(&e.from);
                let to_suffix = e.to.rsplit('/').next().unwrap_or(&e.to);
                let edge_key = keys::edge_key(from_suffix, e.kind.as_str(), to_suffix);
                let mut doc = json!({
                    "_key": edge_key,
                    "_from": e.from,
                    "_to": e.to,
                    "analysis_tier": "semantic",
                    "analyzer": analyzer,
                    "resolution": "semantic",
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

        info!(
            count = edge_docs.len(),
            analyzer, "stored LSP semantic edges"
        );
    }

    let mut patched_count = 0;
    for (rel_path, sym_count, analyzed_at) in &file_patches {
        let fkey = keys::file_key(rel_path);
        let mut patch = json!({
            "analysis_tier": "semantic",
            "analyzer": analyzer,
        });
        if let Value::Object(fields) = &mut patch {
            fields.insert(format!("{metadata_prefix}_analyzed"), json!(true));
            fields.insert(format!("{metadata_prefix}_symbol_count"), json!(sym_count));
            fields.insert(format!("{metadata_prefix}_analyzed_at"), json!(analyzed_at));
        }
        match crud::update_document(db, CODEBASE.files, &fkey, &patch).await {
            Ok(_) => patched_count += 1,
            Err(e) => {
                debug!(file_key = %fkey, error = %e, "failed to patch file document (non-fatal)");
            }
        }
    }
    if patched_count > 0 {
        info!(
            count = patched_count,
            analyzer, "patched file documents with semantic LSP metadata"
        );
    }

    // Recompute `symbol_count` from the authoritative stored set for every
    // LSP-touched file. Enrichment adds symbols (struct fields, methods) beyond
    // the syn primitives that `ingest_file` counted, so the syn-based
    // `symbol_count` is now stale (#126). Counting the actually-stored symbols
    // keeps the denorm consistent — the same "count from the stored set" stance
    // as #113 — and treats the richer RA granularity as canonical. The
    // `file_key` index on `codebase_symbols` keeps the per-file count cheap.
    let touched_fkeys: Vec<String> = file_patches
        .iter()
        .map(|(rel_path, _, _)| keys::file_key(rel_path))
        .collect();
    if !touched_fkeys.is_empty() {
        let n = touched_fkeys.len();
        let aql = "FOR fk IN @fkeys \
                   LET c = LENGTH(FOR s IN @@sym FILTER s.file_key == fk RETURN 1) \
                   UPDATE fk WITH { symbol_count: c } IN @@files";
        let bind = json!({
            "fkeys": touched_fkeys,
            "@sym": CODEBASE.symbols,
            "@files": CODEBASE.files,
        });
        match hades_core::db::query::query(
            db,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Writer,
        )
        .await
        {
            Ok(_) => debug!(
                files = n,
                analyzer, "recomputed symbol_count after semantic LSP enrichment"
            ),
            Err(e) => warn!(
                error = %e,
                analyzer,
                "failed to recompute symbol_count after LSP enrichment (non-fatal)"
            ),
        }
    }

    Ok(SemanticLspStats {
        symbols: sym_count,
        edges: edge_count,
        workspaces,
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
            // Index is keyed by the bare name (call sites use bare names), but
            // the value must be the qualified-name-derived key so edges target
            // the actual stored vertex (#113).
            let skey = keys::symbol_key(&fkey, &sym.qualified_name(), sym.start_line);
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
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
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
            let import_type = sym
                .metadata
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let module = sym
                .metadata
                .get("module")
                .and_then(|v| v.as_str())
                .unwrap_or("");

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
                            && target_path != source_path
                        {
                            let edge_key = keys::edge_key(&source_fkey, "imports", target_skey);
                            if seen.insert(edge_key.clone()) {
                                edges.push(json!({
                                    "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                    "_to": format!("{}/{}", CODEBASE.symbols, target_skey),
                                    "_key": edge_key,
                                    "resolved": true,
                                    "style": "from_import",
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
                        && target_path != source_path
                    {
                        let target_fkey = keys::file_key(target_path);
                        let edge_key = keys::edge_key(&source_fkey, "imports", &target_fkey);
                        if seen.insert(edge_key.clone()) {
                            edges.push(json!({
                                "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                                "_key": edge_key,
                                "resolved": false,
                                "style": "from_import",
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
                        && target_path != source_path
                    {
                        let target_fkey = keys::file_key(target_path);
                        let edge_key = keys::edge_key(&source_fkey, "imports", &target_fkey);
                        if seen.insert(edge_key.clone()) {
                            edges.push(json!({
                                "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                                "_key": edge_key,
                                "resolved": false,
                                "style": "import",
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
    use std::collections::HashSet;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_unparsed_language_label() {
        assert_eq!(unparsed_language_label("core/kernels/adamw.cu"), "cuda");
        assert_eq!(unparsed_language_label("k.cuh"), "cuda");
        assert_eq!(unparsed_language_label("src/foo.cpp"), "cpp");
        assert_eq!(unparsed_language_label("a/b.h"), "c");
        assert_eq!(unparsed_language_label("notes.txt"), "other");
        assert_eq!(unparsed_language_label("Makefile"), "other");
    }

    #[test]
    fn test_analysis_fidelity_is_monotonic_without_explicit_downgrade() {
        assert!(should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            false
        ));
        assert!(should_preserve_tier(
            Some(AnalysisTier::Structural),
            AnalysisTier::Text,
            false
        ));
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            true
        ));
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Structural),
            AnalysisTier::Semantic,
            false
        ));
    }

    #[test]
    fn test_only_semantic_languages_use_dedicated_edge_resolvers() {
        for language in [Language::Rust, Language::Python, Language::Cpp] {
            assert!(uses_semantic_relationship_resolver(
                language,
                AnalysisTier::Semantic
            ));
            assert!(!uses_semantic_relationship_resolver(
                language,
                AnalysisTier::Structural
            ));
        }
        assert!(!uses_semantic_relationship_resolver(
            Language::Go,
            AnalysisTier::Semantic
        ));
    }

    #[test]
    fn test_discover_files_unparsed_ext() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("shader.wgsl"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("shader.vert"), "void main() {}\n").unwrap();
        fs::write(dir.path().join("app.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap();

        // Without the allowlist: only the .py is picked up.
        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1);

        // Explicitly allowlisted extensions are ingested as raw text.
        let allow: HashSet<String> = ["wgsl", "vert"].iter().map(|s| s.to_string()).collect();
        let files = discover_files(dir.path(), None, &allow).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_discover_files_unparsed_single_file() {
        let dir = TempDir::new().unwrap();
        let shader = dir.path().join("backward.wgsl");
        fs::write(&shader, "fn main() {}\n").unwrap();

        // Single unparsed file is rejected without the allowlist...
        assert!(discover_files(&shader, None, &HashSet::new()).is_err());
        // ...and accepted with it.
        let allow: HashSet<String> = ["wgsl"].iter().map(|s| s.to_string()).collect();
        let files = discover_files(&shader, None, &allow).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_single() {
        let dir = TempDir::new().unwrap();
        let py_file = dir.path().join("test.py");
        fs::write(&py_file, "x = 1\n").unwrap();

        let files = discover_files(&py_file, None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("b.rs"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("c.go"), "package demo\n").unwrap();
        fs::write(dir.path().join("d.cpp"), "void run() {}\n").unwrap();
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap();

        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 4); // all registered languages, not .md
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

        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1); // only a.py
    }

    #[test]
    fn test_discover_files_language_override() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("script"), "x = 1\n").unwrap(); // no extension

        // Without override: no files found.
        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 0);

        // With override: extensionless file is included.
        let files = discover_files(dir.path(), Some(Language::Python), &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_override_excludes_non_source() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("script"), "x = 1\n").unwrap(); // no extension — included
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap(); // has extension — excluded
        fs::write(dir.path().join("data.json"), "{}").unwrap(); // has extension — excluded
        fs::write(dir.path().join("real.py"), "x = 1\n").unwrap(); // recognized — included

        let files = discover_files(dir.path(), Some(Language::Python), &HashSet::new()).unwrap();
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
        assert_eq!(edges[0]["resolved"], true);
        assert_eq!(edges[0]["style"], "from_import");
        assert_eq!(edges[0]["source_path"], "core/models.py");
        assert_eq!(edges[0]["symbol_name"], "helper");
        // Should be file→symbol edge
        assert!(
            edges[0]["_to"]
                .as_str()
                .unwrap()
                .contains("codebase_symbols")
        );
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
        assert!(
            to.starts_with("codebase_symbols/"),
            "expected symbol edge, got: {to}"
        );
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
        assert!(
            to.starts_with("codebase_files/"),
            "expected file edge fallback, got: {to}"
        );
    }
}
