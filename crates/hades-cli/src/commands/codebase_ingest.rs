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
#[derive(Default)]
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

    let unparsed_set = normalize_unparsed_ext(unparsed_ext);
    let lang_override = parse_language_arg(language)?;

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
    let base = ingest_base_path(&path);

    // ── Analyzer preflight (#164/#167) ─────────────────────────────────
    // Resolve each needed analyzer (config/env override wins over PATH) and
    // probe it FROM the ingest base, because the rustup shim resolves
    // per-directory. This runs BEFORE any file is touched: `--force` purges a
    // file's semantic edges on the assumption enrichment will rebuild them,
    // so an analyzer that cannot run must stop the ingest up front — after
    // the purge is too late (#164). `--allow-analysis-downgrade` is the
    // explicit override, matching its existing fidelity semantics.
    let needs_rust = files
        .iter()
        .any(|f| is_semantic_target(f, lang_override, &unparsed_set, Language::Rust, "rs"));
    let needs_go = files
        .iter()
        .any(|f| is_semantic_target(f, lang_override, &unparsed_set, Language::Go, "go"));
    let rust_analyzer_cmd = if needs_rust {
        preflight_or_bail(
            "rust-analyzer",
            config.analyzers.rust_analyzer.as_deref(),
            &base,
            allow_analysis_downgrade,
        )?
    } else {
        None
    };
    let gopls_cmd = if needs_go {
        preflight_or_bail(
            "gopls",
            config.analyzers.gopls.as_deref(),
            &base,
            allow_analysis_downgrade,
        )?
    } else {
        None
    };

    // Process each file with per-file error isolation.
    let mut results: Vec<FileResult> = Vec::with_capacity(files.len());
    // File keys whose symbol set was rebuilt this run — the inputs to the
    // post-run dangling-edge sweep (#183).
    let mut rewritten_file_keys: Vec<String> = Vec::new();
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
        let rel_path = rel_path_for(&base, file_path);

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
        // Extensionless scripts are classified by shebang: a `#!…python3` file
        // gets the Python analyzer, a `#!/bin/bash` one has no analyzer and
        // takes the raw-text path rather than erroring out (#183).
        let shebang = if file_ext.is_none() {
            shebang_of(file_path)
        } else {
            None
        };
        let shebang_lang = shebang.and_then(|(_, lang)| lang);
        let has_shebang = shebang.is_some();
        // An explicit `--language` still wins; shebang only fills the gap where
        // there was previously no signal at all.
        let effective_override = lang_override.or(shebang_lang);
        // The unparsed allowlist stays orthogonal to `--language`: an
        // allowlisted extension with no parser takes the raw-text path even when
        // `--language` is set for other files. Only the shebang clause is gated,
        // and only on a shebang-derived language — gating the whole predicate on
        // `effective_override` would send `--unparsed-ext sh` files through
        // whatever `--language` names, and diverge from `is_semantic_target`,
        // which is the split #164 exists to prevent.
        let is_unparsed = Language::from_path(&rel_path).is_none()
            && (file_ext
                .as_deref()
                .is_some_and(|e| unparsed_set.contains(e))
                || (has_shebang && shebang_lang.is_none() && lang_override.is_none()));

        // Track Rust/Go files for post-loop semantic enrichment (parsed only).
        // Uses the SAME predicate as the preflight gate above, so the set the
        // gate protects and the set the phases process cannot diverge — a
        // divergence here is exactly how `--language rust` on non-.rs files
        // would skip the preflight and silently lose enrichment (#164).
        let is_rust_target = is_semantic_target(
            file_path,
            lang_override,
            &unparsed_set,
            Language::Rust,
            "rs",
        );
        if is_rust_target {
            rust_abs_paths.push(file_path.clone());
        }
        let is_go_target =
            is_semantic_target(file_path, lang_override, &unparsed_set, Language::Go, "go");
        if is_go_target {
            go_abs_paths.push(file_path.clone());
        }

        // A post-loop LSP phase will re-supply this file's semantic symbols and
        // edges after the per-file write, so the fidelity guard must not block
        // the rewrite: the purge it is protecting is undone within this same
        // run. Without this, Go — which has no per-file semantic analyzer and so
        // can only ever offer `Structural` — was pinned against re-ingest
        // forever once the gopls phase had stamped the node (#193).
        let reenriched_this_run = (is_rust_target && rust_analyzer_cmd.is_some())
            || (is_go_target && gopls_cmd.is_some());

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
                reenriched_this_run,
            )
            .await
        } else {
            ingest_file(
                &db,
                embedder.as_ref(),
                config,
                file_path,
                &rel_path,
                effective_override,
                &mut imports,
                compile_commands,
                force,
                allow_analysis_downgrade,
                reenriched_this_run,
            )
            .await
        };

        let duration = item_start.elapsed().as_millis() as u64;
        match result {
            Ok(r) => {
                // A file that was actually rewritten (not skipped) had its symbol
                // set purged and rebuilt, so a symbol another file points at may
                // have disappeared. Remember it for the post-run dangling sweep.
                if r.skipped != Some(true) && r.success {
                    rewritten_file_keys.push(keys::file_key(&rel_path));
                }
                results.push(FileResult {
                    duration_ms: duration,
                    ..r
                })
            }
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
    let ra_stats = if !rust_abs_paths.is_empty() && rust_analyzer_cmd.is_some() {
        match run_rust_analyzer_phase(&db, &base, &rust_abs_paths, rust_analyzer_cmd.as_deref())
            .await
        {
            Ok(stats) => {
                info!(
                    symbols = stats.symbols,
                    edges = stats.edges,
                    crates = stats.workspaces,
                    store_errors = stats.store_errors,
                    "rust-analyzer enrichment complete"
                );
                stats
            }
            Err(e) => {
                // `{e:#}` prints the full anyhow context chain — plain `{e}`
                // shows only the outermost context and swallows the underlying
                // cause (#180).
                warn!(error = %format!("{e:#}"), "rust-analyzer enrichment failed, syn-based data retained");
                SemanticLspStats::default()
            }
        }
    } else {
        SemanticLspStats::default()
    };

    // Total enrichment failure is loud, not an info line (#164). The preflight
    // passed, Rust files were ingested, and the phase produced nothing — that
    // is the exact state that previously reported success while the graph
    // silently lost its calls/implements layer. Only the downgrade flag makes
    // proceeding an explicit choice.
    if !rust_abs_paths.is_empty()
        && rust_analyzer_cmd.is_some()
        && ra_stats.workspaces == 0
        && !ra_stats.store_failed
        && !allow_analysis_downgrade
    {
        anyhow::bail!(
            "rust-analyzer enrichment produced nothing across {} Rust file(s) \
             (crates_analyzed = 0) despite a passing preflight. The graph would \
             keep syn symbols but lose calls/implements edges. Investigate the \
             analyzer session logs above, or pass --allow-analysis-downgrade to \
             accept the loss explicitly.",
            rust_abs_paths.len()
        );
    }

    // A store failure is a distinct stage from analysis failure (#180): the
    // analyzer did its job, but the results never reached ArangoDB. Attribute
    // it correctly so operators don't chase analyzer session logs for a
    // database error.
    if ra_stats.store_failed && !allow_analysis_downgrade {
        anyhow::bail!(
            "rust-analyzer analyzed {} crate(s) but storing the enrichment to \
             ArangoDB failed (see the store warnings above for the database \
             error). The graph keeps syn symbols but loses calls/implements \
             edges. Fix the store error and re-run, or pass \
             --allow-analysis-downgrade to accept the loss explicitly.",
            ra_stats.workspaces
        );
    }

    let gopls_stats = if !go_abs_paths.is_empty() && gopls_cmd.is_some() {
        match run_gopls_phase(&db, &base, &go_abs_paths, gopls_cmd.as_deref()).await {
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
                warn!(error = %format!("{error:#}"), "gopls enrichment failed; Tree-sitter Go data retained");
                SemanticLspStats::default()
            }
        }
    } else {
        SemanticLspStats::default()
    };

    // Same store-vs-analysis attribution as the rust-analyzer path (#180):
    // gopls analyzed its modules but the results never reached ArangoDB, so
    // exiting 0 would silently lose the Go calls/implements layer.
    if gopls_stats.store_failed && !allow_analysis_downgrade {
        anyhow::bail!(
            "gopls analyzed {} module(s) but storing the enrichment to ArangoDB \
             failed (see the store warnings above for the database error). The \
             graph keeps Tree-sitter symbols but loses calls/implements edges. \
             Fix the store error and re-run, or pass --allow-analysis-downgrade \
             to accept the loss explicitly.",
            gopls_stats.workspaces
        );
    }

    // Same loud-zero rule as rust-analyzer (#164): a passing preflight with Go
    // files ingested and zero modules analyzed is silent semantic loss, not
    // success.
    if !go_abs_paths.is_empty()
        && gopls_cmd.is_some()
        && gopls_stats.workspaces == 0
        && !allow_analysis_downgrade
    {
        anyhow::bail!(
            "gopls enrichment produced nothing across {} Go file(s) despite a \
             passing preflight. Investigate the session logs above, or pass \
             --allow-analysis-downgrade to accept the loss explicitly.",
            go_abs_paths.len()
        );
    }

    // Report inbound edges left dangling by this run's rebuilds.
    //
    // Purging a file removes only its *outgoing* edges, by design — inbound ones
    // are owned by other source files this run may not have touched. A rebuild
    // that drops a symbol (rename, re-qualification, analyzer change) leaves
    // those pointing at nothing, which fails the `imports_edge_endpoints`
    // invariant in `codebase validate` (#183). Counted, not deleted: the edge
    // records a real dependency, and removing it would erase the only signal
    // that the dependent needs re-ingesting. Runs after the enrichment phases so
    // a symbol rust-analyzer/gopls recreates is not counted as gone.
    let dangling_inbound = count_dangling_inbound(&db, &rewritten_file_keys).await;
    if dangling_inbound > 0 {
        warn!(
            edges = dangling_inbound,
            "inbound edges now point at symbols this run removed; re-run \
             `codebase ingest --force <the same ingest root>` to re-resolve \
             them (--force because the dependents' own symbol_hash is \
             unchanged, and the original root because keys are relative to it \
             -- a narrower path re-bases them and writes duplicate nodes), or \
             run `hades codebase prune-orphans` to drop them"
        );
    }

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
        // Inbound edges now pointing at symbols this run removed. Surfaced in
        // the JSON contract, not just stderr, so an agent parsing stdout sees
        // that the graph needs attention rather than only "completed: N".
        "dangling_inbound_edges": dangling_inbound,
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
            "store_errors": ra_stats.store_errors,
            "store_failed": ra_stats.store_failed,
        },
        "gopls": {
            "symbols": gopls_stats.symbols,
            "edges": gopls_stats.edges,
            "modules_analyzed": gopls_stats.workspaces,
            "store_errors": gopls_stats.store_errors,
            "store_failed": gopls_stats.store_failed,
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
/// Normalize an `--unparsed-ext` allowlist: trim, strip a leading dot, lowercase.
///
/// Shared with `codebase drift` so the same flag value always produces the same
/// discovery set. This is not cosmetic: `discover_files` matches a file's
/// lowercased, dot-less extension against this set, so an entry of `.md` or
/// ` md ` silently matches nothing. If drift normalized differently from ingest,
/// files ingest *did* pick up would be absent from drift's disk set, get reported
/// as `stale`, and — piped into `codebase retire` — have their live nodes
/// deleted. Parity here has to be structural, not conventional.
pub(crate) fn normalize_unparsed_ext(unparsed_ext: &[String]) -> std::collections::HashSet<String> {
    unparsed_ext
        .iter()
        .map(|e| e.trim().trim_start_matches('.').to_lowercase())
        .filter(|e| !e.is_empty())
        .collect()
}

/// Parse a `--language` override, accepting the word forms as well as extensions.
///
/// Shared with `codebase drift` for the same reason as
/// [`normalize_unparsed_ext`]: drift documents that its flags must match the
/// ingest invocation, so it has to accept exactly what ingest accepts. Parsing
/// via `Language::from_extension` alone would reject `rust`, `python`, `golang`,
/// and `cuda` — all of which ingest takes.
pub(crate) fn parse_language_arg(language: Option<&str>) -> Result<Option<Language>> {
    let Some(l) = language else { return Ok(None) };
    let lang = match l.to_lowercase().as_str() {
        "python" | "py" => Language::Python,
        "rust" | "rs" => Language::Rust,
        "c" | "cpp" | "c++" | "cuda" | "cu" => Language::Cpp,
        "go" | "golang" => Language::Go,
        other => {
            bail!("unsupported language: {other}. Supported: python, rust, go, c/c++/cuda")
        }
    };
    Ok(Some(lang))
}

/// The base directory that ingest strips to form a file node's `rel_path`.
///
/// A file node's `_key` is `keys::file_key(rel_path)` where `rel_path` is the
/// path relative to this base — so anything comparing graph keys against the
/// working tree (e.g. `codebase drift`) MUST derive keys through this same
/// function, or every key mismatches and the comparison is meaningless.
pub(crate) fn ingest_base_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    } else {
        path.parent()
            .map(|p| p.canonicalize().unwrap_or_else(|_| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

/// The `rel_path` a file node records: the path relative to the ingest base.
///
/// Single home for this derivation. The ingest loop and `codebase drift` both
/// call it, so drift cannot silently disagree with ingest about which file a key
/// refers to.
pub(crate) fn rel_path_for(base: &Path, file_path: &Path) -> String {
    file_path
        .strip_prefix(base)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string()
}

/// Compute the graph `_key` for a discovered file, relative to `base`.
pub(crate) fn file_key_for(base: &Path, file_path: &Path) -> String {
    keys::file_key(&rel_path_for(base, file_path))
}

/// A file under the ingest root that discovery deliberately did not pick up.
///
/// Recorded rather than dropped so `codebase drift` can report a third bucket.
/// A file that is neither ingested nor reportable is a silent hole: the pair
/// (ingest, drift) otherwise reports a clean sweep over a partially-covered
/// tree, which is a false green rather than a visible gap (#183).
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct UnhandledFile {
    pub path: String,
    pub reason: &'static str,
}

/// The outcome of a discovery walk: what will be ingested, and what will not.
pub(crate) struct Discovery {
    pub files: Vec<PathBuf>,
    pub unhandled: Vec<UnhandledFile>,
}

/// The first line of a file, if it is readable as UTF-8.
///
/// Used only for shebang sniffing, so a binary (invalid UTF-8) simply yields
/// `None` and stays out of discovery.
///
/// The read is capped: `read_line` alone allocates until the first newline, so
/// an extensionless blob with none near the start (a compiled `a.out`, a
/// checked-in artifact, a minified single-line bundle) would be pulled entirely
/// into memory during a directory walk that previously never opened it.
/// A shebang lives in the first handful of bytes or not at all.
fn first_line(path: &Path) -> Option<String> {
    use std::io::{BufRead, BufReader, Read};
    /// Longest plausible shebang line; anything beyond cannot be one.
    const SHEBANG_PROBE_BYTES: u64 = 256;
    let file = std::fs::File::open(path).ok()?;
    let mut line = String::new();
    BufReader::new(file.take(SHEBANG_PROBE_BYTES))
        .read_line(&mut line)
        .ok()?;
    Some(line)
}

/// The analyzer language implied by a file's shebang, plus whether it had one.
///
/// Only consulted for extensionless files: an extension is cheaper and more
/// reliable when present.
/// Only extensionless files are sniffed. An extension is cheaper and more
/// reliable when present, and — critically — the ingest loop applies the same
/// restriction, so admitting an extension-bearing script here would let it pass
/// discovery and then fail with "cannot detect language" instead of the
/// actionable "unsupported file type … use --language or --unparsed-ext".
pub(crate) fn shebang_of(path: &Path) -> Option<(bool, Option<Language>)> {
    if path.extension().is_some() {
        return None;
    }
    let line = first_line(path)?;
    let trimmed = line.trim_end();
    if !Language::is_shebang(trimmed) {
        return None;
    }
    Some((true, Language::from_shebang(trimmed)))
}

pub(crate) fn discover_files(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<Vec<PathBuf>> {
    Ok(discover_files_detailed(path, lang_override, unparsed_set)?.files)
}

/// Walk the tree, returning both the files ingest will process and the ones it
/// will not, each with a reason.
pub(crate) fn discover_files_detailed(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<Discovery> {
    // Whether a path's (lowercased) extension is in the unparsed allowlist.
    let ext_allowed = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| unparsed_set.contains(&e.to_lowercase()))
    };

    if path.is_file() {
        let path_str = path.to_string_lossy();
        if lang_override.is_some()
            || Language::from_path(&path_str).is_some()
            || ext_allowed(path)
            || shebang_of(path).is_some()
        {
            return Ok(Discovery {
                files: vec![path.to_path_buf()],
                unhandled: Vec::new(),
            });
        }
        bail!(
            "unsupported file type: {}. Use --language or --unparsed-ext to override.",
            path.display()
        );
    }

    let mut files = Vec::new();
    let mut unhandled = Vec::new();
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
        let has_ext = entry_path.extension().is_some();

        let (include, reason) = if Language::from_path(&path_str).is_some() {
            // File has a recognized source extension — always include.
            (true, "")
        } else if ext_allowed(entry_path) {
            // Extension is in the unparsed allowlist (e.g. cu,cuh) — include
            // for the parser-free embedding fallback (#121).
            (true, "")
        } else if lang_override.is_some() && !has_ext {
            // Language override active: include extensionless files only
            // (skip .md, .json, images, etc.).
            (true, "")
        } else if !has_ext && shebang_of(entry_path).is_some() {
            // Extensionless script identified by its shebang. `--unparsed-ext`
            // is extension-keyed and so can never name these (#183); without
            // this branch they are invisible to both ingest and drift.
            (true, "")
        } else if has_ext {
            (false, "no handler for extension")
        } else {
            (false, "no extension and no shebang")
        };

        if include {
            files.push(entry_path.to_path_buf());
        } else {
            unhandled.push(UnhandledFile {
                path: path_str.to_string(),
                reason,
            });
        }
    }

    files.sort();
    unhandled.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Discovery { files, unhandled })
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
    reenriched_this_run: bool,
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
                    reenriched_this_run,
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
    if preserve_higher_fidelity(
        db,
        &fkey,
        analysis.analysis_tier,
        allow_analysis_downgrade,
        reenriched_this_run,
    )
    .await?
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
        // Full-source digest, stored alongside the symbol hash so `codebase
        // drift` can see content staleness. `symbol_hash` is deliberately
        // name-only (see compute_symbol_hash), so a rewritten body, changed
        // signature, or edited comment leaves it identical — without this field
        // nothing in the system can tell that stored chunks have gone stale
        // (#183). Recorded only; it does not gate incremental re-ingest.
        "content_hash": hades_core::code::compute_content_hash(&source),
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

    // Remove stale chunks immediately before re-writing them. Chunk inserts are
    // overwrite-by-key, so a re-ingest producing *fewer* chunks than the previous
    // run would otherwise leave the old high-index docs behind — inflating
    // `chunk_count` and stranding chunks that reference symbols the purge above
    // already removed (#159). The unparsed path always did this; the parsed path
    // did not, so shrinking source files accumulated orphans `--force` could not
    // clear.
    //
    // This sits here rather than before the embedder call so the delete→insert
    // window contains no network round-trip: delete→write is not atomic, and a
    // failure between them leaves `chunk_count > 0` with zero stored chunks (the
    // mirror of #159). The window cannot be closed without a transaction, so it
    // is kept as small as possible. The next successful ingest self-heals.
    // Runs unconditionally — a file that drops to zero chunks must still have
    // its old ones removed.
    delete_file_chunks(db, &fkey).await;

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
    reenriched_this_run: bool,
) -> Result<FileResult> {
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;
    let fkey = keys::file_key(rel_path);
    let lang_label = language_label.unwrap_or_else(|| unparsed_language_label(rel_path));

    if preserve_higher_fidelity(
        db,
        &fkey,
        AnalysisTier::Text,
        allow_analysis_downgrade,
        reenriched_this_run,
    )
    .await?
    {
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
        // The parser-free path has no symbols, so its change-detection digest is
        // already the full-source hash. Recorded under both names so drift has a
        // single uniform column across parsed and unparsed files.
        "symbol_hash": content_hash,
        "content_hash": content_hash,
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
    if let Err(e) = hades_core::db::query::remove_docs_by_fields(
        db,
        CODEBASE.embeddings,
        &["file_key"],
        file_key,
    )
    .await
    {
        debug!(file_key, error = %e, "failed to clean up old embeddings (non-fatal)");
    }
}

/// Delete all chunk documents for a file.
///
/// Called before re-chunking on **both** the parsed and unparsed paths so that a
/// re-ingest which produces fewer chunks leaves no orphaned high-index chunk docs
/// behind (overwrite-by-key only updates the chunks that still exist). The parsed
/// path was missing this call until #159.
async fn delete_file_chunks(db: &ArangoPool, file_key: &str) {
    if let Err(e) =
        hades_core::db::query::remove_docs_by_fields(db, CODEBASE.chunks, &["file_key"], file_key)
            .await
    {
        debug!(file_key, error = %e, "failed to clean up old chunks (non-fatal)");
    }
}

/// Count inbound edges left dangling by this run's rebuilds. **Read-only.**
///
/// Deliberately reports rather than deletes. An inbound edge belongs to a file
/// this run may not have touched, and it encodes a real relationship: `b.py`
/// imports something from `a.py`. When a rebuild of `a.py` renames the target
/// symbol, deleting that edge destroys the only record that `b.py` depends on
/// `a.py` — and nothing re-derives it, because `b.py` is itself unchanged and
/// every later ingest skips it. The graph would then pass `validate` and `drift`
/// while silently missing a true relation, which is the same class of false
/// green this change set exists to remove. Re-ingesting the dependent (or an
/// explicit `codebase prune-orphans`) is the honest repair, so the count is
/// surfaced in the JSON summary and the operator decides.
///
/// Scoped two ways so the number means something: the target key must carry the
/// `{file_key}__` prefix of a file this run actually rewrote, AND the target
/// must genuinely not resolve — a symbol recreated by enrichment is not counted.
///
/// Note: symbols of files whose key exceeds the 254-byte budget carry a
/// *truncated* file_key prefix and fall outside this filter; `prune-orphans`
/// remains the global backstop.
async fn count_dangling_inbound(db: &ArangoPool, file_keys: &[String]) -> u64 {
    if file_keys.is_empty() {
        return 0;
    }
    let mut dangling = 0u64;
    for edges in [
        CODEBASE.imports_edges,
        CODEBASE.calls_edges,
        CODEBASE.implements_edges,
    ] {
        // Prefix test first: it is a cheap string comparison that eliminates
        // almost every edge, whereas DOCUMENT() is an unindexable per-edge
        // lookup. Ordering it last made this a full scan of all three edge
        // collections on every run.
        let aql = "\
            LET prefixes = (FOR fk IN @keys RETURN CONCAT(@symbols_name, '/', fk, '__')) \
            RETURN LENGTH( \
                FOR e IN @@edges \
                    FILTER LENGTH(FOR p IN prefixes FILTER STARTS_WITH(e._to, p) LIMIT 1 RETURN 1) > 0 \
                      AND DOCUMENT(e._to) == null \
                    RETURN 1)";
        let bind = json!({
            "@edges": edges,
            "symbols_name": CODEBASE.symbols,
            "keys": file_keys,
        });
        match hades_core::db::query::query(
            db,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        {
            Ok(rows) => {
                dangling += rows
                    .results
                    .first()
                    .and_then(|v| v.as_u64())
                    .unwrap_or_default();
            }
            Err(e) => {
                warn!(collection = edges, error = %e, "failed to check for dangling inbound edges after re-ingest (non-fatal)");
            }
        }
    }
    dangling
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
///
/// `reenriched_this_run` is the escape hatch for languages whose semantic
/// artifacts come from a post-loop LSP phase rather than from the per-file
/// analyzer. Go has no semantic analyzer, so `ingest_file` can only ever offer
/// `Structural`; the gopls phase supplies the semantic symbols and edges
/// afterwards, in this same run. Blocking the rewrite there protects nothing —
/// the purge is immediately followed by re-enrichment — while permanently
/// pinning every `.go` node against re-ingest (#193). C++ and Python have no
/// such phase, so a genuine downgrade there is permanent and still blocked.
fn should_preserve_tier(
    stored: Option<AnalysisTier>,
    incoming: AnalysisTier,
    allow_downgrade: bool,
    reenriched_this_run: bool,
) -> bool {
    !allow_downgrade && !reenriched_this_run && stored.is_some_and(|tier| tier > incoming)
}

/// Enforce monotonic analyzer fidelity before any destructive per-file write.
///
/// Compares against the stored file node's `analysis_tier`, which records the
/// analysis that produced that node's `symbol_hash` and chunks — the LSP phases
/// deliberately leave it alone (see `store_lsp_extractions`).
async fn preserve_higher_fidelity(
    db: &ArangoPool,
    file_key: &str,
    incoming: AnalysisTier,
    allow_downgrade: bool,
    reenriched_this_run: bool,
) -> Result<bool> {
    match crud::get_document(db, CODEBASE.files, file_key).await {
        Ok(doc) => {
            let stored = doc["analysis_tier"].as_str().and_then(AnalysisTier::parse);
            Ok(should_preserve_tier(
                stored,
                incoming,
                allow_downgrade,
                reenriched_this_run,
            ))
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
///
/// Analysis and storage are reported separately (#180): `workspaces` counts
/// crates/modules the analyzer processed, while `store_errors`/`store_failed`
/// describe what happened when writing the results to ArangoDB. A store
/// failure must not be conflated with `workspaces == 0` — that signature is
/// reserved for the analyzer itself producing nothing (#164).
#[derive(Default)]
struct SemanticLspStats {
    /// Symbol documents actually stored (created + updated).
    symbols: usize,
    /// Edge documents actually stored (created + updated).
    edges: usize,
    /// Crates (rust-analyzer) or modules (gopls) successfully analyzed.
    workspaces: usize,
    /// Documents ArangoDB rejected individually (illegal key, bad body).
    /// Non-zero means degraded-but-usable enrichment.
    store_errors: usize,
    /// The store stage failed wholesale (request-level error). Analysis
    /// results exist but none of them reached the database.
    store_failed: bool,
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
    command: Option<&str>,
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

        let session = match RustAnalyzerSession::start_with_options(
            crate_root,
            command,
            hades_core::code::lsp::DEFAULT_INDEX_TIMEOUT_SECS,
        )
        .await
        {
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
    command: Option<&str>,
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
        let session = match GoplsSession::start_with_options(
            module_root,
            command,
            hades_core::code::lsp::DEFAULT_INDEX_TIMEOUT_SECS,
        )
        .await
        {
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

    let mut stored_symbols = 0usize;
    let mut stored_edges = 0usize;
    let mut store_errors = 0usize;

    // Store enriched symbol documents (overwrite=true for idempotent re-runs).
    // Per-document error reporting (#180): one rejected document degrades one
    // symbol, not the whole batch, and ArangoDB's rejection reasons are logged
    // instead of swallowed.
    if !symbol_docs.is_empty() {
        let docs: Vec<Value> = symbol_docs
            .iter()
            .filter_map(|s| serde_json::to_value(s).ok())
            .collect();
        match crud::insert_documents_detailed(db, CODEBASE.symbols, &docs, true).await {
            Ok((result, details)) => {
                stored_symbols = (result.created + result.updated) as usize;
                store_errors += result.errors as usize;
                if result.errors > 0 {
                    warn!(
                        analyzer,
                        rejected = result.errors,
                        stored = stored_symbols,
                        details = %details.iter().take(5).cloned().collect::<Vec<_>>().join(" | "),
                        "ArangoDB rejected some symbol documents (first 5 reasons shown)"
                    );
                }
                info!(
                    count = stored_symbols,
                    analyzer, "stored LSP symbol documents"
                );
            }
            Err(e) => {
                // Request-level failure: nothing was stored. Keep the analysis
                // stats so the caller reports the store as the failed stage
                // rather than pretending the analyzer produced nothing.
                warn!(
                    analyzer,
                    error = %e,
                    "failed to store symbol documents; skipping edge store"
                );
                return Ok(SemanticLspStats {
                    workspaces,
                    store_failed: true,
                    ..Default::default()
                });
            }
        }
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
                match crud::insert_documents_detailed(db, kind.collection(), &docs, true).await {
                    Ok((result, details)) => {
                        stored_edges += (result.created + result.updated) as usize;
                        store_errors += result.errors as usize;
                        if result.errors > 0 {
                            warn!(
                                analyzer,
                                collection = kind.collection(),
                                rejected = result.errors,
                                details = %details.iter().take(5).cloned().collect::<Vec<_>>().join(" | "),
                                "ArangoDB rejected some edge documents (first 5 reasons shown)"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            analyzer,
                            collection = kind.collection(),
                            error = %e,
                            "failed to store edge batch"
                        );
                        return Ok(SemanticLspStats {
                            symbols: stored_symbols,
                            edges: stored_edges,
                            workspaces,
                            store_errors,
                            store_failed: true,
                        });
                    }
                }
            }
        }

        info!(count = stored_edges, analyzer, "stored LSP semantic edges");
    }

    let mut patched_count = 0;
    for (rel_path, sym_count, analyzed_at) in &file_patches {
        let fkey = keys::file_key(rel_path);
        // Deliberately does NOT touch `analysis_tier`/`analyzer`. Those describe
        // the analysis that produced this node's `symbol_hash` and chunks, and
        // enrichment rewrites neither. Stamping the file `semantic` here made a
        // Go node advertise a fidelity its own digest did not have, and — because
        // `preserve_higher_fidelity` compares against exactly this field — meant
        // every later run offered `structural`, lost, and skipped the file
        // permanently, `--force` included (#193). The enrichment is recorded
        // under the prefixed keys below, and the symbols and edges it writes
        // carry `analysis_tier: "semantic"` on themselves.
        let mut patch = json!({});
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
        symbols: stored_symbols,
        edges: stored_edges,
        workspaces,
        store_errors,
        store_failed: false,
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

/// Resolve an analyzer command (config/env override wins over PATH) and probe
/// it from the workspace. Returns the command to use, or None when the
/// operator explicitly accepted the downgrade.
///
/// The probe runs from `workspace` because the rustup shim resolves
/// per-directory (#164): the same `rust-analyzer` can work in a shell and die
/// inside a repo whose rust-toolchain.toml pins a toolchain missing the
/// component. Probing anywhere else validates the wrong toolchain.
fn preflight_or_bail(
    name: &str,
    configured: Option<&str>,
    workspace: &Path,
    allow_analysis_downgrade: bool,
) -> Result<Option<String>> {
    let probe = hades_core::code::lsp::resolve_and_probe(name, configured, workspace);
    match probe.outcome {
        Ok(version) => {
            info!(analyzer = name, %version, source = probe.source, "analyzer preflight passed");
            Ok(Some(probe.command))
        }
        Err(e) if allow_analysis_downgrade => {
            warn!(
                analyzer = name,
                error = %e,
                "analyzer preflight FAILED; proceeding without semantic \
                 enrichment because --allow-analysis-downgrade was passed"
            );
            Ok(None)
        }
        Err(e) => anyhow::bail!(
            "{name} preflight failed ({source}): {e}\n\
             Source files needing it were discovered, and `--force` would purge \
             semantic edges this analyzer rebuilds. Fix the analyzer (for the \
             rustup shim: `rustup component add rust-analyzer --toolchain \
             <the workspace's pinned toolchain>`), pin a binary in hades.yaml \
             under `analyzers.{}`, or pass --allow-analysis-downgrade to \
             proceed without semantic enrichment.",
            name.replace('-', "_"),
            source = probe.source,
        ),
    }
}

/// Is `path` a target for semantic enrichment in `want` language?
///
/// The ONE predicate shared by the preflight gate and the per-file tracking
/// loop, so the set the gate protects and the set the phases process cannot
/// diverge. The language override counts (matching the loop's historical
/// behavior), and unparsed-allowlisted files never count.
fn is_semantic_target(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
    want: Language,
    want_ext: &str,
) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    let is_unparsed = Language::from_path(&path.to_string_lossy()).is_none()
        && ext.as_deref().is_some_and(|e| unparsed_set.contains(e));
    !is_unparsed && (lang_override == Some(want) || ext.as_deref() == Some(want_ext))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use tempfile::TempDir;

    // ── #159 regression: shrinking re-ingest must not leave orphan chunks ──

    /// Count documents in `col` whose `file_key` matches.
    async fn count_by_file_key(pool: &ArangoPool, col: &str, fkey: &str) -> u64 {
        let aql = "FOR d IN @@col FILTER d.file_key == @fk COLLECT WITH COUNT INTO n RETURN n";
        let bind = json!({ "@col": col, "fk": fkey });
        hades_core::db::query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    /// Remove every trace of a test fixture file from the codebase graph.
    /// Shared prefix for every fixture this test writes, across all PIDs.
    const FIXTURE_PREFIX: &str = "__hades_test159_";

    /// Do the codebase collections this test writes to actually exist?
    ///
    /// `ingest_file` is the low-level path and does not bootstrap collections —
    /// the real `codebase ingest` command does that before calling it. Against a
    /// database that is not a code graph, the first insert would fail with
    /// "collection not found" and turn a skippable environment into a hard test
    /// failure. Check first so the skip-if-absent convention actually holds.
    async fn codebase_collections_present(pool: &ArangoPool) -> bool {
        for col in [
            CODEBASE.files,
            CODEBASE.chunks,
            CODEBASE.embeddings,
            CODEBASE.symbols,
        ] {
            let aql = "RETURN LENGTH(FOR d IN @@col LIMIT 1 RETURN 1)";
            let bind = json!({ "@col": col });
            if hades_core::db::query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
                .await
                .is_err()
            {
                return false;
            }
        }
        true
    }

    /// Remove fixtures left behind by *any* previous run, not just this PID.
    ///
    /// Assertion failures unwind before the trailing cleanup, so a failed run
    /// leaks its PID-keyed docs permanently — the next run uses a new PID and
    /// would never reclaim them. Sweeping the shared prefix keeps the test
    /// leak-free in the project-management DB even across failures.
    async fn cleanup_fixture_prefix(pool: &ArangoPool) {
        let stale = {
            let aql = "FOR f IN @@files FILTER STARTS_WITH(f._key, @prefix) RETURN f._key";
            let bind = json!({ "@files": CODEBASE.files, "prefix": FIXTURE_PREFIX });
            hades_core::db::query::query(
                pool,
                aql,
                Some(&bind),
                None,
                false,
                ExecutionTarget::Reader,
            )
            .await
            .map(|r| {
                r.results
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        };
        for fkey in stale {
            cleanup_fixture(pool, &fkey).await;
        }
        // Chunks/embeddings whose file node was already gone are not reachable
        // via the file-key sweep above, so clear them by their own prefix too.
        for col in [CODEBASE.chunks, CODEBASE.embeddings, CODEBASE.symbols] {
            let aql = "FOR d IN @@col FILTER STARTS_WITH(d.file_key, @prefix) REMOVE d IN @@col";
            let bind = json!({ "@col": col, "prefix": FIXTURE_PREFIX });
            let _ = hades_core::db::query::query(
                pool,
                aql,
                Some(&bind),
                None,
                false,
                ExecutionTarget::Writer,
            )
            .await;
        }
    }

    async fn cleanup_fixture(pool: &ArangoPool, fkey: &str) {
        purge_file_symbols_and_edges(pool, fkey).await;
        delete_file_chunks(pool, fkey).await;
        delete_file_embeddings(pool, fkey).await;
        let aql = "FOR d IN @@files FILTER d._key == @fk REMOVE d IN @@files";
        let bind = json!({ "@files": CODEBASE.files, "fk": fkey });
        let _ = hades_core::db::query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Writer,
        )
        .await;
    }

    /// Connect to the integration-test database, or `None` to skip.
    fn test_pool() -> Option<ArangoPool> {
        let socket = std::path::PathBuf::from(
            std::env::var("ARANGO_SOCKET")
                .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
        );
        if !socket.exists() {
            if std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true") {
                panic!(
                    "ARANGO_TESTS is set but socket not found at {}",
                    socket.display()
                );
            }
            eprintln!(
                "skipping: ArangoDB socket not found at {}",
                socket.display()
            );
            return None;
        }
        let Ok(password) = std::env::var("ARANGO_PASSWORD") else {
            eprintln!("skipping: ARANGO_PASSWORD not set");
            return None;
        };
        let client =
            hades_core::db::ArangoClient::with_socket(socket, "bident_burn", "root", &password);
        Some(ArangoPool::new(client.clone(), client))
    }

    /// A re-ingest that produces *fewer* chunks than the previous run must not
    /// leave the old high-index chunk docs behind (#159).
    ///
    /// Chunk inserts are overwrite-by-key, so without an explicit delete the
    /// parsed path kept chunks `N+1..M` from the longer previous version
    /// forever — inflating `chunk_count` and stranding chunks that reference
    /// symbols the pre-write purge had already removed. `--force` could not
    /// clear them, so the graph never converged.
    ///
    /// Runs for every parsed language that owns the parsed path, so a future
    /// language added to `Language` inherits the coverage.
    ///
    /// Requires ArangoDB (skips if the socket is absent, per the integration
    /// test convention). Uses a PID-suffixed fixture path so its keys never
    /// collide with real data, and removes every document it wrote.
    #[tokio::test]
    async fn test_reingest_shrinking_file_leaves_no_orphan_chunks() {
        let Some(pool) = test_pool() else { return };
        if !codebase_collections_present(&pool).await {
            eprintln!("skipping: target database has no codebase collections");
            return;
        }
        // Reclaim anything a previously failed run leaked (any PID).
        cleanup_fixture_prefix(&pool).await;

        let config = HadesConfig::default();
        let pid = std::process::id();

        // (extension, long source generator, short source) per parsed language.
        let cases: Vec<(&str, String, &str)> = vec![
            (
                "rs",
                (0..40)
                    .map(|i| {
                        format!(
                            "/// Padded documentation for generated function {i}, long enough \
                             that the chunker emits several chunks for this file.\n\
                             pub fn generated_{i}(input: u64) -> u64 {{\n\
                             \x20   let mut acc = input;\n\
                             \x20   for step in 0..{i}u64 {{ acc = acc.wrapping_add(step); }}\n\
                             \x20   acc\n\
                             }}\n\n"
                        )
                    })
                    .collect(),
                "pub fn only() -> u64 { 1 }\n",
            ),
            (
                "go",
                std::iter::once("package fixture\n\n".to_string())
                    .chain((0..40).map(|i| {
                        format!(
                            "// Padded documentation for generated function {i}, long enough \
                             that the chunker emits several chunks for this file.\n\
                             func Generated{i}(input uint64) uint64 {{\n\
                             \x20   acc := input\n\
                             \x20   for step := 0; step < {i}; step++ {{ acc += uint64(step) }}\n\
                             \x20   return acc\n\
                             }}\n\n"
                        )
                    }))
                    .collect(),
                "package fixture\n\nfunc Only() uint64 { return 1 }\n",
            ),
        ];

        for (ext, long_src, short_src) in cases {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join(format!("shrink.{ext}"));
            let rel_path = format!("__hades_test159_{pid}/shrink.{ext}");
            let fkey = keys::file_key(&rel_path);

            // Start clean in case this PID's fixture somehow survived.
            cleanup_fixture(&pool, &fkey).await;

            fs::write(&path, &long_src).unwrap();
            let mut imports = ImportContext::default();
            let first = ingest_file(
                &pool,
                None,
                &config,
                &path,
                &rel_path,
                None,
                &mut imports,
                None,
                false,
                false,
                false,
            )
            .await
            .unwrap_or_else(|e| panic!("first ingest failed for .{ext}: {e}"));
            let after_first = count_by_file_key(&pool, CODEBASE.chunks, &fkey).await;
            assert!(
                after_first > 1,
                ".{ext} fixture must produce multiple chunks to exercise the shrink case, \
                 got {after_first}"
            );
            assert_eq!(
                first.num_chunks.unwrap_or(0) as u64,
                after_first,
                ".{ext} first ingest: reported chunk count must match stored docs"
            );

            fs::write(&path, short_src).unwrap();
            let mut imports = ImportContext::default();
            let second = ingest_file(
                &pool,
                None,
                &config,
                &path,
                &rel_path,
                None,
                &mut imports,
                None,
                true,
                false,
                false,
            )
            .await
            .unwrap_or_else(|e| panic!("second ingest failed for .{ext}: {e}"));
            let after_second = count_by_file_key(&pool, CODEBASE.chunks, &fkey).await;

            // The regression: without the delete, after_second would still equal
            // after_first, because overwrite-by-key only rewrote 0..N.
            assert!(
                after_second < after_first,
                ".{ext} shrinking re-ingest left orphan chunks: {after_first} before, \
                 {after_second} after (#159)"
            );
            assert_eq!(
                second.num_chunks.unwrap_or(0) as u64,
                after_second,
                ".{ext} second ingest: reported chunk count must match stored docs (#159)"
            );

            // No chunk may reference a symbol the purge removed (validate #7).
            let dangling = {
                let aql = "FOR c IN @@chunks FILTER c.file_key == @fk \
                           FOR s IN (c.symbols || []) \
                           FILTER DOCUMENT(CONCAT(@syms_name, '/', s)) == null \
                           COLLECT WITH COUNT INTO n RETURN n";
                let bind = json!({
                    "@chunks": CODEBASE.chunks,
                    "syms_name": CODEBASE.symbols,
                    "fk": fkey,
                });
                hades_core::db::query::query_single(
                    &pool,
                    aql,
                    Some(&bind),
                    ExecutionTarget::Reader,
                )
                .await
                .ok()
                .flatten()
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
            };
            assert_eq!(
                dangling, 0,
                ".{ext} chunks reference removed symbols (#159)"
            );

            cleanup_fixture(&pool, &fkey).await;
        }
    }

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
            false,
            false
        ));
        assert!(should_preserve_tier(
            Some(AnalysisTier::Structural),
            AnalysisTier::Text,
            false,
            false
        ));
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            true,
            false
        ));
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Structural),
            AnalysisTier::Semantic,
            false,
            false
        ));
    }

    /// #193: a language whose semantic artifacts come from a post-loop LSP phase
    /// must not be pinned against re-ingest by its own enrichment.
    ///
    /// Go can only ever offer `Structural` from `ingest_file`, so once a node was
    /// stamped `Semantic` the guard fired on every subsequent run and skipped the
    /// file permanently — `--force` included, since the guard runs ahead of it.
    /// When the gopls phase will re-enrich in this same run, the purge it was
    /// protecting is undone immediately, so preserving buys nothing.
    #[test]
    fn test_pending_lsp_reenrichment_releases_the_fidelity_guard() {
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            false,
            true
        ));

        // Without a scheduled phase the guard still holds: a C++ node whose
        // libclang analysis is gone has nothing to restore it this run.
        assert!(should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            false,
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
    fn test_discover_files_includes_extensionless_shebang_scripts() {
        let dir = tempfile::tempdir().unwrap();
        // Extensionless shell script: `--unparsed-ext` is extension-keyed and so
        // can never name this file (#183).
        fs::write(dir.path().join("deploy-thing"), "#!/bin/bash\necho hi\n").unwrap();
        // Extensionless Python script: should be recognized as Python.
        fs::write(dir.path().join("runner"), "#!/usr/bin/env python3\nx = 1\n").unwrap();
        // Extensionless non-script: no shebang, stays out.
        fs::write(dir.path().join("NOTES"), "just prose\n").unwrap();

        let d = discover_files_detailed(dir.path(), None, &HashSet::new()).unwrap();
        let names: Vec<String> = d
            .files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"deploy-thing".to_string()), "got {names:?}");
        assert!(names.contains(&"runner".to_string()), "got {names:?}");
        assert!(!names.contains(&"NOTES".to_string()), "got {names:?}");

        // And the one that stayed out is REPORTED, not silently dropped.
        let unhandled: Vec<&str> = d.unhandled.iter().map(|u| u.reason).collect();
        assert_eq!(d.unhandled.len(), 1, "{:?}", d.unhandled);
        assert_eq!(unhandled[0], "no extension and no shebang");
    }

    #[test]
    fn test_shebang_sniffing_is_extensionless_only() {
        let dir = tempfile::tempdir().unwrap();
        // An extension-bearing script must NOT be admitted by the shebang path:
        // the ingest loop only sniffs extensionless files, so admitting it here
        // would pass discovery and then fail with "cannot detect language"
        // instead of the actionable unsupported-file-type bail.
        let with_ext = dir.path().join("deploy.sh");
        fs::write(&with_ext, "#!/bin/bash\necho hi\n").unwrap();
        assert!(shebang_of(&with_ext).is_none());

        let without_ext = dir.path().join("deploy-thing");
        fs::write(&without_ext, "#!/bin/bash\necho hi\n").unwrap();
        assert!(shebang_of(&without_ext).is_some());
    }

    #[test]
    fn test_first_line_probe_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        // A newline-free blob must not be read whole during a directory walk.
        let blob = dir.path().join("bigblob");
        fs::write(&blob, "x".repeat(2 * 1024 * 1024)).unwrap();
        let line = first_line(&blob).unwrap();
        assert!(line.len() <= 256, "probe read {} bytes", line.len());
        // And it is correctly not treated as a script.
        assert!(shebang_of(&blob).is_none());
    }

    #[test]
    fn test_discover_files_reports_unhandled_extensions() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("data.json"), "{}\n").unwrap();
        fs::write(dir.path().join("Config.toml"), "k = 1\n").unwrap();

        let d = discover_files_detailed(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(d.files.len(), 1, "only the .py is handled");
        // The two unhandled files are counted with a reason rather than falling
        // outside drift's notion of source entirely — that silence was the bug.
        assert_eq!(d.unhandled.len(), 2, "{:?}", d.unhandled);
        assert!(
            d.unhandled
                .iter()
                .all(|u| u.reason == "no handler for extension")
        );
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
