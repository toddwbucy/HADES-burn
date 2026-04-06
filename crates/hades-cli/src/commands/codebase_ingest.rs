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
use hades_core::code::{self, AstChunking, CodeAnalysisError, Language, Symbol};
use hades_core::code::rust_imports;
use hades_core::db::collections::CODEBASE;
use hades_core::db::crud;
use hades_core::db::keys;
use hades_core::db::ArangoPool;
use hades_core::persephone::embedding::EmbeddingClient;
use hades_core::HadesConfig;

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
    /// Python: rel_path → list of imported module names.
    python_imports: HashMap<String, Vec<String>>,
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

    let embedder = EmbeddingClient::connect_default()
        .await
        .context("failed to connect to embedding service")?;

    // Ensure codebase collections exist.
    ensure_collections(&db).await?;

    // Discover source files.
    let files = discover_files(&path, lang_override)?;
    if files.is_empty() {
        let output = json!({
            "success": true,
            "command": "codebase ingest",
            "data": { "total": 0, "message": "no supported source files found" },
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
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
        rust_imports: HashMap::new(),
        rust_file_symbols: HashMap::new(),
    };

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

        let result = ingest_file(
            &db,
            &embedder,
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
                    skipped: None,
                    error: Some(e.to_string()),
                    duration_ms: duration,
                });
            }
        }
    }

    // Resolve Python import graph edges.
    let py_import_edges = resolve_python_imports(&imports.python_imports, &base);
    if !py_import_edges.is_empty() {
        info!(edge_count = py_import_edges.len(), "resolved Python import edges");
        if let Err(e) = crud::insert_documents(&db, CODEBASE.edges, &py_import_edges, true).await {
            warn!(error = %e, "failed to store Python import edges");
        }
    }

    // Resolve Rust import graph edges (file → symbol).
    let rust_symbol_index = rust_imports::build_symbol_index(&imports.rust_file_symbols);
    let rs_import_edges = rust_imports::resolve_rust_imports(&imports.rust_imports, &rust_symbol_index);
    if !rs_import_edges.is_empty() {
        info!(edge_count = rs_import_edges.len(), "resolved Rust import edges");
        if let Err(e) = crud::insert_documents(&db, CODEBASE.edges, &rs_import_edges, true).await {
            warn!(error = %e, "failed to store Rust import edges");
        }
    }

    let total_import_edges = py_import_edges.len() + rs_import_edges.len();

    // Output summary.
    let total = results.len();
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = results.iter().filter(|r| !r.success && r.skipped != Some(true)).count();
    let skipped = results.iter().filter(|r| r.skipped == Some(true)).count();
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let output = json!({
        "success": failed == 0,
        "command": "codebase ingest",
        "data": {
            "total": total,
            "completed": succeeded,
            "failed": failed,
            "skipped": skipped,
            "import_edges": total_import_edges,
            "python_import_edges": py_import_edges.len(),
            "rust_import_edges": rs_import_edges.len(),
            "results": results,
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "duration_ms": duration_ms,
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    if failed > 0 {
        return Err(CodebaseIngestFailure { total, failed }.into());
    }
    Ok(())
}

// ── Collection setup ────────────────────────────────────────────────────

/// Ensure all codebase collections exist, creating any that are missing.
async fn ensure_collections(db: &ArangoPool) -> Result<()> {
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
    embedder: &EmbeddingClient,
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
                skipped: Some(true),
                error: Some(format!("parse error: {msg}")),
                duration_ms: 0,
            });
        }
        Err(e) => return Err(e.into()),
    };

    // Check for incremental skip via symbol_hash.
    let fkey = keys::file_key(rel_path);
    if let Some(true) = check_unchanged(db, &fkey, &analysis.symbol_hash).await? {
        debug!(path = rel_path, "unchanged (same symbol_hash), skipping");
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang.name().to_string()),
            num_symbols: Some(analysis.symbols.len()),
            num_chunks: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }

    // Collect Python import names for later edge resolution.
    if lang == Language::Python {
        let py_imports: Vec<String> = analysis
            .symbols
            .iter()
            .filter(|s| s.kind == hades_core::code::SymbolKind::Import)
            .filter_map(|s| s.metadata.get("module").and_then(|v| v.as_str()).map(String::from))
            .collect();
        if !py_imports.is_empty() {
            imports.python_imports.insert(rel_path.to_string(), py_imports);
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

    // Build file document.
    let file_doc = json!({
        "_key": fkey,
        "path": rel_path,
        "language": lang.name(),
        "metrics": analysis.metrics,
        "symbol_hash": analysis.symbol_hash,
        "symbol_count": analysis.symbols.len(),
        "chunk_count": chunks.len(),
        "total_lines": analysis.metrics.total_lines,
        "status": "PROCESSED",
        "ingested_at": chrono::Utc::now().to_rfc3339(),
    });

    // Build chunk documents.
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
            })
        })
        .collect();

    // Build symbol documents.
    let symbol_docs: Vec<Value> = analysis
        .symbols
        .iter()
        .map(|s| {
            let skey = keys::symbol_key(&fkey, &s.name);
            json!({
                "_key": skey,
                "file_key": fkey,
                "name": s.name,
                "kind": s.kind.to_string(),
                "start_line": s.start_line,
                "end_line": s.end_line,
                "metadata": s.metadata,
            })
        })
        .collect();

    // Build defines edges (file → symbol).
    let define_edges: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| matches!(
            s.kind,
            hades_core::code::SymbolKind::Function
            | hades_core::code::SymbolKind::Class
            | hades_core::code::SymbolKind::Struct
            | hades_core::code::SymbolKind::Enum
            | hades_core::code::SymbolKind::Trait
            | hades_core::code::SymbolKind::Constant
            | hades_core::code::SymbolKind::Macro
        ))
        .map(|s| {
            let skey = keys::symbol_key(&fkey, &s.name);
            json!({
                "_from": format!("{}/{}", CODEBASE.files, fkey),
                "_to": format!("{}/{}", CODEBASE.symbols, skey),
                "type": "defines",
                "file_path": rel_path,
                "symbol_name": s.name,
            })
        })
        .collect();

    // Embed chunks.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embedding_docs = if !chunk_texts.is_empty() {
        match embedder
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
    } else {
        Vec::new()
    };

    // Store everything to ArangoDB (overwrite mode for idempotent re-runs).
    crud::insert_documents(db, CODEBASE.files, &[file_doc], true)
        .await
        .context("failed to store file document")?;

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
        crud::insert_documents(db, CODEBASE.edges, &define_edges, true)
            .await
            .context("failed to store define edges")?;
    }

    let num_symbols = analysis.symbols.len();
    let num_chunks = chunks.len();

    // Transfer Rust symbols into the import index (avoids cloning).
    if lang == Language::Rust {
        imports.rust_file_symbols.insert(
            rel_path.to_string(),
            std::mem::take(&mut analysis.symbols),
        );
    }

    info!(
        path = rel_path,
        language = lang.name(),
        symbols = num_symbols,
        chunks = num_chunks,
        "ingested"
    );

    Ok(FileResult {
        path: rel_path.to_string(),
        success: true,
        language: Some(lang.name().to_string()),
        num_symbols: Some(num_symbols),
        num_chunks: Some(num_chunks),
        skipped: None,
        error: None,
        duration_ms: 0,
    })
}

// ── Incremental check ───────────────────────────────────────────────────

/// Check if a file's symbol_hash is unchanged in the database.
///
/// Returns `Some(true)` if the file exists and has the same hash,
/// `Some(false)` if it exists with a different hash, `None` if not found.
async fn check_unchanged(
    db: &ArangoPool,
    file_key: &str,
    new_hash: &str,
) -> Result<Option<bool>> {
    match crud::get_document(db, CODEBASE.files, file_key).await {
        Ok(doc) => {
            let stored_hash = doc["symbol_hash"].as_str().unwrap_or("");
            Ok(Some(stored_hash == new_hash))
        }
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ── Python import graph resolution ──────────────────────────────────────

/// Resolve Python import statements to file→file edges.
///
/// Only creates edges for imports that resolve to files within the
/// ingested set. External package imports are silently skipped.
fn resolve_python_imports(
    python_imports: &HashMap<String, Vec<String>>,
    _base: &Path,
) -> Vec<Value> {
    // Build a mapping from Python module name → relative file path.
    let mut module_to_file: HashMap<String, String> = HashMap::new();
    for rel_path in python_imports.keys() {
        // Convert path to module name using Path components (platform-safe).
        // "core/models.py" → ["core", "models"] → "core.models"
        let p = Path::new(rel_path);
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        // Collect directory components + file stem.
        let mut parts: Vec<&str> = p
            .parent()
            .map(|parent| parent.components()
                .filter_map(|c| c.as_os_str().to_str())
                .collect())
            .unwrap_or_default();
        parts.push(stem);

        let module = parts.join(".");

        // Strip trailing .__init__ for package init files.
        let module = module
            .strip_suffix(".__init__")
            .unwrap_or(&module)
            .to_string();

        module_to_file.insert(module, rel_path.clone());
    }

    let mut edges = Vec::new();
    for (source_path, imports) in python_imports {
        let source_fkey = keys::file_key(source_path);
        for import_module in imports {
            // Try exact match first, then prefix match for submodule imports.
            let target_path = module_to_file.get(import_module.as_str())
                .or_else(|| {
                    // "core.models.Config" → try "core.models", then "core"
                    let mut parts: Vec<&str> = import_module.split('.').collect();
                    while parts.len() > 1 {
                        parts.pop();
                        let prefix = parts.join(".");
                        if let Some(path) = module_to_file.get(&prefix) {
                            return Some(path);
                        }
                    }
                    None
                });

            if let Some(target_path) = target_path {
                // Don't create self-edges.
                if target_path == source_path {
                    continue;
                }
                let target_fkey = keys::file_key(target_path);
                edges.push(json!({
                    "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                    "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                    "_key": format!("{}_imports_{}", source_fkey, target_fkey),
                    "type": "imports",
                    "source_path": source_path,
                    "target_path": target_path,
                    "module": import_module,
                }));
            }
        }
    }

    // Deduplicate edges by _key (multiple imports from same file to same target).
    let mut seen = std::collections::HashSet::new();
    edges.retain(|e| {
        let key = e["_key"].as_str().unwrap_or("").to_string();
        seen.insert(key)
    });

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

    #[test]
    fn test_resolve_python_imports_basic() {
        let mut imports = HashMap::new();
        imports.insert(
            "core/models.py".to_string(),
            vec!["core.utils".to_string()],
        );
        imports.insert(
            "core/utils.py".to_string(),
            vec!["os".to_string()], // external, should be skipped
        );

        let base = PathBuf::from("/tmp/project");
        let edges = resolve_python_imports(&imports, &base);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["type"], "imports");
        assert_eq!(edges[0]["source_path"], "core/models.py");
        assert_eq!(edges[0]["target_path"], "core/utils.py");
    }

    #[test]
    fn test_resolve_python_imports_no_self_edge() {
        let mut imports = HashMap::new();
        imports.insert(
            "core/models.py".to_string(),
            vec!["core.models".to_string()],
        );

        let base = PathBuf::from("/tmp");
        let edges = resolve_python_imports(&imports, &base);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_resolve_python_imports_init_package() {
        let mut imports = HashMap::new();
        imports.insert(
            "core/__init__.py".to_string(),
            vec![],
        );
        imports.insert(
            "app.py".to_string(),
            vec!["core".to_string()],
        );

        let base = PathBuf::from("/tmp");
        let edges = resolve_python_imports(&imports, &base);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["target_path"], "core/__init__.py");
    }

    #[test]
    fn test_resolve_python_imports_dedup() {
        let mut imports = HashMap::new();
        imports.insert(
            "a.py".to_string(),
            vec!["b".to_string(), "b".to_string()], // duplicate
        );
        imports.insert("b.py".to_string(), vec![]);

        let base = PathBuf::from("/tmp");
        let edges = resolve_python_imports(&imports, &base);
        assert_eq!(edges.len(), 1);
    }
}
