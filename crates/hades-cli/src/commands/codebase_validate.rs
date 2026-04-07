//! Post-ingest validation of codebase graph invariants.
//!
//! Implements the 17 invariants from the codebase graph ontology spec §10.
//! Each invariant runs an AQL query that returns violations; an empty result
//! means the invariant holds.
//!
//! Groups:
//! - Document invariants (1–7): referential integrity and count consistency
//! - Edge invariants (8–11): `_from`/`_to` vertex existence
//! - Uniqueness invariants (12–14): enforced by ArangoDB, reported as pass
//! - Primitive invariants (15–17): kind/lang_kind field correctness

use std::io::Write;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::info;

use hades_core::config::HadesConfig;
use hades_core::db::{ArangoError, ArangoPool};
use hades_core::db::query::{self, ExecutionTarget};

use super::output::{self, OutputFormat};

// ── Invariant definitions ──────────────────────────────────────────

/// All 17 invariants from ontology spec §10.
const INVARIANTS: &[Invariant] = &[
    // Document invariants (1–7)
    Invariant {
        id: 1,
        name: "chunk_file_ref",
        description: "Every codebase_chunks.file_key references an existing codebase_files document",
        aql: Some(
            "FOR c IN codebase_chunks \
               FILTER !DOCUMENT(CONCAT('codebase_files/', c.file_key)) \
               LIMIT @limit \
               RETURN { chunk: c._key, file_key: c.file_key }",
        ),
    },
    Invariant {
        id: 2,
        name: "embedding_chunk_ref",
        description: "Every codebase_embeddings.chunk_key references an existing codebase_chunks document",
        aql: Some(
            "FOR e IN codebase_embeddings \
               FILTER !DOCUMENT(CONCAT('codebase_chunks/', e.chunk_key)) \
               LIMIT @limit \
               RETURN { embedding: e._key, chunk_key: e.chunk_key }",
        ),
    },
    Invariant {
        id: 3,
        name: "embedding_file_ref",
        description: "Every codebase_embeddings.file_key references an existing codebase_files document",
        aql: Some(
            "FOR e IN codebase_embeddings \
               FILTER !DOCUMENT(CONCAT('codebase_files/', e.file_key)) \
               LIMIT @limit \
               RETURN { embedding: e._key, file_key: e.file_key }",
        ),
    },
    Invariant {
        id: 4,
        name: "chunk_count_consistency",
        description: "codebase_files.chunk_count matches actual chunk count",
        aql: Some(
            "FOR f IN codebase_files \
               LET actual = LENGTH(FOR c IN codebase_chunks FILTER c.file_key == f._key RETURN 1) \
               FILTER f.chunk_count != actual \
               LIMIT @limit \
               RETURN { file: f._key, expected: f.chunk_count, actual: actual }",
        ),
    },
    Invariant {
        id: 5,
        name: "embedding_count_consistency",
        description: "codebase_files.embedding_count matches actual embedding count",
        aql: Some(
            "FOR f IN codebase_files \
               LET actual = LENGTH(FOR e IN codebase_embeddings FILTER e.file_key == f._key RETURN 1) \
               FILTER f.embedding_count != actual \
               LIMIT @limit \
               RETURN { file: f._key, expected: f.embedding_count, actual: actual }",
        ),
    },
    Invariant {
        id: 6,
        name: "symbol_count_consistency",
        description: "codebase_files.symbol_count matches actual symbol count (syn-based)",
        aql: Some(
            "FOR f IN codebase_files \
               LET actual = LENGTH(FOR s IN codebase_symbols FILTER s.file_key == f._key RETURN 1) \
               FILTER f.symbol_count != actual \
               LIMIT @limit \
               RETURN { file: f._key, expected: f.symbol_count, actual: actual }",
        ),
    },
    Invariant {
        id: 7,
        name: "chunk_symbols_exist",
        description: "Every key in codebase_chunks.symbols[] exists in codebase_symbols",
        aql: Some(
            "FOR c IN codebase_chunks \
               FILTER IS_ARRAY(c.symbols) AND LENGTH(c.symbols) > 0 \
               FOR sym_key IN c.symbols \
                 FILTER !DOCUMENT(CONCAT('codebase_symbols/', sym_key)) \
                 LIMIT @limit \
                 RETURN DISTINCT { chunk: c._key, missing_symbol: sym_key }",
        ),
    },

    // Edge invariants (8–11)
    Invariant {
        id: 8,
        name: "defines_edge_endpoints",
        description: "codebase_defines_edges: _from in codebase_files, _to in codebase_symbols",
        aql: Some(
            "FOR e IN codebase_defines_edges \
               FILTER !DOCUMENT(e._from) OR !DOCUMENT(e._to) \
               LIMIT @limit \
               RETURN { edge: e._key, _from: e._from, _to: e._to }",
        ),
    },
    Invariant {
        id: 9,
        name: "calls_edge_endpoints",
        description: "codebase_calls_edges: _from and _to in codebase_symbols",
        aql: Some(
            "FOR e IN codebase_calls_edges \
               FILTER !DOCUMENT(e._from) OR !DOCUMENT(e._to) \
               LIMIT @limit \
               RETURN { edge: e._key, _from: e._from, _to: e._to }",
        ),
    },
    Invariant {
        id: 10,
        name: "implements_edge_endpoints",
        description: "codebase_implements_edges: _from and _to in codebase_symbols",
        aql: Some(
            "FOR e IN codebase_implements_edges \
               FILTER !DOCUMENT(e._from) OR !DOCUMENT(e._to) \
               LIMIT @limit \
               RETURN { edge: e._key, _from: e._from, _to: e._to }",
        ),
    },
    Invariant {
        id: 11,
        name: "imports_edge_endpoints",
        description: "codebase_imports_edges: _from in codebase_files, _to in codebase_symbols or codebase_files",
        aql: Some(
            "FOR e IN codebase_imports_edges \
               FILTER !DOCUMENT(e._from) OR !DOCUMENT(e._to) \
               LIMIT @limit \
               RETURN { edge: e._key, _from: e._from, _to: e._to }",
        ),
    },

    // Uniqueness invariants (12–14) — enforced by ArangoDB
    Invariant {
        id: 12,
        name: "unique_keys",
        description: "No two documents in any collection share the same _key (enforced by database)",
        aql: None,
    },
    Invariant {
        id: 13,
        name: "deterministic_edge_keys",
        description: "Same (from, kind, to) triple always produces the same edge _key (verified by unit tests)",
        aql: None,
    },
    Invariant {
        id: 14,
        name: "deterministic_symbol_keys",
        description: "Same (file_key, qualified_name) pair always produces the same symbol _key (verified by unit tests)",
        aql: None,
    },

    // Primitive invariants (15–17)
    Invariant {
        id: 15,
        name: "file_kind_is_file",
        description: "Every codebase_files.kind is \"file\"",
        aql: Some(
            "FOR f IN codebase_files \
               FILTER f.kind != 'file' \
               LIMIT @limit \
               RETURN { file: f._key, kind: f.kind }",
        ),
    },
    Invariant {
        id: 16,
        name: "symbol_kind_is_primitive",
        description: "Every codebase_symbols.kind is one of: module, type, callable, value",
        aql: Some(
            "FOR s IN codebase_symbols \
               FILTER s.kind NOT IN ['module', 'type', 'callable', 'value'] \
               LIMIT @limit \
               RETURN { symbol: s._key, kind: s.kind }",
        ),
    },
    Invariant {
        id: 17,
        name: "no_import_impl_symbols",
        description: "No symbol document exists with lang_kind of \"import\" or \"impl\"",
        aql: Some(
            "FOR s IN codebase_symbols \
               FILTER s.lang_kind IN ['import', 'impl'] \
               LIMIT @limit \
               RETURN { symbol: s._key, lang_kind: s.lang_kind }",
        ),
    },
];

struct Invariant {
    id: u32,
    name: &'static str,
    description: &'static str,
    /// AQL query returning violations. `None` = enforced by design.
    aql: Option<&'static str>,
}

// ── Report types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct ValidationReport {
    invariants: Vec<InvariantResult>,
    summary: ReportSummary,
}

#[derive(Serialize)]
struct InvariantResult {
    id: u32,
    name: String,
    description: String,
    pass: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    violations: Vec<Value>,
    violation_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Serialize)]
struct ReportSummary {
    total: u32,
    passed: u32,
    failed: u32,
    skipped: u32,
}

// ── Public entry point ─────────────────────────────────────────────

/// Maximum number of violations to return per invariant.
const VIOLATION_LIMIT: u32 = 50;

/// `hades codebase validate`
pub async fn run_validate(config: &HadesConfig) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let mut results = Vec::with_capacity(INVARIANTS.len());
    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut skipped = 0u32;

    for inv in INVARIANTS {
        let result = check_invariant(&pool, inv).await?;

        if result.note.is_some() && result.pass {
            skipped += 1;
        } else if result.pass {
            passed += 1;
        } else {
            failed += 1;
        }

        let status = if result.note.is_some() && result.pass {
            "SKIP"
        } else if result.pass {
            "PASS"
        } else {
            "FAIL"
        };
        let mut err = std::io::stderr().lock();
        let _ = writeln!(
            err,
            "  [{status}] #{id:>2} {name}",
            id = result.id,
            name = result.name,
        );

        results.push(result);
    }

    let report = ValidationReport {
        invariants: results,
        summary: ReportSummary {
            total: INVARIANTS.len() as u32,
            passed,
            failed,
            skipped,
        },
    };

    let json_value = serde_json::to_value(&report).context("failed to serialize report")?;
    output::print_output("codebase.validate", json_value, &OutputFormat::Json);

    // Human-readable summary to stderr.
    let mut err = std::io::stderr().lock();
    writeln!(err)?;
    writeln!(
        err,
        "Validation: {passed} passed, {failed} failed, {skipped} skipped ({total} total)",
        total = report.summary.total,
    )?;

    if failed > 0 {
        info!(failed, "validation completed with failures");
    } else {
        info!("all invariants passed");
    }

    Ok(())
}

// ── Per-invariant check ────────────────────────────────────────────

async fn check_invariant(pool: &ArangoPool, inv: &Invariant) -> Result<InvariantResult> {
    let Some(aql) = inv.aql else {
        // Enforced by database design — skip with note.
        return Ok(InvariantResult {
            id: inv.id,
            name: inv.name.to_string(),
            description: inv.description.to_string(),
            pass: true,
            violations: Vec::new(),
            violation_count: 0,
            note: Some("Enforced by database/unit tests — not queried".to_string()),
        });
    };

    let bind_vars = json!({ "limit": VIOLATION_LIMIT });

    let qr = match query::query(
        pool,
        aql,
        Some(&bind_vars),
        None,
        false,
        ExecutionTarget::Reader,
    )
    .await
    {
        Ok(qr) => qr,
        Err(ArangoError::Api { error_num: 1203, message, .. }) => {
            // Collection or view not found — skip gracefully.
            return Ok(InvariantResult {
                id: inv.id,
                name: inv.name.to_string(),
                description: inv.description.to_string(),
                pass: true,
                violations: Vec::new(),
                violation_count: 0,
                note: Some(format!("Skipped: {message}")),
            });
        }
        Err(e) => {
            return Err(e).with_context(|| format!("invariant #{}: AQL query failed", inv.id));
        }
    };

    let violation_count = qr.results.len() as u64;
    let pass = violation_count == 0;

    Ok(InvariantResult {
        id: inv.id,
        name: inv.name.to_string(),
        description: inv.description.to_string(),
        pass,
        violations: qr.results,
        violation_count,
        note: None,
    })
}
