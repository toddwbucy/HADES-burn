//! Declarative schema application — `hades schema apply <file.yaml>`.
//!
//! Parses a YAML schema file into structured operations, validates them,
//! and applies them to an ArangoDB database. The YAML is the canonical
//! source at bootstrap; once a database is in use, the database itself
//! becomes the source of truth and the YAML drifts as a configuration
//! artifact (see `docs/declarative-schema.md`).
//!
//! Pipeline:
//!
//! ```text
//! schema.yaml
//!    ↓
//! [1. parse]      serde_yaml → SchemaFile
//!    ↓
//! [2. validate]   semantic checks (referential integrity, _key uniqueness)
//!    ↓
//! [3. plan]       emit ordered list of Operations
//!    ↓
//! [4. dry-run?]   if --dry-run: serialize plan to JSON, return
//!    ↓
//! [5. apply]      execute each Operation; collect per-op results
//! ```
//!
//! Idempotency: per-document upserts via `crud::insert_documents`
//! (`onDuplicate=replace`), collection creation tolerates 409 Conflict.
//! Re-applying the same YAML to the same DB is a no-op semantically.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use serde_yaml::Mapping;
use tracing::{debug, info};

use crate::db::{ArangoError, ArangoErrorKind, ArangoPool, crud};

/// Placeholder used in plan ops and `_key` suffixes when an edge
/// definition has no `source_field`. Parenthesized so it cannot
/// collide with a user-supplied field name. Must match between
/// `plan()` (operation visibility) and `apply()` (Arango `_key`).
const NO_SOURCE_FIELD: &str = "(none)";

// ── public types ──────────────────────────────────────────────────────

/// Errors that can occur during schema apply.
#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    /// YAML parse failure (syntax error, type mismatch, unknown field).
    #[error("YAML parse error: {0}")]
    Parse(#[from] serde_yaml::Error),

    /// Semantic validation failed. Display formats the list inline.
    #[error("{}", format_validation_errors(.0))]
    Validation(Vec<String>),

    /// "In-use" guard tripped without `--force`.
    #[error("database '{db}' is in use ({reason}); pass --force to override")]
    InUse { db: String, reason: String },

    /// Underlying ArangoDB error.
    #[error("ArangoDB error during apply: {0}")]
    Arango(#[from] ArangoError),

    /// Serialization-related failure inside the applier.
    #[error("internal serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

fn format_validation_errors(errs: &[String]) -> String {
    let lines: Vec<String> = errs.iter().map(|e| format!("  - {e}")).collect();
    format!(
        "schema validation failed: {} error(s):\n{}",
        errs.len(),
        lines.join("\n")
    )
}

/// A parsed schema file.
///
/// Represents the full contents of one `schema.yaml`, with the known
/// sections destructured and any other top-level keys captured as
/// per-collection document seeds.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaFile {
    /// Required collections (document or edge type).
    #[serde(default)]
    pub collections: Vec<CollectionDef>,

    /// Edge definitions to register in `hades_schema`.
    #[serde(default)]
    pub edge_definitions: Vec<EdgeDef>,

    /// Named graphs to create via the gharial API.
    #[serde(default)]
    pub named_graphs: Vec<NamedGraphDef>,

    /// Per-collection document seeds. Collected from arbitrary
    /// top-level YAML keys not matching any of the above sections.
    /// Keys must match a declared collection name (validated).
    #[serde(skip)]
    pub documents: HashMap<String, Vec<Value>>,
}

/// One collection to ensure exists.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionDef {
    /// Collection name (e.g., `"axioms"`, `"smell_specs"`).
    pub name: String,
    /// `"document"` or `"edge"`.
    #[serde(rename = "type")]
    pub collection_type: CollectionType,
}

/// Collection type discriminator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CollectionType {
    Document,
    Edge,
}

impl CollectionType {
    /// ArangoDB collection-type integer (2 = document, 3 = edge).
    pub fn as_u32(&self) -> u32 {
        match self {
            Self::Document => 2,
            Self::Edge => 3,
        }
    }
}

/// One edge definition for `hades_schema`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeDef {
    pub name: String,
    #[serde(default)]
    pub source_field: Option<String>,
    pub from_collections: Vec<String>,
    pub to_collections: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// One named graph to create via gharial.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NamedGraphDef {
    pub name: String,
    pub edges: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Options for `apply`.
#[derive(Debug, Clone, Default)]
pub struct ApplyOptions {
    /// If true, build the plan and return it without executing.
    pub dry_run: bool,
    /// If true, skip the "in-use" guard.
    pub force: bool,
}

/// Result of a (non-dry-run) apply.
#[derive(Debug, Clone, Serialize)]
pub struct ApplyResult {
    pub collections_created: usize,
    pub collections_skipped_existing: usize,
    pub documents_upserted: usize,
    pub edge_definitions_registered: usize,
    pub named_graphs_created: usize,
    pub named_graphs_skipped_existing: usize,
}

/// One planned operation. Public for `--dry-run` JSON output.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Operation {
    CreateCollection { name: String, collection_type: u32 },
    UpsertDocuments { collection: String, count: usize },
    /// One edge_definition document into hades_schema. The
    /// `(name, source_field)` pair is the unique identity — a single
    /// edge collection can carry multiple definitions that differ by
    /// source_field (the historical NL pattern, retained for
    /// generality). Edge defs without a source_field use the literal
    /// `"(none)"` so the operation list still distinguishes them.
    UpsertEdgeDefinition { name: String, source_field: String },
    /// Persist a `schema_meta` document to `hades_schema` so
    /// `RuntimeSchema::load` can find it. Always emitted when
    /// hades_schema is being managed by this apply pass.
    UpsertSchemaMeta,
    /// Persist a `named_graph` document to `hades_schema` so
    /// `RuntimeSchema::load` can deserialize it. Distinct from
    /// `CreateNamedGraph` (the gharial API call): one materializes
    /// the graph in ArangoDB, the other records it in `hades_schema`.
    UpsertNamedGraphDoc { name: String },
    CreateNamedGraph { name: String, edges: Vec<String> },
}

// ── public API ────────────────────────────────────────────────────────

/// Parse a YAML schema string into a [`SchemaFile`].
pub fn parse(yaml_str: &str) -> Result<SchemaFile, ApplyError> {
    // First pass: untyped, so we can extract per-collection document
    // seed blocks (top-level keys that aren't reserved sections).
    let raw: serde_yaml::Value = serde_yaml::from_str(yaml_str)?;
    let map = raw
        .as_mapping()
        .ok_or_else(|| {
            ApplyError::Validation(vec![
                "schema YAML must be a mapping at the top level".into(),
            ])
        })?
        .clone();

    const RESERVED: &[&str] = &["collections", "edge_definitions", "named_graphs"];

    // Second pass: parse the reserved sections via serde with deny_unknown_fields.
    // Build a sub-mapping containing only the reserved keys, then deserialize.
    let mut reserved_map = Mapping::new();
    let mut documents: HashMap<String, Vec<Value>> = HashMap::new();
    for (k, v) in map.into_iter() {
        let key_str = match k.as_str() {
            Some(s) => s.to_string(),
            None => {
                return Err(ApplyError::Validation(vec![
                    "top-level keys must be strings".into(),
                ]));
            }
        };
        if RESERVED.contains(&key_str.as_str()) {
            reserved_map.insert(serde_yaml::Value::String(key_str), v);
        } else {
            // Treat as document seed block: must be a sequence of mappings.
            let docs: Vec<Value> = serde_yaml::from_value(v).map_err(|e| {
                ApplyError::Validation(vec![format!(
                    "section '{key_str}' must be a list of documents: {e}"
                )])
            })?;
            documents.insert(key_str, docs);
        }
    }

    let mut file: SchemaFile =
        serde_yaml::from_value(serde_yaml::Value::Mapping(reserved_map))?;
    file.documents = documents;
    Ok(file)
}

/// Run semantic validation on a parsed [`SchemaFile`].
///
/// Collects all errors and returns them together rather than failing
/// on the first — operators see the full picture of what's wrong.
pub fn validate(file: &SchemaFile) -> Result<(), ApplyError> {
    let mut errors: Vec<String> = Vec::new();

    // Build a set of declared collection names + their types for cross-checks.
    let mut declared: HashMap<&str, CollectionType> = HashMap::new();
    for c in &file.collections {
        if declared.insert(c.name.as_str(), c.collection_type).is_some() {
            errors.push(format!(
                "collection '{}' declared twice",
                c.name
            ));
        }
    }

    // Every document seed block must reference a declared collection.
    for col_name in file.documents.keys() {
        if !declared.contains_key(col_name.as_str()) {
            errors.push(format!(
                "document seed block '{col_name}' references collection \
                 not declared in `collections:`"
            ));
        }
    }

    // Document _key uniqueness within each collection.
    for (col, docs) in &file.documents {
        let mut seen: HashSet<&str> = HashSet::new();
        for (i, doc) in docs.iter().enumerate() {
            let key = doc.get("_key").and_then(|v| v.as_str());
            if let Some(k) = key
                && !seen.insert(k)
            {
                errors.push(format!(
                    "duplicate _key '{k}' in collection '{col}'"
                ));
            }
            if key.is_none() {
                errors.push(format!(
                    "document at '{col}'[{i}] missing required _key field"
                ));
            }
        }
    }

    // edge_definitions: from/to must reference declared collections.
    let mut edge_def_names: HashSet<&str> = HashSet::new();
    for ed in &file.edge_definitions {
        if !edge_def_names.insert(ed.name.as_str()) {
            errors.push(format!("edge_definition '{}' declared twice", ed.name));
        }
        // The edge_definition's own collection should be declared (and edge).
        match declared.get(ed.name.as_str()) {
            Some(CollectionType::Edge) => {}
            Some(CollectionType::Document) => errors.push(format!(
                "edge_definition '{}' references a document-type collection",
                ed.name
            )),
            None => errors.push(format!(
                "edge_definition '{}' references collection not declared in `collections:`",
                ed.name
            )),
        }
        for from_col in &ed.from_collections {
            match declared.get(from_col.as_str()) {
                Some(CollectionType::Document) => {}
                Some(CollectionType::Edge) => errors.push(format!(
                    "edge_definition '{}' has from_collection '{from_col}' \
                     declared as type:edge; endpoints must be type:document",
                    ed.name
                )),
                None => errors.push(format!(
                    "edge_definition '{}' has from_collection '{from_col}' \
                     not declared in `collections:`",
                    ed.name
                )),
            }
        }
        for to_col in &ed.to_collections {
            match declared.get(to_col.as_str()) {
                Some(CollectionType::Document) => {}
                Some(CollectionType::Edge) => errors.push(format!(
                    "edge_definition '{}' has to_collection '{to_col}' \
                     declared as type:edge; endpoints must be type:document",
                    ed.name
                )),
                None => errors.push(format!(
                    "edge_definition '{}' has to_collection '{to_col}' \
                     not declared in `collections:`",
                    ed.name
                )),
            }
        }
    }

    // named_graphs: every edge name must match a declared edge_definition.
    for ng in &file.named_graphs {
        for edge in &ng.edges {
            if !edge_def_names.contains(edge.as_str()) {
                errors.push(format!(
                    "named_graph '{}' references edge '{edge}' \
                     not declared in `edge_definitions:`",
                    ng.name
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(ApplyError::Validation(errors))
    }
}

/// Build the operation plan for a validated [`SchemaFile`].
///
/// Operations are ordered: collections first (so subsequent inserts
/// have a target), then documents, then edge_definitions registration,
/// then named graphs (which require their edge collections to exist).
pub fn plan(file: &SchemaFile) -> Vec<Operation> {
    let mut ops: Vec<Operation> = Vec::new();

    // hades_schema is always present: even an empty schema needs a
    // schema_meta document for `RuntimeSchema::load` to succeed.
    ops.push(Operation::CreateCollection {
        name: "hades_schema".into(),
        collection_type: CollectionType::Document.as_u32(),
    });

    for c in &file.collections {
        ops.push(Operation::CreateCollection {
            name: c.name.clone(),
            collection_type: c.collection_type.as_u32(),
        });
    }

    // Stable order for documents (by collection name, alphabetical).
    let mut doc_collections: Vec<&String> = file.documents.keys().collect();
    doc_collections.sort();
    for col in doc_collections {
        let docs = &file.documents[col];
        if !docs.is_empty() {
            ops.push(Operation::UpsertDocuments {
                collection: col.clone(),
                count: docs.len(),
            });
        }
    }

    for ed in &file.edge_definitions {
        ops.push(Operation::UpsertEdgeDefinition {
            name: ed.name.clone(),
            source_field: ed
                .source_field
                .clone()
                .unwrap_or_else(|| NO_SOURCE_FIELD.into()),
        });
    }

    // Persist named graph metadata to hades_schema so RuntimeSchema::load
    // can find it. The corresponding gharial creation follows.
    for ng in &file.named_graphs {
        ops.push(Operation::UpsertNamedGraphDoc { name: ng.name.clone() });
    }

    // Always emit a schema_meta upsert; it's required by RuntimeSchema::load.
    ops.push(Operation::UpsertSchemaMeta);

    for ng in &file.named_graphs {
        ops.push(Operation::CreateNamedGraph {
            name: ng.name.clone(),
            edges: ng.edges.clone(),
        });
    }

    ops
}

/// Apply a parsed schema file to a database.
pub async fn apply(
    pool: &ArangoPool,
    file: &SchemaFile,
    opts: ApplyOptions,
) -> Result<ApplyResult, ApplyError> {
    // Validation runs before any DB write — a malformed file never
    // touches the database.
    validate(file)?;

    if !opts.force && !opts.dry_run {
        check_not_in_use(pool, file).await?;
    }

    let mut result = ApplyResult {
        collections_created: 0,
        collections_skipped_existing: 0,
        documents_upserted: 0,
        edge_definitions_registered: 0,
        named_graphs_created: 0,
        named_graphs_skipped_existing: 0,
    };

    if opts.dry_run {
        debug!("schema apply dry-run: skipping all writes");
        return Ok(result);
    }

    // 1. Collections. `hades_schema` is always ensured: even a
    //    collections-only YAML needs it for schema_meta so subsequent
    //    `RuntimeSchema::load` succeeds (matches plan()'s ordering).
    ensure_collection(pool, "hades_schema", CollectionType::Document, &mut result).await?;
    for c in &file.collections {
        ensure_collection(pool, &c.name, c.collection_type, &mut result).await?;
    }

    // 2. Document seeds.
    let mut doc_collections: Vec<&String> = file.documents.keys().collect();
    doc_collections.sort();
    for col in doc_collections {
        let docs = &file.documents[col];
        if docs.is_empty() {
            continue;
        }
        let res = crud::insert_documents(pool, col, docs, /* overwrite= */ true).await?;
        let n = (res.created + res.updated) as usize;
        info!(collection = %col, count = n, "upserted documents");
        result.documents_upserted += n;
    }

    // 3. Edge definitions registered into hades_schema. The (name,
    //    source_field) pair forms the unique _key so multiple defs
    //    keyed off different source fields can coexist on one
    //    collection (the historical NL pattern).
    for ed in &file.edge_definitions {
        let sf_key_part = ed.source_field.as_deref().unwrap_or(NO_SOURCE_FIELD);
        let key = format!("edge__{}__{}", ed.name, sf_key_part);
        let doc = json!({
            "_key": key,
            "schema_type": "edge_definition",
            "name": ed.name,
            "source_field": ed.source_field,
            "from_collections": ed.from_collections,
            "to_collections": ed.to_collections,
            "description": ed.description,
        });
        crud::insert_documents(pool, "hades_schema", &[doc], /* overwrite= */ true).await?;
        result.edge_definitions_registered += 1;
        info!(name = %ed.name, "registered edge definition");
    }

    // 4. Named graph documents into hades_schema. Distinct from the
    //    gharial creation below: this is the record `RuntimeSchema::load`
    //    deserializes into `RuntimeNamedGraph`. Field name `edge_definitions`
    //    matches `RuntimeNamedGraph`'s expected layout.
    for ng in &file.named_graphs {
        let doc = json!({
            "_key": format!("named_graph__{}", ng.name),
            "schema_type": "named_graph",
            "name": ng.name,
            "edge_definitions": ng.edges,
            "description": ng.description.clone().unwrap_or_default(),
        });
        crud::insert_documents(pool, "hades_schema", &[doc], /* overwrite= */ true).await?;
        info!(name = %ng.name, "registered named graph document");
    }

    // 5. schema_meta document — required by RuntimeSchema::load. Empty
    //    seeds get an empty `relation_order` (RGCN training is deferred);
    //    `feature_dim` mirrors the runtime default for Jina V4.
    let relation_order: Vec<String> = Vec::new();
    let checksum = crate::graph::runtime_schema::compute_checksum(&relation_order);
    let meta_doc = json!({
        "_key": "meta",
        "schema_type": "schema_meta",
        "schema_version": 1u32,
        "seed_name": Value::Null,
        "relation_order": relation_order,
        "num_relations": 0u32,
        "feature_dim": 2048u32,
        "schema_checksum": checksum,
    });
    crud::insert_documents(pool, "hades_schema", &[meta_doc], /* overwrite= */ true).await?;
    info!("registered schema_meta document");

    // 6. Named graphs via gharial. Idempotent: 409 Conflict (graph
    //    exists) is treated as success; other errors propagate.
    //
    //    Build the edge_definitions lookup once: it depends only on
    //    `file`, not on which graph we're iterating.
    let edge_def_lookup: HashMap<&str, &EdgeDef> = file
        .edge_definitions
        .iter()
        .map(|ed| (ed.name.as_str(), ed))
        .collect();

    for ng in &file.named_graphs {
        let edge_definitions: Vec<Value> = ng
            .edges
            .iter()
            .filter_map(|edge_name| edge_def_lookup.get(edge_name.as_str()).copied())
            .map(|ed| {
                json!({
                    "collection": ed.name,
                    "from": ed.from_collections,
                    "to": ed.to_collections,
                })
            })
            .collect();

        let body = json!({
            "name": ng.name,
            "edgeDefinitions": edge_definitions,
        });

        match pool.writer().post("gharial", &body).await {
            Ok(_) => {
                result.named_graphs_created += 1;
                info!(name = %ng.name, "created named graph");
            }
            Err(e) if e.kind() == ArangoErrorKind::Conflict => {
                result.named_graphs_skipped_existing += 1;
                info!(name = %ng.name, "named graph already exists");
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(result)
}

// ── helpers ────────────────────────────────────────────────────────────

async fn ensure_collection(
    pool: &ArangoPool,
    name: &str,
    collection_type: CollectionType,
    result: &mut ApplyResult,
) -> Result<(), ApplyError> {
    let body = json!({
        "name": name,
        "type": collection_type.as_u32(),
    });
    match pool.writer().post("collection", &body).await {
        Ok(_) => {
            result.collections_created += 1;
            info!(collection = %name, ?collection_type, "created collection");
            Ok(())
        }
        Err(e) if e.kind() == ArangoErrorKind::Conflict => {
            result.collections_skipped_existing += 1;
            debug!(collection = %name, "collection already exists");
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

/// Heuristic for "in-use": the database has user-data beyond an
/// initial empty seed. The codebase universal layer doesn't count
/// (those collections may exist from prior `codebase ingest` runs
/// and shouldn't block schema bootstrap — see
/// `docs/declarative-schema.md` §11).
async fn check_not_in_use(
    pool: &ArangoPool,
    file: &SchemaFile,
) -> Result<(), ApplyError> {
    let universal: HashSet<&str> = crate::db::collections::CODEBASE
        .all_collections()
        .iter()
        .map(|(n, _)| *n)
        .collect();

    // For each user-declared collection, count its existing documents.
    // If any have > 0 docs, treat the DB as in-use.
    for c in &file.collections {
        if universal.contains(c.name.as_str()) {
            // Universal layer is exempt: codebase ingest writes here
            // independently of subject-layer schema bootstrap.
            continue;
        }
        let count_aql = "RETURN LENGTH(@@col)";
        let bind = json!({ "@col": c.name });
        let res = crate::db::query::query_single(
            pool,
            count_aql,
            Some(&bind),
            crate::db::query::ExecutionTarget::Reader,
        )
        .await;
        match res {
            Ok(Some(v)) => {
                let count = v.as_u64().unwrap_or(0);
                if count > 0 {
                    return Err(ApplyError::InUse {
                        db: pool.database().to_string(),
                        reason: format!("collection '{}' contains {count} document(s)", c.name),
                    });
                }
            }
            Ok(None) => {}
            // Missing collection → not in use (yet).
            Err(e) if e.is_not_found() => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

// ── tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal() {
        let yaml = r#"
collections:
  - name: axioms
    type: document
"#;
        let file = parse(yaml).unwrap();
        assert_eq!(file.collections.len(), 1);
        assert_eq!(file.collections[0].name, "axioms");
        assert_eq!(file.collections[0].collection_type, CollectionType::Document);
    }

    #[test]
    fn parse_documents_seed() {
        let yaml = r#"
collections:
  - { name: axioms, type: document }
axioms:
  - _key: foo
    is: ["a", "b"]
"#;
        let file = parse(yaml).unwrap();
        assert_eq!(file.collections.len(), 1);
        assert_eq!(file.documents.get("axioms").unwrap().len(), 1);
        assert_eq!(
            file.documents["axioms"][0]["_key"].as_str(),
            Some("foo")
        );
    }

    #[test]
    fn parse_rejects_unknown_top_level_via_documents_failure() {
        // An unknown top-level key gets treated as a document seed
        // block. If its value isn't a list, we get a clear error.
        let yaml = r#"
collections:
  - { name: axioms, type: document }
unexpected_section: "this should be a list"
"#;
        let err = parse(yaml).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(_)));
    }

    #[test]
    fn validate_rejects_undeclared_seed_block() {
        let yaml = r#"
collections:
  - { name: axioms, type: document }
smell_specs:
  - _key: smell-001
"#;
        let file = parse(yaml).unwrap();
        let err = validate(&file).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(ref errs) if errs.iter().any(|e| e.contains("smell_specs"))));
    }

    #[test]
    fn validate_rejects_missing_key() {
        let yaml = r#"
collections:
  - { name: axioms, type: document }
axioms:
  - is: ["foo"]
"#;
        let file = parse(yaml).unwrap();
        let err = validate(&file).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(ref errs) if errs.iter().any(|e| e.contains("missing required _key"))));
    }

    #[test]
    fn validate_rejects_duplicate_keys() {
        let yaml = r#"
collections:
  - { name: axioms, type: document }
axioms:
  - { _key: foo }
  - { _key: foo }
"#;
        let file = parse(yaml).unwrap();
        let err = validate(&file).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(ref errs) if errs.iter().any(|e| e.contains("duplicate"))));
    }

    #[test]
    fn validate_rejects_edge_def_to_undeclared_collection() {
        let yaml = r#"
collections:
  - { name: compliance_edges, type: edge }
edge_definitions:
  - name: compliance_edges
    from_collections: [codebase_files]
    to_collections: [smell_specs]
"#;
        let file = parse(yaml).unwrap();
        let err = validate(&file).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(ref errs) if
            errs.iter().any(|e| e.contains("codebase_files")) &&
            errs.iter().any(|e| e.contains("smell_specs"))));
    }

    #[test]
    fn validate_rejects_named_graph_referencing_missing_edge_def() {
        let yaml = r#"
collections:
  - { name: compliance_edges, type: edge }
named_graphs:
  - name: graph1
    edges: [missing_edge]
"#;
        let file = parse(yaml).unwrap();
        let err = validate(&file).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(ref errs) if errs.iter().any(|e| e.contains("missing_edge"))));
    }

    #[test]
    fn plan_orders_correctly() {
        let yaml = r#"
collections:
  - { name: axioms, type: document }
  - { name: smell_specs, type: document }
  - { name: compliance_edges, type: edge }
axioms:
  - { _key: a }
edge_definitions:
  - name: compliance_edges
    from_collections: [axioms]
    to_collections: [smell_specs]
named_graphs:
  - name: g1
    edges: [compliance_edges]
"#;
        let file = parse(yaml).unwrap();
        validate(&file).unwrap();
        let ops = plan(&file);
        // hades_schema first, then user collections, then docs, then edges, then graphs
        assert!(matches!(&ops[0], Operation::CreateCollection { name, .. } if name == "hades_schema"));
        // Find the index of each phase
        let last_create = ops.iter().rposition(|op| matches!(op, Operation::CreateCollection { .. })).unwrap();
        let first_upsert = ops.iter().position(|op| matches!(op, Operation::UpsertDocuments { .. })).unwrap();
        let first_edge_def = ops.iter().position(|op| matches!(op, Operation::UpsertEdgeDefinition { .. })).unwrap();
        let first_graph = ops.iter().position(|op| matches!(op, Operation::CreateNamedGraph { .. })).unwrap();
        assert!(last_create < first_upsert, "all collections before any upsert");
        assert!(first_upsert < first_edge_def, "documents before edge defs");
        assert!(first_edge_def < first_graph, "edge defs before named graphs");
    }

    #[test]
    fn full_pipeline_a_realistic_yaml() {
        // Mirror docs/declarative-schema.md §10's example.
        let yaml = r#"
collections:
  - { name: axioms, type: document }
  - { name: smell_specs, type: document }
  - { name: compliance_edges, type: edge }

axioms:
  - _key: testable-functions
    name: "Testable Functions"
    is:
      - "Pure functions with explicit inputs and outputs"
    is_not:
      - "Hidden global state mutations"

smell_specs:
  - _key: smell-010
    tier: static
    pattern: 'unwrap\(\)\s*$'
    description: "Bare .unwrap() outside tests"

edge_definitions:
  - name: compliance_edges
    source_field: source
    from_collections: [axioms]
    to_collections: [smell_specs]

named_graphs:
  - name: code_context_compliance
    edges: [compliance_edges]
"#;
        let file = parse(yaml).unwrap();
        validate(&file).unwrap();
        let ops = plan(&file);
        // 1 (hades_schema) + 3 collections + 2 doc upserts (axioms,
        // smell_specs) + 1 edge def + 1 named_graph_doc + 1 schema_meta
        // + 1 named_graph (gharial)
        assert_eq!(ops.len(), 1 + 3 + 2 + 1 + 1 + 1 + 1);
    }

    #[test]
    fn validate_rejects_edge_typed_endpoint() {
        // from_collections referencing an edge-type collection must be
        // rejected — gharial only allows document collections at endpoints.
        let yaml = r#"
collections:
  - { name: axioms, type: document }
  - { name: smell_specs, type: document }
  - { name: rel_a, type: edge }
  - { name: rel_b, type: edge }
edge_definitions:
  - name: rel_a
    from_collections: [rel_b]
    to_collections: [smell_specs]
"#;
        let file = parse(yaml).unwrap();
        let err = validate(&file).unwrap_err();
        assert!(matches!(err, ApplyError::Validation(ref errs) if
            errs.iter().any(|e| e.contains("rel_b") && e.contains("type:edge"))));
    }
}
