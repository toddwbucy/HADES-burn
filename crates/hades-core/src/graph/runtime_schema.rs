//! Runtime schema — database-local ontology definitions.
//!
//! Each database has a `hades_schema` collection storing its edge definitions,
//! named graphs, and metadata as documents. This replaces the compile-time
//! statics in `schema.rs` for all runtime operations (materialize, graph
//! creation, RGCN training).
//!
//! The compile-time statics remain as **seeds** — written to `_schema` by
//! `db schema init --seed nl`. They are never read in the hot path.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::db::ArangoPool;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors encountered while loading or validating a runtime schema.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("hades_schema collection not found")]
    NoSchemaCollection,

    #[error("schema metadata document missing (expected _key: \"meta\")")]
    NoMetaDocument,

    #[error("failed to query _schema: {0}")]
    Query(String),

    #[error("failed to deserialize schema document: {0}")]
    Deserialize(String),

    #[error("schema validation failed: {0}")]
    Validation(String),
}

// ---------------------------------------------------------------------------
// Runtime types (owned strings, loaded from `_schema`)
// ---------------------------------------------------------------------------

/// Runtime edge collection definition.
///
/// Mirrors [`super::schema::EdgeCollectionDef`] but with owned `String`s
/// so it can be deserialized from ArangoDB documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEdgeDef {
    /// ArangoDB edge collection name.
    pub name: String,
    /// Document field that holds the reference(s).
    pub source_field: String,
    /// Vertex collections that can appear as `_from`.
    pub from_collections: Vec<String>,
    /// Vertex collections that can appear as `_to`.
    pub to_collections: Vec<String>,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Whether `source_field` holds a list of references.
    #[serde(default)]
    pub is_array: bool,
    /// Additional fields to copy from the source doc onto edges.
    #[serde(default)]
    pub edge_attributes: Vec<String>,
    /// Materialization strategy: "standard", "lineage", "cross_paper", "paired_fields".
    #[serde(default = "default_strategy")]
    pub materialize_strategy: String,
    /// Stable RGCN relation index (position in `relation_order`).
    #[serde(default)]
    pub relation_index: Option<u32>,
}

fn default_strategy() -> String {
    "standard".to_string()
}

/// Runtime named graph definition.
///
/// References edge definitions **by name**, not by index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeNamedGraph {
    /// Graph name (used in `GRAPH "name"` AQL clauses).
    pub name: String,
    /// Edge collection names composing this graph.
    pub edge_definitions: Vec<String>,
    /// Human-readable purpose.
    #[serde(default)]
    pub description: String,
}

/// Schema metadata — one per database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMeta {
    /// Incremented on any schema mutation.
    #[serde(default = "default_version")]
    pub schema_version: u32,
    /// Name of the seed used to initialize (e.g. "nl", "empty"), if any.
    #[serde(default)]
    pub seed_name: Option<String>,
    /// Ordered list of edge collection names for RGCN training.
    /// Index position = relation type index. **Append-only.**
    #[serde(default)]
    pub relation_order: Vec<String>,
    /// Cached `relation_order.len()`.
    #[serde(default)]
    pub num_relations: usize,
    /// Embedding feature dimension (e.g. 2048 for Jina V4).
    #[serde(default = "default_feature_dim")]
    pub feature_dim: usize,
    /// SHA-256 of canonicalized `relation_order`.
    #[serde(default)]
    pub schema_checksum: String,
}

fn default_version() -> u32 {
    1
}

fn default_feature_dim() -> usize {
    2048
}

// ---------------------------------------------------------------------------
// RuntimeSchema — the loaded, queryable schema
// ---------------------------------------------------------------------------

/// Complete runtime schema loaded from `hades_schema` collection.
#[derive(Debug, Clone)]
pub struct RuntimeSchema {
    pub meta: SchemaMeta,
    pub edge_definitions: Vec<RuntimeEdgeDef>,
    pub named_graphs: Vec<RuntimeNamedGraph>,
    /// Whether the schema was loaded from the database (true) or from
    /// compile-time NL statics as a fallback (false).
    pub from_database: bool,
}

impl RuntimeSchema {
    /// Load schema from the `_schema` collection.
    ///
    /// If the collection does not exist, falls back to compile-time NL statics
    /// for backward compatibility with databases that predate runtime schema.
    pub async fn load(pool: &ArangoPool) -> Result<Self, SchemaError> {
        // Check if _schema collection exists.
        let collections = crate::db::crud::list_collections(pool, false)
            .await
            .map_err(|e| SchemaError::Query(e.to_string()))?;

        let has_schema = collections.iter().any(|c| c.name == "hades_schema");
        if !has_schema {
            return Self::from_nl_statics();
        }

        // Load all documents from _schema.
        let aql = "FOR d IN hades_schema RETURN d";
        let result = crate::db::query::query(
            pool,
            aql,
            None,
            None,
            false,
            crate::db::query::ExecutionTarget::Reader,
        )
        .await
        .map_err(|e| SchemaError::Query(e.to_string()))?;
        let docs = result.results;

        if docs.is_empty() {
            return Self::from_nl_statics();
        }

        let mut meta: Option<SchemaMeta> = None;
        let mut edge_defs = Vec::new();
        let mut named_graphs = Vec::new();

        for doc in &docs {
            let schema_type: &str = doc
                .get("schema_type")
                .and_then(Value::as_str)
                .unwrap_or("");

            match schema_type {
                "schema_meta" => {
                    meta = Some(
                        serde_json::from_value(doc.clone())
                            .map_err(|e| SchemaError::Deserialize(e.to_string()))?,
                    );
                }
                "edge_definition" => {
                    edge_defs.push(
                        serde_json::from_value(doc.clone())
                            .map_err(|e| SchemaError::Deserialize(e.to_string()))?,
                    );
                }
                "named_graph" => {
                    named_graphs.push(
                        serde_json::from_value(doc.clone())
                            .map_err(|e| SchemaError::Deserialize(e.to_string()))?,
                    );
                }
                _ => {} // Ignore unknown document types.
            }
        }

        let meta = meta.ok_or(SchemaError::NoMetaDocument)?;

        Ok(Self {
            meta,
            edge_definitions: edge_defs,
            named_graphs,
            from_database: true,
        })
    }

    /// Construct a RuntimeSchema from the compile-time NL statics.
    ///
    /// Used as a fallback when `_schema` collection doesn't exist.
    fn from_nl_statics() -> Result<Self, SchemaError> {
        use super::schema::{ALL_EDGE_COLLECTIONS, ALL_NAMED_GRAPHS, EDGE_COLLECTION_NAMES, JINA_DIM};

        let edge_definitions: Vec<RuntimeEdgeDef> = ALL_EDGE_COLLECTIONS
            .iter()
            .map(|edef| {
                let strategy = match (edef.name, edef.source_field) {
                    ("nl_lineage_chain_edges", "chain") => "lineage",
                    ("nl_cross_paper_edges", "from_node") => "cross_paper",
                    _ => "standard",
                };
                RuntimeEdgeDef {
                    name: edef.name.to_string(),
                    source_field: edef.source_field.to_string(),
                    from_collections: edef.from_collections.iter().map(|s| s.to_string()).collect(),
                    to_collections: edef.to_collections.iter().map(|s| s.to_string()).collect(),
                    description: edef.description.to_string(),
                    is_array: edef.is_array,
                    edge_attributes: edef.edge_attributes.iter().map(|s| s.to_string()).collect(),
                    materialize_strategy: strategy.to_string(),
                    relation_index: super::schema::relation_index(edef.name).map(|i| i as u32),
                }
            })
            .collect();

        let named_graphs: Vec<RuntimeNamedGraph> = ALL_NAMED_GRAPHS
            .iter()
            .map(|ng| {
                let edge_names: Vec<String> = ng
                    .edge_collection_indices
                    .iter()
                    .map(|&i| ALL_EDGE_COLLECTIONS[i].name.to_string())
                    .collect();
                RuntimeNamedGraph {
                    name: ng.name.to_string(),
                    edge_definitions: edge_names,
                    description: ng.description.to_string(),
                }
            })
            .collect();

        let relation_order: Vec<String> =
            EDGE_COLLECTION_NAMES.iter().map(|s| s.to_string()).collect();
        let checksum = compute_checksum(&relation_order);

        let meta = SchemaMeta {
            schema_version: 1,
            seed_name: Some("nl".to_string()),
            num_relations: relation_order.len(),
            relation_order,
            feature_dim: JINA_DIM,
            schema_checksum: checksum,
        };

        Ok(Self { meta, edge_definitions, named_graphs, from_database: false })
    }

    /// Look up an edge definition by name.
    pub fn get_edge_def(&self, name: &str) -> Option<&RuntimeEdgeDef> {
        self.edge_definitions.iter().find(|e| e.name == name)
    }

    /// Look up a named graph by name.
    pub fn get_named_graph(&self, name: &str) -> Option<&RuntimeNamedGraph> {
        self.named_graphs.iter().find(|g| g.name == name)
    }

    /// Return the RGCN relation index for an edge collection name.
    pub fn relation_index(&self, collection_name: &str) -> Option<usize> {
        self.meta
            .relation_order
            .iter()
            .position(|n| n == collection_name)
    }

    /// Build a Gharial API payload for a named graph.
    ///
    /// Returns `None` if the graph name isn't found.
    pub fn to_gharial_payload(&self, graph_name: &str) -> Option<Value> {
        let ng = self.get_named_graph(graph_name)?;

        // Group edge definitions by collection name, merging from/to sets.
        let mut edge_map: std::collections::BTreeMap<&str, (std::collections::BTreeSet<&str>, std::collections::BTreeSet<&str>)> =
            std::collections::BTreeMap::new();

        for edge_name in &ng.edge_definitions {
            if let Some(edef) = self.get_edge_def(edge_name) {
                let entry = edge_map.entry(&edef.name).or_default();
                for fc in &edef.from_collections {
                    entry.0.insert(fc);
                }
                for tc in &edef.to_collections {
                    entry.1.insert(tc);
                }
            }
        }

        let edge_defs: Vec<Value> = edge_map
            .into_iter()
            .map(|(collection, (from, to))| {
                serde_json::json!({
                    "collection": collection,
                    "from": from.into_iter().collect::<Vec<_>>(),
                    "to": to.into_iter().collect::<Vec<_>>(),
                })
            })
            .collect();

        Some(serde_json::json!({
            "name": ng.name,
            "edgeDefinitions": edge_defs,
        }))
    }
}

// ---------------------------------------------------------------------------
// Seed generation
// ---------------------------------------------------------------------------

/// Generate `_schema` documents from the compile-time NL statics.
///
/// Returns a list of JSON documents ready for bulk insert into `_schema`.
pub fn nl_seed_documents() -> Vec<Value> {
    use super::schema::{ALL_EDGE_COLLECTIONS, ALL_NAMED_GRAPHS, EDGE_COLLECTION_NAMES, JINA_DIM};

    let mut docs = Vec::with_capacity(ALL_EDGE_COLLECTIONS.len() + ALL_NAMED_GRAPHS.len() + 1);

    // Edge definitions.
    for edef in ALL_EDGE_COLLECTIONS.iter() {
        let strategy = match (edef.name, edef.source_field) {
            ("nl_lineage_chain_edges", "chain") => "lineage",
            ("nl_cross_paper_edges", "from_node") => "cross_paper",
            _ => "standard",
        };
        let key = format!("edge__{}__{}", edef.name, edef.source_field);
        docs.push(serde_json::json!({
            "_key": key,
            "schema_type": "edge_definition",
            "name": edef.name,
            "source_field": edef.source_field,
            "from_collections": edef.from_collections,
            "to_collections": edef.to_collections,
            "description": edef.description,
            "is_array": edef.is_array,
            "edge_attributes": edef.edge_attributes,
            "materialize_strategy": strategy,
            "relation_index": super::schema::relation_index(edef.name),
        }));
    }

    // Named graphs.
    for ng in ALL_NAMED_GRAPHS.iter() {
        let edge_names: Vec<&str> = ng
            .edge_collection_indices
            .iter()
            .map(|&i| ALL_EDGE_COLLECTIONS[i].name)
            .collect();
        docs.push(serde_json::json!({
            "_key": format!("graph__{}", ng.name),
            "schema_type": "named_graph",
            "name": ng.name,
            "edge_definitions": edge_names,
            "description": ng.description,
        }));
    }

    // Metadata.
    let relation_order: Vec<&str> = EDGE_COLLECTION_NAMES.to_vec();
    let relation_order_owned: Vec<String> = relation_order.iter().map(|s| s.to_string()).collect();
    let checksum = compute_checksum(&relation_order_owned);

    docs.push(serde_json::json!({
        "_key": "meta",
        "schema_type": "schema_meta",
        "schema_version": 1,
        "seed_name": "nl",
        "relation_order": relation_order,
        "num_relations": relation_order.len(),
        "feature_dim": JINA_DIM,
        "schema_checksum": checksum,
    }));

    docs
}

/// Generate an empty seed — metadata only, no edge definitions or graphs.
pub fn empty_seed_documents() -> Vec<Value> {
    vec![serde_json::json!({
        "_key": "meta",
        "schema_type": "schema_meta",
        "schema_version": 1,
        "seed_name": serde_json::Value::Null,
        "relation_order": [],
        "num_relations": 0,
        "feature_dim": 2048,
        "schema_checksum": compute_checksum(&[]),
    })]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute SHA-256 checksum of the relation order.
pub fn compute_checksum(relation_order: &[String]) -> String {
    let canonical = serde_json::to_string(relation_order).unwrap_or_default();
    let hash = Sha256::digest(canonical.as_bytes());
    // Convert hash bytes to hex string.
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nl_seed_document_count() {
        let docs = nl_seed_documents();
        // 16 edge defs + 6 named graphs + 1 meta = 23
        assert_eq!(docs.len(), 23);
    }

    #[test]
    fn test_nl_seed_has_meta() {
        let docs = nl_seed_documents();
        let meta = docs.iter().find(|d| d["_key"] == "meta").unwrap();
        assert_eq!(meta["schema_type"], "schema_meta");
        assert_eq!(meta["num_relations"], 22);
        assert_eq!(meta["seed_name"], "nl");
        assert_eq!(meta["feature_dim"], 2048);
    }

    #[test]
    fn test_nl_seed_edge_keys_are_deterministic() {
        let docs = nl_seed_documents();
        let first_edge = docs
            .iter()
            .find(|d| d["schema_type"] == "edge_definition")
            .unwrap();
        let key = first_edge["_key"].as_str().unwrap();
        assert!(key.starts_with("edge__"));
        assert!(key.contains("__"));
    }

    #[test]
    fn test_nl_seed_strategy_assignment() {
        let docs = nl_seed_documents();
        let lineage = docs
            .iter()
            .find(|d| d["name"] == "nl_lineage_chain_edges" && d["source_field"] == "chain")
            .unwrap();
        assert_eq!(lineage["materialize_strategy"], "lineage");

        let cross_paper = docs
            .iter()
            .find(|d| d["name"] == "nl_cross_paper_edges")
            .unwrap();
        assert_eq!(cross_paper["materialize_strategy"], "cross_paper");

        let standard = docs
            .iter()
            .find(|d| d["name"] == "nl_axiom_basis_edges")
            .unwrap();
        assert_eq!(standard["materialize_strategy"], "standard");
    }

    #[test]
    fn test_empty_seed() {
        let docs = empty_seed_documents();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0]["schema_type"], "schema_meta");
        assert_eq!(docs[0]["num_relations"], 0);
        assert!(docs[0]["seed_name"].is_null());
    }

    #[test]
    fn test_checksum_deterministic() {
        let order = vec!["a".to_string(), "b".to_string()];
        let c1 = compute_checksum(&order);
        let c2 = compute_checksum(&order);
        assert_eq!(c1, c2);
        assert!(c1.starts_with("sha256:"));
    }

    #[test]
    fn test_checksum_changes_on_reorder() {
        let order1 = vec!["a".to_string(), "b".to_string()];
        let order2 = vec!["b".to_string(), "a".to_string()];
        assert_ne!(compute_checksum(&order1), compute_checksum(&order2));
    }

    #[test]
    fn test_from_nl_statics() {
        let schema = RuntimeSchema::from_nl_statics().unwrap();
        assert_eq!(schema.meta.num_relations, 22);
        assert_eq!(schema.edge_definitions.len(), 16);
        assert_eq!(schema.named_graphs.len(), 6);
        assert!(schema.meta.schema_checksum.starts_with("sha256:"));
    }

    #[test]
    fn test_runtime_relation_index() {
        let schema = RuntimeSchema::from_nl_statics().unwrap();
        assert_eq!(schema.relation_index("nl_axiom_basis_edges"), Some(0));
        assert_eq!(schema.relation_index("nl_validated_against_edges"), Some(19));
        assert_eq!(schema.relation_index("nonexistent"), None);
    }

    #[test]
    fn test_runtime_gharial_payload() {
        let schema = RuntimeSchema::from_nl_statics().unwrap();
        let payload = schema.to_gharial_payload("nl_core").unwrap();
        assert_eq!(payload["name"], "nl_core");
        let edge_defs = payload["edgeDefinitions"].as_array().unwrap();
        assert_eq!(edge_defs.len(), 3);
    }

    #[test]
    fn test_runtime_gharial_payload_unknown_graph() {
        let schema = RuntimeSchema::from_nl_statics().unwrap();
        assert!(schema.to_gharial_payload("nonexistent").is_none());
    }
}
