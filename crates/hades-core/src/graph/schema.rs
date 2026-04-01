//! NL knowledge graph schema — edge collections and named graphs.
//!
//! Port of `~/git/HADES/core/database/nl_graph_schema.py`.
//!
//! Defines:
//! - 16 `EdgeCollectionDef` instances (materialization metadata)
//! - 22 edge collection names with stable relation indices (for RGCN training)
//! - 6 named graphs composing edges into traversable scopes

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Paper prefixes and concept types
// ---------------------------------------------------------------------------

/// Paper prefixes — each paper contributes a family of typed collections.
pub const PAPER_PREFIXES: &[&str] = &[
    "hope",
    "atlas",
    "lattice",
    "miras",
    "titans",
    "titans_revisited",
    "tnt",
    "trellis",
];

/// Concept types found across papers.
pub const CONCEPT_TYPES: &[&str] = &[
    "abstractions",
    "algorithms",
    "axioms",
    "definitions",
    "equations",
    "lineage",
];

// ---------------------------------------------------------------------------
// Edge collection definition
// ---------------------------------------------------------------------------

/// Definition of an ArangoDB edge collection to be materialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCollectionDef {
    /// ArangoDB edge collection name.
    pub name: &'static str,
    /// Document field that holds the reference(s).
    pub source_field: &'static str,
    /// Vertex collections that can appear as `_from`.
    pub from_collections: Vec<&'static str>,
    /// Vertex collections that can appear as `_to`.
    pub to_collections: Vec<&'static str>,
    /// Human-readable description.
    pub description: &'static str,
    /// Whether `source_field` holds a list of references.
    pub is_array: bool,
    /// Additional fields to copy from the source doc onto edges.
    pub edge_attributes: Vec<&'static str>,
}

/// Definition of an ArangoDB named graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedGraphDef {
    /// Graph name (used in `GRAPH "name"` AQL clauses).
    pub name: &'static str,
    /// Indices into `ALL_EDGE_COLLECTIONS` composing this graph.
    pub edge_collection_indices: Vec<usize>,
    /// Human-readable purpose.
    pub description: &'static str,
}

impl NamedGraphDef {
    /// Build the JSON payload for `POST /_api/gharial`.
    ///
    /// Merges `from`/`to` collections across edge definitions that share
    /// the same collection name, matching the Python `to_gharial_payload()`.
    pub fn to_gharial_payload(&self) -> serde_json::Value {
        let mut seen: BTreeMap<&str, (BTreeSet<&str>, BTreeSet<&str>)> = BTreeMap::new();

        for &idx in &self.edge_collection_indices {
            let edef = &ALL_EDGE_COLLECTIONS[idx];
            let entry = seen.entry(edef.name).or_default();
            entry.0.extend(edef.from_collections.iter().copied());
            entry.1.extend(edef.to_collections.iter().copied());
        }

        let edge_defs: Vec<serde_json::Value> = seen
            .into_iter()
            .map(|(coll_name, (from, to))| {
                serde_json::json!({
                    "collection": coll_name,
                    "from": from.into_iter().collect::<Vec<_>>(),
                    "to": to.into_iter().collect::<Vec<_>>(),
                })
            })
            .collect();

        serde_json::json!({
            "name": self.name,
            "edgeDefinitions": edge_defs,
        })
    }

    /// Return the `EdgeCollectionDef` instances composing this graph.
    pub fn edge_definitions(&self) -> Vec<&'static EdgeCollectionDef> {
        self.edge_collection_indices
            .iter()
            .map(|&i| &ALL_EDGE_COLLECTIONS[i])
            .collect()
    }
}

/// Complete graph schema for the NL knowledge base.
#[derive(Debug, Clone)]
pub struct NlGraphSchema {
    pub edge_collections: &'static [EdgeCollectionDef],
    pub named_graphs: &'static [NamedGraphDef],
}

impl NlGraphSchema {
    /// Look up an edge collection definition by name.
    pub fn get_edge_collection(&self, name: &str) -> Option<&EdgeCollectionDef> {
        self.edge_collections.iter().find(|ec| ec.name == name)
    }

    /// Look up a named graph definition by name.
    pub fn get_named_graph(&self, name: &str) -> Option<&NamedGraphDef> {
        self.named_graphs.iter().find(|ng| ng.name == name)
    }

    /// Return sorted list of all unique edge collection names.
    pub fn all_edge_collection_names(&self) -> Vec<&str> {
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for ec in self.edge_collections {
            names.insert(ec.name);
        }
        names.into_iter().collect()
    }

    /// Return list of all named graph names.
    pub fn all_named_graph_names(&self) -> Vec<&str> {
        self.named_graphs.iter().map(|ng| ng.name).collect()
    }
}

// ---------------------------------------------------------------------------
// Helper functions for building collection lists
// ---------------------------------------------------------------------------

/// Build `{prefix}_{type}` for all paper prefixes.
fn paper_typed(concept_type: &'static str) -> Vec<&'static str> {
    match concept_type {
        "axioms" => vec![
            "hope_axioms", "atlas_axioms", "lattice_axioms", "miras_axioms",
            "titans_axioms", "titans_revisited_axioms", "tnt_axioms", "trellis_axioms",
        ],
        "equations" => vec![
            "hope_equations", "atlas_equations", "lattice_equations", "miras_equations",
            "titans_equations", "titans_revisited_equations", "tnt_equations", "trellis_equations",
        ],
        "definitions" => vec![
            "hope_definitions", "atlas_definitions", "lattice_definitions", "miras_definitions",
            "titans_definitions", "titans_revisited_definitions", "tnt_definitions", "trellis_definitions",
        ],
        "abstractions" => vec![
            "hope_abstractions", "atlas_abstractions", "lattice_abstractions", "miras_abstractions",
            "titans_abstractions", "titans_revisited_abstractions", "tnt_abstractions", "trellis_abstractions",
        ],
        "lineage" => vec![
            "hope_lineage", "atlas_lineage", "lattice_lineage", "miras_lineage",
            "titans_lineage", "titans_revisited_lineage", "tnt_lineage", "trellis_lineage",
        ],
        _ => vec![],
    }
}

/// Axiom collections: `nl_axioms` + all `{paper}_axioms`.
fn axiom_collections() -> Vec<&'static str> {
    let mut v = vec!["nl_axioms"];
    v.extend(paper_typed("axioms"));
    v
}

/// Equation collections: all `{paper}_equations` + `conveyance_equations`.
fn equation_collections() -> Vec<&'static str> {
    let mut v = paper_typed("equations");
    v.push("conveyance_equations");
    v
}

/// Definition collections: all `{paper}_definitions` + `conveyance_definitions`.
fn definition_collections() -> Vec<&'static str> {
    let mut v = paper_typed("definitions");
    v.push("conveyance_definitions");
    v
}

/// Migration target collections: paper-scoped + extra paper collections.
///
/// Reuses the already-leaked `&'static str` from `AXIOM_COMPLIANT_COLLECTIONS`
/// by filtering out NL-global and hecate_specs entries.
fn migration_target_collections() -> Vec<&'static str> {
    // NL-global collections to exclude from migration targets
    const NL_GLOBAL: &[&str] = &[
        "nl_axioms", "nl_reframings", "nl_build_paths", "nl_code_smells",
        "nl_ethnographic_notes", "nl_ethnography", "nl_optimizers",
        "nl_probe_patterns", "nl_roadmap", "nl_system", "nl_toolchain", "nl_articles",
        "hecate_specs",
    ];
    AXIOM_COMPLIANT_COLLECTIONS
        .iter()
        .copied()
        .filter(|s| !NL_GLOBAL.contains(s))
        .collect()
}

/// All vertex collections that can reference axioms (everything except infra).
///
/// In the Python code this is `ALL_VERTEX_COLLECTIONS - {paper_edges, builds, build_runs}`.
///
/// Uses `Box::leak` to produce `&'static str` from dynamically-formatted
/// `{prefix}_{type}` strings. This is safe because the `LazyLock` ensures
/// the leaks happen exactly once at program startup.
static AXIOM_COMPLIANT_COLLECTIONS: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| {
        let mut v = Vec::with_capacity(80);

        // Paper-scoped: prefix × type
        for prefix in PAPER_PREFIXES {
            for ctype in CONCEPT_TYPES {
                let s: &'static str = Box::leak(format!("{prefix}_{ctype}").into_boxed_str());
                v.push(s);
            }
        }

        // Extra paper-scoped
        v.extend([
            "hope_assumptions", "hope_blockers", "hope_code_smells",
            "hope_context_sources", "hope_examples", "hope_extensions",
            "hope_nl_reframings", "hope_optimizers", "hope_probes",
            "hope_python_signatures", "hope_triton_specs", "hope_validation_reports",
            "conveyance_definitions", "conveyance_equations", "conveyance_hypotheses",
            "conveyance_philosophy", "conveyance_protocols", "community_implementations",
        ]);

        // NL-global
        v.extend([
            "nl_axioms", "nl_reframings", "nl_build_paths", "nl_code_smells",
            "nl_ethnographic_notes", "nl_ethnography", "nl_optimizers",
            "nl_probe_patterns", "nl_roadmap", "nl_system", "nl_toolchain", "nl_articles",
        ]);

        // Infra (only hecate_specs; paper_edges/builds/build_runs excluded)
        v.push("hecate_specs");

        v.sort();
        v.dedup();
        v
    });

/// All vertex collections (the full union).
static ALL_VERTEX_COLLECTIONS: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| {
        let mut v = AXIOM_COMPLIANT_COLLECTIONS.clone();
        v.extend(["paper_edges", "builds", "build_runs"]);
        v.sort();
        v.dedup();
        v
    });

// ---------------------------------------------------------------------------
// Edge collection instances (indices 0..15)
// ---------------------------------------------------------------------------

/// All 16 edge collection definitions for materialization.
///
/// **These indices are distinct from [`EDGE_COLLECTION_NAMES`] indices.**
///
/// - `ALL_EDGE_COLLECTIONS` indices are used by
///   [`NamedGraphDef::edge_collection_indices`] and the materialization
///   pipeline. Example: `nl_validated_against_edges` is index **1** here.
/// - `EDGE_COLLECTION_NAMES` indices are RGCN relation-type indices used
///   in [`GraphData::edge_type`]. Example: `nl_validated_against_edges`
///   is index **19** there.
///
/// **Do not reorder either array** — doing so breaks materialization
/// (this array) or trained RGCN models (`EDGE_COLLECTION_NAMES`).
pub static ALL_EDGE_COLLECTIONS: std::sync::LazyLock<Vec<EdgeCollectionDef>> =
    std::sync::LazyLock::new(|| {
        vec![
            // 0: axiom_basis
            EdgeCollectionDef {
                name: "nl_axiom_basis_edges",
                source_field: "axiom_basis",
                from_collections: AXIOM_COMPLIANT_COLLECTIONS.clone(),
                to_collections: axiom_collections(),
                description: "Links any NL concept to its IS axiom. The most universal edge.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 1: validated_against
            EdgeCollectionDef {
                name: "nl_validated_against_edges",
                source_field: "validated_against",
                from_collections: AXIOM_COMPLIANT_COLLECTIONS.clone(),
                to_collections: axiom_collections(),
                description: "Links any NL concept to its IS_NOT anti-axiom.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 2: axiom inherits
            EdgeCollectionDef {
                name: "nl_axiom_inherits_edges",
                source_field: "inherits_from",
                from_collections: paper_typed("axioms"),
                to_collections: vec!["nl_axioms"],
                description: "Paper-level axiom inheritance. {paper}_axioms → nl_axioms.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 3: structural embodiments
            EdgeCollectionDef {
                name: "nl_structural_embodiment_edges",
                source_field: "structural_embodiments",
                from_collections: paper_typed("axioms"),
                to_collections: definition_collections(),
                description: "Axiom IS containers → definitions they structurally embody.",
                is_array: true,
                edge_attributes: vec![],
            },
            // 4: equation depends
            EdgeCollectionDef {
                name: "nl_equation_depends_edges",
                source_field: "depends_on",
                from_collections: equation_collections(),
                to_collections: equation_collections(),
                description: "Equation-to-equation dependency DAG.",
                is_array: true,
                edge_attributes: vec![],
            },
            // 5: definition source equation
            EdgeCollectionDef {
                name: "nl_definition_source_edges",
                source_field: "source_equation",
                from_collections: definition_collections(),
                to_collections: equation_collections(),
                description: "Definition → source equation it derives from.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 6: signature → equation
            EdgeCollectionDef {
                name: "nl_signature_equation_edges",
                source_field: "linked_equation",
                from_collections: vec!["hope_python_signatures", "hope_algorithms"],
                to_collections: equation_collections(),
                description: "Python signature/algorithm → equation it implements.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 7: NL reframing link
            EdgeCollectionDef {
                name: "nl_reframing_link_edges",
                source_field: "nl_reframing_link",
                from_collections: AXIOM_COMPLIANT_COLLECTIONS.clone(),
                to_collections: vec!["nl_reframings", "hope_nl_reframings"],
                description: "Concept → its NL reframing (philosophical lens).",
                is_array: false,
                edge_attributes: vec![],
            },
            // 8: migration provenance
            EdgeCollectionDef {
                name: "nl_migration_edges",
                source_field: "migrated_from",
                from_collections: vec![
                    "nl_axioms", "nl_reframings", "nl_build_paths", "nl_code_smells",
                    "nl_ethnographic_notes", "nl_ethnography", "nl_optimizers",
                    "nl_probe_patterns", "nl_roadmap", "nl_system", "nl_toolchain", "nl_articles",
                ],
                to_collections: migration_target_collections(),
                description: "Provenance from promoted nl_* docs back to original paper source.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 9: lineage chain
            EdgeCollectionDef {
                name: "nl_lineage_chain_edges",
                source_field: "chain",
                from_collections: paper_typed("lineage"),
                to_collections: {
                    let mut v = paper_typed("abstractions");
                    v.extend(equation_collections());
                    v.extend(definition_collections());
                    v.sort();
                    v.dedup();
                    v
                },
                description: "Ordered lineage chains: chain[0]→chain[1]→... with metadata.",
                is_array: true,
                edge_attributes: vec!["name", "type", "description"],
            },
            // 10: hecate trace → equations
            EdgeCollectionDef {
                name: "nl_hecate_trace_edges",
                source_field: "traced_to_equations",
                from_collections: vec!["hecate_specs"],
                to_collections: equation_collections(),
                description: "Hecate build spec → equations it implements.",
                is_array: true,
                edge_attributes: vec![],
            },
            // 11: hecate trace → algorithms
            EdgeCollectionDef {
                name: "nl_hecate_trace_edges",
                source_field: "traced_to_algorithms",
                from_collections: vec!["hecate_specs"],
                to_collections: vec!["hope_algorithms"],
                description: "Hecate build spec → algorithms.",
                is_array: true,
                edge_attributes: vec![],
            },
            // 12: hecate trace → axioms
            EdgeCollectionDef {
                name: "nl_hecate_trace_edges",
                source_field: "traced_to_axioms",
                from_collections: vec!["hecate_specs"],
                to_collections: axiom_collections(),
                description: "Hecate build spec → axioms.",
                is_array: true,
                edge_attributes: vec![],
            },
            // 13: cross-paper edges
            EdgeCollectionDef {
                name: "nl_cross_paper_edges",
                source_field: "from_node",
                from_collections: ALL_VERTEX_COLLECTIONS.clone(),
                to_collections: ALL_VERTEX_COLLECTIONS.clone(),
                description: "Cross-paper concept relationships from paper_edges collection.",
                is_array: false,
                edge_attributes: vec!["name", "description", "relationship", "source_paper", "target_paper"],
            },
            // 14: smell compliance
            EdgeCollectionDef {
                name: "nl_smell_compliance_edges",
                source_field: "smell_compliance",
                from_collections: AXIOM_COMPLIANT_COLLECTIONS.clone(),
                to_collections: vec!["nl_code_smells", "hope_code_smells"],
                description: "Concept → code smell compliance records.",
                is_array: false,
                edge_attributes: vec![],
            },
            // 15: build path counterpart
            EdgeCollectionDef {
                name: "nl_build_path_edges",
                source_field: "counterpart",
                from_collections: vec!["nl_build_paths"],
                to_collections: vec!["nl_reframings"],
                description: "Build path → counterpart NL reframing.",
                is_array: false,
                edge_attributes: vec![],
            },
        ]
    });

// ---------------------------------------------------------------------------
// Edge collection names for RGCN training (22 relations)
// ---------------------------------------------------------------------------

/// Ordered edge collection names used as relation indices for RGCN training.
///
/// **Index positions are stable** — the RGCN model encodes relation types by
/// index. Adding new collections must append; reordering breaks trained models.
///
/// Matches Python `core/graph/loader.py::EDGE_COLLECTIONS`.
pub const EDGE_COLLECTION_NAMES: &[&str] = &[
    "nl_axiom_basis_edges",           //  0
    "nl_axiom_inherits_edges",        //  1
    "nl_axiom_violation_edges",       //  2  (future, empty)
    "nl_cross_paper_edges",           //  3
    "nl_code_callgraph_edges",        //  4  (future, empty)
    "nl_code_equation_edges",         //  5  (future, empty)
    "nl_code_test_edges",             //  6  (future, empty)
    "nl_definition_source_edges",     //  7
    "nl_equation_depends_edges",      //  8
    "nl_equation_source_edges",       //  9
    "nl_hecate_trace_edges",          // 10
    "nl_lineage_chain_edges",         // 11
    "nl_migration_edges",             // 12
    "nl_paper_cross_reference_edges", // 13
    "nl_reframing_link_edges",        // 14
    "nl_signature_equation_edges",    // 15
    "nl_smell_compliance_edges",      // 16
    "nl_smell_source_edges",          // 17
    "nl_structural_embodiment_edges", // 18
    "nl_validated_against_edges",     // 19
    "persephone_edges",               // 20
    "nl_smell_spec_edges",            // 21
];

/// Number of relation types (edge collections) for RGCN training.
pub const NUM_RELATIONS: usize = EDGE_COLLECTION_NAMES.len(); // 22

/// Jina V4 embedding dimension.
///
/// Must match `EmbeddingModelConfig::dimension` in the runtime config.
/// The graph loader (P4.2) validates this at load time.
pub const JINA_DIM: usize = 2048;

/// Look up the relation index for an edge collection name.
///
/// Returns `None` if the collection is not in the training set.
pub fn relation_index(collection_name: &str) -> Option<usize> {
    EDGE_COLLECTION_NAMES
        .iter()
        .position(|&n| n == collection_name)
}

// ---------------------------------------------------------------------------
// Named graph definitions
// ---------------------------------------------------------------------------

/// All 6 named graph definitions.
///
/// Indices reference positions in `ALL_EDGE_COLLECTIONS`.
pub static ALL_NAMED_GRAPHS: std::sync::LazyLock<Vec<NamedGraphDef>> =
    std::sync::LazyLock::new(|| {
        vec![
            // nl_core: axiom compliance backbone
            NamedGraphDef {
                name: "nl_core",
                edge_collection_indices: vec![0, 1, 2], // axiom_basis, validated_against, inherits
                description: "The axiom compliance backbone. IS/IS_NOT + inheritance.",
            },
            // nl_equations: equation dependency network
            NamedGraphDef {
                name: "nl_equations",
                edge_collection_indices: vec![4, 5, 6], // depends, source_eq, signature_eq
                description: "Equation dependency DAG + definitions + signatures.",
            },
            // nl_hierarchy: structural embodiments + lineage
            NamedGraphDef {
                name: "nl_hierarchy",
                edge_collection_indices: vec![3, 9], // structural_embodiment, lineage_chain
                description: "Concept hierarchy: embodiments and lineage chains.",
            },
            // nl_hecate: build spec traceability
            NamedGraphDef {
                name: "nl_hecate",
                edge_collection_indices: vec![10, 11, 12], // hecate traces (eq, alg, axiom)
                description: "Hecate build spec traceability to equations, algorithms, axioms.",
            },
            // nl_cross_paper: cross-paper relationships
            NamedGraphDef {
                name: "nl_cross_paper",
                edge_collection_indices: vec![13], // cross_paper
                description: "Cross-paper concept relationships.",
            },
            // nl_concept_map: everything
            NamedGraphDef {
                name: "nl_concept_map",
                edge_collection_indices: (0..ALL_EDGE_COLLECTIONS.len()).collect(),
                description: "Complete NL knowledge graph — all edges.",
            },
        ]
    });

/// Global schema singleton.
pub static NL_GRAPH_SCHEMA: std::sync::LazyLock<NlGraphSchema> =
    std::sync::LazyLock::new(|| NlGraphSchema {
        edge_collections: &ALL_EDGE_COLLECTIONS,
        named_graphs: &ALL_NAMED_GRAPHS,
    });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_collection_count() {
        assert_eq!(ALL_EDGE_COLLECTIONS.len(), 16);
    }

    #[test]
    fn test_relation_count() {
        assert_eq!(NUM_RELATIONS, 22);
        assert_eq!(EDGE_COLLECTION_NAMES.len(), 22);
    }

    #[test]
    fn test_relation_index_lookup() {
        assert_eq!(relation_index("nl_axiom_basis_edges"), Some(0));
        assert_eq!(relation_index("nl_validated_against_edges"), Some(19));
        assert_eq!(relation_index("persephone_edges"), Some(20));
        assert_eq!(relation_index("nl_smell_spec_edges"), Some(21));
        assert_eq!(relation_index("nonexistent"), None);
    }

    #[test]
    fn test_relation_indices_are_contiguous() {
        for (i, name) in EDGE_COLLECTION_NAMES.iter().enumerate() {
            assert_eq!(
                relation_index(name),
                Some(i),
                "relation index mismatch for {name}"
            );
        }
    }

    #[test]
    fn test_named_graph_count() {
        assert_eq!(ALL_NAMED_GRAPHS.len(), 6);
    }

    #[test]
    fn test_named_graph_names() {
        let names: Vec<&str> = ALL_NAMED_GRAPHS.iter().map(|g| g.name).collect();
        assert!(names.contains(&"nl_core"));
        assert!(names.contains(&"nl_equations"));
        assert!(names.contains(&"nl_hierarchy"));
        assert!(names.contains(&"nl_hecate"));
        assert!(names.contains(&"nl_cross_paper"));
        assert!(names.contains(&"nl_concept_map"));
    }

    #[test]
    fn test_concept_map_contains_all_edges() {
        let concept_map = ALL_NAMED_GRAPHS
            .iter()
            .find(|g| g.name == "nl_concept_map")
            .unwrap();
        assert_eq!(
            concept_map.edge_collection_indices.len(),
            ALL_EDGE_COLLECTIONS.len(),
            "nl_concept_map should reference all edge collections"
        );
    }

    #[test]
    fn test_schema_lookup() {
        let schema = &*NL_GRAPH_SCHEMA;
        assert!(schema.get_edge_collection("nl_axiom_basis_edges").is_some());
        assert!(schema.get_edge_collection("nonexistent").is_none());
        assert!(schema.get_named_graph("nl_core").is_some());
        assert!(schema.get_named_graph("nonexistent").is_none());
    }

    #[test]
    fn test_unique_edge_collection_names() {
        let names = NL_GRAPH_SCHEMA.all_edge_collection_names();
        // 16 defs but some share names (3 hecate traces → 1 name), so fewer unique names
        assert!(names.len() < ALL_EDGE_COLLECTIONS.len());
        assert!(names.contains(&"nl_hecate_trace_edges"));
    }

    #[test]
    fn test_gharial_payload() {
        let nl_core = ALL_NAMED_GRAPHS
            .iter()
            .find(|g| g.name == "nl_core")
            .unwrap();
        let payload = nl_core.to_gharial_payload();
        assert_eq!(payload["name"], "nl_core");
        let edge_defs = payload["edgeDefinitions"].as_array().unwrap();
        assert_eq!(edge_defs.len(), 3); // 3 distinct edge collection names
    }

    #[test]
    fn test_axiom_basis_from_collections_nonempty() {
        let axiom_basis = &ALL_EDGE_COLLECTIONS[0];
        assert_eq!(axiom_basis.name, "nl_axiom_basis_edges");
        assert!(
            !axiom_basis.from_collections.is_empty(),
            "axiom_basis should have from_collections"
        );
        assert!(
            !axiom_basis.to_collections.is_empty(),
            "axiom_basis should have to_collections"
        );
    }

    #[test]
    fn test_hecate_traces_share_collection_name() {
        // Indices 10, 11, 12 all use "nl_hecate_trace_edges"
        assert_eq!(ALL_EDGE_COLLECTIONS[10].name, "nl_hecate_trace_edges");
        assert_eq!(ALL_EDGE_COLLECTIONS[11].name, "nl_hecate_trace_edges");
        assert_eq!(ALL_EDGE_COLLECTIONS[12].name, "nl_hecate_trace_edges");
        // But different source fields
        assert_eq!(ALL_EDGE_COLLECTIONS[10].source_field, "traced_to_equations");
        assert_eq!(ALL_EDGE_COLLECTIONS[11].source_field, "traced_to_algorithms");
        assert_eq!(ALL_EDGE_COLLECTIONS[12].source_field, "traced_to_axioms");
    }

    #[test]
    fn test_paper_prefixes_match_python() {
        assert_eq!(PAPER_PREFIXES.len(), 8);
        assert!(PAPER_PREFIXES.contains(&"hope"));
        assert!(PAPER_PREFIXES.contains(&"titans_revisited"));
    }

    #[test]
    fn test_paper_typed_synced_with_prefixes() {
        // Concept types used in edge collection definitions.
        // Not all CONCEPT_TYPES have paper_typed entries (e.g. "algorithms"
        // only exists as hope_algorithms, not for all papers).
        let edge_types = ["axioms", "equations", "definitions", "abstractions", "lineage"];

        for &ctype in &edge_types {
            let entries = paper_typed(ctype);
            assert_eq!(
                entries.len(),
                PAPER_PREFIXES.len(),
                "paper_typed({ctype:?}) has {} entries but PAPER_PREFIXES has {}",
                entries.len(),
                PAPER_PREFIXES.len(),
            );
            // Each entry should be "{prefix}_{ctype}" in PAPER_PREFIXES order
            for (entry, &prefix) in entries.iter().zip(PAPER_PREFIXES) {
                let expected_suffix = format!("_{ctype}");
                assert!(
                    entry.starts_with(prefix) && entry.ends_with(&expected_suffix),
                    "paper_typed({ctype:?}): expected {prefix}_{ctype}, got {entry}"
                );
            }
        }

        // Unknown concept types should return empty
        assert!(paper_typed("nonexistent").is_empty());
    }
}
