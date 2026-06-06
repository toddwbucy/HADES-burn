# Spec — Pluggable Graph Backend for HADES

**Status:** Draft / RFC (not ratified)
**Date:** 2026-06-06
**Implements:** [pluggable-graph-backend-prd.md](pluggable-graph-backend-prd.md)
**Author:** Todd Bucy + Claude (HADES-Burn session)

This is the *how*. It is grounded in a coupling audit of `hades-core`, `hades-cli`,
and `hades-prefetch` (findings inline as `file:line`). Headline: **~65% of the code is
already backend-portable**; the coupled 35% is concentrated in graph I/O, vector search,
schema/gharial, and transport.

---

## 1. The seam

HADES already routes all model-facing data access through *closed structured operations*
(no raw AQL above `dispatch.rs`). The training subsystem is already DB-agnostic
(safetensors IPC + a DB-unaware Python service, `training.rs`, `hades-prefetch/`). So the
trait boundary sits at the **logical data-access layer**: everything above it
(`dispatch`, `schema_apply` orchestration, the graph loader's IDMap/pooling math, the
RGCN/SAGE orchestration, search fusion, output) stays; everything below it (AQL/Cypher,
gharial/Bolt, ANN functions, REST/Bolt transport) moves into per-backend impls.

```
            CLI / dispatch / methodology / orchestration   ← unchanged, backend-agnostic
                              │
                   ┌──────────▼──────────┐
                   │   GraphBackend trait │   ← the new seam (closed operations only)
                   └──────────┬──────────┘
            ┌─────────────────┴─────────────────┐
   ┌────────▼─────────┐               ┌──────────▼─────────┐
   │  ArangoBackend   │               │   Neo4jBackend     │
   │ HTTP/Unix socket │               │  Bolt + Cypher     │
   │ AQL, gharial,    │               │  native vector idx │
   │ APPROX_NEAR_*    │               │  MERGE/SET         │
   └──────────────────┘               └────────────────────┘
```

## 2. Architecture

- **New module:** `hades-core/src/backend/` with `mod.rs` (the trait + shared types),
  `arango/` (the existing `db/*` code moved/wrapped), and later `neo4j/`.
- **Selection:** `HadesConfig` gains `backend: { kind: "arangodb" | "neo4j", … }`. A
  factory `Backend::from_config(&config) -> Arc<dyn GraphBackend>` replaces direct
  `ArangoPool::from_config`. Per-deployment (one backend per process; PRD D4/NG2).
- **Callers change type, not logic:** functions taking `&ArangoPool` take
  `&dyn GraphBackend` (or `&Arc<dyn GraphBackend>`). The big ones:
  `graph::load`, `export_embeddings`, `schema_apply::apply`, `db_search`,
  `codebase_ingest` write phase, `dispatch` CRUD.

## 3. The `GraphBackend` trait (operation set)

Grouped by contract requirement (PRD §5.1). Signatures illustrative; final shapes land in
P2.

```rust
#[async_trait]
pub trait GraphBackend: Send + Sync {
    // ── R2 graph primitives ────────────────────────────────
    async fn get_document(&self, col: &str, key: &str) -> Result<Option<Value>>;
    async fn upsert_documents(&self, col: &str, docs: &[Value]) -> Result<WriteStats>; // by stable key
    async fn delete_document(&self, col: &str, key: &str) -> Result<()>;
    async fn create_collection(&self, name: &str, kind: CollectionKind) -> Result<()>;
    async fn list_collections(&self, exclude_system: bool) -> Result<Vec<CollectionInfo>>;
    async fn truncate_collection(&self, name: &str) -> Result<()>;
    async fn drop_collection(&self, name: &str) -> Result<()>;
    async fn count(&self, col: &str) -> Result<u64>;

    // ── R1 structural-embedding lifecycle (LEAD) ───────────
    /// Stream (from_id, to_id) for an edge collection. (loader.rs:207)
    async fn scan_edges(&self, col: &str) -> Result<EdgeStream>;
    /// Fetch node-level embeddings by key. (loader.rs:283)
    async fn node_embeddings(&self, col: &str, keys: &[String]) -> Result<Vec<(String, Vec<f32>)>>;
    /// Fetch chunk embeddings grouped by file_key for code-node pooling. (loader.rs:455)
    async fn chunk_embeddings_by_file(&self, col: &str, file_keys: &[String])
        -> Result<Vec<(String, Vec<f32>)>>;
    /// Bulk write structural_embedding onto nodes. (export.rs:187)
    async fn write_structural_embeddings(&self, col: &str, rows: &[(String, Vec<f32>)])
        -> Result<WriteStats>;
    /// R1.4 — keys of nodes new/changed since last embedding (delta for inductive update).
    async fn nodes_needing_embedding(&self, col: &str) -> Result<Vec<String>>;

    // ── R3 search ──────────────────────────────────────────
    async fn vector_knn(&self, profile: &SearchProfile, q: &[f32], k: usize, metric: VectorMetric)
        -> Result<Vec<SimilarityHit>>;
    async fn create_vector_index(&self, col: &str, field: &str, dim: u32, metric: VectorMetric)
        -> Result<()>;
    // fulltext + the hybrid/structural fusion stay above the trait (db_search.rs), calling these.

    // ── R4 traversal ───────────────────────────────────────
    async fn neighbors(&self, start: &NodeRef, edge_types: &[String], depth: u32) -> Result<Vec<NodeRef>>;
    async fn shortest_path(&self, from: &NodeRef, to: &NodeRef, edge_types: &[String]) -> Result<Option<Path>>;

    // ── R5 schema-as-data ──────────────────────────────────
    async fn ensure_named_graph(&self, def: &NamedGraphDef) -> Result<()>; // gharial / projection / no-op
    async fn load_schema_documents(&self) -> Result<Vec<Value>>;           // FOR d IN hades_schema (runtime_schema.rs:191)

    // ── R6 escape hatch (NOT model-facing) ─────────────────
    async fn raw_read(&self, query: &str, binds: &Value) -> Result<QueryResult>; // db aql / db cypher
}
```

**Design notes**
- `query()`-of-raw-AQL is *removed* from the model-facing path; `raw_read` is the
  operator-only escape hatch (`db aql` on Arango, `db cypher` on Neo4j). Mutating raw
  queries stay rejected (the existing invariant).
- Identity: the trait speaks **stable keys** + (collection, key) `NodeRef`, never raw
  `_id`/`_from`/`_to`. The Arango impl maps `NodeRef → "col/key"`; the Neo4j impl maps it
  to a label + key property + a unique constraint.
- Field-name conventions (`file_key`, `parent_key`, `embedding`, `structural_embedding`)
  live in a `BackendSchema`/profile struct (from `db/collections.rs`), not hardcoded in
  queries — so the loader stops embedding `_from`/`_to` literally (today: `loader.rs:207`).

## 4. Data-model mapping

| HADES concept | ArangoDB | Neo4j |
|---|---|---|
| Node collection | document collection | node **label** |
| Edge collection | edge collection | relationship **type** |
| Stable id | `_key` | `key` property + unique constraint per label |
| Edge endpoints | `_from`/`_to` (`"col/key"`) | `(:Label {key})-[:TYPE]->(:Label {key})` |
| Document body | native JSON (nested ok) | node properties (scalars + lists) |
| Nested object (e.g. `principles[]`) | native | **strategy needed** — JSON-string prop *or* sub-nodes (PRD Q1; default: JSON-string for opaque blobs, sub-nodes for things we query) |
| `embedding` / `structural_embedding` | float array prop | `list<float>` prop + vector index |
| Named graph | gharial (`runtime_schema.rs:299`) | no registry; traversal is Cypher over rel-types (`ensure_named_graph` is a no-op/constraint-ensurer) |
| `hades_schema` (schema-as-data) | documents | `(:HadesSchema {schema_type, …})` nodes |

The collection *profiles* (`db/collections.rs`) are app-domain, not engine — they port
as-is and become the `BackendSchema` the trait consults for field names.

## 5. The structural-embedding lifecycle (lead requirement, in detail)

What stays backend-agnostic (do **not** touch): `IDMap` construction, `GraphData` tensor
allocation, the mean-pooling math (`loader.rs:444-503`), edge split / negative sampling /
safetensors (`hades-prefetch/*`), the gRPC training client (`training.rs`), the Python
service (`HadesHeteroSAGE`/`HadesRGCN`), the checkpoint architecture round-trip.

What each backend implements (the only coupled points):
1. **`scan_edges`** per `relation_order` collection → `(from_key, to_key)` → feeds IDMap.
   *(Arango: `FOR e IN @@col RETURN [e._from,e._to]`, loader.rs:207. Neo4j: `MATCH
   (a)-[:TYPE]->(b) RETURN a.key,b.key`.)*
2. **`node_embeddings`** + **`chunk_embeddings_by_file`** → feature assembly. *(Arango:
   loader.rs:283 / loader.rs:455. Neo4j: `MATCH (n:Label) WHERE n.key IN $keys RETURN
   n.key,n.embedding`.)*
3. **`write_structural_embeddings`** → write-back. *(Arango: `FOR u IN @updates UPDATE
   u._key …`, export.rs:187. Neo4j: `UNWIND $rows AS r MATCH (n:Label {key:r.key}) SET
   n.structural_embedding = r.emb`.)*
4. **`nodes_needing_embedding`** (R1.4, **new capability**) → the inductive delta.
   Reuse `codebase ingest`'s existing `symbol_hash`/content-hash + a
   `structural_embedding_of_hash` marker on the node: a node needs (re)embedding iff it
   has no `structural_embedding`, or its content hash ≠ the hash recorded when it was last
   embedded. `graph-embed update` then loads the full graph for context but only *writes*
   the delta (and, as an optimization, can forward-pass a neighborhood subgraph). This is
   the formerly-optional "PR 3b" promoted to a contract requirement.
5. **vector index** over `structural_embedding` for similarity / suspect-set.

Result: SAGE/RGCN "just work" on any backend that implements these five reads/writes;
the model, training loop, and checkpoint logic are untouched.

## 6. Vector / hybrid / structural search

- **`vector_knn`** is the only DB-specific primitive. Arango: `APPROX_NEAR_COSINE` etc.
  with brute-force fallback (`db/vector.rs:120-206`). Neo4j: native vector index
  (`db.index.vector.queryNodes`) with brute-force fallback. Contract specifies **kNN
  semantics + metric**, not the function.
- **Hybrid + structural rerank fusion** stays above the trait (`db_search.rs:118-149` —
  it already fetches vectors and does cosine in Rust; the blend `0.7*score +
  0.3*structural_cosine` is pure arithmetic). It calls `vector_knn` + a fulltext primitive
  and fuses HADES-side. No per-backend fusion code.

## 7. Schema

- `schema apply` orchestration (`schema_apply.rs`) is already AQL-free — it calls CRUD +
  one `ensure_named_graph`. So it ports by changing its pool type to `&dyn GraphBackend`.
- `ensure_named_graph`: Arango builds the gharial payload (`runtime_schema.rs:299-341`);
  Neo4j ensures uniqueness constraints + (optionally) a named GDS graph projection, else
  no-op (traversal is just Cypher).
- `load_schema_documents`: Arango `FOR d IN hades_schema RETURN d`; Neo4j `MATCH
  (d:HadesSchema) RETURN d`. The `SchemaMeta`/`RuntimeEdgeDef` deserialization is shared.

## 8. Transport

- **Arango:** keep the existing HTTP-over-Unix-socket client (`db/transport.rs`,
  `db/pool.rs`) — it already cleanly wraps requests; only `_api/*` paths are Arango-specific.
- **Neo4j:** Bolt protocol (the `neo4rs` crate, or the official driver). Local connection
  (`bolt://127.0.0.1:7687` or a local socket if supported) to satisfy NFR1 latency. Auth +
  TLS per Neo4j config (operator/Weaver-owned where relevant).

## 9. Migration tooling

`hades migrate --from <backend-cfg> --to <backend-cfg>`: read every collection + its
edges + `hades_schema` from source via the trait, write to target via the trait. Because
both sides are the same trait, this is backend-pair-agnostic. Validates by comparing
node/edge counts and re-running a suspect-set query on both. Read-only on source (NFR5).

## 10. Phasing & the de-risk milestone

- **P2 is the keystone:** move `db/*` + the graph I/O + search behind `GraphBackend` as
  `ArangoBackend`, change caller signatures, **prove zero behavior change** (existing
  tests + live smoke). At P2's end HADES is provably backend-agnostic with one backend —
  the riskiest 80% of the design is validated before a line of Neo4j is written.
- **P3/P4:** `Neo4jBackend` + the R1 lifecycle; conformance suite + end-to-end SAGE.
- **P5:** migration tool.

## 11. File-level change map (from the coupling audit)

| Area | Files | Action |
|---|---|---|
| Trait + factory | `backend/mod.rs` (new), `config/*` | add |
| Transport/pool/CRUD/query | `db/{transport,pool,crud,query,collections,index,cache,keys}.rs` | move under `backend/arango/`, implement trait |
| Vector | `db/vector.rs` | behind `vector_knn`; brute-force fallback shared |
| Graph loader | `graph/loader.rs` (esp. :207, :283, :387, :455) | replace AQL with trait calls; keep IDMap/pooling |
| Export | `graph/export.rs` (:187) | replace UPDATE-AQL with `write_structural_embeddings` |
| Schema | `schema_apply.rs`, `graph/runtime_schema.rs` (:191, :299) | pool→trait; `ensure_named_graph` per backend |
| Dispatch | `dispatch.rs` | CRUD/query via trait; `raw_read` escape hatch |
| Ingest write phase | `cli/.../codebase_ingest.rs` (:762-847) | writes via trait; `_key`/endpoints via `NodeRef` |
| Prefetch/training | `hades-prefetch/*`, `training.rs` | **no change** (already agnostic) |

## 12. Testing strategy

- **Behavior-preserving (P2):** the full existing `cargo test` + a live `bident_burn`
  smoke must pass unchanged.
- **Conformance suite:** a backend-parameterized test module exercising R1–R6 against a
  throwaway DB on each backend (the same suite proves Arango and Neo4j). R1 (the lifecycle
  incl. the delta query) is the centerpiece.
- **End-to-end SAGE (P4):** repeat the bident_burn validation (train hetero_sage →
  checkpoint records arch → inductive `update` writes only the delta → embeddings land →
  structural-similarity query sane) on Neo4j.
- **Migration (P5):** round-trip a graph, assert count + suspect-set parity.

## 13. Risks & mitigations

- **Nested docs (RK2):** §4 representation strategy; decide per-field in P2 when we see the
  real document shapes.
- **Vector divergence (RK3):** contract is kNN-semantics; brute-force floor guarantees
  correctness even if a backend lacks ANN.
- **Trait churn:** P2 will reshape signatures as real callers are threaded through; treat
  the trait in §3 as provisional until P2 lands.
- **Highest-risk coupling (audit):** `graph/loader` edge/embedding reads + `db/vector` —
  these get the most design attention and the first conformance tests.

## 14. Open questions

Carried from the PRD: Q1 nested-doc representation, Q2 delta mechanism (lean: reuse
`symbol_hash`), Q3 Neo4j native-index vs GDS, Q4 HADES-migrates-or-not, Q5 selection
granularity. Resolve Q1/Q2 during P2 (they shape the trait); Q3 during P3.
