# PRD — Pluggable Graph Backend for HADES

**Status:** Draft / RFC (not ratified — see [Decisions](#9-decisions-assumed--confirm) / [Open questions](#10-open-questions))
**Date:** 2026-06-06
**Scope:** HADES-Burn — abstract the data layer behind a `GraphBackend` trait so HADES can run on more than one graph database (ArangoDB today, Neo4j next), with the structural-embedding (inductive) lifecycle as the lead requirement.
**Author:** Todd Bucy + Claude (HADES-Burn session)
**Companion:** [pluggable-graph-backend-spec.md](pluggable-graph-backend-spec.md) (the *how*)

---

## 1. Summary

HADES is a context-engineering tool: it builds an axiom-gated knowledge graph that
**constrains model behavior during code generation** (the immune-system graph of
[graph-methodology.md](../graph-methodology.md)). Today it is wired to exactly one
database, ArangoDB. This PRD proposes decoupling HADES from any single graph
database behind a `GraphBackend` trait, so the project graph can live on whichever
backend best fits its growth, security, and licensing profile — without rewriting
the tool. ArangoDB remains first-class; **Neo4j is the first additional backend.**

The non-negotiable, highest-priority capability a backend must provide is the
**inductive structural-embedding lifecycle** — train once, then embed newly-ingested
nodes by forward pass without retraining — because the graph exists to keep a
*growing* codebase's structural signal current for context management.

## 2. Background & motivation

- **The purpose is context management.** The graph constrains generation; its value
  decays if its structural signal lags the code. Codebases grow continuously, so the
  embedding model must be **inductive** (SAGE), not transductive (RGCN). This is why
  SAGE support precedes RGCN in the backend contract — RGCN is a strict subset of the
  inductive lifecycle (train + full re-embed, no delta).
- **The licensing trigger.** From ArangoDB 3.12.5 the Community Edition unlocks all
  Enterprise features but under a new Community License: **100 GiB/cluster cap,
  internal/non-commercial use only, no embedding/OEM/SaaS** (see the licensing
  investigation in the session record). We are at ~52 GB today on a growing graph.
  The current 3.12.4.3 instance can run indefinitely (single node, unserved,
  pre-3.12.5 license, no cap), but tying HADES's *future* to one vendor's licensing
  weather is a strategic risk for a tool whose whole point is reusability.
- **Why this is tractable now.** HADES already decided models never write raw query
  language — it exposes *closed, structured operations* that translate internally
  ([no-raw-aql invariant](../../README.md)). That seam, built for safety, is exactly
  the abstraction boundary needed for multiple backends. The training subsystem is
  *already* backend-agnostic (safetensors IPC + a DB-unaware Python service). A
  coupling audit puts the codebase at **~65% already portable**; the coupled 35% is
  concentrated in graph I/O, vector search, schema, and transport.

## 3. Goals

1. **G1 — Backend abstraction.** A `GraphBackend` trait that all DB-touching code
   calls through; no AQL/Cypher above the trait.
2. **G2 — ArangoDB preserved.** ArangoDB remains a fully-supported backend; the
   refactor behind the trait is **behavior-preserving** and proven against the live DB.
3. **G3 — Neo4j backend.** A second, production-usable backend on Neo4j (Bolt + Cypher
   + native vector index).
4. **G4 — Inductive structural embeddings on every backend.** The full SAGE lifecycle
   (train → checkpoint → inductive `update` of new/changed nodes) works identically on
   each backend; RGCN falls out as the subset.
5. **G5 — Documented minimum-backend contract.** A precise, testable contract so
   "supports HADES" is an explicit bar, and a conformance test suite validates any
   backend against it.
6. **G6 — Migration tooling.** A backend-to-backend dump/load so a graph can move
   between backends without bespoke scripts.

## 4. Non-goals

- **NG1.** Tearing down or upgrading the existing ArangoDB instance. It stays as-is.
- **NG2.** Running two backends *simultaneously in one process* / cross-backend joins.
  (Selection is per-deployment; see [Open questions](#10-open-questions).)
- **NG3.** Exposing a raw cross-DB query language to models. The closed-operation
  invariant stays; a per-backend raw escape hatch (`db aql` / `db cypher`) is allowed
  but is not part of the model-facing surface.
- **NG4.** Solving WeaverTools' database choice. Weaver is decoupled and owns its own
  store; this PRD is about HADES's project graphs. (The trait may benefit Weaver later,
  but that's their call.)
- **NG5.** A generic ORM. The trait is scoped to HADES's operations, not arbitrary
  graph workloads.

## 5. Requirements

### 5.1 The minimum-backend contract (the heart)

A conforming `GraphBackend` MUST provide, in priority order:

**R1 — Structural-embedding lifecycle, inductive-first (LEAD REQUIREMENT).**
  - R1.1 **Graph read for training:** bulk-read nodes, their feature vectors, and
    edges scoped by `relation_order`, into the flat tensor contract
    (`node_features`, `node_collections`, `edge_src/dst`, `edge_type`).
  - R1.2 **Feature assembly:** fetch the embeddings needed to build node features,
    including the per-file chunk-embedding set used for mean-pooling code-node features.
    (The pooling math is HADES-side and backend-agnostic; only the fetch is per-backend.)
  - R1.3 **Embedding write-back:** bulk-upsert `structural_embedding` onto nodes.
  - R1.4 **Delta query (incremental/inductive):** answer *"which nodes are new or whose
    content changed since they were last embedded?"* so `update` embeds only the delta,
    not the whole graph. This is what makes "inductive" real for a continuously-ingested
    codebase.
  - R1.5 **Vector index over `structural_embedding`** for structural-similarity /
    suspect-set queries (the queries that actually feed context).
  - *(RGCN/transductive = R1 without R1.4.)*

**R2 — Graph primitives.** Typed nodes (≈ document collections / labels) and typed
  edges/relationships with from/to; **upsert by stable key** (the `_key` = normalized
  path contract); property/document storage on nodes; create/list/truncate/drop
  collections.

**R3 — Search.** kNN vector search + fulltext, behind the existing `db query`
  hybrid/rerank operations. The fusion logic stays HADES-side; the kNN + text primitives
  are per-backend.

**R4 — Traversal.** Traverse a named set of edge types (the named-graph operations);
  shortest-path / neighborhood as used by HADES.

**R5 — Schema-as-data.** Store and load the `hades_schema` records (schema_meta,
  edge_definition, named_graph) — including `relation_order`, `feature_dim`, `model_type`.

**R6 — Batch + transactions.** Bulk insert/update; atomic multi-write where HADES needs
  consistency (e.g. ingest of a file's nodes+edges).

### 5.2 Functional requirements

- **FR1.** Every existing `hades` CLI command that touches data works on both backends
  (or fails with a clear "not supported on backend X" for any genuinely backend-specific
  command, of which there should be ~none in the model-facing set).
- **FR2.** `codebase ingest` writes nodes/edges/embeddings through the trait; the
  `_key`/identity contract is honored on each backend (mapped, not assumed).
- **FR3.** `graph-embed train` / `update` run the full lifecycle (R1) on either backend;
  the checkpoint's architecture round-trip (rgcn|hetero_sage) is unchanged (it's above
  the trait).
- **FR4.** `schema apply` provisions collections/edges/named-graph/`model_type` on either
  backend; validation (incl. unknown `model_type`) is backend-agnostic.
- **FR5.** `db query` (vector/hybrid/structural) returns equivalent ranked results on
  either backend (allowing for ANN-implementation differences).

### 5.3 Non-functional requirements

- **NFR1 — Native, low-latency.** No container layer. ArangoDB via its existing Unix
  socket; Neo4j via Bolt (local). Latency parity with today is a hard requirement
  (the whole IPC design is latency-driven).
- **NFR2 — Licensing insulation.** No HADES code may depend on a backend feature that
  forces a specific paid/capped license. Backend-specific licensing is the operator's
  concern; HADES must run on at least one genuinely-unencumbered configuration.
- **NFR3 — Behavior-preserving refactor.** Phase 2 (ArangoDB behind the trait) must pass
  the existing test suite + a live smoke against the current DB with **zero behavior
  change**.
- **NFR4 — Conformance-tested.** A backend conformance suite (R1–R6) runs against any
  backend; a backend is "supported" only when it passes.
- **NFR5 — Production data sacrosanct.** Migration/dump tooling is read-only on the
  source unless explicitly writing to a designated target; never mutate a production DB
  as a side effect.

### 5.4 Invariants preserved (must not regress)

- Models never write raw query language (closed operations only).
- Agentic vs human-UI split (project graphs are ingested; Persephone kanban is not).
- Data-is-sacrosanct ACL posture; collection-scoped destruction only, no whole-DB drop
  from the CLI.

## 6. Success criteria

- **SC1.** `cargo test` green + live smoke unchanged after Phase 2 (ArangoBackend) — the
  abstraction is proven *before* Neo4j exists.
- **SC2.** The conformance suite passes for ArangoBackend **and** Neo4jBackend.
- **SC3.** A real codebase graph is ingested, trained (`hetero_sage`), and inductively
  `update`d on Neo4j, with `structural_embedding` landing on nodes and structural-
  similarity queries returning sane results — the same end-to-end validation we ran on
  ArangoDB/bident_burn.
- **SC4.** A graph migrates ArangoDB→Neo4j via the tool with node/edge/embedding counts
  matching and suspect-set queries equivalent.
- **SC5.** No latency regression on the ArangoDB path.

## 7. Phasing / milestones

1. **P1 — Contract + trait design** (this PRD + Spec, ratified).
2. **P2 — Refactor ArangoDB behind `GraphBackend`** *(the de-risk milestone — value lands
   here even with no second backend; HADES becomes provably backend-agnostic).*
3. **P3 — Neo4jBackend** (Bolt + Cypher + native vector; passes conformance).
4. **P4 — Structural-embedding lifecycle on Neo4j** (R1 incl. the delta query) +
   end-to-end SAGE validation.
5. **P5 — Migration tooling** + ArangoDB↔Neo4j validation.
6. **P6 — Docs/skill sweep** (the methodology + `hades` skill describe backend choice).

Rough order-of-magnitude (from the coupling audit): P2 ≈ 1–2 weeks; P3–P4 ≈ 5–8 weeks for
a production Neo4j backend; P5 ≈ 1 week. Estimates firm up after P2.

## 8. Risks

- **RK1 — Leaky abstraction / lowest-common-denominator.** Mitigation: scope the trait to
  HADES operations (not arbitrary graph), let each backend implement idiomatically, and
  gate on the conformance suite.
- **RK2 — Nested-document mismatch.** ArangoDB rich JSON docs (e.g. axioms with nested
  `principles[]`) don't map 1:1 to Neo4j's flatter property model. Mitigation: a defined
  representation strategy (JSON-string property vs decomposed sub-nodes) in the Spec.
- **RK3 — Vector/ANN divergence.** ArangoDB `APPROX_NEAR_*` vs Neo4j native vector index /
  GDS. Mitigation: contract specifies kNN semantics, not implementation; brute-force
  fallback as the floor.
- **RK4 — Effort underestimate.** Graph I/O + vector are the highest-coupling areas.
  Mitigation: P2 measures the real surface before committing to P3.
- **RK5 — Neo4j's own licensing.** Neo4j Community gates RBAC/audit/encryption behind paid
  Enterprise; do not assume Neo4j solves the security-features-for-free goal (it likely
  doesn't). Mitigation: this PRD targets *backend portability*, not "Neo4j as the secure
  free option"; that claim needs separate verification.

## 9. Decisions (assumed — confirm)

- **D1.** General `GraphBackend` trait (N backends), Neo4j as the first concrete second
  backend — not a Neo4j-only fork.
- **D2.** Work happens **in this repo** behind the trait, not in a parallel repo.
- **D3.** ArangoDB stays supported and is the default; whether HADES's *own* project data
  migrates to Neo4j is a later, separate operator decision (the tool enables it).
- **D4.** Backend selected **per-deployment** via config; per-database-simultaneous is out
  of scope (NG2).
- **D5.** Native only (no Docker), per NFR1.

## 10. Open questions

- **Q1.** Nested-document representation on property-graph backends — JSON-string vs
  sub-node decomposition? (Affects axioms, schema docs.)
- **Q2.** Delta-query mechanism — content-hash, ingest timestamp, or a "dirty" flag on
  nodes? (`codebase ingest` already computes a `symbol_hash` for skip-unchanged — reuse it.)
- **Q3.** Neo4j vector: native index vs GDS `knn`/similarity — which is the floor we
  require, and does it satisfy R1.5 at our scale?
- **Q4.** Does HADES ultimately migrate project graphs to Neo4j, or stay ArangoDB-primary
  with Neo4j optional? (Drives P5 priority.)
- **Q5.** Selection granularity — confirm per-deployment (D4) is sufficient, or is
  per-database routing eventually needed?

## 11. Connects to

- [graph-methodology.md](../graph-methodology.md) — the purpose (context management; the
  immune-system graph) and the **ratification rule**: this PRD is itself an artifact in
  the `Issue → PRD → Spec → code` chain and is **not ratified until it proves whole**.
- [declarative-schema.md](../declarative-schema.md) — `relation_order` / `feature_dim` /
  `model_type`, which the contract (R5) must carry across backends.
- [codebase-graph-ontology.md](../codebase-graph-ontology.md) — the `codebase_*`
  collections + `_key` contract the trait must honor.
- Session record — the ArangoDB 3.12.5 licensing investigation that motivated this.
